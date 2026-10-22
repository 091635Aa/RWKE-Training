# -*- coding: utf-8 -*-
"""
自主训练驱动器（spec: <workspace>/.trae/specs/finetune-emocompanion-lora Task 5）
==========================================================================

无人值守"数据 → 训练 → 评估 → 迭代"闭环，全部行为由 系统配置.json 的 微调节 驱动：

    0) 配置加载 + 训练数据五产物齐备检查（缺失时调用 训练数据.py 构建，确定性可复现）；
    1) 轮次循环（有界）：
       a. 训练：微调训练器（lr 取当前档位）→ 非首轮 / --恢复 时 恢复checkpoint
          （优化器状态与 step 连续；恢复后优化器 lr 对齐当前档位）→ 加载数据 → 训练；
       b. 评估：每轮重建 RWKE引擎（适配器已变更必须重挂）→ 评估集逐条剥离聊天模板得
          裸用户消息（引擎.生成 内部自动套 "User: {消息}\\n\\nAssistant:"）→ 固定种子生成
          → 回复落盘 → 一致性评估.评估数据集 六维打分 → 合成分均值；
       c. 落盘：results/微调_回复_<模型>_r{n}.jsonl、results/微调_轮次_<模型>_r{n}.json、
          results/自主训练_状态_<模型>.json（每轮更新，--恢复 续跑依据）；
       d. 判定：合成分均值 >= 一致性达标线 → 达标即停（成功路径）；否则按顺序迭代：
          ①续训（同 lr 续 epoch）②lr 降档（取下一档位，从 checkpoint 继续）
          ③数据扩充（调 训练数据.py 按样本上限重建后重训）④预算耗尽 → 如实停止。
    2) 终局报告：results/微调认证_<模型>.json（达标判定/最佳合成分/逐维明细/迭代历史/结论）
       与 docs/微调报告.md（人可读：结论、轮次对比表；未达标时含差距诊断——最低维度与
       建议，数字全部来自实测落盘结果，不虚报）。

快验模式（--快验）读 微调.快验 节：小预算闭环（每轮步数上限 / 单轮 epochs / 评估生成缩短 /
训练集截断前 N 条）。快验下训练集被截断，数据扩充对训练集无增益，自动跳过扩充动作
（不消耗预算），日志中显式说明。

中断续跑：训练阶段 Ctrl+C 由 微调.py 捕获并保存 checkpoint（摘要.中断=True）后优雅返回；
其他阶段 Ctrl+C 或进程被杀时状态文件保证续接。重跑加 --恢复（可带 checkpoint 路径）：
轮次 / lr档位 / 已用预算 / 待执行动作 全部续接，不重复扣预算。评估阶段中断则续跑时
从 checkpoint 继续训练并重新评估（训练产物以 checkpoint 为准，单调推进）。

用法
----
    python 自主训练.py --模型 RWKV7-World-0.4B --快验 --达标线 40    # 正路径：达标即停
    python 自主训练.py --模型 RWKV7-World-0.4B --快验 --达标线 99    # 诚实路径：预算耗尽未达标
    python 自主训练.py --模型 RWKV7-World-0.4B --快验 --达标线 99 --恢复   # 中断后续跑
    python 自主训练.py --模型 RWKV7-World-0.4B                       # 正式闭环（全量配置）
    python 自主训练.py --模型 RWKV7-World-0.4B --日记状态            # 评估时挂载EmoCompanion日记 State

退出码：0 闭环正常结束（达标 / 预算耗尽均属正常终局，看报告结论）；130 Ctrl+C 中断；
2 显存守卫拦截；1 其他错误。
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_RWKE_DIR = Path(__file__).resolve().parent
if str(_RWKE_DIR) not in sys.path:
    sys.path.insert(0, str(_RWKE_DIR))

import 一致性评估  # noqa: E402（纯 stdlib 规则评分器，导入轻量）

# 微调.六维权重（长键）→ 一致性评估 内部权重键（短键）映射
维度键映射 = {
    "锚点命中率": "锚点",
    "口癖率": "口癖",
    "五维人设符合度": "五维",
    "情感基调匹配": "情感",
    "身份一致性": "身份",
    "流畅性守卫": "流畅",
}
维度名列表 = list(维度键映射)

# 微调节必需键（缺失即拒绝启动：全部行为配置驱动，不允许半配置运行）
必需配置键 = [
    "基座默认", "rank", "alpha", "学习率档位", "epoch上限", "批大小", "梯度累积",
    "序列长度", "数据路径", "一致性达标线", "六维权重", "迭代预算", "评估生成参数", "快验",
]

# 差距诊断：维度 → 建议（未达标时指出最低维度并给出可执行建议）
维度建议映射 = {
    "锚点命中率": "锚点词（萌妹/天鹅之梦/ViewerA/开播第十五天/EmoCompanion的直播间）未稳定出现："
                 "增补人设问答与锚点场景训练样本，或提高训练轮数/样本量",
    "口癖率": "啊/那个 频次偏离设定区间（完全缺失或过量）：检查训练目标的口癖分布，"
             "保持每条目标至少一处自然口癖",
    "五维人设符合度": "温暖词/戏谑能量不足：增补情感化对话与日记文风样本",
    "情感基调匹配": "回复情感词与期望情感不同向：扩充对应情感标签的样本覆盖",
    "身份一致性": "出现出戏短语或未自称EmoCompanion/EmoCompanion：强化身份问答样本（该维度未满分时优先修复）",
    "流畅性守卫": "生成存在硬伤（复读/乱码/过短）：降低评估温度、缩短生成长度或排查训练数据质量",
}


# ======================================================================
# 通用工具
# ======================================================================

def 打印分隔(标题: str) -> None:
    print("\n" + "=" * 60)
    print(f" {标题}")
    print("=" * 60, flush=True)


def 写json(路径: Path, 数据) -> None:
    路径.parent.mkdir(parents=True, exist_ok=True)
    路径.write_text(json.dumps(数据, ensure_ascii=False, indent=2), encoding="utf-8")


def 读系统配置() -> dict:
    路径 = _RWKE_DIR / "系统配置.json"
    with open(路径, "r", encoding="utf-8") as f:
        return json.load(f)


def 释放显存() -> None:
    """训练器 / 引擎实例释放后清理 CUDA 缓存，防止跨阶段显存叠加触发显存守卫。"""
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def 禁用jit_gpu融合() -> None:
    """
    可选禁用 torch.jit 的 GPU 算子融合（NNC/nvrtc）：本机在系统内存紧张时出现过
    "ptxas fatal: Memory allocation failure" → "nvrtc compilation failed"
    （scripted TMix/CMix 反向在运行时编译融合内核时失败）。

    默认不禁用（融合路径快约 3-5 倍，内核派发开销低）；设环境变量
    自主训练_禁用JIT融合=1 时禁用，回退逐算子执行：数值路径不变、稳定但慢。
    进程级设置，对 微调训练前向 与 引擎生成 同时生效。
    """
    import os
    if str(os.environ.get("自主训练_禁用JIT融合", "")).strip() not in ("1", "true", "True"):
        print("[环境] torch.jit GPU 算子融合保持启用"
              "（如遇 ptxas/nvrtc 运行时编译失败：设 自主训练_禁用JIT融合=1 后 --恢复 续跑）")
        return
    try:
        import torch
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        print("[环境] 已禁用 torch.jit GPU 算子融合（逐算子执行，稳定但慢）")
    except Exception as e:
        print(f"[环境] 禁用 jit GPU 融合未生效（{e}），按默认配置继续")


def 剥离模板(提示: str) -> str:
    """评估集提示（"User: {消息}\\n\\nAssistant:"）→ 裸用户消息。

    引擎.生成 内部会重新套聊天模板，因此传给生成接口的必须是裸消息：
    取 "User: " 之后、"\n\nAssistant:" 之前的文本；无模板时原样返回（仅去首尾空白）。
    """
    文本 = str(提示)
    if 文本.startswith("User:"):
        文本 = 文本[len("User:"):]
        if 文本.startswith(" "):
            文本 = 文本[1:]
    结束 = 文本.find("\n\nAssistant:")
    if 结束 >= 0:
        文本 = 文本[:结束]
    return 文本.strip()


def 重建训练数据(样本上限: int) -> dict:
    """调用 训练数据.py 的 构建() 重建训练资产（确定性固定种子，重跑逐字节一致）。"""
    import 训练数据
    return 训练数据.构建(
        训练数据._打标目录默认, 训练数据._种子日记默认, 训练数据._输出目录默认,
        样本上限=int(样本上限), 精选篇数=80,
    )


def 差距诊断(最佳记录: dict, 达标线: float) -> dict:
    """未达标时的差距诊断：差距分值 + 各维均分排序 + 最低维度与可执行建议。"""
    汇总 = 最佳记录["逐维明细"]
    维度分 = {维: float(汇总[f"{维}_均值"]) for 维 in 维度名列表}
    排序 = sorted(维度分.items(), key=lambda kv: kv[1])
    最低维, 最低分 = 排序[0]
    差距 = round(达标线 - float(最佳记录["合成分"]), 1)
    建议 = [维度建议映射[最低维]]
    if len(排序) > 1 and 排序[1][1] < 60.0:
        建议.append(维度建议映射[排序[1][0]])
    return {
        "差距": 差距,
        "最佳轮次": 最佳记录["轮次"],
        "各维均分排序": [{"维度": k, "均分": v} for k, v in 排序],
        "最低维度": 最低维,
        "最低维度均分": 最低分,
        "建议": 建议,
        "通用建议": "可提高训练预算（epochs/步数/样本量）后加 --恢复 续训，"
                   "或检查评估生成参数与训练数据对评估集的覆盖度",
    }


# ======================================================================
# 驱动器
# ======================================================================

class 自主训练驱动器:
    """无人值守闭环驱动器：状态机 + 轮次执行 + 预算判定 + 终局报告。"""

    def __init__(
        self,
        模型: str,
        微调配置: dict,
        快验: bool,
        恢复,
        达标线覆盖,
        输出目录: Path,
        日记状态参数,
    ) -> None:
        self.模型 = 模型
        self.微调配置 = 微调配置
        self.快验 = bool(快验)
        self.恢复 = 恢复
        self.输出目录 = Path(输出目录)
        self.输出目录.mkdir(parents=True, exist_ok=True)
        self.状态路径 = self.输出目录 / f"自主训练_状态_{模型}.json"

        # ---- 达标线解析：命令行覆盖 > 状态续接 > 配置 默认 ----
        self.达标线 = (
            float(达标线覆盖) if 达标线覆盖 is not None
            else float(微调配置["一致性达标线"])
        )

        # ---- 评估权重：微调.六维权重（长键）→ 短键，校验通过才显式下发 ----
        权重长键 = 微调配置.get("六维权重", {})
        if (
            set(权重长键) == set(维度键映射)
            and abs(sum(float(v) for v in 权重长键.values()) - 1.0) < 1e-6
        ):
            self.权重短键 = {维度键映射[k]: float(v) for k, v in 权重长键.items()}
            print("[权重] 微调.六维权重 校验通过，已下发一致性评估")
        else:
            self.权重短键 = None
            print("[权重] 微调.六维权重 缺键或权重和不为 1，交由一致性评估内置默认权重")

        # ---- 日记 State（可选用）：--日记状态 无值取 states/EmoCompanion_状态_{模型}.pth ----
        self.日记状态路径 = self._解析日记状态(日记状态参数)

        # ---- 状态加载 / 初始化（--恢复 读取续跑，否则重置新闭环）----
        self.状态 = self._加载状态()
        self.状态["模型"] = self.模型
        self.状态["达标线"] = self.达标线
        self.状态["快验"] = self.快验
        self.保存状态()

    # ──────────────────────────────────────────────
    # 初始化辅助
    # ──────────────────────────────────────────────

    def _解析日记状态(self, 参数) -> Path | None:
        if not 参数:
            return None
        候选 = (
            _RWKE_DIR / "states" / f"EmoCompanion_状态_{self.模型}.pth"
            if 参数 == "auto" else Path(参数)
        )
        if 候选.exists():
            print(f"[日记状态] 评估时将挂载：{候选}")
            return 候选
        print(f"[日记状态] 指定状态文件不存在（{候选}），按无日记状态继续")
        return None

    def _加载状态(self) -> dict:
        if self.恢复 and self.状态路径.exists():
            状态 = json.loads(self.状态路径.read_text(encoding="utf-8"))
            if 状态.get("模型") != self.模型:
                raise ValueError(
                    f"状态文件模型不符（防错挂）：文件为 '{状态.get('模型')}'，当前为 '{self.模型}'"
                )
            if 状态.get("达标线") != self.达标线:
                print(f"[续跑] 达标线沿用命令行覆盖值 {self.达标线}（状态记录 {状态.get('达标线')}）")
            if "快验" in 状态 and bool(状态["快验"]) != self.快验:
                print(f"[续跑] 状态记录 快验={状态['快验']} 与本次命令行不一致，沿用状态设置")
                self.快验 = bool(状态["快验"])
            状态.setdefault("轮次", 0)
            状态.setdefault("lr档位索引", 0)
            状态.setdefault("已用", {"续训": 0, "降档": 0, "扩充": 0})
            状态.setdefault("各轮合成分", [])
            状态.setdefault("历史", [])
            状态.setdefault("待执行动作", None)
            print(
                f"[续跑] 已读取状态：轮次={状态['轮次']} lr档位索引={状态['lr档位索引']} "
                f"已用={状态['已用']} 各轮合成分={状态['各轮合成分']}"
            )
            return 状态
        if self.恢复:
            print(
                f"[续跑] 未找到状态文件（{self.状态路径}）：按第 1 轮开始，"
                f"checkpoint 存在则续接适配器与优化器状态"
            )
        if self.状态路径.exists():
            print(f"[状态] 存在历史状态文件且未指定 --恢复：重置为新闭环")
        return {
            "模型": self.模型,
            "达标线": self.达标线,
            "快验": self.快验,
            "轮次": 0,
            "lr档位索引": 0,
            "已用": {"续训": 0, "降档": 0, "扩充": 0},
            "各轮合成分": [],
            "历史": [],
            "待执行动作": None,
        }

    # ──────────────────────────────────────────────
    # 数据 / 状态
    # ──────────────────────────────────────────────

    def 检查数据(self) -> None:
        """训练数据五产物齐备检查（4 个配置路径 + 构建报告），缺失时构建。"""
        数据路径 = self.微调配置["数据路径"]
        产物 = {键: _RWKE_DIR / 相对 for 键, 相对 in 数据路径.items()}
        构建报告 = 产物["评估集"].parent / "构建报告.json"
        缺失 = [str(p) for p in list(产物.values()) + [构建报告] if not p.exists()]
        if not 缺失:
            print("[数据] 训练数据五产物齐备（训练集/验证集/评估集/日记精选/构建报告）")
            return
        print(f"[数据] 训练数据产物缺失：{缺失}")
        print("[数据] 调用 训练数据.py 构建() 重建（确定性，固定种子）...")
        报告 = 重建训练数据(1200)
        print(
            f"[数据] 构建完成：train={报告.get('train数')} val={报告.get('val数')} "
            f"评估集={报告.get('评估集条数')}条 日记={报告.get('日记总数')}篇"
        )

    def 准备训练集路径(self) -> Path:
        """正式模式返回配置训练集；快验模式截前 快验.训练样本数 条写临时截断文件。"""
        训练集 = _RWKE_DIR / self.微调配置["数据路径"]["训练集"]
        if not self.快验:
            return 训练集
        上限 = int(self.微调配置["快验"]["训练样本数"])
        行 = [l for l in 训练集.read_text(encoding="utf-8").splitlines() if l.strip()]
        截断 = 行[:上限]
        截断文件 = self.输出目录 / f"快验_训练集截断_{self.模型}.jsonl"
        截断文件.write_text("\n".join(截断) + ("\n" if 截断 else ""), encoding="utf-8")
        print(f"[快验] 训练集截断：{len(截断)}/{len(行)} 条 → {截断文件.name}")
        return 截断文件

    def 保存状态(self) -> None:
        写json(self.状态路径, self.状态)

    def 打印曲线(self) -> None:
        分数 = self.状态["各轮合成分"]
        曲线 = " -> ".join(f"r{i + 1}={s}" for i, s in enumerate(分数))
        print(f"[合成分曲线] {曲线}（达标线 {self.达标线}）", flush=True)

    # ──────────────────────────────────────────────
    # 迭代预算
    # ──────────────────────────────────────────────

    def 选择动作(self):
        """按 续训 → 降档 → 扩充 顺序选下一迭代动作；预算耗尽返回 None。"""
        预算 = self.微调配置["迭代预算"]
        已用 = self.状态["已用"]
        档位数 = len(self.微调配置["学习率档位"])
        if int(已用["续训"]) < int(预算.get("续训轮次", 0)):
            return "续训"
        if (
            int(已用["降档"]) < int(预算.get("lr降档次数", 0))
            and int(self.状态["lr档位索引"]) + 1 < 档位数
        ):
            return "降档"
        if int(已用["扩充"]) < int(预算.get("数据扩充轮次", 0)):
            if self.快验:
                print("[迭代] 快验模式下训练集被截断，数据扩充对训练集无增益：跳过扩充（不消耗预算）")
            else:
                return "扩充"
        return None

    def 消耗预算(self, 动作: str) -> None:
        预算 = self.微调配置["迭代预算"]
        已用 = self.状态["已用"]
        if 动作 == "续训":
            已用["续训"] += 1
            print(f"[迭代] 合成分未达标 → 续训（{已用['续训']}/{预算.get('续训轮次')}，lr 不变继续 epoch）")
        elif 动作 == "降档":
            已用["降档"] += 1
            self.状态["lr档位索引"] = int(self.状态["lr档位索引"]) + 1
            新lr = self.微调配置["学习率档位"][int(self.状态["lr档位索引"])]
            print(
                f"[迭代] 合成分未达标 → lr 降档（{已用['降档']}/{预算.get('lr降档次数')}）"
                f"：切到档位 {self.状态['lr档位索引']} lr={新lr}，从 checkpoint 继续"
            )
        elif 动作 == "扩充":
            已用["扩充"] += 1
            print(f"[迭代] 合成分未达标 → 数据扩充（{已用['扩充']}/{预算.get('数据扩充轮次')}）后重训")

    # ──────────────────────────────────────────────
    # 单轮执行
    # ──────────────────────────────────────────────

    def 执行一轮(self, 轮次号: int, 动作: str):
        """执行一轮完整闭环（训练 → 生成 → 评估 → 落盘）；训练被 Ctrl+C 打断时返回 None。"""
        微调配置 = self.微调配置
        快验配置 = 微调配置["快验"]
        档位 = [float(x) for x in 微调配置["学习率档位"]]
        lr = 档位[int(self.状态["lr档位索引"])]
        本轮epochs = int(快验配置["epochs"]) if self.快验 else int(微调配置["epoch上限"])
        本轮上限步 = int(快验配置["上限步"]) if self.快验 else 0
        评估参数 = dict(微调配置["评估生成参数"])
        最大新token = int(快验配置["最大新Token"]) if self.快验 else int(评估参数["最大新Token"])

        打印分隔(
            f"第 {轮次号} 轮 | 动作：{动作} | lr={lr} | epochs={本轮epochs} | "
            f"步数上限={本轮上限步 if 本轮上限步 > 0 else '不限'} | 最大新Token={最大新token}"
        )

        # ---- 数据扩充动作：按预算样本上限重建训练资产（快验模式不会走到这里）----
        扩充说明 = None
        if 动作 == "扩充":
            样本上限 = int(微调配置["迭代预算"]["数据扩充样本上限"])
            print(f"[数据扩充] 调用 训练数据.py 构建（样本上限 {样本上限}）重建训练资产...")
            报告 = 重建训练数据(样本上限)
            扩充说明 = f"样本上限={样本上限}, train={报告.get('train数')}"
            print(f"[数据扩充] 重建完成：train={报告.get('train数')} val={报告.get('val数')}")

        # ---- 训练（训练器每轮重建，跨轮经 checkpoint 续接优化器状态与 step）----
        训练集路径 = self.准备训练集路径()
        import 微调 as 微调模块
        训练器 = 微调模块.微调训练器(
            self.模型,
            rank=int(微调配置["rank"]),
            alpha=int(微调配置["alpha"]),
            lr=lr,
            序列长度=int(微调配置["序列长度"]),
            累积=int(微调配置["梯度累积"]),
        )
        step起点 = 0
        if 轮次号 > 1 or self.恢复:
            恢复路径 = None if self.恢复 in (None, "auto") else str(self.恢复)
            try:
                训练器.恢复checkpoint(恢复路径)
            except FileNotFoundError:
                备选 = _RWKE_DIR / "适配器" / f"{self.模型}_lora_best.pt"
                if 轮次号 == 1 and 恢复路径 is None and 备选.exists():
                    print("[续接] 默认 checkpoint 不存在，回退最优 checkpoint 续接")
                    训练器.恢复checkpoint(str(备选))
                elif 轮次号 == 1:
                    print("[续接] 未找到任何 checkpoint：按全新适配器开始首轮训练")
                else:
                    raise
            step起点 = int(训练器.step)
            # 恢复后优化器 lr 为 checkpoint 旧值：换档（降档）时必须对齐当前档位
            if abs(float(训练器.优化器.param_groups[0]["lr"]) - lr) > 1e-15:
                训练器.优化器.param_groups[0]["lr"] = lr
                训练器.lr = lr
                print(f"[lr] checkpoint 恢复后对齐当前档位 lr={lr}")
        训练器.加载数据(str(训练集路径))
        目标步 = (step起点 + 本轮上限步) if 本轮上限步 > 0 else 0
        摘要 = 训练器.训练(epochs=本轮epochs, 上限步=目标步)
        del 训练器
        释放显存()
        if 摘要.get("中断"):
            print("[中断] 训练被 Ctrl+C 打断：checkpoint 已由微调核心保存，"
                  "状态待执行动作已保留，可用 --恢复 续跑")
            return None

        # ---- 评估：引擎每轮重建（适配器已变更必须重挂），全程固定种子确定性 ----
        适配器路径 = _RWKE_DIR / "适配器" / f"{self.模型}_lora.pt"
        if not 适配器路径.exists():
            raise FileNotFoundError(f"评估前未找到适配器 checkpoint：{适配器路径}")
        评估开始 = time.time()
        from rwke引擎 import RWKE引擎
        引擎 = RWKE引擎(self.模型, 适配器路径=str(适配器路径))
        if self.日记状态路径:
            引擎.挂载日记状态(str(self.日记状态路径))
            print(f"[日记状态] 已挂载：{self.日记状态路径}")
        print(引擎.信息())
        评估集路径 = _RWKE_DIR / 微调配置["数据路径"]["评估集"]
        评估集 = json.loads(Path(评估集路径).read_text(encoding="utf-8"))
        回复列表 = []
        for 条目 in 评估集:
            裸消息 = 剥离模板(条目.get("提示", ""))
            回复 = 引擎.生成(
                裸消息,
                种子=int(评估参数["种子"]),
                max_new_tokens=最大新token,
                温度=float(评估参数["温度"]),
                top_p=float(评估参数["top_p"]),
                top_k=int(评估参数["top_k"]),
            )
            回复列表.append(回复)
            预览 = 回复[:36].replace("\n", " ")
            print(f"  [生成 {len(回复列表):>2}/{len(评估集)}] {条目.get('编号')} → {预览}")
        del 引擎
        释放显存()

        # ---- 回复落盘 + 六维评估 ----
        回复文件 = self.输出目录 / f"微调_回复_{self.模型}_r{轮次号}.jsonl"
        with open(回复文件, "w", encoding="utf-8") as f:
            for 条目, 回复 in zip(评估集, 回复列表):
                f.write(json.dumps(
                    {"编号": 条目.get("编号"), "回复": 回复}, ensure_ascii=False
                ) + "\n")
        评估结果 = 一致性评估.评估数据集(
            str(评估集路径), 回复列表, 权重=self.权重短键, 达标线=self.达标线
        )
        汇总 = 评估结果["汇总"]
        合成分 = float(汇总["合成分_均值"])

        记录 = {
            "轮次": 轮次号,
            "动作": 动作,
            "lr": lr,
            "lr档位索引": int(self.状态["lr档位索引"]),
            "epochs": 本轮epochs,
            "步数上限": 本轮上限步,
            "step": int(摘要["step"]),
            "本轮新增step": int(摘要["step"]) - step起点,
            "训练样本数": int(摘要["可用样本数"]),
            "评估条数": len(评估集),
            "最终loss": 摘要.get("最终loss"),
            "最优loss": 摘要.get("最优loss"),
            "显存峰值GB": 摘要.get("显存峰值GB"),
            "训练耗时秒": 摘要.get("耗时秒"),
            "评估耗时秒": round(time.time() - 评估开始, 1),
            "合成分": 合成分,
            "达标线": self.达标线,
            "达标": 合成分 >= self.达标线,
            "逐维明细": 汇总,
            "逐条合成分": [
                {"编号": d["编号"], "合成分": d["评估"]["合成分"], "达标": d["达标"]}
                for d in 评估结果["明细"]
            ],
            "日记状态": str(self.日记状态路径) if self.日记状态路径 else None,
            "数据扩充": 扩充说明,
            "回复文件": str(回复文件),
            "评估生成参数": {**评估参数, "最大新Token": 最大新token},
            "时间戳": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        轮次文件 = self.输出目录 / f"微调_轮次_{self.模型}_r{轮次号}.json"
        记录["轮次文件"] = str(轮次文件)
        写json(轮次文件, 记录)
        判定词 = "达标" if 记录["达标"] else "未达标"
        print(
            f"[第 {轮次号} 轮结果] 合成分 {合成分}（达标线 {self.达标线}，{判定词}）| "
            f"step={记录['step']}（本轮+{记录['本轮新增step']}）| "
            f"训练耗时 {记录['训练耗时秒']}s 评估耗时 {记录['评估耗时秒']}s"
        )
        return 记录

    # ──────────────────────────────────────────────
    # 主循环
    # ──────────────────────────────────────────────

    def 运行(self) -> str:
        """轮次主循环，返回终局："达标" / "预算耗尽" / "中断"。"""
        已用 = self.状态["已用"]
        预算 = self.微调配置["迭代预算"]
        打印分隔(
            f"自主训练闭环启动 | 模型={self.模型} | 模式={'快验' if self.快验 else '正式'} | "
            f"达标线={self.达标线} | 起始轮次={int(self.状态['轮次']) + 1}"
        )
        print(
            f"迭代预算：续训 {已用['续训']}/{预算.get('续训轮次')} | "
            f"lr降档 {已用['降档']}/{预算.get('lr降档次数')} | "
            f"数据扩充 {已用['扩充']}/{预算.get('数据扩充轮次')}"
            + ("（快验模式：扩充将自动跳过）" if self.快验 else "")
        )
        while True:
            下一轮 = int(self.状态["轮次"]) + 1
            if 下一轮 == 1:
                动作 = "首轮"
            else:
                待 = self.状态.get("待执行动作")
                if 待 and int(待.get("轮次", -1)) == 下一轮:
                    动作 = 待["动作"]
                    self.状态["lr档位索引"] = int(待.get("lr档位索引", self.状态["lr档位索引"]))
                    print(f"[续接] 第 {下一轮} 轮沿用中断前待执行动作：{动作}（不重复扣预算）")
                else:
                    动作 = self.选择动作()
                    if 动作 is None:
                        print("[停止] 迭代预算耗尽（续训/降档/扩充 均不可用），按实测结果输出未达标报告")
                        self.打印曲线()
                        return "预算耗尽"
                    self.消耗预算(动作)
                    self.状态["待执行动作"] = {
                        "动作": 动作, "轮次": 下一轮, "lr档位索引": self.状态["lr档位索引"],
                    }
                    self.保存状态()
            记录 = self.执行一轮(下一轮, 动作)
            if 记录 is None:
                return "中断"
            self.状态["轮次"] = 下一轮
            self.状态["待执行动作"] = None
            self.状态["各轮合成分"].append(记录["合成分"])
            self.状态["历史"].append(记录)
            self.保存状态()
            self.打印曲线()
            if 记录["达标"]:
                return "达标"

    # ──────────────────────────────────────────────
    # 终局报告
    # ──────────────────────────────────────────────

    def 写终局(self, 结局: str) -> None:
        """写 results/微调认证_<模型>.json 与 docs/微调报告.md（数字全部来自实测）。"""
        历史 = self.状态.get("历史", [])
        已用 = self.状态["已用"]
        预算 = self.微调配置["迭代预算"]
        达标 = 结局 == "达标"
        认证路径 = self.输出目录 / f"微调认证_{self.模型}.json"
        报告路径 = _RWKE_DIR / "docs" / "微调报告.md"

        认证 = {
            "模型": self.模型,
            "模式": "快验" if self.快验 else "正式",
            "达标线": self.达标线,
            "达标判定": 达标,
            "终局": 结局,
            "轮次数": len(历史),
            "预算使用": {"已用": 已用, "配置": dict(预算)},
            "迭代历史": 历史,
            "产物": {
                "适配器checkpoint": str(_RWKE_DIR / "适配器" / f"{self.模型}_lora.pt"),
                "最优checkpoint": str(_RWKE_DIR / "适配器" / f"{self.模型}_lora_best.pt"),
                "状态文件": str(self.状态路径),
                "轮次文件": [r.get("轮次文件") for r in 历史],
                "回复文件": [r.get("回复文件") for r in 历史],
                "认证文件": str(认证路径),
                "报告文件": str(报告路径),
            },
            "微调配置快照": {
                "rank": self.微调配置["rank"],
                "alpha": self.微调配置["alpha"],
                "学习率档位": self.微调配置["学习率档位"],
                "epoch上限": self.微调配置["epoch上限"],
                "梯度累积": self.微调配置["梯度累积"],
                "序列长度": self.微调配置["序列长度"],
                "评估生成参数": self.微调配置["评估生成参数"],
                "快验": self.微调配置["快验"],
            },
            "时间戳": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if 历史:
            最佳 = max(历史, key=lambda r: float(r["合成分"]))
            认证["最佳合成分"] = 最佳["合成分"]
            认证["最佳轮次"] = 最佳["轮次"]
            认证["逐维明细"] = 最佳["逐维明细"]
            认证["评估条数"] = 最佳.get("评估条数")
            认证["训练总步数"] = 最佳["step"]
            if 达标:
                认证["结论"] = (
                    f"经 {len(历史)} 轮闭环迭代（每轮 训练→生成→六维评估），"
                    f"第 {最佳['轮次']} 轮合成分 {最佳['合成分']} 达到达标线 {self.达标线}，"
                    f"闭环以'达标即停'结束。"
                )
            else:
                诊 = 差距诊断(最佳, self.达标线)
                认证["差距诊断"] = 诊
                扩充注记 = "（快验模式下扩充跳过）" if self.快验 else ""
                认证["结论"] = (
                    f"迭代预算耗尽（续训 {已用['续训']}/{预算.get('续训轮次')}、"
                    f"lr降档 {已用['降档']}/{预算.get('lr降档次数')}、"
                    f"数据扩充 {已用['扩充']}/{预算.get('数据扩充轮次')}{扩充注记}），"
                    f"最佳合成分 {最佳['合成分']}（第 {最佳['轮次']} 轮）未达达标线 {self.达标线}，"
                    f"差距 {诊['差距']} 分。本结论如实反映实测结果，不虚报。"
                )
        else:
            认证["最佳合成分"] = None
            认证["最佳轮次"] = None
            认证["逐维明细"] = None
            认证["结论"] = "无已完成轮次（首轮即中断），无评估结果可报告"

        写json(认证路径, 认证)
        print(f"[认证] 已写出 {认证路径}")
        self._写报告md(认证, 报告路径)

    def _写报告md(self, 认证: dict, 报告路径: Path) -> None:
        行 = []
        行.append(f"# EmoCompanion角色微调自主训练报告（{self.模型}）")
        行.append("")
        行.append(f"- 生成时间：{认证['时间戳']}")
        行.append(f"- 运行模式：{'快验（小预算闭环）' if self.快验 else '正式闭环'}")
        行.append(f"- 一致性达标线：{self.达标线}")
        if 认证.get("评估条数"):
            行.append(f"- 评估集：{self.微调配置['数据路径']['评估集']}（{认证['评估条数']} 条，固定种子生成）")
        行.append(f"- 结论：**{认证['结论']}**")
        行.append("")
        历史 = 认证.get("迭代历史", [])
        if 历史:
            行.append("## 轮次对比")
            行.append("")
            行.append("| 轮次 | 动作 | lr | 累计step | 本轮新增步 | 训练样本 | 合成分 | 锚点命中率 | 口癖率 | 五维人设 | 情感基调 | 身份一致性 | 流畅性 |")
            行.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            for r in 历史:
                m = r.get("逐维明细", {})
                行.append(
                    f"| {r['轮次']} | {r['动作']} | {r['lr']} | {r['step']} | {r['本轮新增step']} | "
                    f"{r['训练样本数']} | {r['合成分']} | {m.get('锚点命中率_均值')} | {m.get('口癖率_均值')} | "
                    f"{m.get('五维人设符合度_均值')} | {m.get('情感基调匹配_均值')} | "
                    f"{m.get('身份一致性_均值')} | {m.get('流畅性守卫_均值')} |"
                )
            行.append("")
        诊 = 认证.get("差距诊断")
        if 诊:
            行.append("## 差距诊断（未达标）")
            行.append("")
            行.append(
                f"- 最佳合成分 {认证.get('最佳合成分')}（第 {认证.get('最佳轮次')} 轮）"
                f"距达标线 {self.达标线} 还差 {诊['差距']} 分"
            )
            行.append(f"- 最低维度：**{诊['最低维度']}**（均分 {诊['最低维度均分']}）")
            行.append("- 各维均分排序（升序）：")
            for item in 诊["各维均分排序"]:
                行.append(f"  - {item['维度']}：{item['均分']}")
            行.append("- 建议：")
            for s in 诊["建议"]:
                行.append(f"  - {s}")
            行.append(f"  - {诊['通用建议']}")
            行.append("")
        行.append("## 产物")
        行.append("")
        for 键, 值 in 认证["产物"].items():
            if isinstance(值, list):
                行.append(f"- {键}：{'；'.join(str(x) for x in 值) if 值 else '（无）'}")
            else:
                行.append(f"- {键}：{值}")
        行.append("")
        行.append("> 本报告由 自主训练.py 自动生成：数字均来自实际运行落盘结果（轮次/回复/评估 JSON），"
                  "未达标时如实给出差距与诊断，不含虚报。")
        报告路径.parent.mkdir(parents=True, exist_ok=True)
        报告路径.write_text("\n".join(行) + "\n", encoding="utf-8")
        print(f"[报告] 已写出 {报告路径}")


# ======================================================================
# CLI
# ======================================================================

def main(argv=None) -> int:
    解析器 = argparse.ArgumentParser(
        description="EmoCompanion角色 LoRA 微调自主训练驱动器（数据→训练→评估→迭代 无人值守闭环）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：python 自主训练.py --模型 RWKV7-World-0.4B --快验 --达标线 40",
    )
    解析器.add_argument("--模型", default=None, help="模型矩阵标识（默认取 微调.基座默认）")
    解析器.add_argument("--达标线", type=float, default=None, help="覆盖 微调.一致性达标线")
    解析器.add_argument("--快验", action="store_true",
                        help="小预算闭环（读 微调.快验 节覆盖训练与评估参数）")
    解析器.add_argument("--恢复", nargs="?", const="auto", default=None,
                        help="从状态文件续跑（无值取默认 checkpoint 适配器/{基座}_lora.pt，或给路径）")
    解析器.add_argument("--输出目录", default=None,
                        help="轮次/回复/状态/认证输出目录（默认 rwke/results）")
    解析器.add_argument("--日记状态", nargs="?", const="auto", default=None,
                        help="评估时挂载日记 State（无值取 states/EmoCompanion_状态_{模型}.pth，或给路径）")
    参数 = 解析器.parse_args(argv)

    禁用jit_gpu融合()
    驱动器 = None
    try:
        配置 = 读系统配置()
        微调配置 = 配置.get("微调")
        if not isinstance(微调配置, dict) or not 微调配置:
            print("错误：系统配置.json 缺少 微调节（自主训练驱动器全部行为由该节驱动）")
            return 1
        缺键 = [k for k in 必需配置键 if k not in 微调配置]
        if 缺键:
            print(f"错误：微调节缺少必需键：{缺键}")
            return 1
        模型 = str(参数.模型) if 参数.模型 else str(微调配置["基座默认"])
        if 模型 not in 配置.get("模型矩阵", {}):
            print(f"错误：模型 '{模型}' 不在系统配置模型矩阵中：{list(配置.get('模型矩阵', {}))}")
            return 1
        if int(微调配置.get("批大小", 1)) != 1:
            print(f"[警告] 微调.批大小={微调配置['批大小']}：训练核心按 batch=1 实现，将按 1 执行")
        输出目录 = Path(参数.输出目录) if 参数.输出目录 else (_RWKE_DIR / "results")
        驱动器 = 自主训练驱动器(
            模型=模型, 微调配置=微调配置, 快验=参数.快验, 恢复=参数.恢复,
            达标线覆盖=参数.达标线, 输出目录=输出目录, 日记状态参数=参数.日记状态,
        )
        驱动器.检查数据()
    except KeyboardInterrupt:
        print("\n[中断] 启动阶段被 Ctrl+C 打断")
        return 130
    except Exception as e:
        print(f"错误：{type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    try:
        结局 = 驱动器.运行()
    except KeyboardInterrupt:
        驱动器.保存状态()
        print("\n[中断] 收到 Ctrl+C：状态已保存，可用 --恢复 续跑")
        return 130
    except Exception as e:
        驱动器.保存状态()
        try:
            import 微调 as 微调模块
            if isinstance(e, 微调模块.显存守卫异常):
                print(f"显存守卫：{e}")
                return 2
        except Exception:
            pass
        print(f"[错误] {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    if 结局 == "中断":
        return 130
    驱动器.写终局(结局)
    达标词 = "达标即停" if 结局 == "达标" else "预算耗尽，未达标（见差距诊断）"
    打印分隔(f"闭环结束：{达标词}")
    驱动器.打印曲线()
    print(f"认证报告：{驱动器.输出目录 / ('微调认证_' + 驱动器.模型 + '.json')}")
    print(f"训练报告：{_RWKE_DIR / 'docs' / '微调报告.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
