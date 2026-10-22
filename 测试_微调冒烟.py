# -*- coding: utf-8 -*-
"""
RWKE LoRA 微调冒烟测试（Task 3.4，全自动，不依赖真实训练数据）
==============================================================

基座 RWKV7-World-0.4B + 合成数据（5 条中文样本内联于脚本，目标区均 >= 10 token），
逐项断言并打印 PASS，全程通过退出码 0：

  [冒烟1] 合成数据构建（沿用微调 JSONL schema：任务类型/提示/目标）
  [冒烟2] 训练器构造（含显存守卫）+ 数据加载 + 目标区长度校验
  [冒烟3] 梯度流向：优化器参数全部为适配器 A/B；基座抽查 3 张量 grad=None（冻结验证）
  [冒烟4] 20 步训练（--序列长度 128 --累积 2）：loss 末 5 步均值 < 首 5 步均值
  [冒烟5] 显存峰值 torch.cuda.max_memory_allocated <= 6GB
  [冒烟6] checkpoint 保存 -> 重新加载 -> A/B 张量逐键 allclose
  [冒烟7] 合并等价性：5 个 token 序列上 适配器路径 logits vs 合并后 .pth（RWKV 直载）
          logits 最大绝对差 < 0.1（fp16 量化噪声容忍）且 argtop1 一致率 100%
          （对比走同核路径：合并模型逐 token forward_one，与适配器路径同用 TMix_one，
          隔离 rwkv 包 one/seq 两种 kernel 固有的约 0.05-0.1 量化噪声底；seq 路径差异另行报告）

用法：python 测试_微调冒烟.py
说明：本机无 MSVC cl.exe 时 RWKV CUDA 算子编译失败自动降级纯 PyTorch 路径，
日志中相关 warning 属正常，不影响冒烟结论。
"""

import json
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import 微调  # noqa: E402  （经 rwke引擎 设置环境变量后导入 rwkv，顺序由 微调 保证）
from 微调 import 微调训练器, 合并适配器, 训练前向单步, 计算有效权重缓存, 新零态  # noqa: E402

_RWKE_DIR = Path(__file__).resolve().parent

# 合成数据（World 聊天模板格式；目标区均为完整中文句子，>= 10 token）
_合成样本 = [
    {
        "任务类型": "人设问答",
        "提示": "User: EmoCompanion，你是谁呀？\n\nAssistant:",
        "目标": "我是EmoCompanion呀，一个爱笑的主播少女，欢迎来到EmoCompanion的直播间，今天也要开心哦。",
    },
    {
        "任务类型": "情感化对话",
        "提示": "User: 今天好累啊，什么都不想动。\n\nAssistant:",
        "目标": "那个……抱抱你！累的时候就听EmoCompanion说说话，我陪你一起慢慢充电嘛。",
    },
    {
        "任务类型": "日记续写",
        "提示": "EmoCompanion的日记：开播第十五天，",
        "目标": "今天直播间来了好多新朋友，我把天鹅之梦唱给他们听，弹幕都在说好听，好开心啊。",
    },
    {
        "任务类型": "人设问答",
        "提示": "User: EmoCompanion最喜欢什么歌？\n\nAssistant:",
        "目标": "当然是天鹅之梦啦，每次唱到高音那里，我都感觉自己变成了一只小小的天鹅。",
    },
    {
        "任务类型": "情感化对话",
        "提示": "User: 我考试没考好，好难过。\n\nAssistant:",
        "目标": "咳，没关系的啦！EmoCompanion第一次开播也紧张得同手同脚，我们下次一起加油呀。",
    },
]


def 写合成数据(路径) -> Path:
    """把内联合成样本写成 JSONL（供冒烟与 CLI 验证复用）。"""
    p = Path(路径)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for 样本 in _合成样本:
            f.write(json.dumps(样本, ensure_ascii=False) + "\n")
    return p


def _首末均值断言(loss历史) -> None:
    """loss 末 5 步均值 < 首 5 步均值（下降趋势）。"""
    首5 = sum(loss历史[:5]) / 5.0
    末5 = sum(loss历史[-5:]) / 5.0
    print(f"  loss 曲线：首5步均值={首5:.4f} -> 末5步均值={末5:.4f}")
    assert 末5 < 首5, f"loss 未呈下降趋势：首5步均值={首5:.4f} >= 末5步均值={末5:.4f}"


def main() -> int:
    时间戳 = time.strftime("%Y%m%d_%H%M%S")
    冒烟目录 = _RWKE_DIR / "适配器" / f"_冒烟临时_{时间戳}"
    成功 = False
    try:
        if torch.cuda.is_available():
            print(
                f"[环境] {torch.cuda.get_device_name(0)} | torch {torch.__version__} | "
                f"空闲显存 {torch.cuda.mem_get_info()[0] / 2**30:.2f}GB"
            )

        # ── 冒烟1：合成数据 ──
        数据路径 = 写合成数据(冒烟目录 / "合成_冒烟.jsonl")
        print(f"[冒烟1] 合成数据就绪：{数据路径}（{len(_合成样本)} 条，目标区均 >= 10 token）PASS")

        # ── 冒烟2：训练器构造（显存守卫在构造内先行执行）──
        训练器 = 微调训练器(
            "RWKV7-World-0.4B",
            rank=8,
            alpha=16,
            lr=3e-4,
            序列长度=128,
            累积=2,
            输出目录=冒烟目录,
            日志间隔=5,
            种子=42,
        )
        训练器.加载数据(str(数据路径))
        assert len(训练器.样本缓存) == len(_合成样本), "合成样本不应被跳过"
        for token列表, 目标起点 in 训练器.样本缓存:
            assert len(token列表) - 目标起点 >= 10, "目标区须 >= 10 token"
        print(
            f"[冒烟2] 训练器就绪：策略={训练器.加载策略} 设备={训练器.设备} "
            f"适配器参数量={训练器.适配器.参数量:,} 数据目标区校验通过 PASS"
        )

        # ── 冒烟3：梯度流向（单样本 backward 抽查）──
        token列表, 目标起点 = 训练器.样本缓存[0]
        state = 新零态(训练器.模型)
        缓存 = 计算有效权重缓存(训练器.模型, 训练器.适配器)
        loss合计 = None
        计损数 = 0
        for t in range(len(token列表) - 1):
            logits, state = 训练前向单步(训练器.模型, int(token列表[t]), state, 缓存)
            if t + 1 >= 目标起点:
                目标 = torch.tensor(int(token列表[t + 1]), device=logits.device, dtype=torch.long)
                单步loss = F.cross_entropy(logits.float().squeeze(0), 目标)
                loss合计 = 单步loss if loss合计 is None else loss合计 + 单步loss
                计损数 += 1
        assert 计损数 > 0
        (loss合计 / 计损数).backward()

        优化器参数id = {id(p) for g in 训练器.优化器.param_groups for p in g["params"]}
        适配器参数id = {id(p) for p in 训练器.适配器.参数列表}
        assert 优化器参数id == 适配器参数id, "优化器参数必须且仅为适配器 A/B"
        基座抽查 = ["blocks.0.att.key.weight", "blocks.12.ffn.value.weight", "blocks.23.att.output.weight"]
        for 名称 in 基座抽查:
            assert 训练器.模型.z[名称].grad is None, f"基座权重不应有梯度：{名称}"
        有梯度数 = sum(1 for p in 训练器.适配器.参数列表 if p.grad is not None)
        非零梯度数 = sum(1 for p in 训练器.适配器.参数列表 if p.grad is not None and p.grad.abs().sum().item() > 0)
        assert 有梯度数 == len(训练器.适配器.参数列表), "适配器参数应全部进入计算图"
        assert 非零梯度数 > 0, "首步至少应有 B 侧非零梯度（B 全零初始化时 A 侧梯度为零属预期）"
        print(
            f"[冒烟3] 梯度仅落适配器：优化器参数 {len(适配器参数id)} 张量 / {训练器.适配器.参数量:,} 参数"
            f"（有梯度 {有梯度数}，首步非零 {非零梯度数}——B 全零初始化时仅 B 侧有非零梯度属预期）；"
            f"基座抽查 {基座抽查} grad=None PASS"
        )
        训练器.优化器.zero_grad(set_to_none=True)

        # ── 冒烟4：20 步训练 + loss 下降趋势 ──
        摘要 = 训练器.训练(epochs=12, 上限步=20)
        assert len(训练器.loss历史) == 20, f"应恰好 20 个优化器步，实际 {len(训练器.loss历史)}"
        assert 训练器.step == 20
        _首末均值断言(训练器.loss历史)
        非零B = sum(1 for b in 训练器.适配器.B字典.values() if b.detach().abs().sum().item() > 0)
        assert 非零B == len(训练器.适配器.B字典), "训练后适配器 B 应全部非零（适配器确实生效）"
        print(
            f"[冒烟4] 20 步训练完成：step={训练器.step} 跳过样本={摘要['跳过样本数']} "
            f"B 非零 {非零B}/{len(训练器.适配器.B字典)} 耗时={摘要['耗时秒']}s PASS"
        )

        # ── 冒烟5：显存峰值 ──
        峰值GB = torch.cuda.max_memory_allocated() / 2**30
        assert 峰值GB <= 6.0, f"显存峰值 {峰值GB:.2f}GB 超出 6GB 预算"
        print(f"[冒烟5] 显存峰值 {峰值GB:.2f}GB <= 6GB PASS")

        # ── 冒烟6：checkpoint 往返 ──
        assert 训练器.检查点路径.exists() and 训练器.最优检查点路径.exists()
        ck = torch.load(str(训练器.检查点路径), map_location="cpu")
        assert set(ck.keys()) >= {
            "适配器A", "适配器B", "rank", "alpha", "基座标识", "step", "loss历史", "优化器状态", "序列长度"
        }, f"checkpoint 键不完整：{sorted(ck)}"
        assert ck["step"] == 20 and len(ck["loss历史"]) == 20 and ck["基座标识"] == "RWKV7-World-0.4B"
        全一致 = True
        for 名称 in ck["适配器A"]:
            全一致 &= torch.allclose(ck["适配器A"][名称], 训练器.适配器.A字典[名称].detach().cpu())
            全一致 &= torch.allclose(ck["适配器B"][名称], 训练器.适配器.B字典[名称].detach().cpu())
        assert 全一致 and len(ck["适配器A"]) == len(训练器.适配器.名称列表), "A/B 逐键 allclose 失败"
        print(
            f"[冒烟6] checkpoint 往返：{len(ck['适配器A'])} 键 A/B allclose，"
            f"step/loss历史/优化器状态/序列长度 齐全（{训练器.检查点路径.name} + _best） PASS"
        )

        # ── 冒烟7：合并等价性（合并后 .pth 直接 RWKV 加载）──
        合并文件 = 冒烟目录 / "RWKV7-World-0.4B_冒烟合并.pth"
        元数据 = 合并适配器(训练器.基座文件, 训练器.检查点路径, 合并文件)
        assert 元数据["适配器已合并"] is True, "合并元数据须含 适配器已合并=True"
        assert 合并文件.with_suffix(".json").exists(), "元数据旁车 .json 缺失"
        合并模型 = 微调.RWKV(str(合并文件.with_suffix("")), "cuda fp16")

        验证序列 = []
        for 样本 in _合成样本:
            全文token = list(训练器.分词器.encode(样本["提示"] + 样本["目标"]))
            验证序列.append(全文token[:48])
        一致数 = 0
        最大差 = 0.0
        for 序号, tk in enumerate(验证序列, 1):
            适配器logits = 训练器.适配器前向序列(tk)               # [vocab] fp32（逐 token TMix_one）
            # 合并模型同核对比：逐 token forward_one（与适配器路径同用 TMix_one，
            # 只度量"合并权重 == W_eff"这一恒等本身，隔离 one/seq kernel 噪声底）
            state = 合并模型.generate_zero_state()
            合并logits = None
            for t in range(len(tk)):
                合并logits, state = 合并模型.forward([int(tk[t])], state)
            合并logits = 合并logits.float().reshape(-1)
            差 = float((适配器logits - 合并logits).abs().max().item())
            最大差 = max(最大差, 差)
            同argmax = int(适配器logits.argmax().item()) == int(合并logits.argmax().item())
            一致数 += 1 if 同argmax else 0
            # 信息性：合并模型走 forward_seq（部署预填充路径）的差异（one/seq kernel 噪声）
            seqlogits, _ = 合并模型.forward(list(tk), None)
            seq差 = float((适配器logits - seqlogits.float().reshape(-1)).abs().max().item())
            print(f"  序列{序号}（{len(tk)} token）：同核最大绝对差={差:.5f} "
                  f"seq路径差={seq差:.5f} argtop1一致={同argmax}")
        assert 最大差 < 0.1, f"合并等价性失败：同核最大绝对差 {最大差:.5f} >= 0.1"
        assert 一致数 == len(验证序列), f"argtop1 一致率 {一致数}/{len(验证序列)} != 100%"
        print(f"[冒烟7] 合并等价性：{len(验证序列)} 序列 同核最大绝对差={最大差:.5f}(<0.1) "
              f"argtop1 一致 {一致数}/{len(验证序列)} PASS")

        del 合并模型
        torch.cuda.empty_cache()
        成功 = True
        print("[冒烟] 全部 PASS")
        return 0
    except AssertionError as e:
        print(f"[冒烟失败] 断言不通过：{e}")
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"[冒烟异常] {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1
    finally:
        if 成功:
            shutil.rmtree(冒烟目录, ignore_errors=True)
            print(f"[清理] 冒烟临时目录已删除：{冒烟目录}")
        else:
            print(f"[保留] 冒烟未通过，临时目录留存供排查：{冒烟目录}")


if __name__ == "__main__":
    sys.exit(main())
