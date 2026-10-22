# -*- coding: utf-8 -*-
"""
RWKV-7 纯 PyTorch LoRA 微调核心（Task 3）
==========================================

在冻结基座上训练低秩适配器（LoRA），Windows 单卡 RTX 3080 16GB 可跑：
不依赖 RWKV-PEFT（其仅 Linux/WSL），不依赖 RWKV CUDA 算子（本机无 MSVC cl.exe 时
rwke引擎 自动降级纯 PyTorch 路径，日志中相关 warning 属正常，功能不缺失）。

用法
----
    # 训练（正式数据由 训练数据.py 生成；冒烟可传合成数据）
    python 微调.py --模型 RWKV7-World-0.4B --数据 训练数据/微调数据_train.jsonl --epochs 2

    # 断点续训（同命令加 --恢复，缺省取 输出目录/{基座标识}_lora.pt）
    python 微调.py --模型 RWKV7-World-0.4B --数据 ... --恢复

    # 默认走快的序列前向；--逐token 回退旧路径（数值对照/兜底）
    python 微调.py --模型 RWKV7-G1-1.5B --数据 训练数据/微调数据_train.jsonl --序列长度 256

    # 自动续训（默认 checkpoint 存在即恢复；滚动存点默认每 50 步）
    python 微调.py --模型 RWKV7-G1-1.5B --数据 ... --自动续训 --存点间隔 50

    # 合并 ΔW 进基座导出同格式 .pth（元数据写同名 .json 旁车）
    python 微调.py --模型 RWKV7-World-0.4B --合并 适配器/RWKV7-World-0.4B_EmoCompanion合并.pth

训练前向两条路径（默认序列版，逐token 为对照/兜底）
--------------------------------------------------
- ``训练前向序列``：整条序列一次算完 24 层，投影走官方 ``RWKV_x070_TMix_seq`` /
  ``RWKV_x070_CMix_seq`` 并注入 W_eff，末尾一次算全序列 logits，再对目标区算 fp32
  交叉熵。逐层顺序与 ``训练前向单步`` 严格一致，数学等价（差异仅 gemm/gemv 的 fp16
  舍入级）；把每 token 一次的小 kernel launch 合并为大批量 gemm，实测显著提速。
- ``训练前向单步``（旧路径）：逐 token 前向，保留作数值对照与兜底（--逐token）。
- 层重算（--层重算）：序列前向本身会在 autograd 图中保留整条序列的中间激活（纯
  PyTorch 分支每 token 状态更新产生大量 [H,N,N] 中间量），显存吃紧时对每层
  TMix_seq/CMix_seq 调 torch.utils.checkpoint 分段重算换取显存。

关键实现事实（rwkv pip v0.8.32 model.py 逐行核对）
--------------------------------------------------
- 环境变量（RWKV_V7_ON=1 / RWKV_CUDA_ON）在 rwke引擎 模块顶部先于 import rwkv 设置，
  本模块经 ``import rwke引擎`` 复用其模型工厂 / jit 算子句柄 / World 分词器；
- ``RWKV_x070_TMix_one`` / ``RWKV_x070_CMix_one`` 为 @MyStatic(torch.jit.script) 模块级
  函数（model.py L491/L596），脚本化函数支持 autograd——训练前向在梯度模式下调用即可；
- 训练前向 ``训练前向单步`` 逐行复刻 rwke引擎._前向单步（即 model.py forward_one
  L336-363 的数值路径），仅把 6 个被适配投影换成 W_eff，其余权重直用 z 原值；
- 加载器 ``RWKV_x070.__init__``（model.py L245-265）对 .pth 内**每一个键**执行
  ``tensor.squeeze()`` / ``.to(DEVICE)`` 等张量运算——附加字符串键会使 RWKV() 加载崩溃，
  故合并元数据不写进 .pth，改写同名 .json 旁车文件（本模块已按此决策实现）；
- raw .pth 中 6 类投影权重为 [out, in]（RWKV 保存惯例，加载时 ``.t()`` 成 z 的
  x@W 约定 [in, out]；att 方阵形状不可辨、ffn 非方阵实测验证），合并时镜像运行时
  fp16 权重叠加 ΔW（见 合并适配器 的存储决策）。

LoRA 设计（spec F2/F6；方向约定按 rwkv v0.8.32 实测修正）
------------------------------------------------------
- 挂载点：每层 att.receptance/key/value/output 与 ffn.key/value 共 6 个投影；
- 实测 z 中这些权重为 x@W 约定的 [in, out]（att 为方阵不可辨、ffn 非方阵可验；
  raw .pth 为 [out, in]，加载时 ``.t()``），前向 x @ W_eff（非 F.linear 的 x@W.T）；
- 每个权重 W：A [r, W.shape[1]] kaiming 初始化（fp32）、B [W.shape[0], r] 全零初始化
  （fp32），ΔW = 缩放*(B@A) 与 W 同形相加——相对经典 LoRA A/B 角色随约定互换
  （零初始化落在 x@B 下投影侧），仍严格满足"起步 ΔW=0 数值恒等、首步即可训练"；
- 缩放 = alpha / r；
- 基座 z 全部 requires_grad=False（cuda fp16 或 cpu fp32 策略下均冻结）；
- W_eff = (W.float() + 缩放*(B@A)).to(W.dtype)，每个样本前向现算一次并整条序列复用
  （同一线性层各 token 共享同一 W_eff 张量，autograd 保存的是引用——显存不随序列长度
  平方膨胀；梯度经 cast 节点流回 A/B，基座 W 为常量叶子；ΔW 用 fp64 matmul 保证与
  合并侧逐位一致，见 计算ΔW）。

显存参考（0.4B）：序列长度 128 时峰值约 2-3GB；512 时状态链激活约 6-8GB
（仍在 16GB 卡内；如需严守 6GB 预算用 --序列长度 128-256）。

供 自主训练.py 调用的 API 摘要
------------------------------
- ``微调训练器(模型标识, rank=8, alpha=16, lr=1e-4, 序列长度=512, 累积=4,
  输出目录=None, 日志间隔=10, 种子=42, 逐token=False, 层重算=False, 存点间隔=50)``：
  构造即完成显存守卫 + 基座加载 + 适配器/优化器构建；
  ``.加载数据(数据路径)``、``.恢复checkpoint(路径=None)``、
  ``.训练(epochs=2, 上限步=0) -> dict``（返回训练摘要，含 loss历史/显存峰值/检查点路径）、
  ``.保存checkpoint(路径=None)``、``.适配器前向序列(token列表) -> logits[vocab]``；
- ``训练前向序列(模型, token列表, 权重缓存=None, 层重算=False) -> logits[T,vocab]``（默认路径）、
  ``序列目标交叉熵(logits, token列表, 目标起点) -> loss``；
- ``训练前向单步(模型, token_id, state, 权重缓存=None) -> (logits[1,vocab], state)``（旧路径）；
- ``计算有效权重缓存(模型, 适配器) -> dict``、``计算ΔW(A, B, 缩放) -> Tensor``、
  ``新零态(模型) -> List[Tensor]``；
- ``合并适配器(基座文件, checkpoint路径, 输出文件) -> dict``（元数据，另写同名 .json 旁车）；
- ``显存守卫(基座文件)``：超预算抛 ``显存守卫异常``（CLI 退出码 2）。
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 训练必须走纯 PyTorch 路径：rwkv 的 V7 CUDA 算子（wkv7s）只实现 forward
# （WKV_7.forward 内 torch.no_grad()，且未定义 backward），训练启用 RWKV_CUDA_ON=1
# 会在反传时直接报错。此处置 0 必须**先于 import rwke引擎**（引擎按该变量决定是否编译算子）。
os.environ["RWKV_CUDA_ON"] = "0"

# 复用 rwke引擎 的环境变量设置（RWKV_V7_ON 先于 import rwkv）、CUDA 降级、模型工厂与分词器
import rwke引擎 as 引擎模块  # noqa: E402
from rwke引擎 import (  # noqa: E402
    RWKV,
    RWKV_x070_TMix_one,
    RWKV_x070_CMix_one,
    RWKV_x070_TMix_seq,
    RWKV_x070_CMix_seq,
    TRIE_TOKENIZER,
)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch.utils.checkpoint  # noqa: E402  （--层重算 分段重算用）

_RWKE_DIR = Path(__file__).resolve().parent

# 被适配的 6 类投影（z 中为 x@W 约定的 [in, out]；训练前向 x @ W_eff，ΔW 与 W 同形相加）
_被适配投影后缀 = (
    "att.receptance.weight",
    "att.key.weight",
    "att.value.weight",
    "att.output.weight",
    "ffn.key.weight",
    "ffn.value.weight",
)


class 显存守卫异常(RuntimeError):
    """显存预算校验失败（CLI 退出码 2，绝不静默 OOM）。"""


# ======================================================================
# 模型矩阵 / 基座路径 / 显存守卫
# ======================================================================

def 读取矩阵条目(模型标识: str) -> dict:
    """从系统配置.json 模型矩阵取条目（仅接受 rwkv7 后端）。"""
    配置 = 引擎模块._读系统配置()
    矩阵 = 配置.get("模型矩阵", {})
    if 模型标识 not in 矩阵:
        raise KeyError(f"模型标识 '{模型标识}' 不在系统配置模型矩阵中：{list(矩阵)}")
    条目 = 矩阵[模型标识]
    if 条目.get("后端") != "rwkv7":
        raise ValueError(f"模型 '{模型标识}' 后端为 '{条目.get('后端')}'，本模块仅支持 rwkv7")
    return 条目


def 解析基座文件路径(条目: dict) -> Path:
    """矩阵条目路径 → 绝对 .pth 文件路径（不存在即报错）。"""
    路径 = Path(条目["路径"])
    if not 路径.is_absolute():
        路径 = _RWKE_DIR / 路径
    if 路径.suffix != ".pth":
        路径 = 路径.with_suffix(".pth")
    if not 路径.exists():
        raise FileNotFoundError(f"基座权重不存在：{路径}")
    return 路径


def 估算显存预算GB(基座文件: Path) -> float:
    """
    按基座 .pth 文件大小估算参数量并给出显存预算档位：

    World/G1 系列 pth 为 fp16 存储（参数量 ≈ 字节/2）：
    0.4B 档（<0.8e9 参数）限 6GB；1.5B 档（<2.2e9 参数）限 14GB；更大模型按
    fp16 权重 × 3 保守系数外推。
    """
    参数量估算 = max(os.path.getsize(基座文件), 1) / 2
    if 参数量估算 < 0.8e9:
        return 6.0
    if 参数量估算 < 2.2e9:
        return 14.0
    return max(14.0, 参数量估算 * 2 * 3 / 2**30)


def 显存守卫(基座文件: Path) -> float:
    """
    训练开始前的显存预算校验：torch.cuda.mem_get_info 空闲量 < 预算即抛 显存守卫异常
    （含明确降档建议），绝不静默 OOM。返回预算 GB（供日志）。
    """
    预算GB = 估算显存预算GB(基座文件)
    if not torch.cuda.is_available():
        raise 显存守卫异常(
            f"未检测到 CUDA GPU，本训练核心要求单卡 CUDA（预算 {预算GB:.0f}GB）。"
            "【降档建议】改用更小模型（RWKV7-World-0.4B）"
        )
    空闲字节, _总字节 = torch.cuda.mem_get_info()
    空闲GB = 空闲字节 / 2**30
    if 空闲GB < 预算GB:
        raise 显存守卫异常(
            f"显存不足：GPU 空闲 {空闲GB:.2f}GB < 训练预算 {预算GB:.0f}GB（绝不静默 OOM）。"
            "【降档建议】减小 --序列长度 / --rank；改用更小模型（RWKV7-World-0.4B）；"
            "或释放其他 GPU 占用后重试"
        )
    return 预算GB


# ======================================================================
# LoRA 适配器
# ======================================================================

class LoRA适配器:
    """
    每层 6 个投影的低秩适配器容器：A [r, W.shape[1]] kaiming（fp32）、
    B [W.shape[0], r] 全零（fp32），ΔW = 缩放*(B@A) 与 W 同形相加，起步 ΔW=0 数值恒等；
    缩放 = alpha / rank（方向约定说明见模块 docstring）。基座 z 不在此改动
    （训练前向按名称查 W_eff）。
    """

    def __init__(self, 模型, rank: int = 8, alpha: int = 16, 种子: int = 42) -> None:
        if int(rank) <= 0 or int(alpha) <= 0:
            raise ValueError(f"rank/alpha 必须为正：rank={rank}, alpha={alpha}")
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.缩放 = float(self.alpha) / float(self.rank)

        torch.manual_seed(int(种子))
        z = 模型.z
        self.名称列表: List[str] = []
        self.A字典: Dict[str, torch.Tensor] = {}
        self.B字典: Dict[str, torch.Tensor] = {}
        参数列表: List[torch.Tensor] = []
        for i in range(模型.n_layer):
            for 后缀 in _被适配投影后缀:
                名称 = f"blocks.{i}.{后缀}"
                W = z[名称]  # [out, in]
                出维, 入维 = W.shape
                A = torch.empty(self.rank, 入维, dtype=torch.float32, device=W.device)
                torch.nn.init.kaiming_uniform_(A, a=math.sqrt(5.0))
                B = torch.zeros(出维, self.rank, dtype=torch.float32, device=W.device)
                A.requires_grad_(True)
                B.requires_grad_(True)
                self.A字典[名称] = A
                self.B字典[名称] = B
                self.名称列表.append(名称)
                参数列表.extend([A, B])
        self.参数列表 = 参数列表

    @property
    def 参数量(self) -> int:
        return sum(p.numel() for p in self.参数列表)

    def cpu快照(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """A/B 张量 → CPU fp32 克隆（checkpoint 落盘用，与设备解耦）。"""
        return (
            {k: v.detach().to("cpu").clone() for k, v in self.A字典.items()},
            {k: v.detach().to("cpu").clone() for k, v in self.B字典.items()},
        )

    def 载入快照(self, A字典: Dict[str, torch.Tensor], B字典: Dict[str, torch.Tensor], 设备) -> None:
        """用 checkpoint 的 A/B 原地覆盖参数（.data.copy_ 保持张量身份，优化器状态对齐）。"""
        for 名称 in self.名称列表:
            if 名称 not in A字典 or 名称 not in B字典:
                raise KeyError(f"checkpoint 缺少被适配键：{名称}")
            self.A字典[名称].data.copy_(A字典[名称].to(设备))
            self.B字典[名称].data.copy_(B字典[名称].to(设备))


# ======================================================================
# 训练前向（逐行复刻 rwke引擎._前向单步，仅替换被适配权重）
# ======================================================================

def 计算ΔW(A: torch.Tensor, B: torch.Tensor, 缩放: float) -> torch.Tensor:
    """
    ΔW = 缩放*(B@A)（[out, in]，fp32），训练缓存（GPU）与合并（CPU）共用本函数。

    用 fp64 计算后回 fp32：rwkv 包 import 时开启 TF32（model.py L12），fp32 GPU matmul
    会以 10 位尾数舍入，与合并路径的 CPU 全精度 fp32 结果产生 1 个 fp16 ULP 级的
    权重差异并经 24 层放大到 logits（实测最大 0.156）；fp64 两侧均正确舍入，
    逐元素一致，保证合并等价性。开销可忽略（K=rank 的小矩阵）。
    """
    return (缩放 * torch.mm(B.to(torch.float64), A.to(torch.float64))).to(torch.float32)


def 计算有效权重缓存(模型, 适配器: LoRA适配器) -> Dict[str, torch.Tensor]:
    """
    每个样本前向现算一次 W_eff = (W.float() + 缩放*(B@A)).to(W.dtype)，整条序列复用。

    梯度经 cast 节点流回 A/B；基座 W（requires_grad=False）为常量叶子。同一线性层
    各 token 共享同一 W_eff 张量，autograd 对重复使用只保存引用，显存可控。
    """
    z = 模型.z
    缓存: Dict[str, torch.Tensor] = {}
    for 名称 in 适配器.名称列表:
        W = z[名称]
        A = 适配器.A字典[名称]
        B = 适配器.B字典[名称]
        缓存[名称] = (W.float() + 计算ΔW(A, B, 适配器.缩放)).to(W.dtype)
    return 缓存


def _取权重(z: dict, 权重缓存: Optional[Dict[str, torch.Tensor]], 名称: str) -> torch.Tensor:
    """被适配键取 W_eff，其余直用 z 原值。"""
    if 权重缓存 and 名称 in 权重缓存:
        return 权重缓存[名称]
    return z[名称]


def 新零态(模型) -> List[torch.Tensor]:
    """
    初始 State：模型.generate_zero_state()（rwkv 全局 DEVICE/DTYPE 依赖最后一次加载的
    模型），此处无条件按本模型设备/dtype 对齐（引擎 _对齐状态 同款规则，防多模型并存错位）。
    """
    state = 模型.generate_zero_state()
    设备 = 模型.z["emb.weight"].device
    权重dtype = 模型.z["emb.weight"].dtype
    对齐: List[torch.Tensor] = []
    for i in range(模型.n_layer):
        对齐.append(state[i * 3 + 0].to(设备).to(权重dtype))
        对齐.append(state[i * 3 + 1].to(设备).float())
        对齐.append(state[i * 3 + 2].to(设备).to(权重dtype))
    return 对齐


def 训练前向单步(
    模型,
    token_id: int,
    state: List[torch.Tensor],
    权重缓存: Optional[Dict[str, torch.Tensor]] = None,
):
    """
    逐行复刻 rwke引擎.RWKE引擎._前向单步（rwkv.model forward_one L336-363 数值路径），
    仅把 6 类被适配投影换成 权重缓存 中的 W_eff；可微（训练时勿置于 no_grad）。

    返回 (logits [1, vocab], state)。
    """
    z = 模型.z
    x = z["emb.weight"][int(token_id)]
    v_first = torch.empty_like(x)

    for i in range(模型.n_layer):
        bbb = f"blocks.{i}."
        att = f"blocks.{i}.att."
        ffn = f"blocks.{i}.ffn."

        xx = F.layer_norm(x, (模型.n_embd,), weight=z[bbb + "ln1.weight"], bias=z[bbb + "ln1.bias"])

        xx, state[i * 3 + 0], state[i * 3 + 1], v_first = RWKV_x070_TMix_one(
            i, 模型.n_head, 模型.head_size, xx, state[i * 3 + 0], v_first, state[i * 3 + 1],
            z[att + "x_r"], z[att + "x_w"], z[att + "x_k"], z[att + "x_v"], z[att + "x_a"], z[att + "x_g"],
            z[att + "w0"], z[att + "w1"], z[att + "w2"], z[att + "a0"], z[att + "a1"], z[att + "a2"],
            z[att + "v0"], z[att + "v1"], z[att + "v2"],
            z[att + "g1"], z[att + "g2"], z[att + "k_k"], z[att + "k_a"], z[att + "r_k"],
            _取权重(z, 权重缓存, att + "receptance.weight"),
            _取权重(z, 权重缓存, att + "key.weight"),
            _取权重(z, 权重缓存, att + "value.weight"),
            _取权重(z, 权重缓存, att + "output.weight"),
            z[att + "ln_x.weight"], z[att + "ln_x.bias"],
        )
        x = x + xx

        xx = F.layer_norm(x, (模型.n_embd,), weight=z[bbb + "ln2.weight"], bias=z[bbb + "ln2.bias"])
        xx, state[i * 3 + 2] = RWKV_x070_CMix_one(
            xx, state[i * 3 + 2], z[ffn + "x_k"],
            _取权重(z, 权重缓存, ffn + "key.weight"),
            _取权重(z, 权重缓存, ffn + "value.weight"),
        )
        x = x + xx

    x = F.layer_norm(x, (模型.n_embd,), weight=z["ln_out.weight"], bias=z["ln_out.bias"])
    logits = x @ z["head.weight"]
    return logits.unsqueeze(0), state


def _TMix序列包装(layer_id, H, N, x, x_prev, v_first, state, *其余):
    """torch.utils.checkpoint 用的可调用包装（层重算用，签名逐位透传给官方 TMix_seq）。"""
    return RWKV_x070_TMix_seq(layer_id, H, N, x, x_prev, v_first, state, *其余)


def _CMix序列包装(x, x_prev, x_k, K_, V_):
    """torch.utils.checkpoint 用的可调用包装（层重算用，透传给官方 CMix_seq）。"""
    return RWKV_x070_CMix_seq(x, x_prev, x_k, K_, V_)


# ======================================================================
# 融合算子前向（--融合算子）：Triton chunked RWKV-7 扫描替代逐 token 递推
# ----------------------------------------------------------------------
# 官方纯 PyTorch 回退的 TMix_seq 对 T 做 Python 逐 token 循环（model.py L563-568），
# 每步 state[H,N,N] 更新产生一条长串行链 —— 实测占训练步时的绝大部分。
# flash-linear-attention 的 chunk_rwkv7 用 Triton 分块扫描（含手写 backward）把
# 串行步数从 T 降到 T/chunk，实测递推环节 77.1ms → 1.2ms/层（T=256,H=32,N=64,fwd+bwd）。
#
# 参数映射（与官方回退逐位对齐，已数值核验）：
#   官方  state ← state*decay + state@((-kk)⊗(kk*a)) + v⊗k
#   fla   chunk_rwkv7(q=r, k=k, v=v, a=-kk, b=kk*a, gk=log(decay))
#   末态布局：官方 [H,V,K]，fla [N,H,K,V] → 取 [0].transpose(1,2)
# 精度：fla 内核用快速 exp，与官方纯 fp32 路径有 ~1e-3 相对偏差（非逐位一致），
#       属可接受的算子级舍入差异；A/B 需以 loss 轨迹而非逐位比对验收。
# ======================================================================

_融合句柄 = None


def 融合可用() -> Tuple[bool, str]:
    """探测融合算子可用性（triton + fla.chunk_rwkv7 + CUDA）；返回 (可用, 说明)。"""
    global _融合句柄
    if not torch.cuda.is_available():
        return False, "无 CUDA 设备"
    try:
        from fla.ops.rwkv7 import chunk_rwkv7
    except Exception as e:  # 依赖缺失即如实降级，不静默
        return False, f"fla 不可用（{type(e).__name__}: {e}）"
    _融合句柄 = chunk_rwkv7
    return True, "fla.chunk_rwkv7（Triton 分块扫描，含反向）"


def TMix_融合(
    layer_id: int, H: int, N: int, x, x_prev, v_first, state,
    x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2,
    k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b,
):
    """
    TMix 的融合算子版：投影与门控逐行复刻官方 TMix_seq，仅把 wkv7 递推换成
    fla.chunk_rwkv7（分块并行扫描）；返回值顺序与官方一致
    ``(输出, x[-1,:], 新state, v_first)``，可直接替换 RWKV_x070_TMix_seq 调用点。
    """
    if _融合句柄 is None:
        from fla.ops.rwkv7 import chunk_rwkv7 as _ck
        _ck_句柄 = _ck
    else:
        _ck_句柄 = _融合句柄
    T = x.shape[0]
    xx = torch.cat((x_prev.unsqueeze(0), x[:-1, :])) - x
    xr, xw, xk, xv, xa, xg = (x + xx * x_r, x + xx * x_w, x + xx * x_k,
                              x + xx * x_v, x + xx * x_a, x + xx * x_g)
    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = torch.nn.functional.normalize((k * k_k).view(T, H, N), dim=-1, p=2.0).view(T, H * N)
    k = k * (1 + (a - 1) * k_a)
    if layer_id == 0:
        v_first = v
    else:
        v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

    # 官方乘法衰减 decay = exp(-0.606531*sigmoid(w0+w)) ∈ (0,1)；fla 收 log 衰减
    w_log = (-0.606531 * torch.sigmoid((w0 + w).float())).view(1, T, H, N)
    初始 = None
    if state is not None:
        初始 = state.to(torch.float32).transpose(1, 2).contiguous().reshape(1, H, N, N)
    输出, 末态 = _ck_句柄(
        r.view(1, T, H, N).float(), w_log,
        k.view(1, T, H, N).float(), v.view(1, T, H, N).float(),
        (-kk).view(1, T, H, N).float(), (kk * a).view(1, T, H, N).float(),
        initial_state=初始, output_final_state=True,
    )
    xx = 输出.reshape(T, H * N).to(x.dtype)
    新state = 末态.reshape(H, N, N).transpose(1, 2).to(
        state.dtype if state is not None else torch.float32).contiguous()
    xx = torch.nn.functional.group_norm(
        xx.view(T, H * N), num_groups=H, weight=ln_w, bias=ln_b, eps=64e-5).view(T, H * N)
    xx = xx + ((r * k * r_k).view(T, H, N).sum(dim=-1, keepdim=True) * v.view(T, H, N)).view(T, H * N)
    return (xx * g) @ O_, x[-1, :], 新state, v_first


def 训练前向序列(
    模型,
    token列表: List[int],
    权重缓存: Optional[Dict[str, torch.Tensor]] = None,
    层重算: bool = False,
    融合算子: bool = False,
):
    """
    序列版前向：整条序列一次算完 24 层，投影走官方 RWKV_x070_TMix_seq / CMix_seq 并注入
    W_eff，逐层顺序与 训练前向单步（及 model.py forward_seq L458-487）完全一致：
    ln1 → TMix → 残差 → ln2 → CMix → 残差 → … → ln_out → head。

    与逐 token 路径的差异仅在算子批处理粒度（gemm 而非 gemv）造成的 fp16 舍入级差异，
    数学上等价（见 rwke引擎._适配前向序列 docstring 同款说明）。与单步路径唯一语义差别：
    单步只对末位 token 输出 logits，此处返回全部 T 个位置的 logits 以便整段计损。

    关键接参（勿错）：
    - ``x_prev``（state[i*3+0]）与 ``v_first`` 按官方签名传入；整条序列从零态开始，
      state 为 新零态(模型)，故 x_prev 首行即 zeros（等价单步首 token 的 x_prev=0）；
    - TMix_seq 返回 ``(输出, x[-1,:], state, v_first)``（与 one 版返回顺序不同）：
      第二项是新 att_x_prev 状态、第四项是 v_first；CMix_seq 返回 ``(输出, x[-1,:])``：
      第二项是新 ffn_x_prev 状态；
    - ``层重算=True`` 时对每层的 TMix_seq / CMix_seq 调 torch.utils.checkpoint 分段重算，
      以重算换激活显存（纯 PyTorch 分支每 token 状态更新会保留大量 [H,N,N] 中间量）。

    返回 logits [T, vocab]（dtype 同基座权重；调用方自行 .float() 后计损，与单步一致）。
    """
    z = 模型.z
    T = len(token列表)
    if T <= 0:
        raise ValueError("训练前向序列：token列表 不能为空")
    token张量 = torch.tensor([int(t) for t in token列表], device=z["emb.weight"].device, dtype=torch.long)
    state = 新零态(模型)

    x = z["emb.weight"][token张量]                 # [T, n_embd]
    v_first = torch.empty_like(x)

    for i in range(模型.n_layer):
        bbb = f"blocks.{i}."
        att = f"blocks.{i}.att."
        ffn = f"blocks.{i}.ffn."

        xx = F.layer_norm(x, (模型.n_embd,), weight=z[bbb + "ln1.weight"], bias=z[bbb + "ln1.bias"])

        其余TMix = (
            z[att + "x_r"], z[att + "x_w"], z[att + "x_k"], z[att + "x_v"], z[att + "x_a"], z[att + "x_g"],
            z[att + "w0"], z[att + "w1"], z[att + "w2"], z[att + "a0"], z[att + "a1"], z[att + "a2"],
            z[att + "v0"], z[att + "v1"], z[att + "v2"],
            z[att + "g1"], z[att + "g2"], z[att + "k_k"], z[att + "k_a"], z[att + "r_k"],
            _取权重(z, 权重缓存, att + "receptance.weight"),
            _取权重(z, 权重缓存, att + "key.weight"),
            _取权重(z, 权重缓存, att + "value.weight"),
            _取权重(z, 权重缓存, att + "output.weight"),
            z[att + "ln_x.weight"], z[att + "ln_x.bias"],
        )
        if 融合算子:
            xx, state[i * 3 + 0], state[i * 3 + 1], v_first = TMix_融合(
                i, 模型.n_head, 模型.head_size, xx, state[i * 3 + 0], v_first, state[i * 3 + 1], *其余TMix
            )
        elif 层重算:
            xx, state[i * 3 + 0], state[i * 3 + 1], v_first = torch.utils.checkpoint.checkpoint(
                _TMix序列包装, i, 模型.n_head, 模型.head_size,
                xx, state[i * 3 + 0], v_first, state[i * 3 + 1], *其余TMix,
                use_reentrant=True,
            )
        else:
            xx, state[i * 3 + 0], state[i * 3 + 1], v_first = RWKV_x070_TMix_seq(
                i, 模型.n_head, 模型.head_size, xx, state[i * 3 + 0], v_first, state[i * 3 + 1], *其余TMix
            )
        x = x + xx

        xx = F.layer_norm(x, (模型.n_embd,), weight=z[bbb + "ln2.weight"], bias=z[bbb + "ln2.bias"])
        关键 = _取权重(z, 权重缓存, ffn + "key.weight")
        值 = _取权重(z, 权重缓存, ffn + "value.weight")
        if 层重算:
            xx, state[i * 3 + 2] = torch.utils.checkpoint.checkpoint(
                _CMix序列包装, xx, state[i * 3 + 2], z[ffn + "x_k"], 关键, 值, use_reentrant=True
            )
        else:
            xx, state[i * 3 + 2] = RWKV_x070_CMix_seq(xx, state[i * 3 + 2], z[ffn + "x_k"], 关键, 值)
        x = x + xx

    # 末位 ln_out + head 与单步完全一致；此处对全序列施加以便整段计损（head 未被适配则直用 z）
    x = F.layer_norm(x, (模型.n_embd,), weight=z["ln_out.weight"], bias=z["ln_out.bias"])
    logits = x @ _取权重(z, 权重缓存, "head.weight")     # [T, vocab]
    return logits


def 序列目标交叉熵(
    logits: torch.Tensor,
    token列表: List[int],
    目标起点: int,
) -> Optional[torch.Tensor]:
    """
    对序列 logits [T, vocab] 计算目标区交叉熵：位置 t（logits[t] 预测 token[t+1]）
    满足 t+1 >= 目标起点 才计损；fp32 计算，对计损位取均值——与逐 token 路径
    （各位置 cross_entropy 求和后除以计损数）数学等价。
    """
    T = int(logits.shape[0])
    if T < 2:
        return None
    预测 = logits[:-1]                                    # [T-1, vocab]
    标签 = torch.tensor([int(t) for t in token列表[1:]], device=logits.device, dtype=torch.long)
    位置 = torch.arange(1, T, device=logits.device)       # 对应 预测 第 j 项预测 token[j+1]
    掩码 = 位置 >= int(目标起点)
    if not bool(掩码.any()):
        return None
    return F.cross_entropy(预测[掩码].float(), 标签[掩码])


# ======================================================================
# 训练器
# ======================================================================

class 微调训练器:
    """
    RWKV-7 LoRA 训练器：teacher-forcing 交叉熵（仅目标区间计损）+ 梯度累积 + checkpoint 断点续训。

    构造即完成：显存守卫 → 基座加载（cuda fp16 失败回退 cpu fp32）→ 适配器/优化器构建。
    """

    def __init__(
        self,
        模型标识: str,
        rank: int = 8,
        alpha: int = 16,
        lr: float = 1e-4,
        序列长度: int = 512,
        累积: int = 4,
        输出目录: Optional[str] = None,
        日志间隔: int = 10,
        种子: int = 42,
        逐token: bool = False,
        层重算: bool = False,
        融合算子: bool = False,
        存点间隔: int = 50,
    ) -> None:
        条目 = 读取矩阵条目(模型标识)
        self.基座标识 = 模型标识
        self.基座文件 = 解析基座文件路径(条目)

        预算GB = 显存守卫(self.基座文件)  # 训练开始前校验，绝不静默 OOM

        # ---- 基座加载（策略与引擎同款：cuda fp16 失败回退 cpu fp32）----
        rwkv路径 = str(self.基座文件.with_suffix("")) if self.基座文件.suffix == ".pth" else str(self.基座文件)
        策略 = 引擎模块._推理环境.get("加载策略", "cuda fp16")
        回退策略 = 引擎模块._推理环境.get("CPU回退策略", "cpu fp32")
        try:
            self.模型 = RWKV(rwkv路径, 策略)
            self.加载策略 = 策略
        except Exception as e:
            print(f"策略 '{策略}' 加载失败（{e}），回退 '{回退策略}'")
            self.模型 = RWKV(rwkv路径, 回退策略)
            self.加载策略 = 回退策略
        self.设备 = self.模型.z["emb.weight"].device
        self.权重dtype = self.模型.z["emb.weight"].dtype

        # ---- World 分词器（词表由 rwkv pip 包内捆绑）----
        词表 = os.path.join(os.path.dirname(引擎模块._rwkv包.__file__), "rwkv_vocab_v20230424.txt")
        self.分词器 = TRIE_TOKENIZER(词表)

        # ---- 适配器与优化器（仅 A/B 进优化器）----
        self.适配器 = LoRA适配器(self.模型, rank=rank, alpha=alpha, 种子=种子)
        self.优化器 = torch.optim.AdamW(self.适配器.参数列表, lr=float(lr))

        self.lr = float(lr)
        self.序列长度 = int(序列长度)
        self.累积 = int(累积)
        self.日志间隔 = int(日志间隔)
        self.种子 = int(种子)
        # 前向路径：默认序列版（一次算完 24 层）；逐token=True 走旧逐 token 路径（数值对照/兜底）
        self.逐token = bool(逐token)
        self.层重算 = bool(层重算)
        # 融合算子：Triton chunked 扫描替换逐 token 递推（需 triton + fla，GPU 才有效）
        self.融合算子 = bool(融合算子) and not self.逐token
        if 融合算子 and not self.逐token:
            可用, 说明 = 融合可用()
            if 可用:
                print(f"[融合算子] 已启用：{说明}")
            else:
                self.融合算子 = False
                print(f"[融合算子] 不可用，回退序列逐 token 递推：{说明}")
        self.存点间隔 = max(1, int(存点间隔))

        self.输出目录 = Path(输出目录) if 输出目录 else _RWKE_DIR / "适配器"
        self.检查点路径 = self.输出目录 / f"{模型标识}_lora.pt"
        self.最优检查点路径 = self.输出目录 / f"{模型标识}_lora_best.pt"

        self.样本缓存: List[Tuple[List[int], int]] = []  # (token序列, 目标起点)
        self.跳过样本数 = 0
        self.step = 0
        self.loss历史: List[float] = []
        self.最优loss = float("inf")
        print(
            f"训练器就绪：基座={模型标识}（{self.加载策略}, 预算 {预算GB:.0f}GB）"
            f" n_layer={self.模型.n_layer} n_embd={self.模型.n_embd} | "
            f"rank={self.适配器.rank} alpha={self.适配器.alpha} "
            f"适配器参数量={self.适配器.参数量:,}（基座全部冻结）"
        )

    # ──────────────────────────────────────────────
    # 数据
    # ──────────────────────────────────────────────

    def 加载数据(self, 数据路径: str) -> None:
        """JSONL（每行 {"任务类型","提示","目标",...}）→ 编码并缓存 token 序列与目标起点。"""
        p = Path(数据路径)
        if not p.exists():
            raise FileNotFoundError(f"训练数据不存在：{p}")
        总行数 = 0
        坏行 = 0
        self.样本缓存 = []
        for 行号, 行 in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            行 = 行.strip()
            if not 行:
                continue
            总行数 += 1
            try:
                样本 = json.loads(行)
            except json.JSONDecodeError:
                坏行 += 1
                continue
            if not isinstance(样本, dict):
                坏行 += 1
                continue
            结果 = self._样本转token(样本)
            if 结果 is None:
                self.跳过样本数 += 1
                continue
            self.样本缓存.append(结果)
        self.跳过样本数 += 坏行
        if not self.样本缓存:
            raise ValueError(f"数据集中没有可用样本：{p}（{总行数} 行，其中不可用 {self.跳过样本数} 条）")
        print(f"数据加载：{len(self.样本缓存)} 条可用样本（截断/空目标跳过 {self.跳过样本数} 条），来源 {p}")

    def _样本转token(self, 样本: dict) -> Optional[Tuple[List[int], int]]:
        """
        样本 → (token 序列, 目标起点)。提示与目标分别编码（保证 loss 边界精确）后拼接，
        超长从左截断到 --序列长度；截断后目标区须非空且至少存在 1 个计损位置，否则返回 None。
        """
        提示token = list(self.分词器.encode(str(样本.get("提示", ""))))
        目标token = list(self.分词器.encode(str(样本.get("目标", ""))))
        全部 = 提示token + 目标token
        if len(全部) > self.序列长度:
            全部 = 全部[: self.序列长度]  # 从左截断：保留提示与目标开头
        目标起点 = min(len(提示token), len(全部))
        # 计损位置 t 满足：t+1 属于目标区且 t ≤ N-2 → 目标区非空且总长 ≥2 才有梯度信号
        if len(全部) < 2 or 目标起点 >= len(全部):
            return None
        return 全部, 目标起点

    # ──────────────────────────────────────────────
    # checkpoint（保存 / 恢复）
    # ──────────────────────────────────────────────

    def 保存checkpoint(self, 路径=None, 标记: str = "") -> Path:
        """A/B（CPU fp32）+ rank/alpha + 基座标识 + step/loss历史 + 优化器状态 + 序列长度。"""
        p = Path(路径) if 路径 else self.检查点路径
        p.parent.mkdir(parents=True, exist_ok=True)
        A快照, B快照 = self.适配器.cpu快照()
        内容 = {
            "适配器A": A快照,
            "适配器B": B快照,
            "rank": self.适配器.rank,
            "alpha": self.适配器.alpha,
            "基座标识": self.基座标识,
            "step": self.step,
            "loss历史": list(self.loss历史),
            "优化器状态": self.优化器.state_dict(),
            "序列长度": self.序列长度,
        }
        torch.save(内容, str(p))
        if 标记:
            print(f"  checkpoint 已保存（{标记}）：{p}", flush=True)
        return p

    def 恢复checkpoint(self, 路径=None) -> None:
        """从 checkpoint 恢复 A/B/优化器状态/step/loss历史（基座标识与 rank/alpha 不符即拒绝，防错挂）。"""
        p = Path(路径) if 路径 else self.检查点路径
        if not p.exists():
            raise FileNotFoundError(f"checkpoint 不存在：{p}")
        内容 = torch.load(str(p), map_location="cpu")
        if 内容.get("基座标识") != self.基座标识:
            raise ValueError(
                f"checkpoint 基座标识不符（防错挂）：文件为 '{内容.get('基座标识')}'，当前为 '{self.基座标识}'"
            )
        if int(内容.get("rank", self.适配器.rank)) != self.适配器.rank or \
                int(内容.get("alpha", self.适配器.alpha)) != self.适配器.alpha:
            raise ValueError(
                f"checkpoint rank/alpha 不符：文件为 ({内容.get('rank')}, {内容.get('alpha')})，"
                f"当前为 ({self.适配器.rank}, {self.适配器.alpha})"
            )
        self.适配器.载入快照(内容["适配器A"], 内容["适配器B"], self.设备)
        self.优化器.load_state_dict(内容["优化器状态"])
        self.step = int(内容.get("step", 0))
        self.loss历史 = [float(x) for x in 内容.get("loss历史", [])]
        self.最优loss = min(self.loss历史) if self.loss历史 else float("inf")
        print(f"已恢复 checkpoint：{p}（step={self.step}，loss历史 {len(self.loss历史)} 条）")

    # ──────────────────────────────────────────────
    # 训练
    # ──────────────────────────────────────────────

    def 训练步(self, token列表: List[int], 目标起点: int) -> Optional[float]:
        """
        单样本：按模式分派——默认序列版（一次算完 24 层，快），逐token=True 走旧逐 token 路径
        （数值对照/兜底）。返回该样本平均 loss（未除累积）。
        """
        if self.逐token:
            return self._训练步逐token(token列表, 目标起点)
        return self._训练步序列(token列表, 目标起点)

    def _训练步序列(self, token列表: List[int], 目标起点: int) -> Optional[float]:
        """
        序列路径：整条序列一次前向（注入 W_eff），对目标区一次算 fp32 cross_entropy
        （均值），再除以梯度累积步数 backward。数学上与逐 token 路径等价。
        """
        缓存 = 计算有效权重缓存(self.模型, self.适配器)
        logits = 训练前向序列(self.模型, token列表, 缓存, 层重算=self.层重算,
                              融合算子=self.融合算子)
        样本loss = 序列目标交叉熵(logits, token列表, 目标起点)
        if 样本loss is None:
            return None
        (样本loss / self.累积).backward()
        return float(样本loss.detach().item())

    def _训练步逐token(self, token列表: List[int], 目标起点: int) -> Optional[float]:
        """
        单样本：顺序单 token 前向（t 从 0..N-2），对 t+1 属于目标区的位置累计交叉熵
        （fp32 计算），样本均值除以梯度累积步数后 backward。返回该样本平均 loss（未除累积）。
        """
        state = 新零态(self.模型)
        缓存 = 计算有效权重缓存(self.模型, self.适配器)
        loss合计: Optional[torch.Tensor] = None
        计损数 = 0
        for t in range(len(token列表) - 1):
            logits, state = 训练前向单步(self.模型, int(token列表[t]), state, 缓存)
            if t + 1 >= 目标起点:
                目标 = torch.tensor(int(token列表[t + 1]), device=logits.device, dtype=torch.long)
                单步loss = F.cross_entropy(logits.float().squeeze(0), 目标)
                loss合计 = 单步loss if loss合计 is None else loss合计 + 单步loss
                计损数 += 1
        if 计损数 == 0:
            return None
        样本loss = loss合计 / 计损数
        (样本loss / self.累积).backward()
        return float(样本loss.detach().item())

    def _步进(self, 组loss累计: float, 组内样本数: int) -> None:
        """优化器 step + step 计数 + loss 历史 + 日志 + 最优/滚动 checkpoint。"""
        self.优化器.step()
        self.优化器.zero_grad(set_to_none=True)
        self.step += 1
        步loss = 组loss累计 / max(组内样本数, 1)
        self.loss历史.append(步loss)
        if not math.isfinite(步loss):
            self.保存checkpoint(self.检查点路径, 标记="异常（loss 非有限）")
            raise RuntimeError(
                f"step {self.step} loss 非有限（{步loss}）：fp16 前向数值溢出。"
                "【降档建议】减小 --lr 或 --序列长度，从 checkpoint --恢复 后重训"
            )
        if 步loss < self.最优loss:
            self.最优loss = 步loss
            self.保存checkpoint(self.最优检查点路径, 标记=f"step{self.step} 最优")
        # 滚动存点：每 --存点间隔 步写默认 checkpoint，崩溃后最多丢 N 步（epoch 内也有落盘）
        if self.step % self.存点间隔 == 0:
            self.保存checkpoint(self.检查点路径, 标记=f"滚动存点 step{self.step}")
        if self.step % self.日志间隔 == 0 or self.step == 1:
            当前lr = self.优化器.param_groups[0]["lr"]
            峰值GB = torch.cuda.max_memory_allocated() / 2**30 if self.设备.type == "cuda" else 0.0
            print(f"[step {self.step}] loss={步loss:.4f} lr={当前lr:.3e} 显存峰值={峰值GB:.2f}GB", flush=True)

    def 训练(self, epochs: int = 2, 上限步: int = 0) -> dict:
        """
        主训练循环：每 epoch 洗牌（固定种子可复现）；每累积步 step 一次；epoch 末 /
        最优 step / KeyboardInterrupt / 结束时保存 checkpoint。返回训练摘要 dict（并落盘 JSON）。
        """
        if not self.样本缓存:
            raise RuntimeError("先调用 加载数据()")
        epochs = int(epochs)
        上限步 = int(上限步)
        cuda可用 = self.设备.type == "cuda"
        if cuda可用:
            torch.cuda.reset_peak_memory_stats()
        开始时间 = time.time()
        中断 = False
        达上限 = (上限步 > 0 and self.step >= 上限步)
        组内样本数 = 0
        组loss累计 = 0.0
        if 达上限:
            print(f"step {self.step} 已达 --上限步 {上限步}，跳过训练")

        print(
            f"开始训练：epochs={epochs} 上限步={上限步 or '不限'} 累积={self.累积} "
            f"序列长度={self.序列长度} lr={self.lr} 样本={len(self.样本缓存)} "
            f"前向={'逐token' if self.逐token else '序列'}"
            f"{'（层重算）' if (self.层重算 and not self.逐token) else ''} "
            f"存点间隔={self.存点间隔}"
        )
        self.优化器.zero_grad(set_to_none=True)
        try:
            for epoch in range(1, epochs + 1):
                if 达上限:
                    break
                顺序 = list(range(len(self.样本缓存)))
                random.Random(self.种子 * 1000 + epoch).shuffle(顺序)
                组内样本数 = 0
                组loss累计 = 0.0
                for idx in 顺序:
                    token列表, 目标起点 = self.样本缓存[idx]
                    loss = self.训练步(token列表, 目标起点)
                    if loss is None:
                        self.跳过样本数 += 1
                        continue
                    组内样本数 += 1
                    组loss累计 += loss
                    if 组内样本数 >= self.累积:
                        self._步进(组loss累计, 组内样本数)
                        组内样本数 = 0
                        组loss累计 = 0.0
                        if 上限步 and self.step >= 上限步:
                            达上限 = True
                            break
                if 达上限:
                    break
                if 组内样本数 > 0:  # epoch 末不足一个累积组：按当前累计梯度补一步
                    self._步进(组loss累计, 组内样本数)
                    组内样本数 = 0
                    if 上限步 and self.step >= 上限步:
                        达上限 = True
                        break
                self.保存checkpoint(self.检查点路径, 标记=f"epoch{epoch} 末")
        except KeyboardInterrupt:
            print("收到 KeyboardInterrupt：保存 checkpoint 后优雅停止")
            中断 = True

        if 组内样本数 > 0:  # 中断时补进尾组（正常结束/达上限时已清零）
            self._步进(组loss累计, 组内样本数)
            组内样本数 = 0
        self.保存checkpoint(self.检查点路径, 标记="最终")

        峰值GB = torch.cuda.max_memory_allocated() / 2**30 if cuda可用 else 0.0
        摘要 = {
            "基座标识": self.基座标识,
            "加载策略": self.加载策略,
            "rank": self.适配器.rank,
            "alpha": self.适配器.alpha,
            "lr": self.lr,
            "序列长度": self.序列长度,
            "累积": self.累积,
            "前向模式": "逐token" if self.逐token else "序列",
            "层重算": bool(self.层重算 and not self.逐token),
            "融合算子": bool(self.融合算子),
            "存点间隔": self.存点间隔,
            "epochs请求": epochs,
            "上限步": 上限步,
            "step": self.step,
            "最终loss": self.loss历史[-1] if self.loss历史 else None,
            "最优loss": self.最优loss if math.isfinite(self.最优loss) else None,
            "loss历史": list(self.loss历史),
            "可用样本数": len(self.样本缓存),
            "跳过样本数": self.跳过样本数,
            "适配器参数量": self.适配器.参数量,
            "显存峰值GB": round(峰值GB, 3),
            "耗时秒": round(time.time() - 开始时间, 1),
            "中断": 中断,
            "检查点": str(self.检查点路径),
            "最优检查点": str(self.最优检查点路径) if self.最优检查点路径.exists() else None,
        }
        self.输出目录.mkdir(parents=True, exist_ok=True)
        摘要路径 = self.输出目录 / f"{self.基座标识}_训练摘要.json"
        with open(摘要路径, "w", encoding="utf-8") as f:
            json.dump(摘要, f, ensure_ascii=False, indent=2)
        状态词 = "中断" if 中断 else ("达上限" if 达上限 else "完成")
        print(
            f"训练{状态词}：step={self.step} 最终loss={摘要['最终loss']} "
            f"最优loss={摘要['最优loss']} 显存峰值={摘要['显存峰值GB']}GB 耗时={摘要['耗时秒']}s"
        )
        print(f"训练摘要已保存：{摘要路径}")
        return 摘要

    # ──────────────────────────────────────────────
    # 观测辅助（合并等价性验证 / 下游评估）
    # ──────────────────────────────────────────────

    @torch.no_grad()
    def 适配器前向序列(self, token列表: List[int]) -> torch.Tensor:
        """基座 + 适配器 前向整条序列，返回末位 logits [vocab]（fp32）。"""
        state = 新零态(self.模型)
        缓存 = 计算有效权重缓存(self.模型, self.适配器)
        logits = None
        for t in range(len(token列表)):
            logits, state = 训练前向单步(self.模型, int(token列表[t]), state, 缓存)
        return logits.squeeze(0).float()


# ======================================================================
# 合并（ΔW 并入基座，导出同格式 .pth；元数据写同名 .json 旁车）
# ======================================================================

def 合并适配器(基座文件: Path, checkpoint路径: Path, 输出文件: Path) -> dict:
    """
    torch.load 基座 .pth（平铺张量 dict，map_location='cpu'），对每个被适配键
    W += 缩放*(B@A)，保存为 RWKV() 可直接加载的同构 .pth。

    存储决策（关键）：被适配键先镜像 RWKV 加载后的"运行时权重"（raw→fp16，与加载器
    ``.to(DTYPE)`` 逐位一致；cuda fp16 部署策略下的 z 即此值），叠加 fp64 精度的 ΔW 后
    以 fp16 存储。原因：World 系列基座 pth 实测为 bf16 存储（7 位尾数），若按原 dtype
    保存，ΔW 会被 bf16 截断，与运行时 fp16 路径产生实测 0.1+ 的 logits 偏差（fp16 量化
    噪声容忍度 0.1 会被击穿）；改存 fp16 后"合并模型加载后的 z'"与"训练前向 W_eff"
    逐位相等，合并等价性严格成立。其余键保持原样（加载器逐键 .to(DTYPE) 兼容混合 dtype）。

    元数据（"适配器已合并"/基座标识/rank/alpha/step）写同名 .json 旁车：已核对
    rwkv.model L255-265 对 pth 内每个键做张量运算，附加字符串键会使加载崩溃。
    （raw 中被适配权重为 [out, in] 惯例，加载 .t() 后 z 为 x@W 的 [in, out]；
    合并侧与训练侧共用 计算ΔW，方向自洽。）
    """
    基座文件 = Path(基座文件)
    checkpoint路径 = Path(checkpoint路径)
    输出文件 = Path(输出文件)
    if not 基座文件.exists():
        raise FileNotFoundError(f"基座权重不存在：{基座文件}")
    if not checkpoint路径.exists():
        raise FileNotFoundError(f"适配器 checkpoint 不存在：{checkpoint路径}")

    内容 = torch.load(str(checkpoint路径), map_location="cpu")
    A字典 = 内容["适配器A"]
    B字典 = 内容["适配器B"]
    缩放 = float(内容["alpha"]) / float(内容["rank"])
    print(
        f"合并：基座 {基座文件.name} + 适配器 {checkpoint路径.name}"
        f"（rank={内容['rank']} alpha={内容['alpha']} step={内容.get('step')}）"
    )

    权重 = torch.load(str(基座文件), map_location="cpu", mmap=True)
    缺失 = [k for k in A字典 if k not in 权重]
    if 缺失:
        raise KeyError(f"基座 pth 缺少被适配键（前 3 个）：{缺失[:3]}")
    for 名称 in A字典:
        ΔW = 计算ΔW(A字典[名称], B字典[名称], 缩放)              # 与 z 同形状的 ΔW（fp32）
        原始 = 权重[名称]
        # 镜像运行时权重（raw→fp16，与加载器 .to(DTYPE) 一致）→ 叠加 ΔW → 以 fp16 存储，
        # 使合并模型加载后的 z' 与训练前向 W_eff 逐位相等（详见函数 docstring 存储决策）
        运行时z形 = 原始.to(torch.float16).t().float() + ΔW      # z 形状（x@W 约定）
        权重[名称] = 运行时z形.t().contiguous().to(torch.float16)
    输出文件.parent.mkdir(parents=True, exist_ok=True)
    torch.save(权重, str(输出文件))

    元数据 = {
        "适配器已合并": True,
        "基座标识": 内容.get("基座标识", ""),
        "基座文件": 基座文件.name,
        "适配器checkpoint": checkpoint路径.name,
        "rank": 内容["rank"],
        "alpha": 内容["alpha"],
        "step": 内容.get("step", 0),
    }
    旁车路径 = 输出文件.with_suffix(".json")
    with open(旁车路径, "w", encoding="utf-8") as f:
        json.dump(元数据, f, ensure_ascii=False, indent=2)
    print(f"合并完成：{输出文件}（元数据旁车 {旁车路径.name}）")
    return 元数据


def 默认checkpoint路径(模型标识: str, 输出目录=None) -> Path:
    """默认 checkpoint 文件：{输出目录}/{基座标识}_lora.pt。"""
    目录 = Path(输出目录) if 输出目录 else _RWKE_DIR / "适配器"
    return 目录 / f"{模型标识}_lora.pt"


# ======================================================================
# CLI
# ======================================================================

def main(argv=None) -> int:
    解析器 = argparse.ArgumentParser(
        description="RWKV-7 纯 PyTorch LoRA 微调核心（Windows 单卡；训练 / 断点续训 / 合并）"
    )
    解析器.add_argument("--模型", required=True, help="模型矩阵标识（如 RWKV7-World-0.4B）")
    解析器.add_argument("--数据", default=None, help="训练数据 JSONL 路径（每行 {任务类型,提示,目标,...}）")
    解析器.add_argument("--epochs", type=int, default=2, help="训练轮数（默认 2）")
    解析器.add_argument("--rank", type=int, default=8, help="LoRA 秩（默认 8）")
    解析器.add_argument("--alpha", type=int, default=16, help="LoRA alpha（默认 16）")
    解析器.add_argument("--lr", type=float, default=1e-4, help="学习率（默认 1e-4）")
    解析器.add_argument("--序列长度", type=int, default=512, help="序列截断长度（默认 512）")
    解析器.add_argument("--累积", type=int, default=4, help="梯度累积步数（默认 4）")
    解析器.add_argument("--上限步", type=int, default=0, help="优化器步数上限（默认 0=不限）")
    解析器.add_argument("--输出目录", default=None, help="checkpoint/摘要输出目录（默认 rwke/适配器）")
    解析器.add_argument("--恢复", nargs="?", const="auto", default=None,
                        help="从 checkpoint 恢复（无值取默认 {输出目录}/{基座标识}_lora.pt，或给路径）")
    解析器.add_argument("--合并", default=None, metavar="输出.pth",
                        help="合并模式：把 checkpoint 的 ΔW 并入基座导出该路径（元数据写同名 .json）")
    解析器.add_argument("--日志间隔", type=int, default=10, help="日志打印步间隔（默认 10）")
    解析器.add_argument("--种子", type=int, default=42, help="适配器初始化与洗牌种子（默认 42）")
    解析器.add_argument("--逐token", action="store_true",
                        help="回退到旧逐 token 前向（默认走快的序列前向；用于数值对照/兜底）")
    解析器.add_argument("--层重算", action="store_true",
                        help="序列前向对每层做 torch.utils.checkpoint 分段重算（省显存、略慢）")
    解析器.add_argument("--融合算子", action="store_true",
                        help="用 fla.chunk_rwkv7（Triton 分块扫描，含反向）替换逐 token 递推；"
                             "需 triton-windows + flash-linear-attention（实测递推提速 ~60×，精度 ~1e-3）")
    解析器.add_argument("--存点间隔", type=int, default=50,
                        help="滚动 checkpoint 落盘间隔步数（默认 50；崩溃后最多丢这么多步）")
    解析器.add_argument("--自动续训", action="store_true",
                        help="启动时若默认 checkpoint 存在则自动恢复（等价默认 --恢复 auto）")
    参数 = 解析器.parse_args(argv)

    try:
        条目 = 读取矩阵条目(参数.模型)
        基座文件 = 解析基座文件路径(条目)

        if 参数.合并:
            if 参数.恢复 and 参数.恢复 != "auto":
                checkpoint路径 = Path(参数.恢复)
            else:
                checkpoint路径 = 默认checkpoint路径(参数.模型, 参数.输出目录)
            合并适配器(基座文件, checkpoint路径, Path(参数.合并))
            return 0

        if not 参数.数据:
            print("错误：训练模式需要 --数据（JSONL 路径；合并请用 --合并）")
            return 1
        训练器 = 微调训练器(
            参数.模型,
            rank=参数.rank,
            alpha=参数.alpha,
            lr=参数.lr,
            序列长度=参数.序列长度,
            累积=参数.累积,
            输出目录=参数.输出目录,
            日志间隔=参数.日志间隔,
            种子=参数.种子,
            逐token=参数.逐token,
            层重算=参数.层重算,
            融合算子=参数.融合算子,
            存点间隔=参数.存点间隔,
        )
        训练器.加载数据(参数.数据)
        if 参数.恢复:
            训练器.恢复checkpoint(None if 参数.恢复 == "auto" else 参数.恢复)
        elif 参数.自动续训:
            # 自动续训：默认 checkpoint 存在即恢复，否则从头训练
            if 训练器.检查点路径.exists():
                print(f"--自动续训：发现默认 checkpoint {训练器.检查点路径}，自动恢复")
                训练器.恢复checkpoint(None)
            else:
                print(f"--自动续训：未找到默认 checkpoint {训练器.检查点路径}，从头训练")
        训练器.训练(参数.epochs, 参数.上限步)
        return 0
    except 显存守卫异常 as e:
        print(f"显存守卫：{e}")
        return 2
    except KeyboardInterrupt:
        print("中断")
        return 0
    except Exception as e:
        print(f"错误：{e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
