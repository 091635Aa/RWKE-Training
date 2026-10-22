# -*- coding: utf-8 -*-
"""
双加速验证：推理（CUDA 图）× 训练（Triton 融合算子）实测对照
==============================================================

    python 验证_双加速.py 推理
    python 验证_双加速.py 训练
    python 验证_双加速.py 全部

推理三路（同一零态、同一 token 序列，32 步）：
    A 逐层   : 快速步进=False            （旧路径，纯 PyTorch 逐层）
    B 整环   : 快速步进=True, 图=False   （TS 整环 jit.script 单次调用）
    C 整环+图: 快速步进=True, 图=True    （现役默认，CUDA Graph 回放）
输出 ms/token 与相对 A 的倍率，并给出三路 logits 最大差（数值对照）。

训练两路（同数据同种子同超参，8 个样本 fwd+bwd 计时）：
    融合算子=False : 官方 TMix_seq（逐 token 递推）
    融合算子=True  : fla.chunk_rwkv7（Triton 分块扫描）
输出 s/step 与 loss 对照。

注意：训练侧需要独占显存（1.5B + LoRA 约 14GB），跑前请先停掉其他 GPU 任务。
"""
import os
import sys
import tempfile
import shutil
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("RWKV_V7_ON", "1")
os.environ["RWKV_CUDA_ON"] = "0"  # 算子只在推理侧可选；本脚本走纯 PyTorch，保证与基线可比

_RWKE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RWKE_DIR))

import torch  # noqa: E402

模型标识 = "RWKV7-G1-1.5B"
数据路径 = _RWKE_DIR / "训练数据" / "微调数据_train.jsonl"
令牌 = [(59 + s * 37) % 60000 for s in range(64)]


# ======================================================================
# 推理
# ======================================================================

def 跑推理(步数: int = 32) -> None:
    from rwke引擎 import RWKE引擎

    print(f"[推理] 加载 {模型标识} …", flush=True)
    引擎 = RWKE引擎(模型标识)

    def 计时(路径: str, 次数: int = 64):
        引擎.快速步进 = 路径 != "逐层"
        if 路径 == "整环+图":
            引擎._图 = None
            引擎._图权重版本 = -1
        else:
            引擎._图 = False  # 不建图：None 会被惰性重建
            引擎._图权重版本 = 引擎._权重版本
        引擎._ts权重版本 = -1  # 强制重建 TS 模块，消除首次编译开销
        st = [t.clone() for t in 引擎.模型.generate_zero_state()]
        st = [t.to(引擎.设备) for t in st]
        for tok in 令牌[:8]:  # 预热（含图捕获 / jit 编译）
            lg, st, _ = 引擎._前向单步(tok, st)
        torch.cuda.synchronize()
        起 = time.perf_counter()
        for tok in 令牌[8 : 8 + 次数]:
            lg, st, _ = 引擎._前向单步(tok, st)
        torch.cuda.synchronize()
        耗时 = time.perf_counter() - 起
        return 耗时 / 次数 * 1000.0, lg.squeeze(0).float().clone()

    print(f"\n{'路径':<10}{'ms/token':>12}{'倍率':>8}")
    结果 = {}
    for 路径 in ("逐层", "整环", "整环+图"):
        ms, _ = 计时(路径)
        结果[路径] = ms
        基准 = 结果["逐层"]
        print(f"{路径:<10}{ms:>12.3f}{(基准 / ms):>7.2f}×")

    # 数值对照：三路各自跑 32 步，比末位 logits
    向量 = {}
    for 路径 in ("逐层", "整环", "整环+图"):
        引擎.快速步进 = 路径 != "逐层"
        if 路径 == "整环+图":
            引擎._图 = None
            引擎._图权重版本 = -1
        else:
            引擎._图 = False
            引擎._图权重版本 = 引擎._权重版本
        引擎._ts权重版本 = -1
        st = [t.clone() for t in 引擎.模型.generate_zero_state()]
        st = [t.to(引擎.设备) for t in st]
        for i, tok in enumerate(令牌[:步数]):
            lg, st, _ = 引擎._前向单步(tok, st)
            if i == 步数 - 1:
                向量[路径] = lg.squeeze(0).float().clone()
    print(f"\n[数值] 32 步后 logits 最大差："
          f"整环 vs 逐层 = {(向量['整环'] - 向量['逐层']).abs().max().item():.3e}，"
          f"整环+图 vs 整环 = {(向量['整环+图'] - 向量['整环']).abs().max().item():.3e}"
          "（应为 0：同算子重排）")

    # 端到端：生成 128 token 实测
    print("\n[端到端] 生成 128 token：")
    for 路径 in ("逐层", "整环+图"):
        引擎.快速步进 = 路径 != "逐层"
        if 路径 == "整环+图":
            引擎._图 = None
            引擎._图权重版本 = -1
        else:
            引擎._图 = False
            引擎._图权重版本 = 引擎._权重版本
        引擎._ts权重版本 = -1
        引擎._末state = None
        引擎.回响池.清空()
        引擎._会话token集合 = set()
        起 = time.perf_counter()
        引擎.生成("你好，简单介绍一下你自己", max_new_tokens=128, 种子=7, 保留池=True)
        耗时 = time.perf_counter() - 起
        print(f"  {路径:<8} {耗时:.2f}s（{128 / 耗时:.1f} token/s）")


# ======================================================================
# 训练
# ======================================================================

def 跑训练(样本数: int = 6, 预热: int = 2) -> None:
    from 微调 import 微调训练器, 融合可用

    可用, 说明 = 融合可用()
    print(f"[训练] 融合算子可用性：{可用}（{说明}）")
    if not 可用:
        print("[训练] 融合算子不可用，跳过 —— 训练侧维持官方序列路径")
        return

    结果 = {}
    for 融合 in (False, True):
        临时 = Path(tempfile.mkdtemp(prefix="双加速_训练_"))
        try:
            print(f"\n[训练] 融合算子={融合} 加载基座 …", flush=True)
            训练器 = 微调训练器(
                模型标识, rank=16, alpha=32, lr=1e-4, 序列长度=256, 累积=4,
                输出目录=str(临时), 日志间隔=9999, 种子=42, 融合算子=融合, 存点间隔=10**9,
            )
            训练器.加载数据(str(数据路径))
            样本 = 训练器.样本缓存[: 预热 + 样本数]
            for s in 样本[:预热]:
                训练器.优化器.zero_grad(set_to_none=True)
                训练器.训练步(*s)
            torch.cuda.synchronize()
            起 = time.perf_counter()
            losses = []
            for s in 样本[预热:]:
                训练器.优化器.zero_grad(set_to_none=True)
                losses.append(训练器.训练步(*s))
            torch.cuda.synchronize()
            单步 = (time.perf_counter() - 起) / max(len(样本) - 预热, 1)
            结果[融合] = (单步, losses)
            print(f"[训练] 融合算子={融合}  s/step={单步:.2f}  "
                  f"loss={[('None' if x is None else f'{x:.4f}') for x in losses]}")
            del 训练器
        finally:
            shutil.rmtree(临时, ignore_errors=True)
            torch.cuda.empty_cache()

    if len(结果) == 2:
        提速 = 结果[False][0] / 结果[True][0]
        print(f"\n[训练结论] {结果[False][0]:.2f} → {结果[True][0]:.2f} s/step，提速 {提速:.2f}×")


if __name__ == "__main__":
    模式 = sys.argv[1] if len(sys.argv) > 1 else "全部"
    if 模式 in ("推理", "全部"):
        跑推理()
    if 模式 in ("训练", "全部"):
        跑训练()