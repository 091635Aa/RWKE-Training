# -*- coding: utf-8 -*-
"""
集成验证：rwke引擎 快速步进（TS 整环）vs 旧路径（逐层 jit）
- 场景1 裸基座：30 步 logits/状态/hidden 三方比对
- 场景2 挂载EmoCompanion LoRA：W_eff 是否正确进入 TS 模块（版本失效重建机制）
- 场景3 真实 生成() 冒烟 + 速度（快速步进开关对比）
"""
import os
os.environ["RWKV_V7_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "0"
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import time
import torch
from pathlib import Path

_RWKE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RWKE_DIR))
from rwke引擎 import RWKE引擎  # noqa: E402

print("[加载] RWKV7-G1-1.5B …")
引擎 = RWKE引擎("RWKV7-G1-1.5B")
state0 = 引擎.模型.generate_zero_state()
if state0 and state0[0].device != 引擎.设备:
    state0 = [t.to(引擎.设备) for t in state0]

步序列 = [(59 + s * 37) % 65535 for s in range(30)]


def _跑30步(路径: str):
    """路径: 'legacy' | 'fast'。返回 (logits列表, 末状态, hidden列表)。"""
    引擎.快速步进 = (路径 == "fast")
    引擎._ts权重版本 = -1  # 强制重建（切回 legacy 再切回 fast 时也要重建，模拟真实时序）
    st = [t.clone() for t in state0]
    lgs, hds = [], []
    for tok in 步序列:
        lg, st, hd = 引擎._前向单步(tok, st)
        lgs.append(lg.squeeze(0).float().clone())
        hds.append(hd.clone())
    return lgs, st, hds


def _比对(场景: str) -> None:
    lg_l, st_l, hd_l = _跑30步("legacy")
    lg_f, st_f, hd_f = _跑30步("fast")
    d_logits = max((a - b).abs().max().item() for a, b in zip(lg_l, lg_f))
    d_hidden = max((a - b).abs().max().item() for a, b in zip(hd_l, hd_f))
    d_kv = max(
        (st_l[i * 3 + 1].float() - st_f[i * 3 + 1].float()).abs().max().item()
        for i in range(引擎.n_layer)
    )
    print(f"[{场景}] 30步 logits最大差={d_logits:.3e}  hidden最大差={d_hidden:.3e}  kv最大差={d_kv:.3e}")
    return d_logits


差1 = _比对("场景1 裸基座")

# ---- 场景2：挂EmoCompanion LoRA（W_eff 进入 TS 模块）----
print("[挂载适配器] EmoCompanion LoRA …")
引擎._挂载适配器(str(_RWKE_DIR / "适配器" / "RWKV7-G1-1.5B_lora_best.pt"))
差2 = _比对("场景2 挂LoRA")

# ---- 场景3：真实 生成() 冒烟 + 速度 ----
import 对话系统  # noqa: E402  （复用心情调制逻辑路径）

for 路径 in ("fast", "legacy"):
    引擎.快速步进 = (路径 == "fast")
    引擎._ts权重版本 = -1
    引擎.回响池.清空()
    引擎._会话token集合 = set()
    引擎._末state = None
    t0 = time.perf_counter()
    回复 = 引擎.生成("你好，简单介绍一下你自己", max_new_tokens=64, 种子=7, 保留池=True)
    耗时 = time.perf_counter() - t0
    print(f"[生成-{路径}] {耗时:.2f}s（64 token 上限）→ {回复[:60]}")

print(f"\n结论：裸基座差={差1:.3e}，挂LoRA差={差2:.3e}（fp16 跨核噪声 <3 合格）")
