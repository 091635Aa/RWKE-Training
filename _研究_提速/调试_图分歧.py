# -*- coding: utf-8 -*-
"""二分定位：CUDA Graph replay vs eager TS 的分歧来源（逐步 diff + 确定性检验）。"""
import os
os.environ["RWKV_V7_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "0"
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
from pathlib import Path
from typing import List

_RWKE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RWKE_DIR))
from rwke引擎 import RWKE引擎  # noqa: E402

print("[加载] …")
引擎 = RWKE引擎("RWKV7-G1-1.5B")
state0 = 引擎.模型.generate_zero_state()
if state0 and state0[0].device != 引擎.设备:
    state0 = [t.to(引擎.设备) for t in state0]
引擎.快速步进 = True
_ = 引擎._前向单步(59, [t.clone() for t in state0])
ts模块 = 引擎._ts模块

token_buf = torch.zeros((), dtype=torch.int64, device=引擎.设备)
logits_buf = torch.zeros(引擎.vocab_size, dtype=引擎.权重dtype, device=引擎.设备)
state_in = [t.clone() for t in state0]

token_buf.fill_(59)
_ = ts模块(token_buf, state_in)
torch.cuda.synchronize()
图 = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        lg, ns, hd = ts模块(token_buf, state_in)
        for i in range(len(state_in)):
            state_in[i].copy_(ns[i])
        logits_buf.copy_(lg.view(-1))
torch.cuda.current_stream().wait_stream(s)
with torch.cuda.graph(图):
    lg, ns, hd = ts模块(token_buf, state_in)
    for i in range(len(state_in)):
        state_in[i].copy_(ns[i])
    logits_buf.copy_(lg.view(-1))

# ---- 确定性检验：同起点 replay 两次 vs eager 两次 ----
for i in range(len(state_in)):
    state_in[i].copy_(state0[i])
token_buf.fill_(59)
图.replay()
r1 = logits_buf.float().clone()
# 重置状态（replay 一次已演进）
for i in range(len(state_in)):
    state_in[i].copy_(state0[i])
token_buf.fill_(59)
图.replay()
r2 = logits_buf.float().clone()
print(f"replay 确定性（两次同输入）: max diff = {(r1 - r2).abs().max().item():.3e}")

# eager 同输入两次
st_e = [t.clone() for t in state0]
lg_e1, ns_e, _ = ts模块(token_buf, st_e)
lg_e2, _, _ = ts模块(token_buf, st_e)
print(f"eager   确定性（两次同输入）: max diff = {(lg_e1.float() - lg_e2.float()).abs().max().item():.3e}")

# ---- 第 0 步：图 vs eager（同起点同 token）----
for i in range(len(state_in)):
    state_in[i].copy_(state0[i])
token_buf.fill_(59)
图.replay()
st_e = [t.clone() for t in state0]
lg_e, ns_e, _ = ts模块(token_buf, st_e)
print(f"第0步 图 vs eager logits diff = {(logits_buf.float() - lg_e.float()).abs().max().item():.3e}")

# ---- 逐步追踪 6 步：两条独立链 ----
state_g = None  # 图链：图缓冲自续
st_e2 = [t.clone() for t in state0]
for i in range(len(state_in)):
    state_in[i].copy_(state0[i])
for k in range(6):
    tok = (59 + k * 37) % 65535
    lg_e, st_e2, _ = ts模块(torch.tensor(tok, dtype=torch.int64, device=引擎.设备), st_e2)
    token_buf.fill_(tok)
    图.replay()
    d = (logits_buf.float() - lg_e.float()).abs().max().item()
    print(f"step {k}: token={tok}  图vs eager diff = {d:.3e}")
