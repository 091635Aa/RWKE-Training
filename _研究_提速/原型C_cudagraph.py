# -*- coding: utf-8 -*-
"""
原型C：CUDA Graph 捕获 TS 整环步进 + 惩罚向量化 A/B
====================================================
P1 惩罚应用 A/B：Python 逐 token 循环（旧）vs GPU 缓冲向量化（新），n=400
P2 CUDA Graph：捕获 ts模块+状态回拷 整环 → replay 裸速 + 30 步等价性
"""
import os
os.environ["RWKV_V7_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "0"
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import time
import torch
from pathlib import Path
from typing import List

_RWKE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RWKE_DIR))
from rwke引擎 import RWKE引擎  # noqa: E402

_结果: dict = {"设备": torch.cuda.get_device_name(0)}


def 计时(函数, 次数, 预热=5):
    for _ in range(预热):
        函数()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(次数):
        函数()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / 次数 * 1000


# ======================================================================
# P1 惩罚应用 A/B（n=400 会话 token）
# ======================================================================
print("===== P1 惩罚应用 A/B（n=400）=====")
N = 400
ids = list(range(100, 100 + N))
logits = torch.randn(1, 65536, device="cuda:0", dtype=torch.float32)
buf = torch.tensor(ids, dtype=torch.int64, device="cuda:0")


def _旧():
    for t in ids:
        logits[0, t] /= 1.05


def _新():
    logits.view(-1)[buf] /= 1.05


毫秒旧 = 计时(_旧, 50)
毫秒新 = 计时(_新, 50)
print(f"旧 Python 循环 n={N}: {毫秒旧:8.2f} ms/步")
print(f"新 GPU 向量化   n={N}: {毫秒新:8.2f} ms/步   ({毫秒旧 / 毫秒新:.0f}x)")
_结果["P1_惩罚旧_ms"] = round(毫秒旧, 2)
_结果["P1_惩罚新_ms"] = round(毫秒新, 2)

# ======================================================================
# P2 CUDA Graph 捕获 TS 整环步进
# ======================================================================
print("\n===== P2 原型C：CUDA Graph =====")
print("[加载] RWKV7-G1-1.5B …")
引擎 = RWKE引擎("RWKV7-G1-1.5B")
引擎.快速步进 = True
state0 = 引擎.模型.generate_zero_state()
if state0 and state0[0].device != 引擎.设备:
    state0 = [t.to(引擎.设备) for t in state0]

# 先触发 TS 模块构建
_ = 引擎._前向单步(59, [t.clone() for t in state0])
ts模块 = 引擎._ts模块

# 静态缓冲
state_in = [t.clone() for t in state0]
token_buf = torch.zeros((), dtype=torch.int64, device=引擎.设备)
logits_buf = torch.zeros(引擎.vocab_size, dtype=引擎.权重dtype, device=引擎.设备)
hidden_buf = torch.zeros(引擎.n_embd, dtype=引擎.权重dtype, device=引擎.设备)

token_buf.fill_(59)
_ = ts模块(token_buf, state_in)  # 预热（分配图外池）
torch.cuda.synchronize()

# 捕获：TS 调用 + 状态回拷（新状态 → 输入缓冲，replay 间自续）
图 = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):  # 侧流预热
        lg, ns, hd = ts模块(token_buf, state_in)
        for i in range(len(state_in)):
            state_in[i].copy_(ns[i])
        logits_buf.copy_(lg.view(-1))
        hidden_buf.copy_(hd)
torch.cuda.current_stream().wait_stream(s)
try:
    with torch.cuda.graph(图):
        lg, ns, hd = ts模块(token_buf, state_in)
        for i in range(len(state_in)):
            state_in[i].copy_(ns[i])
        logits_buf.copy_(lg.view(-1))
        hidden_buf.copy_(hd)
    捕获成功 = True
    print("[捕获] CUDA Graph 捕获成功")
except Exception as e:
    捕获成功 = False
    print(f"[捕获] 失败：{type(e).__name__}: {str(e)[:200]}")

if 捕获成功:
    # 速度
    毫秒图 = 计时(lambda: 图.replay(), 60, 10)
    毫秒ts = 计时(lambda: ts模块(token_buf, state_in), 60, 5)
    print(f"(a) TS 步进（无图）        : {毫秒ts:8.2f} ms/步 → {1000 / 毫秒ts:.1f} tok/s")
    print(f"(b) CUDA Graph replay     : {毫秒图:8.2f} ms/步 → {1000 / 毫秒图:.1f} tok/s  ({毫秒ts / 毫秒图:.2f}x)")
    _结果["P2_ts步进_ms"] = round(毫秒ts, 2)
    _结果["P2_graph_ms"] = round(毫秒图, 2)

    # 等价性：三路对比（图 replay / 裸 TS eager / 引擎 fast），同起点同 token 序列
    # 注：三条链都必须逐帧函数式重绑状态，否则该链每步从零态重跑（假分歧）
    state_jit = [t.clone() for t in state0]
    st_e = [t.clone() for t in state0]
    for i in range(len(state_in)):
        state_in[i].copy_(state0[i])
    最大差裸 = 0.0
    最大差引擎 = 0.0
    for k in range(30):
        tok = (59 + k * 37) % 65535
        lg_j, st_j, hd_j = 引擎._前向单步(tok, state_jit)  # 引擎 fast 路径
        state_jit = st_j  # 函数式重绑
        token_buf.fill_(tok)
        图.replay()
        d_引擎 = (lg_j.float() - logits_buf.float()).abs().max().item()
        最大差引擎 = max(最大差引擎, d_引擎)
    print(f"30步 logits 最大差 图vs引擎fast（重绑修正后）: {最大差引擎:.3e}（应为 0）")
    _结果["P2_图等价差_引擎"] = 最大差引擎

    # 注意：graph 捕获使用静态输入缓冲 state_in；引擎集成时状态进出经
    # 「轮首 copy-in + 图内自续」——多轮/软重锚需 copy-in 一次（72 张量 ~0.4ms）
    print("[注] 图内状态自续：replay 前无需逐帧拷贝，轮首一次 copy-in 即可")

print("\n结果:", _结果)
