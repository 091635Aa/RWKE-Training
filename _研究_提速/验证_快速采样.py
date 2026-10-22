# -*- coding: utf-8 -*-
"""快速采样 A/B：速度对比 + 分布一致性抽检（同一 logits 大样本采样，对比 top 命中分布）。"""
import os
os.environ["RWKV_V7_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "0"
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import time
import torch

_RWKE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _RWKE_DIR)
from rwke引擎 import RWKE引擎  # noqa: E402

print("[加载] …")
引擎 = RWKE引擎("RWKV7-G1-1.5B")

torch.manual_seed(123)
logits = (torch.randn(1, 引擎.vocab_size, device=引擎.设备) * 3.0)

def 计时(fn, n=200, 预热=10):
    for _ in range(预热):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000

标准 = 计时(lambda: 引擎._采样(logits, 1.0, 0.9, 50, 快速=False))
快速 = 计时(lambda: 引擎._采样(logits, 1.0, 0.9, 50, 快速=True))
print(f"标准采样: {标准:.3f} ms/步")
print(f"快速采样: {快速:.3f} ms/步  ({标准 / 快速:.2f}x)")

# ---- 分布一致性抽检：各采 3000 样本，对比 top-10 token 命中占比与样本熵 ----
def 大样本(快速, n=3000):
    torch.manual_seed(777)
    计数 = {}
    for _ in range(n):
        t = int(引擎._采样(logits, 1.0, 0.9, 50, 快速=快速).item())
        计数[t] = 计数.get(t, 0) + 1
    top = sorted(计数.items(), key=lambda x: -x[1])[:10]
    熵 = -sum((c / n) * __import__("math").log(c / n) for c in 计数.values())
    return top, 熵, len(计数)

top_s, 熵_s, 唯一_s = 大样本(False)
top_f, 熵_f, 唯一_f = 大样本(True)
print(f"标准: 唯一token {唯一_s}, 样本熵 {熵_s:.3f}, top5 {top_s[:5]}")
print(f"快速: 唯一token {唯一_f}, 样本熵 {熵_f:.3f}, top5 {top_f[:5]}")
集s = set(t for t, _ in top_s)
集f = set(t for t, _ in top_f)
交 = len(集s & 集f)
print(f"top10 重合度: {交}/10")
print(f"分布结论: 唯一数差 {abs(唯一_s-唯一_f)}、熵差 {abs(熵_s-熵_f):.3f}（<0.1 视为分布一致）")
