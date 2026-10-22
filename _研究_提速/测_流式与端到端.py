# -*- coding: utf-8 -*-
"""真流式首字延迟测量 + 3 轮端到端对照。"""
import json
import sys
import time

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = "http://127.0.0.1:8102"

print("---- 真流式首字延迟（3 次） ----")
for i in range(3):
    body = {"model": "rwkv7-g1-1.5b", "max_tokens": 128, "stream": True,
            "messages": [{"role": "user", "content": "你好呀，今天过得怎么样？"}]}
    t0 = time.perf_counter()
    首字 = None
    总字符 = 0
    with requests.post(f"{BASE}/v1/chat/completions", json=body, stream=True, timeout=300) as r:
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            数据 = line[6:]
            if 数据 == "[DONE]":
                break
            块 = json.loads(数据)
            if "error" in 块:
                print("错误:", 块["error"]); break
            段 = 块["choices"][0]["delta"].get("content", "")
            if 段 and 首字 is None:
                首字 = time.perf_counter() - t0
            总字符 += len(段)
    总时 = time.perf_counter() - t0
    print(f"第{i+1}次: 首字 {首字*1000 if 首字 else -1:.0f}ms | 完整 {总时:.2f}s | {总字符} 字符")

print("\n---- 非流式 3 轮对话（对照） ----")
msgs = []
for 问 in ["你好呀，今天过得怎么样？", "我下午想去看电影，有什么推荐吗？", "看完电影晚上适合干什么？"]:
    msgs.append({"role": "user", "content": 问})
    t0 = time.perf_counter()
    r = requests.post(f"{BASE}/v1/chat/completions", timeout=300, json={
        "model": "rwkv7-g1-1.5b", "max_tokens": 128, "messages": msgs})
    t1 = time.perf_counter()
    回复 = r.json()["choices"][0]["message"]["content"]
    msgs.append({"role": "assistant", "content": 回复})
    print(f"问: {问}")
    print(f"答({t1-t0:.2f}s): {回复[:80]}")
    print()

r = requests.get(f"{BASE}/api/emotion", timeout=10).json()
print("情感端点:", json.dumps(r, ensure_ascii=False))
