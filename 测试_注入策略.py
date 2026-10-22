# -*- coding: utf-8 -*-
"""
RWKE 注入策略对比测试：条纹治理 + 情感能力增幅量化
====================================================

对同一引擎的三种注入策略各跑 100 轮连续对话（并行执行），
量化对比"条纹问题"（跨轮/单条重复模式）与情感能力增幅。

策略
----
S1 基线_裸        ：无注入（对照组）
S2 现行注入       ：λ=扫描表值，原版偏置语义
S3 缓和注入       ：偏置 RMS 归一（强度有界 0.5λ）+ 单轮前强后弱衰减（PIS）+ 池容量 32

指标
----
情感能力指数      = 情感词密度 × 1000（每千字情感词数）
情感能力增幅%     = (指数_策略 / 指数_基线 − 1) × 100
条纹率            = 0.5×跨轮重复率（相邻轮回复前 16 字相同的比例）
                    + 0.5×单条重复占比均值（单条内最长重复 run / 长度）
有效回复率        = 长度 ≥4 字回复占比

用法
----
    python 测试_注入策略.py                    # 0.4B 三策略并行各 100 轮
    python 测试_注入策略.py --模型 RWKV7-World-1.5B
    python 测试_注入策略.py --自动标定          # 缓和注入增幅<70% 时有界λ爬坡（最多3档）
"""

import argparse
import gc
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import rwke引擎 as 引擎模块  # noqa: E402
from rwke引擎 import RWKE引擎  # noqa: E402
from 测试_对比 import 情感词密度, 计算重复率  # noqa: E402
from 测试_连续对话 import 剧本消息, 记忆题, 记忆命中  # noqa: E402

_RWKE_DIR = Path(__file__).resolve().parent
_结果目录 = _RWKE_DIR / "results"
_报告目录 = _RWKE_DIR / "docs"


# ======================================================================
# 条纹指标
# ======================================================================

def 最长重复run占比(文本: str) -> float:
    """单条内最长连续重复字符 run 占比（"花花花花"→高；正常文本→低）。"""
    if not 文本:
        return 0.0
    最长 = 当前 = 1
    for i in range(1, len(文本)):
        当前 = 当前 + 1 if 文本[i] == 文本[i - 1] else 1
        最长 = max(最长, 当前)
    return 最长 / len(文本)


def 条纹率(轮记录: List[dict]) -> float:
    """0.5×跨轮重复率（相邻轮回复前16字相同） + 0.5×单条重复run占比均值。"""
    if not 轮记录:
        return 1.0
    回复s = [r["回复"] for r in 轮记录]
    跨轮重复 = sum(
        1 for i in range(1, len(回复s))
        if len(回复s[i]) >= 16 and len(回复s[i - 1]) >= 16 and 回复s[i][:16] == 回复s[i - 1][:16]
    ) / max(len(回复s) - 1, 1)
    run均值 = statistics.fmean([最长重复run占比(t) for t in 回复s if t])
    return round(0.5 * 跨轮重复 + 0.5 * run均值, 4)


# ======================================================================
# 单策略 100 轮
# ======================================================================

def 构造策略引擎(配置: dict) -> RWKE引擎:
    """顺序构造引擎（rwkv pip 的 DEVICE 全局不支持并发构造；轮级并行不受影响）。"""
    模型标识 = 配置["模型标识"]
    引擎 = RWKE引擎(
        模型标识,
        λ=配置.get("λ"),
        偏置归一=配置.get("偏置归一", False),
        注入衰减=配置.get("注入衰减", False),
        池容量=配置.get("池容量"),
    )
    引擎.启用注入 = 配置["注入"]
    引擎.启用捕获 = 配置["注入"]
    if 配置.get("挂载"):
        状态路径 = _RWKE_DIR / "states" / f"RoleB_状态_{模型标识}.pth"
        if 状态路径.exists():
            引擎.挂载日记状态(str(状态路径), 模型标识)
    引擎.回响池.清空()
    return 引擎


def 跑策略(引擎: RWKE引擎, 配置: dict, 轮数: int, max_new_tokens: int) -> dict:
    记录 = []
    for 轮号 in range(1, 轮数 + 1):
        消息 = 剧本消息(轮号 if 轮号 <= 100 else 100)
        起 = time.perf_counter()
        try:
            回复 = 引擎.生成(消息, max_new_tokens=max_new_tokens,
                             保留池=True, 延续对话=(轮号 > 1))
        except Exception as e:
            回复 = ""
            print(f"  [{配置['名称']}] 第{轮号}轮异常：{type(e).__name__}: {e}")
        记录.append({"轮号": 轮号, "回复": 回复, "延迟秒": round(time.perf_counter() - 起, 2)})

    有效 = [r for r in 记录 if len(r["回复"]) >= 4]
    密度均值 = statistics.fmean([情感词密度(r["回复"]) for r in 有效]) if 有效 else 0.0
    state字节 = sum(t.numel() * t.element_size() for t in 引擎._末state) if 引擎._末state else 0
    记忆命中数 = sum(1 for 轮号 in 记忆题
                    if 轮号 <= 轮数 and 记忆命中(记录[轮号 - 1]["回复"], 轮号))

    结果 = {
        "名称": 配置["名称"],
        "λ": 配置.get("λ"),
        "变体": {k: 配置.get(k, False) for k in ("偏置归一", "注入衰减")},
        "池容量": 配置.get("池容量"),
        "情感能力指数": round(密度均值 * 1000, 2),  # 每千字情感词数
        "条纹率": 条纹率(记录),
        "有效回复率": round(len(有效) / 轮数, 3),
        "均字": round(statistics.fmean([len(r["回复"]) for r in 记录]), 1),
        "记忆命中": f"{记忆命中数}/{len(记忆题)}",
        "均轮延迟秒": round(statistics.fmean([r["延迟秒"] for r in 记录]), 2),
        "stateKB": round(state字节 / 1024, 1),
        "记录": 记录,
    }
    del 引擎
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 结果


# ======================================================================
# 主流程（并行 + 有界自动标定）
# ======================================================================

def main() -> int:
    解析器 = argparse.ArgumentParser(description="RWKE 注入策略对比：条纹治理 + 情感能力增幅")
    解析器.add_argument("--模型", default="RWKV7-World-0.4B")
    解析器.add_argument("--轮数", type=int, default=100)
    解析器.add_argument("--max-new-tokens", type=int, default=48)
    解析器.add_argument("--自动标定", action="store_true",
                        help="缓和注入增幅<70%% 时有界 λ 爬坡（最多 3 档，达标即停）")
    解析器.add_argument("--顺序", action="store_true", help="禁用并行（顺序执行三策略）")
    参数 = 解析器.parse_args()

    配置 = 引擎模块._读系统配置()
    条目 = 配置["模型矩阵"][参数.模型]
    表项 = 配置["扫描表"][str(条目["hidden_dim"])]
    λ表值 = float(表项[0])
    剧本轮数 = min(参数.轮数, 100)

    策略定义 = [
        {"名称": "S1_基线_裸", "注入": False, "λ": None},
        {"名称": "S2_现行注入", "注入": True, "λ": λ表值},
        {"名称": "S3_缓和注入", "注入": True, "λ": λ表值,
         "偏置归一": True, "注入衰减": True, "池容量": 32, "挂载": True},
    ]
    策略定义 = [{**c, "模型标识": 参数.模型} for c in 策略定义]

    def 跑全部(策略s: List[dict]) -> List[dict]:
        # 引擎必须顺序构造（rwkv DEVICE 全局竞态）；轮执行可并行（GPU 算子释放 GIL）
        引擎s = [构造策略引擎(c) for c in 策略s]
        if 参数.顺序 or len(策略s) == 1:
            return [跑策略(e, c, 剧本轮数, 参数.max_new_tokens)
                    for e, c in zip(引擎s, 策略s)]
        with ThreadPoolExecutor(max_workers=len(策略s)) as 池:
            return list(池.map(lambda ec: 跑策略(ec[0], ec[1], 剧本轮数, 参数.max_new_tokens),
                               zip(引擎s, 策略s)))

    print(f"[配置] 模型={参数.模型} λ表值={λ表值} 轮数={剧本轮数}×{len(策略定义)}策略"
          f"（{'并行' if not 参数.顺序 else '顺序'}）")
    t0 = time.perf_counter()
    结果s = 跑全部(策略定义)
    print(f"[完成] 三策略 {剧本轮数} 轮耗时 {time.perf_counter() - t0:.0f}s")

    # ---- 自动标定（有界：最多 3 档 λ 爬坡，达标即停）----
    标定记录 = []
    基线指数 = 结果s[0]["情感能力指数"]
    s3 = 结果s[2]
    s3增幅 = (s3["情感能力指数"] / max(基线指数, 1e-9) - 1) * 100
    if 参数.自动标定 and (s3增幅 < 70 or s3["条纹率"] > 0.15):
        print(f"[标定] S3 增幅 {s3增幅:.0f}% / 条纹率 {s3['条纹率']} 未达目标（≥70%、≤0.15），λ 爬坡…")
        for λ爬 in [0.2, 0.3, 0.4]:
            if λ爬 <= λ表值:
                continue
            爬 = dict(策略定义[2], λ=λ爬, 名称=f"S3_缓和_λ{λ爬}")
            爬引擎 = 构造策略引擎(爬)
            r = 跑策略(爬引擎, 爬, 剧本轮数, 参数.max_new_tokens)
            r["增幅%"] = round((r["情感能力指数"] / max(基线指数, 1e-9) - 1) * 100, 1)
            标定记录.append({k: r[k] for k in ("名称", "λ", "情感能力指数", "增幅%", "条纹率", "有效回复率")})
            print(f"  [爬坡] λ={λ爬}: 指数={r['情感能力指数']} 增幅={r['增幅%']}% 条纹率={r['条纹率']}")
            if r["增幅%"] >= 70 and r["条纹率"] <= 0.15:
                print(f"  [标定] 达标，停止爬坡（有界自动停止）")
                r["名称"] = "S3_缓和_标定最优"
                结果s[2] = r
                break
        else:
            结果s[2]["名称"] = "S3_缓和_未达标"

    # ---- 汇总增幅 ----
    for r in 结果s:
        r["情感能力增幅%"] = round((r["情感能力指数"] / max(基线指数, 1e-9) - 1) * 100, 1)
    for r in 标定记录:
        pass  # 增幅已在循环内写入

    _结果目录.mkdir(exist_ok=True)
    json路径 = _结果目录 / f"注入策略_{参数.模型}.json"
    json路径.write_text(json.dumps(
        {"模型": 参数.模型, "轮数": 剧本轮数, "基线指数": 基线指数,
         "标定记录": 标定记录, "策略": 结果s},
        ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 控制台汇总 ----
    print("\n" + "═" * 72)
    print(f"{'策略':　<18}{'情感指数':>8}{'增幅%':>8}{'条纹率':>8}{'有效回复':>8}{'均字':>6}{'StateKB':>8}")
    for r in 结果s:
        print(f"{r['名称']:　<18}{r['情感能力指数']:>8}{r['情感能力增幅%']:>8}"
              f"{r['条纹率']:>8}{r['有效回复率']:>8}{r['均字']:>6}{r['stateKB']:>8}")
    if 标定记录:
        print("标定爬坡：")
        for m in 标定记录:
            print(f"  {m['名称']}: 指数={m['情感能力指数']} 增幅={m['增幅%']}% 条纹率={m['条纹率']}")
    print(f"[数据] {json路径}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
