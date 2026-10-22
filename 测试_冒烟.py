# -*- coding: utf-8 -*-
"""
RWKE 冒烟测试：四组对照
========================

默认对 RWKV7-World-0.4B 快速冒烟（可 --模型 指定矩阵内其他模型），对比四组输出：

    G1 裸 RWKV          ：启用注入=False，启用捕获=False，零态起步
    G2 回响注入(默认λ)   ：启用注入=True， 启用捕获=True， 零态起步
    G3 挂载日记State     ：启用注入=False，启用捕获=False，日记 State 起步
    G4 挂载State+注入    ：启用注入=True， 启用捕获=True， 日记 State 起步

每组采集：回复文本、首步延迟（含预填充）、步延迟中位数、总耗时、
峰值显存（allocated/reserved）、逐步语义熵均值（V1 同口径）、非有限 logits 步数、
回响池大小。日记 State 缺失时自动经 日记状态.生成日记状态 现场生成。

验证标准（tasks.md Task 6）
--------------------------
1. 四组均产出非空中文文本；
2. 无 NaN/Inf、无崩溃；
3. λ>0 组（G2）语义熵与裸组（G1）有可测差异（|ΔH| > 1e-6）。

用法
----
    python 测试_冒烟.py                       # 0.4B 默认冒烟
    python 测试_冒烟.py --模型 RWKV7-World-1.5B --max-new-tokens 128
    python 测试_冒烟.py --跳过日记状态         # 只跑 G1/G2
"""

import argparse
import gc
import re
import statistics
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import rwke引擎 as 引擎模块  # noqa: E402  （先行导入：确保 RWKV_V7_ON 等环境先于 rwkv 生效）
from rwke引擎 import RWKE引擎  # noqa: E402
from 日记状态 import 生成日记状态  # noqa: E402

_RWKE_DIR = Path(__file__).resolve().parent

# 语义熵（与工作区统一口径一致：softmax 熵，nats；-inf 槽位压到 -1e4）
def 计算语义熵(logits: torch.Tensor) -> float:
    if logits.dim() == 2:
        logits = logits[0]
    logits = logits.clone().float()
    logits[logits == float("-inf")] = -1e4
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-12)
    return float(-(probs * log_probs).sum().item())


def _含中文(文本: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", 文本))


# ======================================================================
# 单组执行
# ======================================================================

def _跑一组(
    引擎: RWKE引擎,
    组名: str,
    启用注入: bool,
    启用捕获: bool,
    挂载日记: bool,
    状态路径: Optional[str],
    消息: str,
    max_new_tokens: int,
) -> dict:
    引擎.启用注入 = 启用注入
    引擎.启用捕获 = 启用捕获

    if 挂载日记:
        元数据 = 引擎.挂载日记状态(状态路径, 引擎.模型标识)
        print(f"  [挂载] {Path(状态路径).name}（来源元数据：{元数据.get('模型标识', '?')}，"
              f"保存于 {元数据.get('时间戳', '?')}）")
    else:
        引擎.清除挂载状态()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        基线显存 = torch.cuda.memory_allocated() / 1024 ** 2

    熵列表: List[float] = []
    非有限步数 = [0]
    步延迟: List[float] = []
    上次时刻 = [time.perf_counter()]

    def logits_callback(步: int, logits: torch.Tensor) -> None:
        熵列表.append(计算语义熵(logits))
        if bool(torch.isnan(logits).any().item()) or bool(torch.isinf(logits).any().item()):
            非有限步数[0] += 1

    def 轮次回调(当前步数: int, 池) -> None:
        现在 = time.perf_counter()
        步延迟.append(现在 - 上次时刻[0])
        上次时刻[0] = 现在

    起点 = time.perf_counter()
    try:
        文本 = 引擎.生成(
            消息,
            max_new_tokens=max_new_tokens,
            logits_callback=logits_callback,
            轮次回调=轮次回调,
        )
        崩溃 = False
    except Exception as e:
        文本 = ""
        崩溃 = True
        print(f"  [异常] {组名} 生成失败：{type(e).__name__}: {e}")
    总耗时 = time.perf_counter() - 起点

    结果 = {
        "组名": 组名,
        "启用注入": 启用注入,
        "启用捕获": 启用捕获,
        "挂载日记State": 挂载日记,
        "文本": 文本,
        "字符数": len(文本),
        "含中文": _含中文(文本),
        "生成步数": len(熵列表),
        "语义熵均值": statistics.fmean(熵列表) if 熵列表 else float("nan"),
        "语义熵末步": 熵列表[-1] if 熵列表 else float("nan"),
        "非有限logits步数": 非有限步数[0],
        "崩溃": 崩溃,
        "总耗时秒": round(总耗时, 3),
        "首步延迟秒(含预填充)": round(步延迟[0], 4) if 步延迟 else None,
        "步延迟中位秒": round(statistics.median(步延迟[1:]), 4) if len(步延迟) > 1 else None,
        "tokens每秒": round((len(步延迟) - 1) / (总耗时 - 步延迟[0]), 2)
                      if len(步延迟) > 1 and 总耗时 > 步延迟[0] else None,
        "回响池大小": 引擎.回响池.大小,
    }

    if torch.cuda.is_available():
        结果["峰值显存MB_allocated"] = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
        结果["峰值显存MB_reserved"] = round(torch.cuda.max_memory_reserved() / 1024 ** 2, 1)
        结果["基线显存MB"] = round(基线显存, 1)

    return 结果


# ======================================================================
# 主流程
# ======================================================================

def main() -> int:
    解析器 = argparse.ArgumentParser(description="RWKE 冒烟测试：四组对照（裸/注入/挂State/挂State+注入）")
    解析器.add_argument("--模型", default="RWKV7-World-0.4B", help="模型矩阵标识（默认 RWKV7-World-0.4B）")
    解析器.add_argument("--消息", default="RoleB，今天过得怎么样？", help="四组共用的测试消息")
    解析器.add_argument("--max-new-tokens", type=int, default=96, help="每组最大生成 token 数（默认 96）")
    解析器.add_argument("--日记", default=None, help="日记文件路径（生成日记 State 用；默认取内置候选）")
    解析器.add_argument("--段大小", type=int, default=512, help="日记流式预填充每段 token 数（默认 512）")
    解析器.add_argument("--跳过日记状态", action="store_true", help="跳过 G3/G4（不生成、不挂载日记 State）")
    参数 = 解析器.parse_args()

    配置 = 引擎模块._读系统配置()
    矩阵 = 配置.get("模型矩阵", {})
    if 参数.模型 not in 矩阵:
        print(f"错误：模型标识 '{参数.模型}' 不在模型矩阵中：{list(矩阵)}")
        return 2

    # ---- 模型文件存在性 ----
    模型路径 = Path(矩阵[参数.模型]["路径"])
    if not 模型路径.is_absolute():
        模型路径 = _RWKE_DIR / 模型路径
    if not 模型路径.exists():
        print(f"错误：模型文件不存在：{模型路径}")
        print("请先运行：python RWKE/下载模型.py   （或 python RWKE/下载模型.py --check 查看状态）")
        return 2

    # ---- 日记 State 准备（缺失自动生成）----
    状态路径 = _RWKE_DIR / "states" / f"RoleB_状态_{参数.模型}.pth"
    需要 = not 参数.跳过日记状态
    if 需要 and not 状态路径.exists():
        print(f"[准备] 日记 State 缺失，现场生成：{状态路径}")
        try:
            生成日记状态(参数.模型, 参数.日记, 段大小=参数.段大小)
        except Exception as e:
            print(f"[准备] 日记 State 生成失败（{e}），跳过 G3/G4")
            需要 = False
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if 参数.跳过日记状态:
        需要 = False

    # ---- 引擎（单实例，四组间切换开关与挂载）----
    print(f"\n[引擎] 加载 {参数.模型} ...")
    引擎 = RWKE引擎(参数.模型)
    print(f"[引擎] {引擎.信息()}\n")

    组定义 = [
        ("G1_裸RWKV", False, False),
        ("G2_回响注入", True, True),
    ]
    if 需要:
        组定义 += [
            ("G3_挂载日记State", False, False),
            ("G4_挂载State+注入", True, True),
        ]

    结果列表 = []
    for 组名, 注入, 捕获 in 组定义:
        挂载 = 组名.startswith(("G3", "G4"))
        print(f"── {组名} （注入={注入}，捕获={捕获}，挂载State={挂载}）──")
        结果 = _跑一组(
            引擎, 组名, 注入, 捕获, 挂载,
            str(状态路径) if 需要 else None,
            参数.消息, 参数.max_new_tokens,
        )
        预览 = 结果["文本"][:80].replace("\n", " ")
        print(f"  文本({结果['字符数']}字)：{预览}{'...' if 结果['字符数'] > 80 else ''}")
        print(f"  步数={结果['生成步数']}，语义熵均值={结果['语义熵均值']:.4f}，"
              f"非有限步={结果['非有限logits步数']}，池大小={结果['回响池大小']}，"
              f"步延迟中位={结果['步延迟中位秒']}s，tokens/s={结果['tokens每秒']}"
              + (f"，峰值显存={结果['峰值显存MB_allocated']}MB(alloc)/"
                 f"{结果['峰值显存MB_reserved']}MB(reserved)" if torch.cuda.is_available() else ""))
        print()
        结果列表.append(结果)

    # ---- 汇总判定 ----
    按名 = {r["组名"]: r for r in 结果列表}
    检查 = []

    文本ok = all(r["字符数"] > 0 and r["含中文"] for r in 结果列表)
    检查.append(("四组均产出非空中文文本", 文本ok))

    无崩溃 = all(not r["崩溃"] for r in 结果列表)
    非有限ok = all(r["非有限logits步数"] == 0 for r in 结果列表)
    检查.append(("无 NaN/Inf logits", 非有限ok))
    检查.append(("无崩溃", 无崩溃))

    熵差异ok = False
    熵差异值 = None
    if "G2_回响注入" in 按名 and "G1_裸RWKV" in 按名:
        熵差异值 = 按名["G2_回响注入"]["语义熵均值"] - 按名["G1_裸RWKV"]["语义熵均值"]
        熵差异ok = abs(熵差异值) > 1e-6
    检查.append((f"λ>0 组语义熵与裸组有可测差异（ΔH={熵差异值 if 熵差异值 is not None else 'N/A'}）", 熵差异ok))

    print("═" * 60)
    print("冒烟判定")
    print("═" * 60)
    全部通过 = True
    for 名称, 通过 in 检查:
        print(f"  [{'PASS' if 通过 else 'FAIL'}] {名称}")
        全部通过 = 全部通过 and 通过

    if 熵差异值 is not None:
        print(f"\n  语义熵对比：G1裸={按名['G1_裸RWKV']['语义熵均值']:.4f}  "
              f"G2注入={按名['G2_回响注入']['语义熵均值']:.4f}  ΔH={熵差异值:+.6f}")
    if 需要 and "G3_挂载日记State" in 按名 and "G1_裸RWKV" in 按名:
        print(f"  文本差异：G1裸 vs G3挂State {'不同（State 影响生成）' if 按名['G1_裸RWKV']['文本'] != 按名['G3_挂载日记State']['文本'] else '相同（注意！）'}")

    print(f"\n结论：{'冒烟通过' if 全部通过 else '冒烟未通过'}")
    return 0 if 全部通过 else 1


if __name__ == "__main__":
    sys.exit(main())
