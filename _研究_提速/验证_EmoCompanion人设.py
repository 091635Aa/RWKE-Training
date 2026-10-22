# -*- coding: utf-8 -*-
"""
EmoCompanion人设 A/B 对比验证（训练完成后跑）
======================================

对比两个适配器 checkpoint 在同一组锚点问题上的表现：
  - 锚点命中：回复是否含期望锚点词（新人主播/开播第十五天/EmoCompanion的直播间/天鹅之梦/萌妹/ViewerA）
  - 自称正确：以"EmoCompanion/我"自称，不出现他人名自称（RoleB/ChatGPT 等）
  - 口癖：啊/那个/清嗓子
  - 复读率：连续重复 ≥3 字片段的轮次占比
  - 连贯性：回复长度（过短=退化）

用法：
    python "_研究_提速/验证_EmoCompanion人设.py"                    # 新旧 best 对比
    python "_研究_提速/验证_EmoCompanion人设.py" --A 路径1 --B 路径2
注意：需 GPU 空闲（训练跑完后再执行）。
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ["RWKV_V7_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "0"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_RWKE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RWKE_DIR))

_锚点问题 = [
    ("你是谁呀？", ["新人主播", "EmoCompanion"]),
    ("你的直播间叫什么名字？", ["EmoCompanion的直播间"]),
    ("今天开播多少天啦？", ["第十五天", "15"]),
    ("天鹅之梦是什么呀？", ["天鹅之梦", "礼物"]),
    ("大家都叫你萌妹，你开心吗？", ["萌妹", "开心"]),
    ("听说有个观众叫ViewerA？", ["ViewerA"]),
    ("你是谁呀？做主播难吗？", ["主播", "难"]),
    ("今天直播都做了什么呀？", []),
]
_口癖词 = ["啊", "那个", "清嗓子"]
_坏自称 = ["RoleB", "ChatGPT", "AI助手", "人工智能"]


def 跑一组(引擎, 适配器路径, 卡文本):
    if 适配器路径:
        引擎._挂载适配器(适配器路径)
    else:
        引擎._适配权重缓存 = None
        引擎.适配器标识 = None
        引擎._权重版本 += 1
    # 人设卡预热（与服务器同构）
    引擎.快速步进 = True
    引擎.每轮字数 = 384
    预热 = 引擎.生成(卡文本, 保留池=False, 种子=7)
    引擎.每轮字数 = 128
    明细 = []
    for 问题, 期望锚点 in _锚点问题:
        回复 = 引擎.生成(问题, 保留池=True, 种子=42)
        低 = 回复
        锚命中 = sum(1 for a in 期望锚点 if a in 低) if 期望锚点 else None
        锚率 = None if 锚命中 is None else round(锚命中 / max(len(期望锚点), 1), 2)
        自称坏 = [w for w in _坏自称 if w in 低]
        口癖 = sum(1 for w in _口癖词 if w in 低)
        复读 = bool(re.search(r"(.{1,6})\1{2,}", 回复))
        明细.append({
            "问题": 问题, "回复": 回复[:80],
            "锚点率": 锚率, "坏自称": 自称坏,
            "口癖数": 口癖, "中间复读": 复读, "长度": len(回复),
        })
        引擎.清空回响()
    引擎.清空回响()
    return 明细


def 汇总(明细):
    有锚 = [m for m in 明细 if m["锚点率"] is not None]
    return {
        "锚点命中率": round(sum(m["锚点率"] for m in 有锚) / max(len(有锚), 1), 2),
        "坏自称轮数": sum(1 for m in 明细 if m["坏自称"]),
        "口癖覆盖率": round(sum(1 for m in 明细 if m["口癖数"] > 0) / len(明细), 2),
        "复读轮数": sum(1 for m in 明细 if m["中间复读"]),
        "平均长度": round(sum(m["长度"] for m in 明细) / len(明细), 1),
    }


def main():
    解析器 = argparse.ArgumentParser()
    解析器.add_argument("--A", default=str(_RWKE_DIR / "适配器" / "RWKV7-G1-1.5B_lora_best.pt"))
    解析器.add_argument("--B", default=str(_RWKE_DIR / "备份_快验_step75_lora_best.pt") if False else "")
    解析器.add_argument("--B2", default="")
    参数 = 解析器.parse_args()

    from rwke引擎 import RWKE引擎
    print("[加载] RWKV7-G1-1.5B …")
    引擎 = RWKE引擎("RWKV7-G1-1.5B")
    卡 = (_RWKE_DIR / "人设卡_EmoCompanion.txt").read_text(encoding="utf-8").strip()

    组列表 = [("新best", 参数.A)]
    if 参数.B:
        组列表.append(("对比B", 参数.B))
    if 参数.B2:
        组列表.append(("对比B2", 参数.B2))
    if len(组列表) == 1:
        组列表.append(("裸基座", ""))  # 默认附裸基座对照

    报告 = {"时间": time.strftime("%Y-%m-%d %H:%M:%S"), "组": {}}
    for 名称, 路径 in 组列表:
        print(f"\n===== {名称}（{路径 or '裸基座'}）=====")
        明细 = 跑一组(引擎, 路径 if 路径 else None, 卡)
        总 = 汇总(明细)
        print(json.dumps(总, ensure_ascii=False))
        for m in 明细:
            print(f"  {m['问题']} → 锚{m['锚点率']} 口癖{m['口癖数']} 复读{m['中间复读']} | {m['回复'][:46]}")
        报告["组"][名称] = {"汇总": 总, "明细": 明细}

    out = _RWKE_DIR / "_研究_提速" / "EmoCompanion人设对比.json"
    out.write_text(json.dumps(报告, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已落盘：{out}")


if __name__ == "__main__":
    main()
