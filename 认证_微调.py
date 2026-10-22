# -*- coding: utf-8 -*-
"""
微调认证：微调前后同评估集对比 + 连续对话人设稳定性抽检（spec Task 6）

用法：
    python 认证_微调.py --模型 RWKV7-World-0.4B
    python 认证_微调.py --模型 RWKV7-G1-1.5B --轮数 100
输出：
    results/微调对比认证_<模型>.json   —— 裸基座 vs 挂适配器 逐维 before/after
    results/连续对话抽检_<模型>.json   —— 100 轮稳定性指标（前段/后段漂移）
    docs/微调报告.md 追加认证节
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import 一致性评估 as 评估器
from rwke引擎 import RWKE引擎

_RWKE_DIR = Path(__file__).resolve().parent


def _剥离模板(提示: str) -> str:
    m = re.search(r"User:\s*(.*?)\n\nAssistant:", 提示, re.S)
    return m.group(1).strip() if m else 提示.strip()


def _加载评估集(路径: Path) -> list:
    return json.loads(路径.read_text(encoding="utf-8-sig"))


def _四gram重复率(文本: str) -> float:
    grams = [文本[i:i + 4] for i in range(len(文本) - 3)]
    if not grams:
        return 0.0
    return (len(grams) - len(set(grams))) / len(grams)


def 生成组回复(引擎: RWKE引擎, 评估集: list, 生成参数: dict) -> list:
    回复列表 = []
    for i, 条目 in enumerate(评估集):
        消息 = _剥离模板(条目["提示"])
        回复 = 引擎.生成(
            消息,
            种子=int(生成参数.get("种子", 42)),
            max_new_tokens=int(生成参数.get("最大新Token", 96)),
            温度=float(生成参数.get("温度", 1.0)),
            top_p=float(生成参数.get("top_p", 0.9)),
            top_k=int(生成参数.get("top_k", 50)),
        )
        回复列表.append(回复)
        print(f"    [{i + 1}/{len(评估集)}] {回复[:40]}...")
    return 回复列表


def 对比认证(模型标识: str, 评估集: list, 适配器路径: Path, 生成参数: dict, 达标线: float) -> dict:
    权重 = 评估器.获取权重()
    组结果 = {}
    for 组名, 适配器 in (("裸基座", None), ("挂适配器", str(适配器路径))):
        print(f"  [{组名}] 构造引擎...")
        引擎 = RWKE引擎(模型标识, 适配器路径=适配器)
        print(f"    {引擎.信息()}")
        回复列表 = 生成组回复(引擎, 评估集, 生成参数)
        明细 = []
        for 条目, 回复 in zip(评估集, 回复列表):
            明细.append({
                "编号": 条目["编号"],
                "回复": 回复,
                "评估": 评估器.评估单条(回复, 条目.get("期望情感"), 条目.get("期望锚点"),
                                  条目.get("提示"), 权重=权重),
            })
        组结果[组名] = {
            "汇总": 评估器.汇总统计([d["评估"] for d in 明细], 达标线=达标线),
            "明细": 明细,
        }
        del 引擎
        import torch
        torch.cuda.empty_cache()
    差值 = round(组结果["挂适配器"]["汇总"]["合成分_均值"]
               - 组结果["裸基座"]["汇总"]["合成分_均值"], 1)
    return {"模型标识": 模型标识, "适配器": str(适配器路径),
            "达标线": 达标线, "合成分差值": 差值, "组结果": 组结果}


def 稳定性抽检(模型标识: str, 评估集: list, 适配器路径: Path, 日记状态: Path,
              生成参数: dict, 轮数: int, 权重: dict) -> dict:
    引擎 = RWKE引擎(模型标识, 适配器路径=str(适配器路径))
    if 日记状态 and 日记状态.exists():
        元数据 = 引擎.挂载日记状态(str(日记状态))
        print(f"  已挂载日记 State：{日记状态.name}")
    else:
        print("  （无日记 State，纯适配器续话）")
    出戏词 = ["作为一个AI", "语言模型", "人工智能", "训练数据", "我无法", "抱歉，我不能"]
    轮记录 = []
    for 轮 in range(1, 轮数 + 1):
        条目 = 评估集[(轮 - 1) % len(评估集)]
        消息 = _剥离模板(条目["提示"])
        回复 = 引擎.生成(消息, 延续对话=True, 保留池=True,
                       种子=int(生成参数.get("种子", 42)) + 轮,
                       max_new_tokens=int(生成参数.get("最大新Token", 96)),
                       温度=float(生成参数.get("温度", 1.0)),
                       top_p=float(生成参数.get("top_p", 0.9)),
                       top_k=int(生成参数.get("top_k", 50)))
        轮记录.append({
            "轮": 轮, "提问": 消息[:30], "回复": 回复,
            "合成分": 评估器.评估单条(回复, 条目.get("期望情感"), 条目.get("期望锚点"),
                                条目.get("提示"), 权重=权重)["合成分"],
            "出戏": any(w in 回复 for w in 出戏词),
            "自称": ("EmoCompanion" in 回复) or ("EmoCompanion" in 回复),
        })
        if 轮 % 10 == 0:
            print(f"    轮 {轮}/{轮数} 完成")
    前段 = [r["合成分"] for r in 轮记录[:10]]
    后段 = [r["合成分"] for r in 轮记录[-10:]]
    全文 = "".join(r["回复"] for r in 轮记录)
    重复率列表 = [_四gram重复率(r["回复"]) for r in 轮记录 if len(r["回复"]) >= 8]
    结果 = {
        "模型标识": 模型标识, "轮数": 轮数,
        "日记状态": str(日记状态) if 日记状态 and 日记状态.exists() else None,
        "前10轮合成分均值": round(sum(前段) / len(前段), 1),
        "后10轮合成分均值": round(sum(后段) / len(后段), 1),
        "漂移": round(sum(后段) / len(后段) - sum(前段) / len(前段), 1),
        "唯一回复率%": round(len(set(r["回复"] for r in 轮记录)) / len(轮记录) * 100, 1),
        "平均4gram重复率": round(sum(重复率列表) / len(重复率列表), 3) if 重复率列表 else 0.0,
        "出戏轮数": sum(1 for r in 轮记录 if r["出戏"]),
        "自称出现轮数": sum(1 for r in 轮记录 if r["自称"]),
        "全程合成分均值": round(sum(r["合成分"] for r in 轮记录) / len(轮记录), 1),
        "全文净字数": len(re.sub(r"\s", "", 全文)),
        "轮记录": 轮记录,
    }
    return 结果


def main() -> int:
    解析器 = argparse.ArgumentParser(description="微调认证：对比 + 稳定性抽检")
    解析器.add_argument("--模型", required=True)
    解析器.add_argument("--适配器", default=None, help="默认 适配器/{模型}_lora.pt")
    解析器.add_argument("--日记状态", default=None, help="默认 states/EmoCompanion_状态_{模型}.pth")
    解析器.add_argument("--评估集", default=str(_RWKE_DIR / "训练数据" / "评估集.json"))
    解析器.add_argument("--轮数", type=int, default=100)
    解析器.add_argument("--最大新Token", type=int, default=96)
    解析器.add_argument("--跳过稳定性", action="store_true")
    参数 = 解析器.parse_args()

    配置 = json.loads((_RWKE_DIR / "系统配置.json").read_text(encoding="utf-8"))
    微调配置 = 配置.get("微调", {})
    达标线 = float(微调配置.get("一致性达标线", 95))
    生成参数 = dict(微调配置.get("评估生成参数", {}))
    生成参数["最大新Token"] = 参数.最大新Token
    适配器路径 = Path(参数.适配器) if 参数.适配器 else _RWKE_DIR / "适配器" / f"{参数.模型}_lora.pt"
    日记状态 = Path(参数.日记状态) if 参数.日记状态 else _RWKE_DIR / "states" / f"EmoCompanion_状态_{参数.模型}.pth"
    评估集 = _加载评估集(Path(参数.评估集))
    权重 = 评估器.获取权重()

    print(f"[1/2] 对比认证：{参数.模型}（达标线 {达标线}）")
    对比 = 对比认证(参数.模型, 评估集, 适配器路径, 生成参数, 达标线)
    对比路径 = _RWKE_DIR / "results" / f"微调对比认证_{参数.模型}.json"
    对比路径.write_text(json.dumps(对比, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  裸基座 合成分均值 = {对比['组结果']['裸基座']['汇总']['合成分_均值']}")
    print(f"  挂适配器 合成分均值 = {对比['组结果']['挂适配器']['汇总']['合成分_均值']}")
    print(f"  差值 = {对比['合成分差值']}（已写 {对比路径.name}）")

    抽检结果 = None
    if not 参数.跳过稳定性:
        print(f"[2/2] 稳定性抽检：{参数.轮数} 轮连续对话")
        抽检结果 = 稳定性抽检(参数.模型, 评估集, 适配器路径, 日记状态, 生成参数, 参数.轮数, 权重)
        抽检路径 = _RWKE_DIR / "results" / f"连续对话抽检_{参数.模型}.json"
        抽检路径.write_text(json.dumps(抽检结果, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  前10轮均值 {抽检结果['前10轮合成分均值']} → 后10轮 {抽检结果['后10轮合成分均值']}"
              f"（漂移 {抽检结果['漂移']}），唯一回复率 {抽检结果['唯一回复率%']}%，出戏 {抽检结果['出戏轮数']} 轮")

    # ---- 报告追加 ----
    报告 = _RWKE_DIR / "docs" / "微调报告.md"
    裸 = 对比["组结果"]["裸基座"]["汇总"]
    挂 = 对比["组结果"]["挂适配器"]["汇总"]
    行 = ["", f"## 认证追加：{参数.模型}（{datetime.now().isoformat(timespec='seconds')}）", ""]
    行.append(f"- 对比认证：裸基座 {裸['合成分_均值']} → 挂适配器 {挂['合成分_均值']}（差 {对比['合成分差值']}，达标线 {达标线}）")
    行.append(f"- 逐维 before/after：锚点 {裸['锚点命中率_均值']}→{挂['锚点命中率_均值']}｜口癖 {裸['口癖率_均值']}→{挂['口癖率_均值']}"
              f"｜五维 {裸['五维人设符合度_均值']}→{挂['五维人设符合度_均值']}｜情感 {裸['情感基调匹配_均值']}→{挂['情感基调匹配_均值']}"
              f"｜身份 {裸['身份一致性_均值']}→{挂['身份一致性_均值']}｜流畅 {裸['流畅性守卫_均值']}→{挂['流畅性守卫_均值']}")
    if 抽检结果:
        行.append(f"- 稳定性：{抽检结果['轮数']} 轮连续对话，前10轮 {抽检结果['前10轮合成分均值']} → 后10轮 {抽检结果['后10轮合成分均值']}"
                  f"（漂移 {抽检结果['漂移']}），唯一回复率 {抽检结果['唯一回复率%']}%，出戏 {抽检结果['出戏轮数']} 轮，"
                  f"自称出现 {抽检结果['自称出现轮数']} 轮")
    行.append(f"- 判定：{'达标' if 挂['合成分_均值'] >= 达标线 else '未达标（差距 ' + str(round(达标线 - 挂['合成分_均值'], 1)) + '，最低维度见上）'}")
    with open(报告, "a", encoding="utf-8") as f:
        f.write("\n".join(行) + "\n")
    print(f"报告已追加：{报告}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
