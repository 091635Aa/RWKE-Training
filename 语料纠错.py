# -*- coding: utf-8 -*-
"""
ASR 语料拼音纠错（打标结果 → 逐字同音错字检出与回写）
=====================================================

背景
----
`打标/数据层/打标结果/*.json` 的 transcript 来自直播间语音转写（ASR），
主要错误形态是**逐字同音错认**：同一个音被写成别的字。
典型：角色名"EmoCompanion"被转写成"EmoCompanion"，"ViewerA"→"桑鱼"，"天鹅之梦"→"天娥之梦"。

本工具不靠语言模型猜，只用**拼音等价**这一硬约束定位候选：
把语料里所有 k 字窗口按拼音展开，凡是与目标词拼音逐字相交、但写法不同的窗口，
即为同音候选（附上下文与原频次）。这样既不漏（同音必被捕获），也能靠频次
把"高频正确形"与"低频错字形"分开。

子命令
------
    扫描   只读：目标词拼音等价扫描 + 通用 2/3 字同音分组报告 → results/纠错_扫描.json
    纠错   对已落盘的训练资产做同音归一化（--目录，默认 RWKE/训练数据；--干跑 只统计）

设计约束
--------
**打标源（<workspace>/\打标\\数据层\\打标结果）只读不改**，与 P 线口径一致
（本地角色挂载与情感注入综合技术方案.md：「已做 EmoCompanion→EmoCompanion 同音归一化，源 F 盘未改动」）。
归一化在构建期进行：`训练数据.py` 的 构建() 对 分段/种子日记 调 归一化对象()。

用法
----
    python 语料纠错.py 扫描
    python 语料纠错.py 纠错 --干跑
    python 语料纠错.py 纠错
"""

import argparse
import json
import re
import shutil
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from pypinyin.pinyin_dict import pinyin_dict

_RWKE_DIR = Path(__file__).resolve().parent
_结果目录 = _RWKE_DIR / "results"


# ======================================================================
# 路径定位：打标结果 / 日记语料（工程内无 打标/，真实数据在上一层 <workspace>/打标）
# ======================================================================

def 定位打标目录() -> Path:
    for 候选 in (
        _RWKE_DIR.parent / "打标" / "数据层" / "打标结果",
        _RWKE_DIR.parent.parent / "打标" / "数据层" / "打标结果",
    ):
        if 候选.is_dir():
            return 候选
    raise FileNotFoundError("未找到 打标结果 目录（检查 打标/数据层/打标结果）")


# ======================================================================
# 拼音层
# ======================================================================

def _无调(拼音: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", 拼音) if not unicodedata.combining(c))


_字拼音缓存: dict = {}


def 字拼音集(字: str) -> tuple:
    """该字所有读音（去调、去重、排序）；非汉字返回空元组。"""
    if 字 in _字拼音缓存:
        return _字拼音缓存[字]
    原始 = pinyin_dict.get(ord(字))
    结果 = tuple(sorted({_无调(p) for p in 原始.split(",")})) if 原始 else ()
    _字拼音缓存[字] = 结果
    return 结果


def _是汉字(字: str) -> bool:
    return "\u4e00" <= 字 <= "\u9fff" or "\u3400" <= 字 <= "\u4dbf"


def 同音(甲: str, 乙: str) -> bool:
    """逐字读音有交集（多音字任一读法相同即算同音）。"""
    if len(甲) != len(乙):
        return False
    return all(set(a) & set(b) for a, b in zip((字拼音集(c) for c in 甲), (字拼音集(c) for c in 乙)))


def 拼音串(文本: str) -> str:
    """可读拼音串（取每个字的首个读音），用于分组键。"""
    return " ".join(sorted(字拼音集(c))[0] if 字拼音集(c) else c for c in 文本)


# ======================================================================
# 目标词表：角色设定中的正确形（人设卡_EmoCompanion.txt / 训练数据.py _锚点词表 同步）
# ======================================================================

目标词表 = {
    "角色名": ["EmoCompanion"],
    "直播间名": ["EmoCompanion的直播间"],
    "锚点": ["萌妹", "天鹅之梦", "ViewerA", "开播第十五天", "新人主播"],
    "口癖": ["那个"],
    # 直播场景高频词（ASR 常错认）
    "场景": ["直播间", "灯牌", "粉丝", "关注", "点赞", "榜一", "家人们", "宝宝", "下播"],
}


# ======================================================================
# 语料读取
# ======================================================================

def _取transcript(数据: dict) -> str:
    部分 = []
    顶层 = 数据.get("transcript")
    if isinstance(顶层, str) and 顶层.strip():
        部分.append(顶层.strip())
    for 段 in (数据.get("时间轴分段") or []):
        if isinstance(段, dict):
            文 = 段.get("transcript")
            if isinstance(文, str) and 文.strip():
                部分.append(文.strip())
    return "\n".join(部分)


def 读语料(打标目录: Path, 缓存: Path):
    """返回 [(文件, transcript)]；带本地缓存（打标 JSON 变大时删缓存即可）。"""
    if 缓存.exists():
        数据 = json.loads(缓存.read_text(encoding="utf-8"))
        return [(Path(p), t) for p, t in 数据]
    条 = []
    for 文件 in sorted(打标目录.glob("*.json")):
        try:
            数据 = json.loads(文件.read_text(encoding="utf-8"))
        except Exception:
            continue
        文 = _取transcript(数据)
        if 文:
            条.append((str(文件), 文))
    缓存.write_text(json.dumps(条, ensure_ascii=False), encoding="utf-8")
    return [(Path(p), t) for p, t in 条]


# ======================================================================
# 扫描
# ======================================================================

def 扫目标词(全部文本: str, 词: str, 上下文数: int = 4) -> dict:
    """找与 词 逐字同音、但写法不同的窗口，按频次排序。"""
    k = len(词)
    if k > len(全部文本):
        return {}
    计数: Counter = Counter()
    上下文: dict = defaultdict(list)
    for i in range(len(全部文本) - k + 1):
        窗口 = 全部文本[i:i + k]
        if 窗口 == 词 or not all(_是汉字(c) for c in 窗口):
            continue
        if " " in 窗口 or "\n" in 窗口:
            continue
        if 同音(窗口, 词):
            计数[窗口] += 1
            if len(上下文[窗口]) < 上下文数:
                上下文[窗口].append(全部文本[max(0, i - 10): i + k + 10].replace("\n", "/"))
    return {
        "正确形": 词,
        "现役频次": 全部文本.count(词),
        "候选": [{"写法": w, "频次": c, "上下文": 上下文[w]} for w, c in 计数.most_common(40)],
    }


def 通用同音分组(全部文本: str, 字数: int = 2, 最小主形频次: int = 30,
                 倍率: int = 10, 上限: int = 120) -> list:
    """
    通用检出：把语料里所有 字数 字窗口按拼音串分组；组内最高频写法视作主形，
    低频写法（≤ 主形/倍率）视作候选错字。仅作候选清单，不自动回写。
    """
    计数: Counter = Counter()
    for i in range(len(全部文本) - 字数 + 1):
        窗口 = 全部文本[i:i + 字数]
        if all(_是汉字(c) for c in 窗口):
            计数[窗口] += 1
    分组: dict = defaultdict(list)
    for 写法, 次 in 计数.items():
        if 次 < 2:
            continue
        分组[拼音串(写法)].append((写法, 次))
    结果 = []
    for 键, 成员 in 分组.items():
        if len(成员) < 2:
            continue
        成员.sort(key=lambda x: -x[1])
        主形, 主频 = 成员[0]
        if 主频 < 最小主形频次:
            continue
        异形 = [(w, c) for w, c in 成员[1:] if c * 倍率 <= 主频]
        if 异形:
            结果.append({"拼音": 键, "主形": 主形, "主形频次": 主频,
                         "异形": [{"写法": w, "频次": c} for w, c in 异形[:8]]})
    结果.sort(key=lambda r: -r["主形频次"])
    return 结果[:上限]


def 命令扫描(参数) -> int:
    打标目录 = 定位打标目录()
    缓存 = _结果目录 / "纠错_语料缓存.json"
    开始 = time.time()
    条 = 读语料(打标目录, 缓存)
    全部文本 = "\n".join(t for _, t in 条)
    print(f"[扫描] 文件 {len(条)} 个 / 字符 {len(全部文本)} / 读入 {time.time() - 开始:.1f}s")

    报告 = {"文件数": len(条), "总字符": len(全部文本), "目标词": {}, "通用分组": {}}
    for 类别, 词列表 in 目标词表.items():
        for 词 in 词列表:
            项 = 扫目标词(全部文本, 词, 上下文数=参数.上下文)
            报告["目标词"][f"{类别}:{词}"] = 项
            print(f"\n== {类别} {词}（现役 {项['现役频次']} 次）==")
            for c in 项["候选"][:12]:
                print(f"   {c['写法']}×{c['频次']}  {c['上下文'][0] if c['上下文'] else ''}")
    for n in (2, 3):
        项 = 通用同音分组(全部文本, 字数=n)
        报告["通用分组"][str(n)] = 项
        print(f"\n== 通用 {n} 字同音分组（主形频次≥30、异形 ≤主形/10）共 {len(项)} 组 ==")
        for r in 项[:30]:
            print(f"   {r['主形']}({r['主形频次']}) ← " +
                  ", ".join(f"{x['写法']}×{x['频次']}" for x in r["异形"]))

    报告路径 = _结果目录 / "纠错_扫描.json"
    报告路径.write_text(json.dumps(报告, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[扫描] 报告 → {报告路径}（总耗时 {time.time() - 开始:.1f}s）")
    return 0


# ======================================================================
# 同音归一化（构建期调用：打标源保持原样，只在训练资产里归一）
# ======================================================================

# 依据：P 线权威数据已做「EmoCompanion→EmoCompanion」同音归一化（本地角色挂载与情感注入综合技术方案.md），
# 但只替换了字面「EmoCompanion」，漏掉同音的 EmoCompanion/元元/源源/袁袁 等写法（P 线残留 EmoCompanion 514 处）。
# 这里按拼音等价补全，且直播间专名同步归一（P 线为「EmoCompanion的直播间」921 处，0 处「EmoCompanion的直播间」）。
#
# 顺序敏感：长词优先。zhconv 先做繁→简（圓圓→EmoCompanion、燈牌→灯牌、直播間→直播间）。
映射表: dict = {
    "EmoCompanion": "EmoCompanion",
    "EmoCompanion": "EmoCompanion",
    "源源": "EmoCompanion",
    "元元": "EmoCompanion",
    "袁袁": "EmoCompanion",
    "袁媛": "EmoCompanion",
    "袁圆": "EmoCompanion",
    "元圆": "EmoCompanion",
    "园园": "EmoCompanion",
    "媛圆": "EmoCompanion",
    # 锚点/场景同音错认（证据见 results/纠错_扫描.json）
    "桑鱼": "ViewerA",
    "桑宇": "ViewerA",
    "登牌": "灯牌",
    "等牌": "灯牌",
}

# 固定搭配保护：出现在这些短语里的字面不归一（逐字同音但语义无关）
保护短语: list = ["源源不断", "EmoCompanion胖胖", "EmoCompanion满满", "元元本本"]


def 归一化文本(文本: str) -> str:
    """繁→简 + 同音归一。构建训练资产时对文本调用（幂等）。"""
    if not 文本:
        return 文本
    try:
        import zhconv
        文本 = zhconv.convert(文本, "zh-cn")
    except Exception:
        pass
    保护 = {}
    for i, 短语 in enumerate(保护短语):
        if 短语 in 文本:
            保护[f"\x01{i}\x01"] = 短语
            文本 = 文本.replace(短语, f"\x01{i}\x01")
    for 错, 对 in sorted(映射表.items(), key=lambda kv: -len(kv[0])):
        文本 = 文本.replace(错, 对)
    for 占位, 短语 in 保护.items():
        文本 = 文本.replace(占位, 短语)
    return 文本


def 归一化对象(对象):
    """递归归一化 dict/list/str（训练资产落盘前统一调用）。"""
    if isinstance(对象, str):
        return 归一化文本(对象)
    if isinstance(对象, list):
        return [归一化对象(x) for x in 对象]
    if isinstance(对象, dict):
        return {k: 归一化对象(v) for k, v in 对象.items()}
    return 对象


def 命令纠错(参数) -> int:
    """对已有训练资产原地归一（打标源不动）；--干跑 只统计。"""
    目录 = Path(参数.目录) if 参数.目录 else _RWKE_DIR / "训练数据"
    备份目录 = 目录 / "_归一化备份"
    命中 = Counter()
    改文件 = 0
    for 文件 in sorted(目录.glob("*.jsonl")) + sorted(目录.glob("*.json")):
        if 文件.name.startswith("_") or 备份目录.name in 文件.parts:
            continue
        原文 = 文件.read_text(encoding="utf-8")
        try:
            数据 = [json.loads(x) for x in 原文.splitlines() if x.strip()] \
                if 文件.suffix == ".jsonl" else json.loads(原文)
        except Exception:
            continue
        新数据 = 归一化对象(数据)
        新文 = ("\n".join(json.dumps(x, ensure_ascii=False) for x in 新数据) + "\n") \
            if 文件.suffix == ".jsonl" else json.dumps(新数据, ensure_ascii=False, indent=2)
        if 新文 == 原文:
            continue
        for 错 in 映射表:
            命中[错] += 原文.count(错)
        改文件 += 1
        if 参数.干跑:
            continue
        备份目录.mkdir(parents=True, exist_ok=True)
        备份 = 备份目录 / 文件.name
        if not 备份.exists():
            shutil.copy2(文件, 备份)
        文件.write_text(新文, encoding="utf-8")

    print(f"[归一化]{'（干跑）' if 参数.干跑 else ''} 目录 {目录} 待改/已改 {改文件} 个文件")
    for 错, 对 in 映射表.items():
        if 命中[错]:
            print(f"   {错} → {对}：{命中[错]} 处")
    if not 参数.干跑 and 改文件:
        print(f"[归一化] 备份 → {备份目录}")
        _结果目录.joinpath("纠错_语料缓存.json").unlink(missing_ok=True)
    return 0


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="ASR 语料拼音纠错")
    子 = p.add_subparsers(dest="命令", required=True)
    s = 子.add_parser("扫描", help="同音错字扫描（只读）")
    s.add_argument("--上下文", type=int, default=4)
    s.set_defaults(执行=命令扫描)
    c = 子.add_parser("纠错", help="对训练资产做同音归一化（打标源不动）")
    c.add_argument("--目录", help="待归一化的资产目录（默认 RWKE/训练数据）")
    c.add_argument("--干跑", action="store_true", help="只统计不落盘")
    c.set_defaults(执行=命令纠错)
    参数 = p.parse_args()
    return 参数.执行(参数)


if __name__ == "__main__":
    raise SystemExit(main())