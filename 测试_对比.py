# -*- coding: utf-8 -*-
"""
RWKE 多模型认证对比测试与报告
==============================

认证矩阵（tasks.md Task 7.1）：
    每个矩阵内 RWKV 模型 × {裸, 注入} + 基线 Qwen2.5-1.5B-Instruct × {裸, 注入}

统一协议：T=1.0, top_p=0.9, top_k=50, repetition_penalty=1.05, seed=42；
提示集取自"chinese-adorable 高情商中文对话"数据集的 user 字段（HF 本地缓存），
TuringBench 人类参考集取同源 girl 字段（真人中文回复）。

指标
----
情感（7.2）：语义熵（工作区统一口径）、4-gram 重复率（V1架构.计算重复率 移植）、
             情感词密度（cnsenti Sentiment，失败回退本地简易词表）、
             TuringBench 检出率（TF-IDF(1-2gram,30k,sublinear)+LR(max_iter=1000) 移植）
成本（7.3）：tokens/s、中位步延迟、峰值显存、0/512/2048 上下文深度延迟曲线、
             KV Cache vs RWKV State 内存占用对照
标定（7.4）：--标定维度 N 对扫描表外 hidden_dim 跑裸 vs 注入标定 λ/γ/τ 并回写系统配置.json
报告（7.5）：results/*.json 断点分片 + docs/RWKE测试报告.md（含横向对比表与达标判定）

用法
----
    python 测试_对比.py                        # 全矩阵认证（已就位模型）
    python 测试_对比.py --模型 RWKV7-World-0.4B  # 子集
    python 测试_对比.py --快速                  # 短生成管线验证
    python 测试_对比.py --强制                  # 忽略断点重跑
    python 测试_对比.py --标定维度 1792         # 扫描表外维度标定回写
"""

import argparse
import gc
import glob
import json
import platform
import random
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, List, Optional

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import rwke引擎 as 引擎模块  # noqa: E402 （先行导入：环境变量先于 rwkv 生效）
from rwke引擎 import RWKE引擎  # noqa: E402

_RWKE_DIR = Path(__file__).resolve().parent
_结果目录 = _RWKE_DIR / "results"
_报告目录 = _RWKE_DIR / "docs"

# 统一采样协议（系统配置.json 采样协议；与统一生成器一致）
采样参数 = dict(温度=1.0, top_p=0.9, top_k=50, 重复惩罚=1.05)
种子基数 = 42

# HF 本地缓存提示集（chinese-adorable 高情商中文对话；user=prompt，girl=真人回复）
_HF提示集通配 = str(Path.home() / ".cache" / "huggingface" / "hub"
                    / "datasets--sunorme--chinese-adorable-high-emotional-intelligence-chat"
                    / "snapshots" / "*" / "chinese-adorable-high-emotional-intelligence-chat.json")
_工作区备选 = _RWKE_DIR.parent / "语义回响" / "图灵测试" / "样本_60条_优化.json"
_数据来源 = "未知"

# 上下文深度曲线采样点（tasks.md 7.3）
深度点 = [0, 512, 2048]
深度生成步 = 16


# ======================================================================
# 通用指标（7.2）
# ======================================================================

def 计算语义熵(logits: torch.Tensor) -> float:
    """单 token 位置语义熵（工作区统一口径：softmax 熵，nats；-inf 槽位压到 -1e4）。"""
    if logits.dim() == 2:
        logits = logits[0]
    logits = logits.clone().float()
    logits[logits == float("-inf")] = -1e4
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-12)
    return float(-(probs * log_probs).sum().item())


def 计算重复率(文本: str, n: int = 4) -> float:
    """n-gram 重复率（V1架构.py 计算重复率 移植，字符级）：重复冗余占比。"""
    if not 文本:
        return 0.0
    文本 = str(文本)
    总长 = len(文本) - n + 1
    if 总长 <= 0:
        return 0.0
    ngram列表 = [文本[i:i + n] for i in range(总长)]
    出现次数: dict = {}
    for g in ngram列表:
        出现次数[g] = 出现次数.get(g, 0) + 1
    重复数 = sum(次数 - 1 for 次数 in 出现次数.values() if 次数 > 1)
    return round(重复数 / len(ngram列表), 4)


# ---- 情感词密度：cnsenti 优先，本地简易词表兜底 ----
_简易正面词 = {"开心", "高兴", "快乐", "幸福", "喜悦", "兴奋", "愉快", "甜蜜", "喜欢", "爱",
               "温暖", "温柔", "安慰", "美好", "感动", "棒", "爽", "舒服", "满意", "幸运",
               "期待", "放心", "甜", "乖", "可爱", "治愈", "亲爱", "想我", "想你", "笑"}
_简易负面词 = {"难过", "悲伤", "伤心", "痛苦", "失落", "心碎", "沮丧", "哭泣", "委屈", "心疼",
               "绝望", "崩溃", "心累", "扎心", "孤独", "焦虑", "烦", "累", "害怕", "担心",
               "生气", "讨厌", "对不起", "自责", "自卑", "迷茫", "没用", "差劲", "压力", "哭"}

try:
    from cnsenti import Sentiment as _CnSentiment
    _情感分析器 = _CnSentiment()
    _情感词表来源 = "cnsenti-0.0.7"

    def 情感词密度(文本: str) -> float:
        if not 文本:
            return 0.0
        r = _情感分析器.sentiment_count(文本)
        return round((r.get("pos", 0) + r.get("neg", 0)) / max(r.get("words", 0), 1), 4)
except Exception:  # cnsenti 不可用时本地词表兜底
    _情感词表来源 = "本地简易词表（cnsenti 不可用）"

    def 情感词密度(文本: str) -> float:
        if not 文本:
            return 0.0
        词数 = max(len(文本), 1)
        命中 = sum(1 for w in (_简易正面词 | _简易负面词) if w in 文本)
        return round(命中 / 词数, 4)


# ======================================================================
# TuringBench 检测器（7.2，移植自 统一_run_turingbench.py）
# ======================================================================

def 图灵检出实验(生成函数: Callable[[str, int, int], str], 对话: List[dict],
                 训练对数: int = 60, 测试对数: int = 30, 种子基数: int = 42,
                 日志前缀: str = "") -> dict:
    """
    TF-IDF(1-2gram)+LR 检测器：人类(girl) vs AI(目标模型生成)，检出率=AI 文本被判 AI 比例。

    生成函数(消息, 轮次, 种子) -> 生成文本（由各后端注入采样协议）。
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import accuracy_score

    random.seed(种子基数)
    池 = list(对话)
    random.shuffle(池)
    训练对话 = 池[:训练对数]
    测试对话 = 池[训练对数:训练对数 + 测试对数]
    if len(测试对话) < 测试对数 // 2:
        raise ValueError(f"对话数据不足：训练 {len(训练对话)} / 测试 {len(测试对话)}")

    AI训练文本 = []
    for i, d in enumerate(训练对话):
        AI训练文本.append(生成函数(d["user"], i, 种子基数))
        if (i + 1) % 20 == 0:
            print(f"    {日志前缀}图灵训练生成 {i + 1}/{len(训练对话)}", flush=True)
    AI测试文本 = []
    for i, d in enumerate(测试对话):
        AI测试文本.append(生成函数(d["user"], 训练对数 + i, 种子基数 + 1))
        if (i + 1) % 10 == 0:
            print(f"    {日志前缀}图灵测试生成 {i + 1}/{len(测试对话)}", flush=True)

    X人类 = [d["girl"] for d in 训练对话]
    X训练 = X人类 + AI训练文本
    y训练 = [1] * len(X人类) + [0] * len(AI训练文本)
    X人类测试 = [d["girl"] for d in 测试对话]
    X测试 = X人类测试 + AI测试文本
    y测试 = [1] * len(X人类测试) + [0] * len(AI测试文本)

    检测器 = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), max_features=30000, sublinear_tf=True),
        LogisticRegression(max_iter=1000),
    )
    检测器.fit(X训练, y训练)
    AI判AI = sum(1 for t in AI测试文本 if 检测器.predict([t])[0] == 0)
    真人判AI = sum(1 for t in X人类测试 if 检测器.predict([t])[0] == 0)
    return {
        "detection_accuracy": round(AI判AI / max(len(AI测试文本), 1), 4),
        "human_likeness": round((len(AI测试文本) - AI判AI) / max(len(AI测试文本), 1), 4),
        "人类文本误判率": round(真人判AI / max(len(X人类测试), 1), 4),
        "检测器训练准确率": round(accuracy_score(y训练, 检测器.predict(X训练)), 4),
        "检测器留出测试准确率": round(accuracy_score(y测试, 检测器.predict(X测试)), 4),
        "AI文本数": len(AI测试文本), "真人文本数": len(X人类测试),
    }


def 加载提示数据(指定路径: Optional[str] = None, 最小需求: int = 90) -> List[dict]:
    """加载 {user, girl} 对话列表：--数据 指定 > HF 本地缓存 > 工作区固定样本。"""
    global _数据来源
    候选: List[Path] = []
    if 指定路径:
        候选.append(Path(指定路径))
    候选 += [Path(p) for p in glob.glob(_HF提示集通配)]
    候选.append(_工作区备选)

    for p in 候选:
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "样本" in data:  # 工作区固定样本格式
            data = data["样本"]
        有效 = [
            {"user": str(d.get("user", "")).strip(), "girl": str(d.get("girl", "")).strip()}
            for d in data if isinstance(d, dict)
        ]
        有效 = [d for d in 有效
                if len(d["user"]) >= 2 and len(d["user"]) <= 80
                and len(d["girl"]) >= 2 and len(d["girl"]) <= 200]
        if len(有效) >= 最小需求:
            print(f"[数据] 提示集：{p}（{len(有效)} 条有效）")
            _数据来源 = p.name
            return 有效
    raise FileNotFoundError(
        f"未找到含 ≥{最小需求} 条有效对话的提示集（--数据 指定，或确认 HF 缓存存在）")


# ======================================================================
# 生成后端
# ======================================================================

class RWKV后端:
    """RWKE引擎 封装：裸/注入可切换，统一采样协议。"""

    名称前缀 = "rwkv7"

    def __init__(self, 模型标识: str, 条目: dict, λ: float, γ: float, τ: float):
        self.模型标识 = 模型标识
        self.条目 = 条目
        self.引擎 = RWKE引擎(模型标识, λ=λ, γ=γ, τ=τ)
        self.hidden_dim = self.引擎.hidden_dim
        self.λ, self.γ, self.τ = λ, γ, τ
        self.主设备 = self.引擎.设备
        self._上次步延迟: Optional[List[float]] = None

    def 切换模式(self, 注入: bool) -> None:
        self.引擎.启用注入 = 注入
        self.引擎.启用捕获 = 注入
        self.引擎.清除挂载状态()

    def 生成(self, 消息: str, 轮次: int, 种子: int, max_new_tokens: int,
            收集熵: Optional[List[float]] = None) -> str:
        def logits_callback(步, logits):
            if 收集熵 is not None:
                收集熵.append(计算语义熵(logits))

        步延迟: List[float] = []
        上次 = [time.perf_counter()]

        def 轮次回调(步数, 池):
            现在 = time.perf_counter()
            步延迟.append(现在 - 上次[0])
            上次[0] = 现在

        文本 = self.引擎.生成(
            消息,
            种子=种子 + 轮次,
            max_new_tokens=max_new_tokens,
            logits_callback=logits_callback,
            轮次回调=轮次回调,
        )
        self._上次步延迟 = 步延迟
        return 文本

    def 步延迟(self) -> Optional[List[float]]:
        return self._上次步延迟

    def 释放(self) -> None:
        self.引擎 = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class Qwen后端:
    """V1架构 复用：_加载模型与分词器 + 回响注入器，裸/注入可切换。"""

    名称前缀 = "qwen25"

    def __init__(self, 模型标识: str, 条目: dict, λ: float, γ: float, τ: float):
        import sys as _sys
        _v1根 = str(_RWKE_DIR.parent / "emocompanion-ai-emotion" / "最后版本")
        if _v1根 not in _sys.path:
            _sys.path.insert(0, _v1根)
        from 核心引擎.推理架构.V1架构 import 回响注入器, _加载模型与分词器
        from 核心引擎.推理架构.回响池 import 语义回响池 as V1池

        路径 = Path(条目["路径"])
        if not 路径.is_absolute():
            路径 = _RWKE_DIR / 路径
        if not 路径.exists():
            raise FileNotFoundError(f"Qwen 基线模型未就位：{路径}")

        self.模型标识 = 模型标识
        self.模型, self.分词器, _ = _加载模型与分词器(str(路径), "fp16")
        self.hidden_dim = int(self.模型.config.hidden_size)
        self.λ, self.γ, self.τ = λ, γ, τ
        self.池 = V1池(self.hidden_dim, max_pool_size=1024, decay_gamma=γ)
        self.注入器 = 回响注入器(
            self.模型, self.池,
            lambda_strength=λ, uncertainty_threshold=τ,
            projection_seed=42, last_n_layers=4,
        )
        self.主设备 = self.模型.device
        self._上次步延迟: Optional[List[float]] = None

    def 切换模式(self, 注入: bool) -> None:
        self.注入器.启用注入 = 注入
        self.注入器.启用捕获 = 注入
        self.池.清空()

    def 生成(self, 消息: str, 轮次: int, 种子: int, max_new_tokens: int,
            收集熵: Optional[List[float]] = None) -> str:
        种子值 = 种子 + 轮次
        torch.manual_seed(种子值)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(种子值)

        消息列表 = [{"role": "user", "content": 消息}]
        提示 = self.分词器.apply_chat_template(消息列表, tokenize=False, add_generation_prompt=True)
        ids = self.分词器(提示, return_tensors="pt").input_ids.to(self.模型.device)
        输入长 = ids.shape[1]

        步延迟: List[float] = []
        上次 = [time.perf_counter()]

        def logits_callback(步, logits):
            if 收集熵 is not None:
                收集熵.append(计算语义熵(logits))

        def 轮次回调(步数, 池):
            现在 = time.perf_counter()
            步延迟.append(现在 - 上次[0])
            上次[0] = 现在

        out = self.注入器.生成(
            ids,
            max_new_tokens=max_new_tokens,
            temperature=采样参数["温度"], top_p=采样参数["top_p"], top_k=采样参数["top_k"],
            repetition_penalty=采样参数["重复惩罚"],
            eos_token_id=self.分词器.eos_token_id,
            logits_callback=logits_callback,
            tokenizer=self.分词器,
            轮次回调=轮次回调,
        )
        self._上次步延迟 = 步延迟
        return self.分词器.decode(out[0, 输入长:], skip_special_tokens=True).strip()

    def 步延迟(self) -> Optional[List[float]]:
        return self._上次步延迟

    def 释放(self) -> None:
        self.模型 = None
        self.注入器 = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ======================================================================
# 成本测量（7.3）
# ======================================================================

def 显存峰值MB() -> dict:
    if not torch.cuda.is_available():
        return {}
    return {
        "峰值显存MB_allocated": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1),
        "峰值显存MB_reserved": round(torch.cuda.max_memory_reserved() / 1024 ** 2, 1),
    }


def _state字节(state: List[torch.Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in state)


def _past字节(past) -> Optional[int]:
    """transformers KV 缓存字节数（兼容 DynamicCache / tuple 两种形态）。"""
    try:
        if hasattr(past, "key_cache"):
            总 = sum(k.numel() * k.element_size() for k in past.key_cache)
            总 += sum(v.numel() * v.element_size() for v in past.value_cache)
            return 总
        总 = 0
        for 层 in past:
            for t in 层:
                总 += t.numel() * t.element_size()
        return 总
    except Exception:
        return None


def 深度延迟曲线_序列(后端, 深度列表: List[int], 生成步: int = 深度生成步) -> dict:
    """
    底座级（不含回响注入）上下文深度延迟曲线：预填充到深度 D，测逐步前向延迟中位。
    RWKV：forward_seq 预填充 + _前向单步；Qwen：model 前向 + past_key_values 增量。
    同时返回该深度下的 State/KV 缓存字节数 → KV vs State 内存对照。
    """
    结果 = {}
    填充句 = "今天天气很好，我心情不错。"
    for D in 深度列表:
        引擎_obj = 后端
        if isinstance(后端, RWKV后端):
            e = 后端.引擎
            分词器 = e.分词器
            填充ids = (分词器.encode(填充句) * (D // 8 + 2))[:D]
            if D > 0:
                state = e.模型.generate_zero_state()
                _, state = e.模型.forward(填充ids, state)
            else:
                state = e.模型.generate_zero_state()
            lat = []
            token_id = 42
            logits = None
            for _ in range(生成步):
                t0 = time.perf_counter()
                logits, state, _ = e._前向单步(token_id, state)
                lat.append(time.perf_counter() - t0)
                token_id = int(logits.argmax(dim=-1).item())
            结果[str(D)] = {
                "步延迟中位ms": round(statistics.median(lat) * 1000, 2),
                "缓存字节": _state字节(state),
                "缓存形态": "RWKV State（O(1)）",
            }
        else:
            m, tok = 后端.模型, 后端.分词器
            填充ids = tok(填充句, return_tensors="pt").input_ids.to(m.device)
            填充ids = 填充ids.repeat(1, max(D // 填充ids.shape[1] + 1, 1))[:, :max(D, 1)]
            with torch.no_grad():
                out = m(input_ids=填充ids, use_cache=True)
                past = out.past_key_values
                下一个 = int(out.logits[:, -1, :].argmax(dim=-1).item())
                lat = []
                t张量 = out.logits[:, -1, :].argmax(dim=-1).reshape(1, 1)  # 必须 [1,1]，1 维会破坏 RoPE 形状
                for _ in range(生成步):
                    t0 = time.perf_counter()
                    out = m(input_ids=t张量, past_key_values=past, use_cache=True)
                    lat.append(time.perf_counter() - t0)
                    past = out.past_key_values
                    t张量 = out.logits[:, -1, :].argmax(dim=-1).reshape(1, 1)
            字节 = _past字节(past)
            结果[str(D)] = {
                "步延迟中位ms": round(statistics.median(lat) * 1000, 2),
                "缓存字节": 字节,
                "缓存形态": "Transformer KV Cache（O(n)）",
            }
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 结果


# ======================================================================
# 认证流程
# ======================================================================

def 认证单模式(后端, 模式名: str, 注入: bool, 提示: List[str], 图灵对话: List[dict],
              max_new_tokens: int, 图灵训练: int, 图灵测试: int, 快速: bool) -> dict:
    """对单一模型单一模式执行：情感提示集生成 + 指标 + 图灵检出 + 成本。"""
    后端.切换模式(注入)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    文本列表: List[str] = []
    熵列表: List[float] = []
    for i, p in enumerate(提示):
        熵: List[float] = []
        文本 = 后端.生成(p, 轮次=i, 种子=种子基数, max_new_tokens=max_new_tokens, 收集熵=熵)
        文本列表.append(文本)
        熵列表.append(statistics.fmean(熵) if 熵 else float("nan"))
        print(f"    [{模式名}] 提示{i + 1}/{len(提示)}：{len(文本)}字 熵={熵列表[-1]:.3f}", flush=True)

    生成函数 = lambda 消息, 轮次, 种子: 后端.生成(消息, 轮次, 种子,
                                              max_new_tokens=64 if not 快速 else 48)
    图灵 = 图灵检出实验(生成函数, 图灵对话, 图灵训练, 图灵测试, 种子基数,
                     日志前缀=f"[{模式名}]")

    记录 = {
        "模式": 模式名,
        "注入": 注入,
        "语义熵均值": round(statistics.fmean([e for e in 熵列表 if e == e]), 4) if any(e == e for e in 熵列表) else None,
        "四gram重复率": round(statistics.fmean([计算重复率(t) for t in 文本列表]), 4),
        "情感词密度": round(statistics.fmean([情感词密度(t) for t in 文本列表]), 4),
        "平均字符数": round(statistics.fmean([len(t) for t in 文本列表]), 1),
        "图灵": 图灵,
        "文本样例": 文本列表[0][:120],
    }
    # 成本：取最后一次生成的步延迟（后端记录）
    延迟 = 后端.步延迟()
    if 延迟 and len(延迟) > 2:
        记录["步延迟中位秒"] = round(statistics.median(延迟[1:]), 4)
        记录["tokens每秒"] = round((len(延迟) - 1) / max(sum(延迟[1:]), 1e-6), 2)
    记录.update(显存峰值MB())
    return 记录


def 认证模型(模型标识: str, 数据: List[dict], 参数: argparse.Namespace) -> Optional[dict]:
    """单模型完整认证：裸 vs 注入 + 深度延迟曲线 + λ* 建议。"""
    配置 = 引擎模块._读系统配置()
    矩阵 = 配置.get("模型矩阵", {})
    条目 = 矩阵[模型标识]
    hidden_dim = int(条目["hidden_dim"])
    扫描表 = 配置.get("扫描表", {})
    表项 = 扫描表.get(str(hidden_dim))
    if 表项 is None:
        print(f"[跳过] {模型标识}：hidden_dim={hidden_dim} 不在扫描表，先用 --标定维度 {hidden_dim} 标定")
        return None
    λ, γ, τ = float(表项[0]), float(表项[1]), float(表项[2])

    分片路径 = _结果目录 / f"认证_{模型标识}.json"
    if 分片路径.exists() and not 参数.强制:
        print(f"[断点] {模型标识} 已有认证分片，跳过（--强制 重跑）：{分片路径.name}")
        return json.loads(分片路径.read_text(encoding="utf-8"))

    print(f"\n{'=' * 60}\n认证 {模型标识}（hidden_dim={hidden_dim}, λ={λ}, γ={γ}, τ={τ}）\n{'=' * 60}")
    if 条目.get("后端") == "rwkv7":
        后端 = RWKV后端(模型标识, 条目, λ, γ, τ)
    else:
        后端 = Qwen后端(模型标识, 条目, λ, γ, τ)
    print(f"  {后端.名称前缀} 后端就绪，hidden_dim={后端.hidden_dim}")

    提示 = [d["user"] for d in 数据[:参数.提示数]]
    裸记录 = 认证单模式(后端, "裸", False, 提示, 数据, 参数.max_new_tokens,
                        参数.图灵训练, 参数.图灵测试, 参数.快速)
    注入记录 = 认证单模式(后端, "注入", True, 提示, 数据, 参数.max_new_tokens,
                      参数.图灵训练, 参数.图灵测试, 参数.快速)

    # ---- 达标判定（先于成本项：即使深度曲线/λ扫描中断，情感认证结果也已落盘）----
    判定 = {
        "无NaN": 裸记录["语义熵均值"] is not None and 注入记录["语义熵均值"] is not None,
        "注入有效差异": None, "无重复退化": None, "生成可持续": None,
    }
    if 判定["无NaN"]:
        ΔH = abs((注入记录["语义熵均值"] or 0) - (裸记录["语义熵均值"] or 0))
        判定["注入有效差异"] = ΔH > 0.05
        重复退化 = 注入记录["四gram重复率"] > 裸记录["四gram重复率"] + 0.15
        判定["无重复退化"] = not 重复退化
        # 防伪通过：注入导致第一步后立即 eos（平均字符数骤降）视为不可持续
        判定["生成可持续"] = 注入记录["平均字符数"] >= max(8.0, 裸记录["平均字符数"] * 0.5)
    达标 = all(v for v in 判定.values() if v is not None)

    记录 = {
        "模型标识": 模型标识, "后端": 条目.get("后端"), "hidden_dim": hidden_dim,
        "参数适配": {"λ": λ, "γ": γ, "τ": τ, "投影种子": 42, "末N层捕获": 4},
        "环境": _环境快照(),
        "裸": 裸记录, "注入": 注入记录,
        "深度延迟曲线": {},
        "达标判定": {**判定, "达标": 达标},
    }
    _结果目录.mkdir(exist_ok=True)
    分片路径.write_text(json.dumps(记录, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [保存] {分片路径.name}（情感认证部分）  达标={'YES' if 达标 else 'NO'}")

    print(f"  [成本] 上下文深度延迟曲线（0/512/2048）...")
    记录["深度延迟曲线"] = 深度延迟曲线_序列(后端, [d for d in 深度点 if d <= 2048], 深度生成步)

    # ---- RWKV 专属：λ* 轻量标定建议 ----
    if 条目.get("后端") == "rwkv7" and not 参数.跳过标定建议:
        记录["标定建议"] = _λ扫描建议(后端, 裸记录, 提示[:min(4, len(提示))], 参数)
    后端.释放()
    分片路径.write_text(json.dumps(记录, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [保存] {分片路径.name}（含成本项）")
    return 记录


def _λ扫描建议(后端: RWKV后端, 裸记录: dict, 提示: List[str], 参数: argparse.Namespace) -> dict:
    """λ 粗扫（短生成）：在 4-gram 重复率可控的前提下选语义熵变化最显著的 λ*。"""
    当前λ = 后端.λ
    候选 = sorted({0.05, 0.1, 0.2, round(当前λ, 3)})
    扫描 = []
    for λc in 候选:
        后端.引擎.λ强度 = λc
        后端.切换模式(True)
        文本s, 熵s = [], []
        for i, p in enumerate(提示):
            熵: List[float] = []
            文本s.append(后端.生成(p, 轮次=100 + i, 种子=种子基数 + 7,
                                max_new_tokens=48 if 参数.快速 else 64, 收集熵=熵))
            熵s.append(statistics.fmean(熵) if 熵 else 0.0)
        重复率 = statistics.fmean([计算重复率(t) for t in 文本s])
        熵均值 = statistics.fmean(熵s) if 熵s else 0.0
        均字 = statistics.fmean([len(t) for t in 文本s])
        扫描.append({"λ": λc, "语义熵均值": round(熵均值, 4),
                     "四gram重复率": 重复率, "平均字符数": round(均字, 1)})
        print(f"    [λ扫描] λ={λc}: 熵={熵均值:.3f} 4gram重复率={重复率:.3f} 均字={均字:.1f}", flush=True)
    后端.引擎.λ强度 = 当前λ

    裸重复率 = 裸记录["四gram重复率"]
    最低字数 = max(8.0, 裸记录["平均字符数"] * 0.5)
    合格 = [s for s in 扫描 if s["四gram重复率"] <= 裸重复率 + 0.15 and s["平均字符数"] >= 最低字数]
    最佳 = max(合格, key=lambda s: abs(s["语义熵均值"] - (裸记录["语义熵均值"] or 0))) if 合格 \
        else max(扫描, key=lambda s: (s["平均字符数"], -s["四gram重复率"]))
    return {"扫描": 扫描, "推荐λ": 最佳["λ"],
            "依据": ("重复率可控且生成可持续下 |ΔH| 最大" if 合格
                     else "全部候选不可持续/退化，取字符数最多且重复率最低者")}


# ======================================================================
# 标定回写（7.4）
# ======================================================================

def 标定并回写(维度: int, 数据: List[dict], 参数: argparse.Namespace) -> None:
    """对扫描表外 hidden_dim 跑裸 vs 注入标定 λ/γ/τ 并回写系统配置.json。"""
    配置 = 引擎模块._读系统配置()
    矩阵 = 配置.get("模型矩阵", {})
    扫描表 = 配置.get("扫描表", {})
    if str(维度) in 扫描表:
        print(f"[标定] hidden_dim={维度} 已在扫描表：{扫描表[str(维度)]}（如需重标定请手动修改）")
        return
    候选 = [(k, v) for k, v in 矩阵.items()
            if int(v["hidden_dim"]) == 维度 and v.get("后端") == "rwkv7"]
    if not 候选:
        print(f"[标定] 矩阵中不存在 hidden_dim={维度} 的 RWKV 模型，无法标定")
        return
    模型标识, 条目 = 候选[0]

    # γ/τ 取邻近维度均值兜底
    邻近 = [(int(k), v) for k, v in 扫描表.items()]
    邻近.sort(key=lambda kv: abs(kv[0] - 维度))
    if 邻近:
        γ = 邻近[0][1][1]
        τ = 邻近[0][1][2]
    else:
        γ, τ = 0.07, 0.09

    print(f"[标定] {模型标识}（hidden_dim={维度}）：λ 粗扫 + γ={γ} τ={τ}（邻近维度）")
    后端 = RWKV后端(模型标识, 条目, 0.1, γ, τ)  # λ 初值占位，扫描中覆盖
    提示 = [d["user"] for d in 数据[:min(6, len(数据))]]

    裸文本, 裸熵 = [], []
    后端.切换模式(False)
    for i, p in enumerate(提示):
        熵: List[float] = []
        裸文本.append(后端.生成(p, 轮次=i, 种子=种子基数, max_new_tokens=64, 收集熵=熵))
        裸熵.append(statistics.fmean(熵) if 熵 else 0.0)
    裸熵均值, 裸重复率 = statistics.fmean(裸熵), statistics.fmean([计算重复率(t) for t in 裸文本])

    扫描 = []
    for λc in [0.05, 0.1, 0.2, 0.4]:
        后端.引擎.λ强度 = λc
        后端.切换模式(True)
        文本s, 熵s = [], []
        for i, p in enumerate(提示):
            熵: List[float] = []
            文本s.append(后端.生成(p, 轮次=50 + i, 种子=种子基数 + 7, max_new_tokens=64, 收集熵=熵))
            熵s.append(statistics.fmean(熵) if 熵 else 0.0)
        重复率 = statistics.fmean([计算重复率(t) for t in 文本s])
        扫描.append({"λ": λc, "语义熵均值": round(statistics.fmean(熵s), 4), "四gram重复率": 重复率})
        print(f"    λ={λc}: 熵={扫描[-1]['语义熵均值']} 重复率={重复率}", flush=True)

    合格 = [s for s in 扫描 if s["四gram重复率"] <= 裸重复率 + 0.15]
    λ星 = (max(合格, key=lambda s: abs(s["语义熵均值"] - 裸熵均值))["λ"] if 合格
           else min(扫描, key=lambda s: s["四gram重复率"])["λ"])
    后端.释放()

    扫描表[str(维度)] = [λ星, γ, τ]
    配置["扫描表"] = 扫描表
    配置路径 = _RWKE_DIR / "系统配置.json"
    备份 = _结果目录 / f"系统配置_标定前_{int(time.time())}.json"
    _结果目录.mkdir(exist_ok=True)
    备份.write_text(json.dumps(引擎模块._读系统配置(), ensure_ascii=False, indent=2), encoding="utf-8")
    配置路径.write_text(json.dumps(配置, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[标定] 已回写 {配置路径.name}：扫描表[{维度}] = [{λ星}, {γ}, {τ}]（备份：{备份.name}）")


# ======================================================================
# 报告（7.5）
# ======================================================================

def _环境快照() -> dict:
    快照 = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda可用": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        快照["gpu"] = torch.cuda.get_device_name(0)
        快照["显存总量GB"] = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
    return 快照


def 生成报告(记录列表: List[dict], 数据来源: str, 参数: argparse.Namespace) -> Path:
    _报告目录.mkdir(exist_ok=True)
    汇总 = {"生成时间": time.strftime("%Y-%m-%d %H:%M:%S"), "数据来源": 数据来源,
            "协议": {**采样参数, "种子": 种子基数, "max_new_tokens": 参数.max_new_tokens},
            "记录": 记录列表}
    汇总路径 = _结果目录 / "认证汇总.json"
    汇总路径.write_text(json.dumps(汇总, ensure_ascii=False, indent=2), encoding="utf-8")

    行 = []
    行.append("# RWKE 测试报告：多模型认证对比\n")
    行.append(f"> 生成时间：{汇总['生成时间']}　|　提示集：{数据来源}\n")
    行.append(f"> 统一协议：T={采样参数['温度']}, top_p={采样参数['top_p']}, top_k={采样参数['top_k']}, "
              f"repetition_penalty={采样参数['重复惩罚']}, seed={种子基数}, "
              f"max_new_tokens={参数.max_new_tokens}\n")

    环境 = 记录列表[0]["环境"] if 记录列表 else _环境快照()
    行.append("\n## 1. 测试环境\n")
    行.append(f"- Python {环境.get('python')} / torch {环境.get('torch')} / CUDA={环境.get('cuda可用')}"
              + (f" / {环境.get('gpu')}（{环境.get('显存总量GB')}GB）" if 环境.get("gpu") else ""))
    行.append(f"- 情感词表来源：{_情感词表来源}\n")

    行.append("\n## 2. 矩阵横向对比\n")
    行.append("| 模型 | hidden_dim | λ/γ/τ | 模式 | 语义熵 | 4gram重复率 | 情感词密度 | "
              "图灵检出率↓ | 类人性↑ | 峰值显存MB | 达标 |")
    行.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in 记录列表:
        p = r["参数适配"]
        for 模式 in ("裸", "注入"):
            m = r[模式]
            行.append(
                f"| {r['模型标识']} | {r['hidden_dim']} | {p['λ']}/{p['γ']}/{p['τ']} | {模式} "
                f"| {m.get('语义熵均值', '-')} | {m.get('四gram重复率', '-')} | {m.get('情感词密度', '-')} "
                f"| {m['图灵']['detection_accuracy']} | {m['图灵']['human_likeness']} "
                f"| {m.get('峰值显存MB_allocated', '-')} | {'YES' if 模式 == '注入' and r['达标判定']['达标'] else '-'} |")

    行.append("\n## 3. 成本：上下文深度延迟曲线 与 KV vs State 内存对照\n")
    行.append("| 模型 | 深度0 延迟ms | 深度512 延迟ms | 深度2048 延迟ms | 深度2048 缓存MB | 缓存形态 |")
    行.append("|---|---|---|---|---|---|")
    for r in 记录列表:
        曲线 = r.get("深度延迟曲线", {})
        cells = [f"{曲线.get(str(d), {}).get('步延迟中位ms', '-')}" for d in 深度点]
        缓存 = 曲线.get("2048", {})
        缓存MB = round(缓存.get("缓存字节", 0) / 1024 ** 2, 2) if 缓存.get("缓存字节") else "-"
        行.append(f"| {r['模型标识']} | {cells[0]} | {cells[1]} | {cells[2]} | {缓存MB} | {缓存.get('缓存形态', '-')} |")
    行.append("\n> RWKV State 尺寸与上下文深度无关（O(1)）；Transformer KV 随深度线性增长（O(n)）。"
              "延迟曲线在底座级（不含回响注入）测得。\n")

    行.append("\n## 4. 各模型认证记录\n")
    for r in 记录列表:
        行.append(f"\n### {r['模型标识']}（后端 {r['后端']}，hidden_dim={r['hidden_dim']}）\n")
        行.append(f"- 参数适配：λ={r['参数适配']['λ']}, γ={r['参数适配']['γ']}, τ={r['参数适配']['τ']}, "
                  f"投影种子={r['参数适配']['投影种子']}, 末N层捕获={r['参数适配']['末N层捕获']}")
        行.append(f"- 裸：熵={r['裸'].get('语义熵均值')} 4gram重复率={r['裸'].get('四gram重复率')} "
                  f"情感密度={r['裸'].get('情感词密度')} 图灵检出率={r['裸']['图灵']['detection_accuracy']}")
        行.append(f"- 注入：熵={r['注入'].get('语义熵均值')} 4gram重复率={r['注入'].get('四gram重复率')} "
                  f"情感密度={r['注入'].get('情感词密度')} 图灵检出率={r['注入']['图灵']['detection_accuracy']}")
        判定 = r["达标判定"]
        行.append(f"- 达标判定：{'**通过**' if 判定.get('达标') else '**未通过**'}"
                  f"（无NaN={判定.get('无NaN')}，注入有效差异={判定.get('注入有效差异')}，"
                  f"无重复退化={判定.get('无重复退化')}，生成可持续={判定.get('生成可持续')}）")
        if "标定建议" in r:
            建议 = r["标定建议"]
            行.append(f"- **λ 标定建议：{建议['推荐λ']}**（{建议['依据']}）；扫描："
                      + "；".join(f"λ={s['λ']}→熵{s['语义熵均值']}/重{s['四gram重复率']}" for s in 建议["扫描"]))
        行.append(f"- 文本样例（裸）：{r['裸'].get('文本样例', '')[:80]}")
        行.append(f"- 文本样例（注入）：{r['注入'].get('文本样例', '')[:80]}")

    行.append("\n## 5. 结论\n")
    通过 = [r["模型标识"] for r in 记录列表 if r["达标判定"].get("达标")]
    未过 = [r["模型标识"] for r in 记录列表 if not r["达标判定"].get("达标")]
    行.append(f"- 认证通过：{通过 or '无'}")
    行.append(f"- 认证未通过：{未过 or '无'}"
              + ("（多为扫描表插值 λ 偏强导致重复退化，按第 4 节 λ 标定建议回写后复测）" if 未过 else ""))
    报告路径 = _报告目录 / "RWKE测试报告.md"
    报告路径.write_text("\n".join(行), encoding="utf-8")
    print(f"\n[报告] {报告路径}")
    print(f"[汇总] {汇总路径}")
    return 报告路径


# ======================================================================
# CLI
# ======================================================================

def main() -> int:
    解析器 = argparse.ArgumentParser(description="RWKE 多模型认证对比测试与报告")
    解析器.add_argument("--模型", nargs="*", default=None, help="模型矩阵标识子集（默认矩阵内全部已就位模型）")
    解析器.add_argument("--数据", default=None, help="提示集 JSON 路径（默认 HF 缓存 > 工作区样本）")
    解析器.add_argument("--提示数", type=int, default=10, help="情感指标提示条数（默认 10）")
    解析器.add_argument("--max-new-tokens", type=int, default=256, help="情感指标生成上限（默认 256）")
    解析器.add_argument("--图灵训练", type=int, default=60, help="TuringBench 训练对数（默认 60）")
    解析器.add_argument("--图灵测试", type=int, default=30, help="TuringBench 测试对数（默认 30）")
    解析器.add_argument("--快速", action="store_true", help="快速模式：短生成管线验证")
    解析器.add_argument("--强制", action="store_true", help="忽略断点分片重跑")
    解析器.add_argument("--跳过标定建议", action="store_true", help="跳过 RWKV λ 扫描建议")
    解析器.add_argument("--标定维度", type=int, default=None, help="对扫描表外 hidden_dim 标定并回写（7.4）")
    参数 = 解析器.parse_args()

    if 参数.快速:
        参数.提示数 = min(参数.提示数, 4)
        参数.max_new_tokens = min(参数.max_new_tokens, 96)
        参数.图灵训练 = min(参数.图灵训练, 30)
        参数.图灵测试 = min(参数.图灵测试, 15)

    try:
        数据 = 加载提示数据(参数.数据, 最小需求=参数.图灵训练 + max(参数.图灵测试, 1))
    except FileNotFoundError as e:
        print(f"错误：{e}")
        return 2

    配置 = 引擎模块._读系统配置()
    矩阵 = 配置.get("模型矩阵", {})

    # 7.4：标定回写模式（独立于认证流程）
    if 参数.标定维度 is not None:
        标定并回写(参数.标定维度, 数据, 参数)
        return 0

    标识列表 = 参数.模型
    if not 标识列表:
        标识列表 = []
        for k, v in 矩阵.items():
            路径 = Path(v["路径"])
            if not 路径.is_absolute():
                路径 = _RWKE_DIR / 路径
            if 路径.exists():
                标识列表.append(k)
            else:
                print(f"[缺模型] {k} 未就位（{路径}），跳过认证")
    print(f"[矩阵] 本轮认证：{标识列表}")

    记录列表 = []
    for 标识 in 标识列表:
        try:
            记录 = 认证模型(标识, 数据, 参数)
            if 记录:
                记录列表.append(记录)
        except Exception as e:
            print(f"[失败] {标识}：{type(e).__name__}: {e}")
            traceback.print_exc()

    if not 记录列表:
        print("错误：无任何模型完成认证")
        return 1
    生成报告(记录列表, _数据来源, 参数)
    return 0


if __name__ == "__main__":
    sys.exit(main())
