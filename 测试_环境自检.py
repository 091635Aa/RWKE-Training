# -*- coding: utf-8 -*-
"""
RWKE（RWKV-Kernel-Engine）环境自检脚本
=====================================

用途：
    在将 SemanticEcho 迁移到 RWKV-7（Goose，纯 RNN 架构）之前，
    逐项检查本机运行环境是否就绪，并输出中文分节报告。

检查项：
    1. Python 版本
    2. torch 版本、CUDA 可用性（torch.cuda.is_available()）、设备名、显存总量/可用
    3. rwkv 包版本；RWKV_V7_ON（RWKV-7 支持）能力判断
    4. nvcc（CUDA 编译器）可用性，据此给出 RWKV_CUDA_ON 设置建议
    5. 模型矩阵文件存在性（必需/可选）
    6. 其他关键依赖 import 检查（numpy、transformers、peft、sklearn、jieba、tqdm）

设计约定：
    - 每一项检查独立 try/except，任何一项失败都不会抛出异常；
    - 无论缺失什么，脚本一律以退出码 0 结束（缺依赖时输出明确指引而非报错中断）。
"""

import os
import sys
import platform
import shutil
import subprocess
import inspect
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# 输出编码：Windows 控制台默认 GBK，强制切换 UTF-8，避免中文乱码
# ---------------------------------------------------------------------------
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ---------------------------------------------------------------------------
# 路径常量：脚本位于 RWKE 根目录下，模型矩阵相对该目录解析
# ---------------------------------------------------------------------------
RWKE_DIR = Path(__file__).resolve().parent
MODELS_DIR = RWKE_DIR / "models"

# 模型矩阵：期望路径（相对 models 目录）、说明、是否必需、下载指引
MODEL_MATRIX = [
    {
        "path": "RWKV7-World-0.4B.pth",
        "desc": "RWKV7-World-0.4B（hidden_dim=1024, n_layer=24）",
        "required": True,
        "guide": (
            'huggingface-cli download BlinkDL/rwkv-7-world RWKV7-World-0.4B.pth '
            '--local-dir "models"（国内可将 HF_ENDPOINT 设为 https://hf-mirror.com）'
        ),
    },
    {
        "path": "RWKV7-World-1.5B.pth",
        "desc": "RWKV7-World-1.5B（hidden_dim=2048, n_layer=24）",
        "required": True,
        "guide": (
            'huggingface-cli download BlinkDL/rwkv-7-world RWKV7-World-1.5B.pth '
            '--local-dir "models"（国内可将 HF_ENDPOINT 设为 https://hf-mirror.com）'
        ),
    },
    {
        "path": "Qwen2.5-1.5B-Instruct",
        "desc": "基线 Qwen2.5-1.5B-Instruct（HF 目录格式）",
        "required": True,
        "guide": (
            'huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct '
            '--local-dir "models/Qwen2.5-1.5B-Instruct"'
        ),
    },
    {
        "path": "RWKV7-G1-1.5B.pth",
        "desc": "RWKV7-G1-1.5B（可选）",
        "required": False,
        "guide": (
            'huggingface-cli download BlinkDL/rwkv7-g1 RWKV7-G1-1.5B.pth '
            '--local-dir "models"（国内可将 HF_ENDPOINT 设为 https://hf-mirror.com）'
        ),
    },
]

# 其他关键依赖：(import 模块名, pip 发行包名)
OTHER_DEPS = [
    ("numpy", "numpy"),
    ("transformers", "transformers"),
    ("peft", "peft"),
    ("sklearn", "scikit-learn"),
    ("jieba", "jieba"),
    ("tqdm", "tqdm"),
]

# rwkv 包支持 RWKV-7 的最低版本（官方说明 0.8.27 起支持 V7）
RWKV_V7_MIN_VERSION = (0, 8, 27)

LINE = "-" * 66
HEAD = "=" * 66

# 缺失项汇总：[(缺失项名称, 安装/下载指引), ...]
MISSING_ITEMS = []


def add_missing(name: str, guide: str) -> None:
    """记录一个缺失项，供最后统一汇总输出。"""
    MISSING_ITEMS.append((name, guide))


def parse_version(ver) -> tuple:
    """把 '0.8.27a0' 之类的版本字符串解析成可比较的数字元组 (0, 8, 27)。"""
    nums = []
    for part in str(ver).split("."):
        digits = ""
        for ch in part:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        nums.append(int(digits))
    return tuple(nums)


def run_section(title: str, func) -> None:
    """统一执行某一节检查：打印分节标题，并兜底捕获一切异常。"""
    print()
    print(LINE)
    print(f"【{title}】")
    print(LINE)
    try:
        func()
    except Exception as exc:  # 任何未预期错误都不中断整体自检
        print(f"  [异常] 本节检查出现未预期错误：{exc!r}")
        add_missing(f"检查项「{title}」", "请根据上方错误信息排查本机环境后重新运行本脚本")


# ---------------------------------------------------------------------------
# 检查 1：Python 版本
# ---------------------------------------------------------------------------
def check_python() -> None:
    ver = platform.python_version()
    impl = platform.python_implementation()
    print(f"  [信息] 解释器路径：{sys.executable}")
    print(f"  [信息] 系统：{platform.system()} {platform.release()}（{platform.machine()}）")
    if parse_version(ver) >= (3, 9):
        print(f"  [OK] Python 版本：{ver}（{impl}，推荐 >= 3.9）")
    else:
        print(f"  [警告] Python 版本：{ver}（{impl}）低于推荐版本 3.9，新版 torch 等依赖可能不支持")
        add_missing(
            "Python >= 3.9",
            "建议安装 Python 3.9 ~ 3.12：https://www.python.org/downloads/",
        )


# ---------------------------------------------------------------------------
# 检查 2：torch 版本、CUDA 可用性、设备名、显存
# ---------------------------------------------------------------------------
def check_torch() -> None:
    try:
        import torch
    except Exception as exc:
        print(f"  [缺失] 无法 import torch（{exc.__class__.__name__}: {exc}）")
        add_missing(
            "torch",
            "按本机 CUDA 版本安装对应构建，例如：pip install torch "
            "--index-url https://download.pytorch.org/whl/cu124；"
            "详见 https://pytorch.org/get-started/locally/",
        )
        return

    print(f"  [OK] torch 版本：{torch.__version__}")

    if not torch.cuda.is_available():
        print("  [警告] CUDA 不可用（torch.cuda.is_available() = False）")
        print("         将以 CPU / 纯 PyTorch 模式运行，推理速度会显著变慢")
        add_missing(
            "CUDA 版 torch（可选但强烈推荐）",
            "按本机 CUDA 版本安装 GPU 构建，例如：pip install torch "
            "--index-url https://download.pytorch.org/whl/cu124；"
            "详见 https://pytorch.org/get-started/locally/",
        )
        return

    print("  [OK] CUDA 可用：torch.cuda.is_available() = True")

    # 设备名
    try:
        print(f"  [OK] GPU 设备：{torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"  [警告] 无法获取 GPU 设备名：{exc!r}")

    # 显存总量 / 可用
    total_bytes = None
    free_bytes = None
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    except Exception:
        try:
            total_bytes = torch.cuda.get_device_properties(0).total_memory
        except Exception:
            pass
    if total_bytes is not None:
        total_gib = total_bytes / 1024 ** 3
        if free_bytes is not None:
            free_gib = free_bytes / 1024 ** 3
            print(f"  [OK] 显存：可用 {free_gib:.1f} GiB / 总量 {total_gib:.1f} GiB")
        else:
            print(f"  [OK] 显存总量：{total_gib:.1f} GiB（可用量查询不支持，已跳过）")
    else:
        print("  [警告] 无法读取显存信息")


# ---------------------------------------------------------------------------
# 检查 3：rwkv 包版本与 RWKV_V7_ON（RWKV-7 支持）
# ---------------------------------------------------------------------------
def check_rwkv() -> None:
    try:
        import rwkv
    except Exception as exc:
        print(f"  [缺失] 无法 import rwkv（{exc.__class__.__name__}: {exc}）")
        add_missing(
            "rwkv 包",
            'pip install "rwkv>=0.8.27"（目标版本 0.8.32；0.8.27 起支持 RWKV-7，'
            "使用前需设置环境变量 RWKV_V7_ON=1）",
        )
        return

    # 包版本：优先取模块属性，失败则回退到包元数据
    version = getattr(rwkv, "__version__", None)
    if not version:
        try:
            from importlib.metadata import version as _pkg_version
            version = _pkg_version("rwkv")
        except Exception:
            version = None
    print(f"  [OK] rwkv 包已安装，版本：{version or '未知'}")

    # RWKV-7 支持检测：优先看 rwkv.model 源码是否含 V7 相关逻辑，
    # 无法确认源码时按版本号（>= 0.8.27）兜底判断
    v7_support = False
    try:
        import rwkv.model as rwkv_model
        src = inspect.getsource(rwkv_model)
        v7_support = ("RWKV_V7_ON" in src) or ("RWKV7" in src) or ("v7" in src.lower())
    except Exception:
        pass
    if v7_support:
        print("  [OK] 检测到 rwkv.model 中存在 RWKV-7 支持逻辑（支持 RWKV_V7_ON=1 开关）")
    else:
        ver_tuple = parse_version(version) if version else ()
        if ver_tuple and ver_tuple >= RWKV_V7_MIN_VERSION:
            print(f"  [OK] rwkv 版本 >= 0.8.27，按官方说明支持 RWKV-7（需设置 RWKV_V7_ON=1）")
        else:
            print("  [警告] 无法确认 rwkv 包的 RWKV-7 支持能力，可能版本过低")
            add_missing(
                "rwkv 包（>= 0.8.27，支持 RWKV-7）",
                'pip install -U "rwkv>=0.8.27"（目标版本 0.8.32），'
                "并设置环境变量 RWKV_V7_ON=1",
            )

    # 当前环境变量状态（仅提示，不算缺失项）
    v7_on = os.environ.get("RWKV_V7_ON")
    cuda_on = os.environ.get("RWKV_CUDA_ON")
    print(f"  [信息] 当前 RWKV_V7_ON   = {v7_on!r}（加载 RWKV-7 模型前需设为 1）")
    print(f"  [信息] 当前 RWKV_CUDA_ON = {cuda_on!r}（1=编译 CUDA 算子；0/未设=纯 PyTorch）")


# ---------------------------------------------------------------------------
# 检查 4：nvcc（CUDA 编译器）可用性 → 给出 RWKV_CUDA_ON 建议
# ---------------------------------------------------------------------------
def check_nvcc() -> None:
    nvcc_path = shutil.which("nvcc")
    if nvcc_path is None:
        print("  [缺失] 未在 PATH 中找到 nvcc（CUDA 编译器）")
        print("         → 建议：设置 RWKV_CUDA_ON=0（纯 PyTorch 回退模式，无需编译即可运行）")
        print("         如需启用 CUDA 算子加速：请安装 NVIDIA CUDA Toolkit 并将 bin 加入 PATH")
        add_missing(
            "nvcc（CUDA 编译器，可选）",
            "安装 NVIDIA CUDA Toolkit：https://developer.nvidia.com/cuda-downloads，"
            "安装后将 bin 目录加入 PATH；或直接采用 RWKV_CUDA_ON=0 纯 PyTorch 模式运行",
        )
        return

    print(f"  [OK] 找到 nvcc：{nvcc_path}")
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        text = (result.stdout or "") + (result.stderr or "")
        release_line = ""
        for line in text.splitlines():
            if "release" in line.lower():
                release_line = line.strip()
                break
        if release_line:
            print(f"  [OK] {release_line}")
        else:
            print("  [OK] nvcc 可正常调用")
        print("         → 建议：设置 RWKV_CUDA_ON=1 以启用 CUDA 算子加速")
        print("           （若编译失败，仍可回退 RWKV_CUDA_ON=0 纯 PyTorch 模式）")
    except Exception as exc:
        print(f"  [警告] nvcc 存在但调用失败：{exc!r}")
        print("         → 建议：设置 RWKV_CUDA_ON=0（纯 PyTorch 回退模式）")
        add_missing(
            "nvcc 可正常调用",
            "检查 CUDA Toolkit 安装与 PATH 配置；或直接采用 RWKV_CUDA_ON=0 纯 PyTorch 模式",
        )


# ---------------------------------------------------------------------------
# 检查 5：模型矩阵文件存在性
# ---------------------------------------------------------------------------
def check_models() -> None:
    print(f"  [信息] 模型根目录：{MODELS_DIR}")
    if not MODELS_DIR.is_dir():
        print("  [缺失] models 目录不存在（首次运行前需下载模型，见下方指引）")

    for item in MODEL_MATRIX:
        path = MODELS_DIR / item["path"]
        tag = "必需" if item["required"] else "可选"

        if path.is_file():
            size_gib = path.stat().st_size / 1024 ** 3
            print(f"  [OK] [{tag}] {item['desc']}")
            print(f"        路径：{path}（{size_gib:.2f} GiB）")
        elif path.is_dir():
            # HF 目录格式：以 config.json 是否存在判断完整性
            if (path / "config.json").is_file():
                print(f"  [OK] [{tag}] {item['desc']}（HF 目录格式，含 config.json）")
                print(f"        路径：{path}")
            else:
                print(f"  [警告] [{tag}] {item['desc']}：目录存在但缺少 config.json，可能下载不完整")
                print(f"        路径：{path}")
                add_missing(f"模型：{item['desc']}（目录不完整）", item["guide"])
        else:
            print(f"  [缺失] [{tag}] {item['desc']}")
            print(f"        期望路径：{path}")
            if item["required"]:
                add_missing(f"模型（必需）：{item['desc']}", f"期望路径：{path}\n         下载：{item['guide']}")
            else:
                print("        （可选组件，缺失不阻塞主流程）")
                add_missing(f"模型（可选）：{item['desc']}", f"期望路径：{path}\n         下载：{item['guide']}")


# ---------------------------------------------------------------------------
# 检查 6：其他关键依赖 import 检查
# ---------------------------------------------------------------------------
def check_other_deps() -> None:
    for mod_name, dist_name in OTHER_DEPS:
        try:
            module = __import__(mod_name)
            version = getattr(module, "__version__", None)
            if not version:
                try:
                    from importlib.metadata import version as _pkg_version
                    version = _pkg_version(dist_name)
                except Exception:
                    version = "未知"
            print(f"  [OK] {dist_name}：{version}")
        except Exception as exc:
            print(f"  [缺失] {dist_name}（import {mod_name} 失败：{exc.__class__.__name__}: {exc}）")
            add_missing(f"Python 包 {dist_name}", f"pip install {dist_name}")


# ---------------------------------------------------------------------------
# 汇总：缺失项与指引
# ---------------------------------------------------------------------------
def print_summary() -> None:
    print()
    print(HEAD)
    print("【缺失项汇总与安装/下载指引】")
    print(HEAD)
    if not MISSING_ITEMS:
        print("  [OK] 全部检查项通过，环境就绪，可以开始 RWKE 开发。")
    else:
        print(f"  共发现 {len(MISSING_ITEMS)} 项缺失/待处理：")
        print()
        for idx, (name, guide) in enumerate(MISSING_ITEMS, 1):
            print(f"  {idx}. {name}")
            print(f"     指引：{guide}")
            print()

    # 运行前提醒（不作为缺失项计数）
    print(LINE)
    print("  [提醒] 运行 RWKV-7 引擎前请确认：")
    print("    1) 设置环境变量 RWKV_V7_ON=1（开启 RWKV-7 支持）")
    print("    2) 有 nvcc 时可设 RWKV_CUDA_ON=1 加速；无 nvcc 或编译失败时设 RWKV_CUDA_ON=0（纯 PyTorch 回退）")
    print('    3) Windows PowerShell 设置示例：$env:RWKV_V7_ON="1" / $env:RWKV_CUDA_ON="0"')
    print('       Windows CMD 设置示例：set RWKV_V7_ON=1 / set RWKV_CUDA_ON=0')
    print(HEAD)


def main() -> int:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(HEAD)
    print(" RWKE（RWKV-Kernel-Engine）环境自检报告")
    print(f" 检查时间：{now}")
    print(f" 脚本位置：{Path(__file__).resolve()}")
    print(HEAD)

    run_section("1/6 Python 版本", check_python)
    run_section("2/6 torch 与 CUDA", check_torch)
    run_section("3/6 rwkv 包与 RWKV_V7_ON 支持", check_rwkv)
    run_section("4/6 nvcc 与 RWKV_CUDA_ON 建议", check_nvcc)
    run_section("5/6 模型矩阵文件存在性", check_models)
    run_section("6/6 其他关键依赖 import 检查", check_other_deps)

    print_summary()

    print()
    print("自检完成。按规格要求，无论缺失什么都以退出码 0 结束。")
    # 规格要求：缺失依赖时输出明确指引而非抛异常，因此固定返回 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
