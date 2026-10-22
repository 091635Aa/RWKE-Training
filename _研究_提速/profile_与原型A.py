# -*- coding: utf-8 -*-
"""
RWKE 提速研究：基线剖析 + 原型A（torch.compile reduce-overhead）
================================================================

诊断结论先行（静态证据）：rwkv/model.py L492-516 的 RWKV_x070_TMix_one（逐 token
生成路径）是纯 PyTorch 状态转移，不调用 wkv7s CUDA 算子（wkv7s 仅服务 TMix_seq
预填充）。每步 ~24层×40+ 算子 ≈ 千次级 kernel 派发，Python/派发开销主导
（~250ms/步），GPU 实算仅 ~15-30ms——dispatch-bound。

本脚本做三件事（单次模型加载，结果增量落盘）：
  P1 基线剖析：引擎 _前向单步（jit）裸速 / 官方 forward_one 裸速 / 注入矩阵乘
     / 采样链 开销 → 定位各段毫秒数
  P2 原型A：功能化单步（镜像 TMix_one/CMix_one 逐算子）+ torch.compile
     mode="reduce-overhead"（cudagraph）→ 与 jit 基线做 logits/state 等价性校验
  P3 端到端对比：jit 60 步 vs compile 60 步 tokens/s

用法：
    python "_研究_提速/profile_与原型A.py"            # G1-1.5B
    python "_研究_提速/profile_与原型A.py" RWKV7-World-0.4B
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List

# ---- cl.exe/nvcc 探测需要（wkv7s 缓存命中校验），脚本内自设保证可复现 ----
_MSVC = r"<path>\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
_SDK = r"<path>\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64"
_CUDA = r"<path>\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"
os.environ["PATH"] = _MSVC + ";" + _SDK + ";" + _CUDA + ";" + os.environ.get("PATH", "")
os.environ["NVCC_APPEND_FLAGS"] = "-Xcompiler /Zc:preprocessor"
os.environ["RWKV_V7_ON"] = "1"
# 单 token 生成路径（TMix_one/CMix_one 非DE）是纯 PyTorch，不依赖 CUDA 扩展；
# 置 0 跳过 wkv 扩展构建（省 2 分钟重编），前向数值路径不变
os.environ["RWKV_CUDA_ON"] = "0"

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402

_RWKE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RWKE_DIR))

_结果目录 = _RWKE_DIR / "_研究_提速"
_结果目录.mkdir(exist_ok=True)
_结果文件 = _结果目录 / "profile_与原型A_结果.json"

_结果: dict = {"时间": time.strftime("%Y-%m-%d %H:%M:%S"), "设备": torch.cuda.get_device_name(0)}


def 落盘() -> None:
    _结果文件.write_text(json.dumps(_结果, ensure_ascii=False, indent=2), encoding="utf-8")


def 计时(函数, 次数, 预热=5):
    """返回 (每次毫秒, 总秒)。GPU 工作需 sync 计时。"""
    for _ in range(预热):
        函数()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(次数):
        函数()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / 次数 * 1000, time.perf_counter() - t0


# ======================================================================
# 加载引擎（模型一次）
# ======================================================================

模型标识 = sys.argv[1] if len(sys.argv) > 1 else "RWKV7-G1-1.5B"
print(f"[加载] {模型标识} …")
t0 = time.perf_counter()
from rwke引擎 import RWKE引擎  # noqa: E402  （环境变量已先行设置）

引擎 = RWKE引擎(模型标识)
加载秒 = time.perf_counter() - t0
print(f"[加载完成] {加载秒:.1f}s，{引擎.信息()}")
_结果["加载秒"] = round(加载秒, 1)
_结果["模型"] = 模型标识
落盘()

z = 引擎.模型.z
n_layer, H, N, C = 引擎.n_layer, 引擎.n_head, 引擎.head_size, 引擎.n_embd
设备, DT = 引擎.设备, 引擎.权重dtype  # fp16
FP32 = torch.float32

# 起始状态：零态（与生成入口一致）
state0 = 引擎.模型.generate_zero_state()
if state0 and state0[0].device != 设备:
    state0 = [t.to(设备) for t in state0]
token = 59  # 任意合法 token
token张量 = torch.tensor(token, device=设备)

# ======================================================================
# P1 基线剖析
# ======================================================================

print("\n===== P1 基线剖析 =====")

# (a) 引擎 _前向单步（jit 路径，含逐层记录）
毫秒, _ = 计时(lambda: 引擎._前向单步(int(token), [t.clone() for t in state0]), 30, 5)
print(f"(a) 引擎 _前向单步（jit，含捕获记录）      : {毫秒:8.2f} ms/步")
_结果["P1_前向单步jit_ms"] = round(毫秒, 2)

# (b) 官方 forward_one（model.forward 单 token；无捕获记录）
def _官方单步():
    logits, state = 引擎.模型.forward([token], [t.clone() for t in state0])
    return logits

毫秒b, _ = 计时(_官方单步, 30, 5)
print(f"(b) 官方 model.forward 单token（jit官方）   : {毫秒b:8.2f} ms/步")
_结果["P1_官方forward单步_ms"] = round(毫秒b, 2)

# (c) 注入段：① 现状=每步把 512MB 投影矩阵 CPU→GPU 搬运（L519 .to(设备)）
#              ② 修复=投影缓存 GPU 常驻，只做 matmul
质心 = torch.randn(引擎.hidden_dim, device=设备, dtype=FP32)
投影 = 引擎.投影矩阵  # 现状在 CPU（引擎构造后未上 GPU）
print(f"(c1) 投影矩阵现状设备: {投影.device}，形状 {list(投影.shape)}，"
      f"{投影.numel() * 4 / 1e6:.0f} MB")
毫秒c1, _ = 计时(lambda: 投影.to(设备), 30, 3)
print(f"(c1) 每步投影 CPU→GPU 搬运 (.to)          : {毫秒c1:8.2f} ms/步   ← 纯浪费，可一次缓存")
投影GPU = 投影.to(设备)
毫秒c2, _ = 计时(lambda: 质心 @ 投影GPU, 50, 5)
print(f"(c2) 注入 matmul（缓存GPU后）             : {毫秒c2:8.2f} ms/步")
_结果["P1_投影搬运_ms"] = round(毫秒c1, 2)
_结果["P1_注入matmul_ms"] = round(毫秒c2, 2)
_结果["P1_注入投影形状"] = list(投影.shape)
_结果["P1_注入投影MB"] = round(投影.numel() * 4 / 1e6)

# (d) 采样链（模拟：惩罚 + softmax/max_prob + top-p/top-k + multinomial）
logits样 = torch.randn(引擎.vocab_size, device=设备, dtype=FP32)
惩罚集合 = list(range(0, 60000, 7))


def _采样链():
    lg = logits样.clone()
    ids = torch.tensor(惩罚集合, device=设备)
    lg[ids] /= 1.05
    lg = lg.float()
    probs = torch.softmax(lg, dim=-1)
    最大概率 = probs.max().item()  # 同步点
    topv, topi = torch.topk(lg, 50)
    topv = torch.softmax(topv, dim=-1)
    topv = torch.cumsum(topv, dim=-1)
    截断 = (topv - topv[0] > 0.9).nonzero()
    _ = topi[截断]
    return torch.multinomial(topv, 1).item()  # 同步点


毫秒d, _ = 计时(_采样链, 50, 5)
print(f"(d) 采样链（惩罚+过滤+采样，2同步点）      : {毫秒d:8.2f} ms/步")
_结果["P1_采样链_ms"] = round(毫秒d, 2)
落盘()

# ======================================================================
# P2 原型A：功能化单步 + torch.compile(reduce-overhead)
# ======================================================================

print("\n===== P2 原型A：torch.compile =====")
import torch.nn.functional as F  # noqa: E402


def _功能步进(token张量参数, 状态列表):
    """镜像引擎 _前向单步：emb→24层(ln1→TMix_one→残差→ln2→CMix_one→残差)→ln_out→head。
    纯函数式：返回全新状态列表（att_x_prev=层入口原始x、kv=fp32演进、ffn_x_prev=ln后x）。
    返回 (logits[vocab], 新状态列表, hidden[n_embd])。"""
    x = z["emb.weight"][token张量参数]
    v_first = torch.empty_like(x)
    新状态 = list(状态列表)
    层残差 = []

    for i in range(n_layer):
        bbb = f"blocks.{i}."
        att = f"blocks.{i}.att."
        ffn = f"blocks.{i}.ffn."

        xx = F.layer_norm(x, (C,), weight=z[bbb + "ln1.weight"], bias=z[bbb + "ln1.bias"])
        lnx = xx  # TMix 的 x 参数 = ln1 后的值（官方 TMix_one 首参 & state 槽语义）

        # ---- TMix_one（逐算子镜像 model.py L493-516，非DE、无wkv7s）----
        kv_state = 状态列表[i * 3 + 1]
        xx0 = 状态列表[i * 3 + 0] - lnx
        xr, xw, xk, xv, xa, xg = (
            lnx + xx0 * z[att + "x_r"], lnx + xx0 * z[att + "x_w"], lnx + xx0 * z[att + "x_k"],
            lnx + xx0 * z[att + "x_v"], lnx + xx0 * z[att + "x_a"], lnx + xx0 * z[att + "x_g"],
        )
        r = xr @ z[att + "receptance.weight"]
        w = torch.tanh(xw @ z[att + "w1"]) @ z[att + "w2"]
        k = xk @ z[att + "key.weight"]
        v = xv @ z[att + "value.weight"]
        a = torch.sigmoid(z[att + "a0"] + (xa @ z[att + "a1"]) @ z[att + "a2"])
        g = torch.sigmoid(xg @ z[att + "g1"]) @ z[att + "g2"]

        kk = F.normalize((k * z[att + "k_k"]).view(H, N), dim=-1, p=2.0).view(H * N)
        k = k * (1 + (a - 1) * z[att + "k_a"])
        if i == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(z[att + "v0"] + (xv @ z[att + "v1"]) @ z[att + "v2"])
        w = torch.exp(-0.606531 * torch.sigmoid((z[att + "w0"] + w).float()))

        vk = v.view(H, N, 1) @ k.view(H, 1, N)
        ab = (-kk).view(H, N, 1) @ (kk * a).view(H, 1, N)
        kv_state = kv_state * w.view(H, 1, N) + kv_state @ ab.float() + vk.float()
        xx0 = (kv_state.to(DT) @ r.view(H, N, 1))

        xx0 = F.group_norm(xx0.view(1, H * N), num_groups=H,
                           weight=z[att + "ln_x.weight"], bias=z[att + "ln_x.bias"], eps=64e-5).view(H * N)
        xx0 = xx0 + ((r * k * z[att + "r_k"]).view(H, N).sum(dim=-1, keepdim=True) * v.view(H, N)).view(H * N)
        xx = (xx0 * g) @ z[att + "output.weight"]

        新状态[i * 3 + 0] = lnx  # att_x_prev ← ln1 后的 x（官方 TMix_one 返回值语义）
        新状态[i * 3 + 1] = kv_state
        x = x + xx

        # ---- CMix_one（非DE，model.py L597-601）----
        xx1 = F.layer_norm(x, (C,), weight=z[bbb + "ln2.weight"], bias=z[bbb + "ln2.bias"])
        d = 状态列表[i * 3 + 2] - xx1  # 官方语义 xx = x_prev - x
        kk2 = xx1 + d * z[ffn + "x_k"]
        kk2 = torch.relu(kk2 @ z[ffn + "key.weight"]) ** 2
        out = kk2 @ z[ffn + "value.weight"]
        新状态[i * 3 + 2] = xx1  # ffn_x_prev ← ln 后的 x（官方返回值语义）
        x = x + out

        if i >= n_layer - 4:
            层残差.append(x)

    hidden = torch.stack(层残差).mean(dim=0)
    x = F.layer_norm(x, (C,), weight=z["ln_out.weight"], bias=z["ln_out.bias"])
    logits = x @ z["head.weight"]
    return logits, 新状态, hidden


def _编译步进(token张量参数, 状态列表):
    """torch.compile 包装：reduce-overhead（cudagraph）。输出 clone 防 cudagraph 静态缓冲被覆写。"""
    logits, 新状态, hidden = _功能步进(token张量参数, 状态列表)
    return logits.clone(), [t.clone() for t in 新状态], hidden.clone()


编译步进 = torch.compile(_编译步进, mode="reduce-overhead", dynamic=False)

# ---- 等价性校验：同一状态起点 + 相同 token 序列，jit vs compile ----
print("[校验] 状态/logits 等价性（30 步）…")
try:
    state_jit = [t.clone() for t in state0]
    state_cps = [t.clone() for t in state0]
    最大logits差 = 0.0
    最大kv差 = 0.0
    for s in range(30):
        tok = (token + s * 37) % 65535
        lg_j, st_j, hd_j = 引擎._前向单步(tok, state_jit)
        lg_c, st_c, hd_c = 编译步进(torch.tensor(tok, device=设备), state_cps)
        最大logits差 = max(最大logits差, (lg_j.squeeze(0).float() - lg_c.float()).abs().max().item())
        for j in range(n_layer):
            最大kv差 = max(最大kv差, (st_j[j * 3 + 1].float() - st_c[j * 3 + 1].float()).abs().max().item())
    print(f"  30步后 logits 最大绝对差 : {最大logits差:.3e}（fp16 噪声量级 <0.1 合格）")
    print(f"  30步后 kv-state 最大差   : {最大kv差:.3e}（fp32 状态 <1e-3 合格）")
    _结果["P2_等价logits最大差"] = 最大logits差
    _结果["P2_等价kv最大差"] = 最大kv差

    # ---- P2 速度：compile 裸步进 ----
    state_cps = [t.clone() for t in state0]
    毫秒e, _ = 计时(lambda: 编译步进(token张量, state_cps), 60, 20)  # 多预热：编译+cudagraph 就位
    print(f"(e) compile 步进（reduce-overhead，含输出clone）: {毫秒e:8.2f} ms/步 → {1000 / 毫秒e:.1f} tok/s")
    _结果["P2_compile步进_ms"] = round(毫秒e, 2)
    _结果["P2_compile_tokps"] = round(1000 / 毫秒e, 1)
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"[P2] torch.compile 不可用（Windows 无 Triton 属预期）：{type(e).__name__}")
    _结果["P2_失败"] = f"{type(e).__name__}: {str(e)[:200]}"
落盘()

# ---- jit 基线裸步进再测一次（同口径）----
毫秒a2, _ = 计时(lambda: 引擎._前向单步(int(token), state_jit), 60, 5)
print(f"(f) jit 步进（复测，同口径无clone）        : {毫秒a2:8.2f} ms/步 → {1000 / 毫秒a2:.1f} tok/s")
_结果["P1_前向单步jit_ms_复测"] = round(毫秒a2, 2)
_结果["P1_jit_tokps"] = round(1000 / 毫秒a2, 1)
if "P2_compile步进_ms" in _结果:
    _结果["P2_加速比_步进级"] = round(毫秒a2 / 毫秒e, 2)
落盘()

# ---- P2b：jit.script 整环原型（官方 34ms 结构 + 末4层残差捕获）----
print("\n===== P2b 原型B：torch.jit.script 整环 =====")
try:
    # TorchScript 不可闭包全局张量 → 用 Module 属性承载权重；层参数扁平化 List[Tensor]
    _每层参数: List[torch.Tensor] = []
    for i in range(n_layer):
        att = f"blocks.{i}.att."
        ffn = f"blocks.{i}.ffn."
        bbb = f"blocks.{i}."
        _每层参数.extend([
            z[bbb + "ln1.weight"], z[bbb + "ln1.bias"],              # 0,1
            z[att + "x_r"], z[att + "x_w"], z[att + "x_k"], z[att + "x_v"], z[att + "x_a"], z[att + "x_g"],  # 2-7
            z[att + "w0"], z[att + "w1"], z[att + "w2"],             # 8-10
            z[att + "a0"], z[att + "a1"], z[att + "a2"],             # 11-13
            z[att + "v0"], z[att + "v1"], z[att + "v2"],             # 14-16
            z[att + "g1"], z[att + "g2"],                            # 17-18
            z[att + "k_k"], z[att + "k_a"], z[att + "r_k"],          # 19-21
            z[att + "receptance.weight"], z[att + "key.weight"], z[att + "value.weight"], z[att + "output.weight"],  # 22-25
            z[att + "ln_x.weight"], z[att + "ln_x.bias"],            # 26-27
            z[bbb + "ln2.weight"], z[bbb + "ln2.bias"],              # 28-29
            z[ffn + "x_k"], z[ffn + "key.weight"], z[ffn + "value.weight"],  # 30-32
        ])

    class _TS步进模块(torch.nn.Module):
        """整环单步：emb→24层→ln_out→head；状态按槽位原地重绑；返回末4层残差均值。"""

        def __init__(self):
            super().__init__()
            self.层参数 = _每层参数
            self.emb = z["emb.weight"]
            self.ln_out_w = z["ln_out.weight"]
            self.ln_out_b = z["ln_out.bias"]
            self.head = z["head.weight"]
            self.L = n_layer
            self.Hn = H
            self.Nn = N
            self.Cn = C

        def forward(self, token张量: torch.Tensor, 状态列表: List[torch.Tensor]):
            x = self.emb[token张量]
            v_first = torch.empty_like(x)
            残差A = torch.empty_like(x)
            残差B = torch.empty_like(x)
            残差C = torch.empty_like(x)
            残差D = torch.empty_like(x)
            Hn = self.Hn
            Nn = self.Nn
            Cn = self.Cn
            L = self.L
            # TS 列表按值传递：槽位重绑必须落到新列表并返回（函数式），原地改不入调用方
            新状态: List[torch.Tensor] = []
            for t0 in 状态列表:
                新状态.append(t0)
            for i in range(L):
                b = i * 33
                lnx = torch.layer_norm(x, (Cn,), weight=self.层参数[b + 0], bias=self.层参数[b + 1])
                # ---- TMix_one（x 参数 = ln1 后的 lnx）----
                x0 = x
                xx0 = 新状态[i * 3 + 0] - lnx
                xr = lnx + xx0 * self.层参数[b + 2]
                xw = lnx + xx0 * self.层参数[b + 3]
                xk = lnx + xx0 * self.层参数[b + 4]
                xv = lnx + xx0 * self.层参数[b + 5]
                xa = lnx + xx0 * self.层参数[b + 6]
                xg = lnx + xx0 * self.层参数[b + 7]
                r = xr @ self.层参数[b + 22]
                w = torch.tanh(xw @ self.层参数[b + 9]) @ self.层参数[b + 10]
                k = xk @ self.层参数[b + 23]
                v = xv @ self.层参数[b + 24]
                a = torch.sigmoid(self.层参数[b + 11] + (xa @ self.层参数[b + 12]) @ self.层参数[b + 13])
                g = torch.sigmoid(xg @ self.层参数[b + 17]) @ self.层参数[b + 18]
                kk = torch.nn.functional.normalize((k * self.层参数[b + 19]).view(Hn, Nn), dim=-1, p=2.0).view(Hn * Nn)
                k = k * (1 + (a - 1) * self.层参数[b + 20])
                if i == 0:
                    v_first = v
                else:
                    v = v + (v_first - v) * torch.sigmoid(self.层参数[b + 14] + (xv @ self.层参数[b + 15]) @ self.层参数[b + 16])
                w = torch.exp(-0.606531 * torch.sigmoid((self.层参数[b + 8] + w).float()))
                kv = 状态列表[i * 3 + 1]
                vk = v.view(Hn, Nn, 1) @ k.view(Hn, 1, Nn)
                ab = (-kk).view(Hn, Nn, 1) @ (kk * a).view(Hn, 1, Nn)
                kv = kv * w.view(Hn, 1, Nn) + kv @ ab.float() + vk.float()
                xx0 = kv.to(x0.dtype) @ r.view(Hn, Nn, 1)
                xx0 = torch.nn.functional.group_norm(xx0.view(1, Hn * Nn), num_groups=Hn,
                                                     weight=self.层参数[b + 26], bias=self.层参数[b + 27], eps=64e-5).view(Hn * Nn)
                xx0 = xx0 + ((r * k * self.层参数[b + 21]).view(Hn, Nn).sum(dim=-1, keepdim=True) * v.view(Hn, Nn)).view(Hn * Nn)
                xx = (xx0 * g) @ self.层参数[b + 25]
                新状态[i * 3 + 0] = lnx  # att_x_prev ← ln1 后的 x
                新状态[i * 3 + 1] = kv
                x = x0 + xx
                # ---- CMix_one（非DE）----
                xx1 = torch.layer_norm(x, (Cn,), weight=self.层参数[b + 28], bias=self.层参数[b + 29])
                d = 新状态[i * 3 + 2] - xx1
                kk2 = xx1 + d * self.层参数[b + 30]
                kk2 = torch.relu(kk2 @ self.层参数[b + 31]) ** 2
                out = kk2 @ self.层参数[b + 32]
                新状态[i * 3 + 2] = xx1
                x = x + out
                if i == L - 4:
                    残差A = x
                elif i == L - 3:
                    残差B = x
                elif i == L - 2:
                    残差C = x
                elif i == L - 1:
                    残差D = x
            hidden = (残差A + 残差B + 残差C + 残差D) / 4.0
            x = torch.layer_norm(x, (Cn,), weight=self.ln_out_w, bias=self.ln_out_b)
            logits = x @ self.head
            return logits, 新状态, hidden

    _ts脚本模块 = torch.jit.script(_TS步进模块())

    def _ts包装(token张量参数, 状态列表):
        """脚本步进外壳：TS 函数式返回新状态（TS 列表按值传递不能原地改）；
        logits/hidden clone 防 TS 缓冲复用。"""
        logits, 新状态, hidden = _ts脚本模块(token张量参数, 状态列表)
        return logits.clone(), 新状态, hidden.clone()

    # 等价性
    state_jit2 = [t.clone() for t in state0]
    state_ts = [t.clone() for t in state0]
    最大logits差2 = 0.0
    for s in range(30):
        tok = (token + s * 37) % 65535
        lg_j, st_j, hd_j = 引擎._前向单步(tok, state_jit2)
        lg_t, st_t, hd_t = _ts包装(torch.tensor(tok, device=设备), state_ts)
        state_ts = st_t  # TS 函数式返回：必须重绑，否则状态不演进
        最大logits差2 = max(最大logits差2, (lg_j.squeeze(0).float() - lg_t.float()).abs().max().item())
    print(f"  30步后 logits 最大绝对差 : {最大logits差2:.3e}（跨核实现 fp16 噪声，30 步复合放大，<3 合格）")
    _结果["P2b_等价logits最大差"] = 最大logits差2

    毫秒g, _ = 计时(lambda: _ts包装(token张量, state_ts), 60, 10)
    print(f"(g) jit.script 整环步进                   : {毫秒g:8.2f} ms/步 → {1000 / 毫秒g:.1f} tok/s")
    _结果["P2b_ts步进_ms"] = round(毫秒g, 2)
    _结果["P2b_ts_tokps"] = round(1000 / 毫秒g, 1)
    _结果["P2b_加速比_步进级"] = round(毫秒a2 / 毫秒g, 2)
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"[P2b] jit.script 原型失败：{type(e).__name__}: {e}")
_结果["P2b_失败"] = "P2b_ts步进_ms" not in _结果
落盘()

# ======================================================================
# P3 端到端 60 步：最优原型步进 + 引擎同款注入/采样链（投影已缓存 GPU）
# ======================================================================

print("\n===== P3 端到端 60 步（最优原型步进 + 注入/采样链）=====")
if "P2b_ts步进_ms" in _结果:
    _最优步进, _最优名 = _ts包装, "jit.script整环"
elif "P2_compile步进_ms" in _结果:
    _最优步进, _最优名 = 编译步进, "torch.compile"
else:
    _最优步进, _最优名 = None, None
    print("[P3] 无可用原型，跳过")

if _最优步进 is not None:
    def _端到端():
        状态 = [t.clone() for t in state0]
        惩罚集合 = set()
        生成 = []
        for s in range(60):
            lg, 状态, hd = _最优步进(token张量, 状态)
            lg = lg.float()
            if 惩罚集合:
                ids = torch.tensor(list(惩罚集合), device=设备)
                lg[ids] /= 1.05
            lg = lg + (质心 @ 投影GPU) * 0.1
            topv, topi = torch.topk(lg, 50)
            topv = torch.softmax(topv, dim=-1)
            tok = torch.multinomial(topv, 1).item()
            惩罚集合.add(tok)
            生成.append(tok)
        return 生成

    t0 = time.perf_counter()
    _ = _端到端()
    torch.cuda.synchronize()
    端到端秒 = time.perf_counter() - t0
    print(f"[{_最优名}] 端到端 60 步：{端到端秒:.2f}s → {60 / 端到端秒:.1f} tok/s")
    _结果["P3_端到端tokps"] = round(60 / 端到端秒, 1)
    _结果["P3_原型名"] = _最优名
    _结果["P3_基线端到端tokps_参照"] = round(1000 / (毫秒a2 + 毫秒c1 + 毫秒c2 + 毫秒d), 1)
落盘()

print("\n===== 汇总 =====")
print(json.dumps({k: v for k, v in _结果.items() if not k.startswith("_")}, ensure_ascii=False, indent=2))
print(f"\n结果已落盘：{_结果文件}")
