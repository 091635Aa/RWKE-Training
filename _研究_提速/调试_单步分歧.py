# -*- coding: utf-8 -*-
"""单步分歧定位：引擎 _前向单步 vs TS 模块，逐层逐槽比对，找第一个分歧点。"""
import os
os.environ["RWKV_V7_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "0"
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
from pathlib import Path
from typing import List

_RWKE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RWKE_DIR))
from rwke引擎 import RWKE引擎  # noqa: E402

引擎 = RWKE引擎("RWKV7-G1-1.5B")
z = 引擎.模型.z
n_layer, H, N, C = 引擎.n_layer, 引擎.n_head, 引擎.head_size, 引擎.n_embd
设备, DT = 引擎.设备, 引擎.权重dtype

state0 = 引擎.模型.generate_zero_state()
if state0 and state0[0].device != 设备:
    state0 = [t.to(设备) for t in state0]

# ---- TS 模块（与研究脚本同构）----
_每层参数: List[torch.Tensor] = []
for i in range(n_layer):
    att = f"blocks.{i}.att."; ffn = f"blocks.{i}.ffn."; bbb = f"blocks.{i}."
    _每层参数.extend([
        z[bbb+"ln1.weight"], z[bbb+"ln1.bias"],
        z[att+"x_r"], z[att+"x_w"], z[att+"x_k"], z[att+"x_v"], z[att+"x_a"], z[att+"x_g"],
        z[att+"w0"], z[att+"w1"], z[att+"w2"],
        z[att+"a0"], z[att+"a1"], z[att+"a2"],
        z[att+"v0"], z[att+"v1"], z[att+"v2"],
        z[att+"g1"], z[att+"g2"],
        z[att+"k_k"], z[att+"k_a"], z[att+"r_k"],
        z[att+"receptance.weight"], z[att+"key.weight"], z[att+"value.weight"], z[att+"output.weight"],
        z[att+"ln_x.weight"], z[att+"ln_x.bias"],
        z[bbb+"ln2.weight"], z[bbb+"ln2.bias"],
        z[ffn+"x_k"], z[ffn+"key.weight"], z[ffn+"value.weight"],
    ])

class _TS步进模块(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.层参数 = _每层参数
        self.emb = z["emb.weight"]
        self.ln_out_w = z["ln_out.weight"]
        self.ln_out_b = z["ln_out.bias"]
        self.head = z["head.weight"]
        self.L = n_layer; self.Hn = H; self.Nn = N; self.Cn = C

    def forward(self, token张量: torch.Tensor, 状态列表: List[torch.Tensor]):
        x = self.emb[token张量]
        v_first = torch.empty_like(x)
        Hn = self.Hn; Nn = self.Nn; Cn = self.Cn; L = self.L
        # TS 列表按值传递：槽位重绑必须落到【新列表】并返回（函数式），原地改不入调用方
        新状态: List[torch.Tensor] = []
        for t0 in 状态列表:
            新状态.append(t0)
        for i in range(L):
            b = i * 33
            lnx = torch.layer_norm(x, (Cn,), weight=self.层参数[b+0], bias=self.层参数[b+1])
            x0 = x
            xx0 = 新状态[i*3+0] - lnx
            xr = lnx + xx0 * self.层参数[b+2]
            xw = lnx + xx0 * self.层参数[b+3]
            xk = lnx + xx0 * self.层参数[b+4]
            xv = lnx + xx0 * self.层参数[b+5]
            xa = lnx + xx0 * self.层参数[b+6]
            xg = lnx + xx0 * self.层参数[b+7]
            r = xr @ self.层参数[b+22]
            w = torch.tanh(xw @ self.层参数[b+9]) @ self.层参数[b+10]
            k = xk @ self.层参数[b+23]
            v = xv @ self.层参数[b+24]
            a = torch.sigmoid(self.层参数[b+11] + (xa @ self.层参数[b+12]) @ self.层参数[b+13])
            g = torch.sigmoid(xg @ self.层参数[b+17]) @ self.层参数[b+18]
            kk = torch.nn.functional.normalize((k * self.层参数[b+19]).view(Hn, Nn), dim=-1, p=2.0).view(Hn*Nn)
            k = k * (1 + (a - 1) * self.层参数[b+20])
            if i == 0:
                v_first = v
            else:
                v = v + (v_first - v) * torch.sigmoid(self.层参数[b+14] + (xv @ self.层参数[b+15]) @ self.层参数[b+16])
            w = torch.exp(-0.606531 * torch.sigmoid((self.层参数[b+8] + w).float()))
            kv = 状态列表[i*3+1]
            vk = v.view(Hn, Nn, 1) @ k.view(Hn, 1, Nn)
            ab = (-kk).view(Hn, Nn, 1) @ (kk * a).view(Hn, 1, Nn)
            kv = kv * w.view(Hn, 1, Nn) + kv @ ab.float() + vk.float()
            xx0 = kv.to(x0.dtype) @ r.view(Hn, Nn, 1)
            xx0 = torch.nn.functional.group_norm(xx0.view(1, Hn*Nn), num_groups=Hn,
                                                 weight=self.层参数[b+26], bias=self.层参数[b+27], eps=64e-5).view(Hn*Nn)
            xx0 = xx0 + ((r * k * self.层参数[b+21]).view(Hn, Nn).sum(dim=-1, keepdim=True) * v.view(Hn, Nn)).view(Hn*Nn)
            xx = (xx0 * g) @ self.层参数[b+25]
            新状态[i*3+0] = lnx
            新状态[i*3+1] = kv
            x = x0 + xx
            xx1 = torch.layer_norm(x, (Cn,), weight=self.层参数[b+28], bias=self.层参数[b+29])
            d = 新状态[i*3+2] - xx1
            kk2 = xx1 + d * self.层参数[b+30]
            kk2 = torch.relu(kk2 @ self.层参数[b+31]) ** 2
            out = kk2 @ self.层参数[b+32]
            新状态[i*3+2] = xx1
            x = x + out
        x = torch.layer_norm(x, (Cn,), weight=self.ln_out_w, bias=self.ln_out_b)
        logits = x @ self.head
        return logits, 新状态, x

ts = torch.jit.script(_TS步进模块())

# ---- 单步比对 ----
tok = 59
sj = [t.clone() for t in state0]
st = [t.clone() for t in state0]
lg_j, stj, _hd = 引擎._前向单步(tok, sj)
lg_t, st, _xt2 = ts(torch.tensor(tok, device=设备), st)
print(f"单步 logits 差: {(lg_j.squeeze(0).float() - lg_t.float()).abs().max().item():.3e}")

# ---- 第 0 层值对照：att_x_prev 应为 ln1(emb[59]) ----
lnx直接 = torch.nn.functional.layer_norm(z["emb.weight"][tok], (C,),
                                          weight=z["blocks.0.ln1.weight"], bias=z["blocks.0.ln1.bias"])
print("engine sj[0][:4] :", sj[0][:4].tolist())
print("ts     st[0][:4] :", st[0][:4].tolist())
print("直接lnx   [:4]   :", lnx直接[:4].tolist())
print("engine sj[0] vs lnx 差:", (sj[0].float() - lnx直接.float()).abs().max().item())
print("ts     st[0] vs lnx 差:", (st[0].float() - lnx直接.float()).abs().max().item())

# kv 第0层对照：零态下 kv_new 应= vk.float()
print("engine sj[1] 范数:", sj[1].norm().item())
print("ts     st[1] 范数:", st[1].norm().item())

print("层 | att_x_prev差 | kv差 | ffn_x_prev差")
for i in range(n_layer):
    d0 = (sj[i*3+0].float() - st[i*3+0].float()).abs().max().item()
    d1 = (sj[i*3+1].float() - st[i*3+1].float()).abs().max().item()
    d2 = (sj[i*3+2].float() - st[i*3+2].float()).abs().max().item()
    flag = " ←" if max(d0, d1, d2) > 1e-2 else ""
    if i < 6 or max(d0, d1, d2) > 1e-2:
        print(f"{i:2d} | {d0:.3e} | {d1:.3e} | {d2:.3e}{flag}")
