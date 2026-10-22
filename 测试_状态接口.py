# -*- coding: utf-8 -*-
"""
RWKE 统一状态接口 · 单元测试
============================
全部使用合成张量，不加载任何模型。

覆盖用例：
1. 五操作对称性（RWKV状态 / KV缓存状态）：初始化→前进→克隆→保存→加载 全链路；
2. 克隆独立性（深拷贝语义）：克隆体上"前进"不影响原状态；
3. 序列化往返一致性：保存后加载的张量逐元素相等，元数据完整；
4. 模型标识校验：错误标识加载应抛出 ValueError；
5. RWKV-7 扁平 List 结构（n_layer*3 张量）在克隆/序列化中的结构保持。

运行方式：
    cd RWKE
    python 测试_状态接口.py
"""

import os
import tempfile
import traceback

import torch

from 状态接口 import RWKV状态, KV缓存状态, 推理状态, 加载状态


# ----------------------------------------------------------------------
# 合成状态构造器
# ----------------------------------------------------------------------

def 合成rwkv状态(n_layer=4, hidden_dim=8, n_head=2, head_size=4):
    """构造合成 RWKV-7 state：扁平 List[torch.Tensor]，长 n_layer*3，
    每层 3 张量：att_x_prev [n_embd]、att_kv [n_head, head_size, head_size] (fp32)、
    ffn_x_prev [n_embd]——与 rwkv pip 包 generate_zero_state() 实测结构一致。"""
    状态 = []
    for _ in range(n_layer):
        状态.append(torch.randn(hidden_dim))                                          # att_x_prev
        状态.append(torch.randn(n_head, head_size, head_size, dtype=torch.float32))   # att_kv
        状态.append(torch.randn(hidden_dim))                                          # ffn_x_prev
    return 状态


def 合成kv状态(n_layer=4, n_head_kv=2, head_dim=4, 序列长=3):
    """构造合成 past_key_values：每层 (k, v) 二元组，各 [batch, n_head_kv, seq_len, head_dim]。"""
    形状 = (1, n_head_kv, 序列长, head_dim)
    return tuple(
        (torch.randn(*形状), torch.randn(*形状))
        for _ in range(n_layer)
    )


# ----------------------------------------------------------------------
# 用例
# ----------------------------------------------------------------------

结果记录 = []


def 用例(名称):
    def 装饰器(函数):
        def 执行():
            try:
                函数()
                结果记录.append((名称, True, ''))
                print(f"  ✓ {名称}")
            except Exception:
                结果记录.append((名称, False, traceback.format_exc()))
                print(f"  ✗ {名称}")
                traceback.print_exc()
        执行.名称 = 名称
        return 执行
    return 装饰器


@用例('抽象基类可被子类实现（接口完整性）')
def 测试抽象接口():
    # 抽象基类不可直接实例化
    try:
        推理状态()
        raise AssertionError('推理状态 抽象类不应能直接实例化')
    except TypeError:
        pass
    # 两个适配器都实现了全部五个操作
    for cls in (RWKV状态, KV缓存状态):
        实例 = cls()
        for 方法 in ('初始化', '前进', '克隆', '保存', '加载'):
            assert callable(getattr(实例, 方法)), f"{cls.__name__} 缺少操作 '{方法}'"


@用例('RWKV状态 五操作对称性（初始化→前进→克隆→保存→加载）')
def 测试rwkv五操作():
    原始 = 合成rwkv状态()
    状态 = RWKV状态()
    状态.初始化()
    assert 状态.原始状态 is None, '初始化后应为 None'
    状态.前进(原始)
    assert len(状态) == 4 * 3, '前进后应为 12 个张量（4 层 × 3）'
    克隆 = 状态.克隆()
    assert isinstance(克隆, RWKV状态), '克隆应返回同类型'
    with tempfile.TemporaryDirectory() as 目录:
        路径 = os.path.join(目录, 'rwkv_state.pth')
        元数据 = 克隆.保存(路径, 模型标识='RWKV7-World-0.4B', hidden_dim=1024, n_layer=24)
        加载后, 元数据2 = RWKV状态.加载(路径)
        assert isinstance(加载后.原始状态, list), '加载后应保持扁平 List 结构'
        assert len(加载后.原始状态) == 12, '加载后应保持 12 个张量'
        for 原张量, 载张量 in zip(加载后.原始状态, 克隆.原始状态):
            assert torch.equal(原张量, 载张量), '加载后的张量应与保存前逐元素相等'
        assert 元数据['格式'] == 'rwkv_state'
        assert 元数据2['模型标识'] == 'RWKV7-World-0.4B'


@用例('KV缓存状态 五操作对称性（初始化→前进→克隆→保存→加载）')
def 测试kv五操作():
    原始 = 合成kv状态()
    状态 = KV缓存状态()
    状态.初始化()
    assert 状态.原始状态 is None, '初始化后应为 None'
    状态.前进(原始)
    assert len(状态) == 4, '前进后应为 4 层'
    克隆 = 状态.克隆()
    assert isinstance(克隆, KV缓存状态), '克隆应返回同类型'
    with tempfile.TemporaryDirectory() as 目录:
        路径 = os.path.join(目录, 'kv_state.pth')
        元数据 = 克隆.保存(路径, 模型标识='Qwen2.5-1.5B-Instruct', hidden_dim=1536, n_layer=28)
        加载后, 元数据2 = KV缓存状态.加载(路径)
        for 层原, 层载 in zip(加载后.原始状态, 克隆.原始状态):
            for 原张量, 载张量 in zip(层原, 层载):
                assert torch.equal(原张量, 载张量), '加载后的张量应与保存前逐元素相等'
        assert 元数据['格式'] == 'kv_cache'
        assert 元数据2['hidden_dim'] == 1536


@用例('克隆独立性：RWKV 克隆体前进不影响原状态（深拷贝语义）')
def 测试rwkv克隆独立():
    原始 = 合成rwkv状态()
    状态 = RWKV状态(原始)
    克隆 = 状态.克隆()
    # 克隆应保持扁平 List 结构且叶子张量独立
    assert isinstance(克隆.原始状态, list) and len(克隆.原始状态) == len(原始), \
        '克隆应保持扁平 List 结构'
    # 在克隆体上模拟"前进"：原地修改克隆体张量，再整体替换为新的状态增量
    克隆.原始状态[0].add_(999.0)                   # 原地修改克隆体叶子张量
    assert not torch.equal(克隆.原始状态[0], 原始[0]), '克隆体张量应已改变'
    新增量 = 合成rwkv状态()                        # 整体替换式前进
    克隆.前进(新增量)
    # 原状态不受任何影响（深拷贝语义）
    assert torch.equal(状态.原始状态[0], 原始[0]), '原地修改克隆体不应影响原状态'
    assert 状态.原始状态[0].abs().max() < 100, '原状态张量不应出现 +999 偏移'
    # 克隆体确实已替换为新增量
    assert 克隆.原始状态 is 新增量, '克隆体前进后应指向新增量'


@用例('克隆独立性：KV缓存 克隆体修改不影响原状态（深拷贝语义）')
def 测试kv克隆独立():
    原始 = 合成kv状态()
    状态 = KV缓存状态(原始)
    克隆 = 状态.克隆()
    # 在克隆体上原地修改
    k0 = 克隆.原始状态[0][0]
    k0.add_(777.0)
    # 原状态不受影响
    assert torch.equal(状态.原始状态[0][0], 原始[0][0]), '原地修改克隆体不应影响原状态'
    assert not torch.equal(克隆.原始状态[0][0], 原始[0][0]), '克隆体应已发生改变'


@用例('序列化往返一致性：.pth 保存/加载张量逐元素相等且元数据完整')
def 测试序列化往返():
    with tempfile.TemporaryDirectory() as 目录:
        # RWKV（扁平 List 逐叶子比对）
        rwkv路径 = os.path.join(目录, 'RoleB_状态_RWKV7-World-1.5B.pth')
        rwkv状态 = RWKV状态(合成rwkv状态(n_layer=24, hidden_dim=16))
        rwkv状态.保存(rwkv路径, 模型标识='RWKV7-World-1.5B', hidden_dim=2048, n_layer=24)
        rwkv载, 元数据 = 加载状态(rwkv路径)
        assert 元数据['格式'] == 'rwkv_state' and 元数据['模型标识'] == 'RWKV7-World-1.5B'
        assert 元数据['hidden_dim'] == 2048 and 元数据['n_layer'] == 24
        assert 元数据['时间戳'], '时间戳元数据应存在'
        assert isinstance(rwkv载, RWKV状态)
        assert len(rwkv载.原始状态) == 24 * 3, '加载后应保持 72 个张量（24 层 × 3）'
        for 原t, 载t in zip(rwkv状态.原始状态, rwkv载.原始状态):
            assert torch.equal(原t, 载t.to(原t.device)), '往返后张量应逐元素相等'
        # KV
        kv路径 = os.path.join(目录, 'kv_状态_Qwen.pth')
        kv状态 = KV缓存状态(合成kv状态(n_layer=28))
        kv状态.保存(kv路径, 模型标识='Qwen2.5-1.5B-Instruct', hidden_dim=1536, n_layer=28)
        kv载, kv元 = 加载状态(kv路径)
        assert kv元['格式'] == 'kv_cache' and isinstance(kv载, KV缓存状态)
        for 层原, 层载 in zip(kv状态.原始状态, kv载.原始状态):
            for 原t, 载t in zip(层原, 层载):
                assert torch.equal(原t, 载t.to(原t.device)), '往返后张量应逐元素相等'


@用例('模型标识校验：错误标识加载抛 ValueError')
def 测试模型标识校验():
    with tempfile.TemporaryDirectory() as 目录:
        路径 = os.path.join(目录, 'RoleB_状态_RWKV7-World-0.4B.pth')
        RWKV状态(合成rwkv状态()).保存(路径, 模型标识='RWKV7-World-0.4B', hidden_dim=1024, n_layer=24)
        # 正确标识 → 通过
        加载状态(路径, 期望模型标识='RWKV7-World-0.4B')
        # 错误标识 → 抛异常
        try:
            加载状态(路径, 期望模型标识='RWKV7-World-1.5B')
            raise AssertionError('错误模型标识应抛出 ValueError')
        except ValueError as 异常:
            assert '模型标识不匹配' in str(异常), f'异常信息应可读，实际：{异常}'
        # 不校验（None）→ 通过
        加载状态(路径, 期望模型标识=None)


@用例('格式校验：非 RWKE 状态文件/未知格式抛 ValueError')
def 测试格式校验():
    with tempfile.TemporaryDirectory() as 目录:
        # 保存一个 KV 格式文件，尝试按 RWKV 格式直接加载
        路径 = os.path.join(目录, '混合.pth')
        KV缓存状态(合成kv状态()).保存(路径, 模型标识='M', hidden_dim=8, n_layer=2)
        try:
            RWKV状态.加载(路径)
            raise AssertionError('KV 文件不应能按 RWKV 状态加载')
        except ValueError:
            pass
        # 完全无关的 .pth
        杂项路径 = os.path.join(目录, '杂项.pth')
        torch.save({'随机': [1, 2, 3]}, 杂项路径)
        try:
            加载状态(杂项路径)
            raise AssertionError('非 RWKE 状态文件应抛出 ValueError')
        except ValueError:
            pass


@用例('GPU 张量保存回退 CPU（无 GPU 环境等价验证：非 CPU 设备张量转 CPU 序列化）')
def 测试设备无关序列化():
    # 模拟"GPU 上生成、CPU 保存"：若本机有 CUDA 则用 cuda 张量，否则跳过设备分支
    原始 = 合成rwkv状态(n_layer=2, hidden_dim=4)
    if torch.cuda.is_available():
        gpu状态 = [t.to('cuda') for t in 原始]
        with tempfile.TemporaryDirectory() as 目录:
            路径 = os.path.join(目录, 'gpu_state.pth')
            RWKV状态(gpu状态).保存(路径, 模型标识='M', hidden_dim=4, n_layer=2)
            载, _ = 加载状态(路径)
            assert 载.原始状态[0].device.type == 'cpu', '加载后张量应在 CPU（map_location）'
    else:
        # 无 GPU：验证 CPU 路径本身不受影响
        with tempfile.TemporaryDirectory() as 目录:
            路径 = os.path.join(目录, 'cpu_state.pth')
            RWKV状态(原始).保存(路径, 模型标识='M', hidden_dim=4, n_layer=2)
            载, _ = 加载状态(路径)
            assert 载.原始状态[0].device.type == 'cpu'


# ----------------------------------------------------------------------
# 汇总执行
# ----------------------------------------------------------------------

def main():
    print('=' * 64)
    print('RWKE 统一状态接口 · 单元测试（合成张量，无需模型）')
    print('=' * 64)

    测试函数 = [
        测试抽象接口,
        测试rwkv五操作,
        测试kv五操作,
        测试rwkv克隆独立,
        测试kv克隆独立,
        测试序列化往返,
        测试模型标识校验,
        测试格式校验,
        测试设备无关序列化,
    ]
    for 函数 in 测试函数:
        函数()

    通过 = sum(1 for _, 成, _ in 结果记录 if 成)
    总数 = len(结果记录)
    print('-' * 64)
    for 名称, 成, 错误 in 结果记录:
        if not 成:
            print(f'失败用例：{名称}\n{错误}')
    print(f'汇总：{通过}/{总数} 通过')
    if 通过 != 总数:
        raise SystemExit(1)
    print('全部通过 ✅')


if __name__ == '__main__':
    main()
