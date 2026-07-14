"""JAX 的 RNN wiring(连接拓扑)。"""

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np


def fully_connected(output_size: int, input_size: int, **_):
    """创建全连接掩码。

    Args:
        output_size (int): 输出大小
        input_size (int): 输入大小

    Returns:
        array: 形状 (output, input_size) 的全 1 掩码
    """
    return jnp.ones((output_size, input_size))


def fully_connected_no_self(output_size: int, input_size: int, **_):
    """类似 fully_connected,但主对角线为 0。"""
    mask = fully_connected(output_size, input_size)
    assert output_size < input_size, f"output_size {output_size} unexpectedly larger than input_size {input_size}"
    rem = input_size - output_size
    return mask - jnp.concatenate([jnp.zeros((output_size, rem)), jnp.eye(output_size)], axis=1)


def random(output_size: int, input_size: int, key=None, sparsity=0.5, **_):
    """随机稀疏掩码。

    Args:
        output_size (int): 隐空间大小
        input_size (int): 输入大小
        key (_type_, optional): jax 随机 key,默认 None。
        sparsity (float, optional): 元素为 0 的概率,默认 0.5。

    Returns:
        _type_: 形状 (num_units, num_units+input_size) 的 0/1 掩码
    """
    if key is None:
        key = jrandom.PRNGKey(0)
    mask = jrandom.bernoulli(key, 1 - sparsity, shape=(output_size, input_size))
    mask = jnp.array(mask, dtype=float)
    return mask


def ncp(num_units: int, input_size: int, interneurons: int, key=None, sparsity=0.3):
    """神经电路策略(NCP)wiring。"""
    assert num_units >= interneurons, f"num_units ({num_units}) must be greater equal interneurons ({interneurons})"
    output_size = num_units - interneurons
    if key is None:
        key = jrandom.PRNGKey(0)
    mask = jnp.zeros((num_units, input_size))
    # 中间神经元接收来自输入和中间神经元的连接
    mask = mask.at[-interneurons:, :-output_size].set(
        jrandom.bernoulli(key, 1 - sparsity, shape=(interneurons, input_size - output_size))
    )
    # 所有神经元都接收来自中间神经元的连接
    mask = mask.at[:, -interneurons:].set(jrandom.bernoulli(key, 1 - sparsity, shape=(num_units, interneurons)))

    # state_strings = [f'o{j}' for j in range(output_neurons)]
    # state_strings += [f'r{j}' for j in range(num_units - interneurons - output_neurons)]
    # state_strings += [f'h{j}' for j in range(interneurons)]
    # inputs_strings = [f'i{j}' for j in range(input_size)] + state_strings
    # print('    ' + ' '.join(inputs_strings))
    # for j, line in enumerate(mask):
    #     print(state_strings[j] + ' ' + str(line))
    return mask


def make_mask_initializer(wiring_name: str, bias=True, **kwargs):
    """按给定 wiring 名称创建掩码。"""

    def make_mask(key, shape, dtype):
        mask = globals()[wiring_name](*shape, key=key, **kwargs)
        if bias:
            # 强制使 bias 可见
            mask = mask.at[:, -1].set(1.0)
        return mask

    return make_mask
