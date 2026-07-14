"""JAX 模型的工具函数。"""

from functools import partial
import os
import json
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
import orbax.checkpoint
from jax.tree_util import tree_map, tree_reduce
import jax.tree_util as jtu


class JAX_RNG:
    """便于管理 jax PRNG 的基类。"""

    def __init__(self, rng) -> None:
        """初始化内部 rng。"""
        self._rng = rng

    @property
    def rng(self):
        """切分内部 rng。"""
        self._rng, rng = jrandom.split(self._rng)
        return rng


def symlog(x):
    """对称对数。"""
    return jnp.sign(x) * jnp.log(jnp.abs(x) + 1)


def sigmoid_between(x, lower, upper):
    """将输入经 sigmoid 映射到 [lower, upper] 区间。"""
    return (upper - lower) * jax.nn.sigmoid(x) + lower


@jax.jit
def preprocess_img(img):
    """将 RGB 图像转为灰度图。"""
    import dm_pix as pix

    return pix.rgb_to_grayscale(jnp.array(img / 255.0, dtype=jnp.float32))


def tree_norm(tree, **kwargs):
    """树中所有元素的范数之和。"""
    return tree_reduce(lambda x, y: x + jnp.linalg.norm(y, **kwargs), tree, initializer=0)


def leaf_norms(tree):
    """返回叶子名及其范数构成的字典。"""
    flattened, _ = jtu.tree_flatten_with_path(tree)
    flattened = {jtu.keystr(k): v for k, v in flattened}
    return {k: tree_reduce(lambda x, y: x + jnp.linalg.norm(y), v, initializer=0) for k, v in flattened.items()}


@partial(jax.jit, static_argnames=["batch_size"])
def zeros_like_tree(tree, batch_size=None):
    """创建带 batch 维的全零 pytree。"""
    if batch_size is not None:
        return tree_map(lambda x: jnp.zeros((batch_size,) + x.shape), tree)
    else:
        return tree_map(lambda x: jnp.zeros_like(x), tree)


def tree_stack(trees):
    """取一组树,把每个对应叶子做 stack。

    例如,给定两棵树 ((a, b), c) 和 ((a', b'), c'),返回
    ((stack(a, a'), stack(b, b')), stack(c, c'))。
    便于把一组对象转成可喂给 vmapped 函数的形式。
    取自 https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = jax.tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(leaf) for leaf in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


def checkpointing(path, fresh=False, hparams: dict = None):
    """在给定路径设置 checkpoint。

    Returns:
        params : PyTree
                恢复的参数;若未找到 checkpoint 或 fresh 为 True 则为 None。
        save_model : Callable
                将给定 PyTree 保存的函数 (PyTree->None)
        hparams : dict
                将给定超参以 json 形式与模型参数一起存储
    """
    path = os.path.abspath(path)
    hparams_file_path = os.path.join(path, "hparams.json")

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    orbax_path = os.path.join(path, "ckpt")

    def save_model(_params):
        return checkpointer.save(orbax_path, _params, force=True)

    restored_params = None
    restored_hparams = "{}"
    print(path, end=": ")
    exists = os.path.exists(path)
    if not exists:
        print("No checkpoint found")
    else:
        if fresh:
            print("Overwriting existing checkpoint")
        else:
            restored_params = checkpointer.restore(orbax_path)
            print("Restored model from checkpoint")
            if os.path.exists(hparams_file_path):
                with open(hparams_file_path) as f:
                    restored_hparams = json.load(f)

    if (not exists or fresh) and hparams is not None:
        os.makedirs(path, exist_ok=True)
        with open(hparams_file_path, "w") as f:
            json.dump(hparams, f)

    return (restored_params, restored_hparams), save_model


def mse_loss(y_hat, y):
    """均方误差。"""
    return jnp.mean((y - y_hat) ** 2)


def bce_loss(y_hat, y):
    """二元交叉熵。"""
    return optax.sigmoid_binary_cross_entropy(y_hat, y)


def mae_loss(y_hat, y):
    """平均绝对误差。"""
    return jnp.mean(jnp.abs(y - y_hat))


def make_vmap_model(_model, **kwargs):
    @jax.jit
    def _vmap_model(_p, _input):
        return jax.vmap(jax.tree_util.Partial(_model.apply, _p, **kwargs))(_input)

    return _vmap_model


def make_validate(_model, test_data, **kwargs):
    vmap_model = make_vmap_model(_model, **kwargs)

    @jax.jit
    def _validate(_p):
        y_hat, _ = vmap_model(_p, test_data)
        return mse_loss(y_hat, test_data)

    return _validate
