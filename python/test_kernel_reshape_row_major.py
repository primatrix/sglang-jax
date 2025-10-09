import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def reshape_kernel_ref_default(kv_ref, o_ref):
    """方法1：reference 直接 reshape（列优先行为）"""
    reshaped_ref = kv_ref.reshape(8, 128)
    o_ref[...] = reshaped_ref[...]


def reshape_kernel_manual_loop(kv_ref, o_ref):
    """方法2：手动循环实现行优先 reshape"""
    kv = kv_ref[...]
    r, l = kv.shape  # 4, 256
    folds = l // 128  # 2

    for i in range(r):
        for j in range(folds):
            o_ref[i * folds + j, :] = kv[i, j * 128 : (j + 1) * 128]


def reshape_kernel_ref_transpose(kv_ref, o_ref):
    """方法3：尝试在 reference 上组合 reshape + transpose"""
    # 尝试：reshape(4,2,128) -> transpose(1,0,2) -> reshape(8,128) 的逆操作
    # 如果 ref reshape 是列优先，我们需要先做逆变换
    # 但这可能不可行，因为我们不能在 ref 上做复杂操作
    reshaped_ref = kv_ref.reshape(4, 2, 128)
    # 注意：reference 可能不支持 transpose
    # transposed_ref = reshaped_ref.transpose(0, 1, 2)
    # o_ref[...] = transposed_ref.reshape(8, 128)[...]
    pass


def reshape_kernel_slice_copy(kv_ref, o_ref):
    """方法4：通过切片直接访问实现行优先"""
    # 不加载整个数据，而是直接通过 reference 切片
    r, l = kv_ref.shape  # 4, 256
    folds = l // 128  # 2

    for i in range(r):
        for j in range(folds):
            o_ref[i * folds + j, :] = kv_ref[i, j * 128 : (j + 1) * 128]


if __name__ == "__main__":
    # 创建测试数据
    tmp_kv = jnp.array(
        [
            jnp.arange(256) + 0,  # 第0行: 0, 1, 2, ..., 255
            jnp.arange(256) + 1000,  # 第1行: 1000, 1001, ..., 1255
            jnp.arange(256) + 2000,  # 第2行: 2000, 2001, ..., 2255
            jnp.arange(256) + 3000,  # 第3行: 3000, 3001, ..., 3255
        ],
        dtype=jnp.float32,
    )

    # 标准行优先 reshape 作为参考
    expected = tmp_kv.reshape(8, 128)
    print("期望的行优先 reshape:")
    print("expected[0, :5] =", expected[0, :5])
    print("expected[1, :5] =", expected[1, :5])
    print("expected[2, :5] =", expected[2, :5])
    print()

    # 方法1：reference 默认 reshape（列优先）
    print("方法1: Reference 默认 reshape (列优先)")
    result1 = pl.pallas_call(
        reshape_kernel_ref_default,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )(tmp_kv)
    print("result1[0, :5] =", result1[0, :5])
    print("result1[1, :5] =", result1[1, :5])
    print("匹配行优先:", jnp.allclose(result1, expected))
    print()

    # 方法2：手动循环
    print("方法2: 手动循环实现行优先")
    result2 = pl.pallas_call(
        reshape_kernel_manual_loop,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )(tmp_kv)
    print("result2[0, :5] =", result2[0, :5])
    print("result2[1, :5] =", result2[1, :5])
    print("匹配行优先:", jnp.allclose(result2, expected))
    print()

    # 方法4：切片复制
    print("方法4: Reference 切片直接复制")
    result4 = pl.pallas_call(
        reshape_kernel_slice_copy, out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32)
    )(tmp_kv)
    print("result4[0, :5] =", result4[0, :5])
    print("result4[1, :5] =", result4[1, :5])
    print("匹配行优先:", jnp.allclose(result4, expected))
    print()

    print("=" * 80)
    print("结论：")
    if jnp.allclose(result2, expected):
        print("✅ 方法2（手动循环）可以实现行优先 reshape")
    if jnp.allclose(result4, expected):
        print("✅ 方法4（切片复制）可以实现行优先 reshape")
    if not jnp.allclose(result1, expected):
        print("❌ 方法1（reference reshape）是列优先，不是行优先")
