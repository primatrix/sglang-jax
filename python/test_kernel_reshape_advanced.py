import time

import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def reshape_row_major_loop(kv_ref, o_ref):
    """方法A：先加载，再循环（row-major）"""
    kv = kv_ref[...]
    r, l = kv.shape
    folds = l // 128

    for i in range(r):
        for j in range(folds):
            o_ref[i * folds + j, :] = kv[i, j * 128 : (j + 1) * 128]


def reshape_row_major_slice(kv_ref, o_ref):
    """方法B：直接切片复制（row-major）"""
    r, l = kv_ref.shape
    folds = l // 128

    for i in range(r):
        for j in range(folds):
            o_ref[i * folds + j, :] = kv_ref[i, j * 128 : (j + 1) * 128]


def reshape_col_major_ref(kv_ref, o_ref):
    """方法C：reference reshape（col-major）"""
    reshaped_ref = kv_ref.reshape(8, 128)
    o_ref[...] = reshaped_ref[...]


def reshape_with_bitcast(kv_ref, o_ref):
    """方法D：组合 bitcast + reshape（模拟 flash_attention.py 的方式）"""
    # 先 bitcast 到 uint32，再 reshape
    r, l = kv_ref.shape
    folds = l // 128

    # 尝试使用 bitcast
    ref_u32 = kv_ref.bitcast(jnp.uint32)
    reshaped = ref_u32.reshape(r * folds, 128)
    o_ref[...] = reshaped.bitcast(jnp.float32)[...]


if __name__ == "__main__":
    print("=" * 80)
    print("Pallas Kernel Reshape 方法对比测试")
    print("=" * 80)

    # 测试数据
    tmp_kv = jnp.array(
        [
            jnp.arange(256) + 0,
            jnp.arange(256) + 1000,
            jnp.arange(256) + 2000,
            jnp.arange(256) + 3000,
        ],
        dtype=jnp.float32,
    )

    expected_row_major = tmp_kv.reshape(8, 128)
    expected_col_major = tmp_kv.reshape(4, 2, 128).transpose(1, 0, 2).reshape(8, 128)

    print("\n1. 标准参考结果")
    print("-" * 40)
    print("行优先 reshape[1, :5] =", expected_row_major[1, :5])
    print("列优先 reshape[1, :5] =", expected_col_major[1, :5])

    # 方法A：加载后循环
    print("\n2. 方法A: 先加载数据，再循环 (row-major)")
    print("-" * 40)
    result_a = pl.pallas_call(
        reshape_row_major_loop, out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32)
    )(tmp_kv)
    print("result[1, :5] =", result_a[1, :5])
    print("匹配行优先:", jnp.allclose(result_a, expected_row_major))

    # 方法B：直接切片
    print("\n3. 方法B: Reference 直接切片 (row-major)")
    print("-" * 40)
    result_b = pl.pallas_call(
        reshape_row_major_slice, out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32)
    )(tmp_kv)
    print("result[1, :5] =", result_b[1, :5])
    print("匹配行优先:", jnp.allclose(result_b, expected_row_major))

    # 方法C：reference reshape
    print("\n4. 方法C: Reference reshape (col-major)")
    print("-" * 40)
    result_c = pl.pallas_call(
        reshape_col_major_ref, out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32)
    )(tmp_kv)
    print("result[1, :5] =", result_c[1, :5])
    print("匹配列优先:", jnp.allclose(result_c, expected_col_major))

    # 方法D：bitcast + reshape
    print("\n5. 方法D: Bitcast + Reshape (col-major)")
    print("-" * 40)
    result_d = pl.pallas_call(
        reshape_with_bitcast, out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32)
    )(tmp_kv)
    print("result[1, :5] =", result_d[1, :5])
    print("匹配列优先:", jnp.allclose(result_d, expected_col_major))

    print("\n" + "=" * 80)
    print("总结：")
    print("=" * 80)
    print("✅ 行优先 reshape: 使用手动循环或 reference 切片")
    print("   - 方法A: kv[...] 然后循环复制")
    print("   - 方法B: 直接 kv_ref[i, j*128:(j+1)*128]")
    print()
    print("✅ 列优先 reshape: 使用 reference.reshape()")
    print("   - 方法C: kv_ref.reshape(...)")
    print("   - 方法D: kv_ref.bitcast().reshape().bitcast()")
    print()
    print("💡 选择建议:")
    print("   - 需要行优先（标准 reshape）→ 方法B（切片，更高效）")
    print("   - 需要列优先（交错数据）→ 方法C（zero-cost 视图变换）")
    print("   - Flash Attention 使用列优先是为了高效处理交错的 KV 数据")
