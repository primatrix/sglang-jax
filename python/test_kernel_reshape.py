import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def reshape_kernel(kv_ref, o_ref):
    reshaped_ref = kv_ref.reshape(8, 128)
    o_ref[...] = reshaped_ref[...]


if __name__ == "__main__":
    tmp_kv = jnp.array(
        [
            jnp.arange(256) + 0,  # 第0行: 0, 1, 2, ..., 255
            jnp.arange(256) + 1000,  # 第1行: 1000, 1001, ..., 1255
            jnp.arange(256) + 2000,  # 第2行: 2000, 2001, ..., 2255
            jnp.arange(256) + 3000,  # 第3行: 3000, 3001, ..., 3255
        ],
        dtype=jnp.float32,
    )

    print("\n原始数据 tmp_kv 形状:", tmp_kv.shape)
    print("tmp_kv[0, :10] =", tmp_kv[0, :10])  # 第0行前10个
    print("tmp_kv[0, 128:138] =", tmp_kv[0, 128:138])  # 第0行后半段前10个
    print("tmp_kv[1, :10] =", tmp_kv[1, :10])  # 第1行前10个

    strided_loaded = pl.pallas_call(
        reshape_kernel, out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32)
    )(tmp_kv)

    print("\nReshape 后的数据形状:", strided_loaded.shape)
    print("strided_loaded[0, :10] =", strided_loaded[0, :10])  # 新第0行前10个
    print("strided_loaded[0, -10:] =", strided_loaded[0, -10:])  # 新第0行后10个
    print("strided_loaded[1, :10] =", strided_loaded[1, :10])  # 新第1行前10个
    print("strided_loaded[1, -10:] =", strided_loaded[1, -10:])  # 新第1行后10个
    print("strided_loaded[2, :10] =", strided_loaded[2, :10])  # 新第2行前10个
    print("strided_loaded[3, :10] =", strided_loaded[3, :10])  # 新第3行前10个
    print("strided_loaded[4, :10] =", strided_loaded[4, :10])  # 新第4行前10个
    print("strided_loaded[5, :10] =", strided_loaded[5, :10])  # 新第5行前10个

    print("\n对比分析:")
    print("标准 reshape 预期: (4, 256) -> (8, 128)")
    print("  新行0 应该是: 原始行0的 [0:128]")
    print("  新行1 应该是: 原始行0的 [128:256]")
    print("  新行2 应该是: 原始行1的 [0:128]")
    print("\n实际 TPU kernel reshape:")
    print("  新行0 实际是:", strided_loaded[0, 0], "...", strided_loaded[0, -1])
    print("  新行1 实际是:", strided_loaded[1, 0], "...", strided_loaded[1, -1])
    print("  新行2 实际是:", strided_loaded[2, 0], "...", strided_loaded[2, -1])

    # 使用普通 numpy reshape 对比
    print("\n普通 JAX reshape 对比:")
    normal_reshape = tmp_kv.reshape(8, 128)
    print("normal_reshape[0, :10] =", normal_reshape[0, :10])
    print("normal_reshape[1, :10] =", normal_reshape[1, :10])
    print("normal_reshape[2, :10] =", normal_reshape[2, :10])

    print("\n" + "=" * 80)
    print("TPU Pallas kernel reference reshape 规律分析:")
    print("=" * 80)
    print("原始形状: (4, 256)")
    print("目标形状: (8, 128)")
    print("\n实际映射关系 (TPU kernel):")
    print("  新行0 ← 原始 [0, 0:128]")
    print("  新行1 ← 原始 [1, 0:128]")
    print("  新行2 ← 原始 [2, 0:128]")
    print("  新行3 ← 原始 [3, 0:128]")
    print("  新行4 ← 原始 [0, 128:256]")
    print("  新行5 ← 原始 [1, 128:256]")
    print("  新行6 ← 原始 [2, 128:256]")
    print("  新行7 ← 原始 [3, 128:256]")
    print("\n这相当于: tmp_kv.reshape(4, 2, 128).transpose(1, 0, 2).reshape(8, 128)")

    # 验证这个规律
    equivalent_reshape = tmp_kv.reshape(4, 2, 128).transpose(1, 0, 2).reshape(8, 128)
    print("\n验证等价变换:")
    print("equivalent[0, :10] =", equivalent_reshape[0, :10])
    print("equivalent[1, :10] =", equivalent_reshape[1, :10])
    print("equivalent[4, :10] =", equivalent_reshape[4, :10])
    print("\n与 kernel 结果匹配:", jnp.allclose(strided_loaded, equivalent_reshape))
