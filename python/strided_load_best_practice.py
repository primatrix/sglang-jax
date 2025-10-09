"""
Pallas Kernel Strided Load 最佳实践

总结三种主要方案及其适用场景
"""

import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl


# ============================================================================
# 方案1: 直接 2D 切片（推荐用于行优先语义）
# ============================================================================
def strided_load_direct_2d(kv_ref, o_ref, *, start, step, chunk_size=128):
    """
    最简单高效的方案：不需要 reshape，直接在原始 2D 数据上做 strided access

    优点：
    - ✅ 不需要 reshape
    - ✅ 代码清晰易懂
    - ✅ 行优先语义（标准 NumPy/JAX 行为）
    - ✅ 无额外数据复制

    适用场景：
    - 原始数据是 2D，需要行优先的 strided load
    - 不需要后续在 (*,128) 形状上做其他操作
    """
    r, l = kv_ref.shape
    folds = l // chunk_size
    output_rows = (r * folds - start + step - 1) // step

    for out_idx in range(output_rows):
        idx_1d = start + out_idx * step
        src_row = idx_1d // folds
        src_col_start = (idx_1d % folds) * chunk_size
        o_ref[out_idx, :] = kv_ref[src_row, src_col_start : src_col_start + chunk_size]


# ============================================================================
# 方案2: 列优先 reshape + strided load（flash_attention.py 的方式）
# ============================================================================
def strided_load_col_major_reshape(kv_ref, o_ref, *, start, step):
    """
    使用 reference reshape（列优先）+ 直接 strided indexing

    优点：
    - ✅ Zero-cost reshape（仅内存视图变换）
    - ✅ 代码简洁

    缺点：
    - ❌ 列优先语义（不同于标准 NumPy/JAX）

    适用场景：
    - 数据本身是交错排列的（如 Flash Attention 的 KV cache）
    - 列优先语义正好符合你的需求
    """
    r, l = kv_ref.shape
    folds = l // 128
    reshaped = kv_ref.reshape(r * folds, 128)
    o_ref[...] = reshaped[start::step]


# ============================================================================
# 方案3: 列优先 reshape + 索引映射（兼顾两种语义）
# ============================================================================
def strided_load_with_index_mapping(kv_ref, o_ref, *, start, step, chunk_size=128):
    """
    先做列优先 reshape，然后通过索引映射实现行优先语义

    优点：
    - ✅ Zero-cost reshape
    - ✅ 行优先语义
    - ✅ reshape 后的数据可用于其他操作

    缺点：
    - ❌ 需要索引转换计算

    适用场景：
    - 需要 reshape 后的 (*,128) 形状用于后续操作
    - 同时想要行优先的 strided load 语义
    """
    r, l = kv_ref.shape
    folds = l // chunk_size

    # 列优先 reshape
    reshaped = kv_ref.reshape(r * folds, chunk_size)
    output_rows = (r * folds - start + step - 1) // step

    for out_idx in range(output_rows):
        row_major_idx = start + out_idx * step

        # 索引转换：行优先 -> 列优先
        src_i = row_major_idx // folds
        src_j = row_major_idx % folds
        col_major_idx = src_j * r + src_i

        if col_major_idx < r * folds:
            o_ref[out_idx, :] = reshaped[col_major_idx, :]


# ============================================================================
# 示例和性能对比
# ============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("Pallas Kernel Strided Load 最佳实践")
    print("=" * 80)

    # 测试数据
    data = jnp.array(
        [
            jnp.arange(256) + 0,
            jnp.arange(256) + 1000,
            jnp.arange(256) + 2000,
            jnp.arange(256) + 3000,
        ],
        dtype=jnp.float32,
    )

    start, step = 1, 2
    output_size = (8 - start + step - 1) // step

    print("\n测试数据: (4, 256)")
    print(f"Strided load 参数: start={start}, step={step}")
    print(f"预期输出: {output_size} 行")

    # 标准参考
    expected = data.reshape(8, 128)[start::step]
    print(f"\n标准行优先结果:")
    print(f"  第0行: [{expected[0,0]:.0f}...{expected[0,-1]:.0f}]")
    print(f"  第1行: [{expected[1,0]:.0f}...{expected[1,-1]:.0f}]")

    # 方案1: 直接 2D 切片
    print("\n" + "=" * 80)
    print("方案1: 直接 2D 切片（推荐）")
    print("=" * 80)
    result1 = pl.pallas_call(
        lambda kv, o: strided_load_direct_2d(kv, o, start=start, step=step),
        out_shape=jax.ShapeDtypeStruct((output_size, 128), jnp.float32),
    )(data)
    print(f"  结果匹配: {jnp.allclose(result1, expected)}")
    print(f"  第0行: [{result1[0,0]:.0f}...{result1[0,-1]:.0f}]")
    print("\n  💡 推荐场景: 默认选择，简单直接")

    # 方案2: 列优先 reshape
    print("\n" + "=" * 80)
    print("方案2: 列优先 reshape（flash_attention.py 方式）")
    print("=" * 80)
    result2 = pl.pallas_call(
        lambda kv, o: strided_load_col_major_reshape(kv, o, start=start, step=step),
        out_shape=jax.ShapeDtypeStruct((output_size, 128), jnp.float32),
    )(data)
    print(f"  结果匹配: {jnp.allclose(result2, expected)}")
    print(f"  第0行: [{result2[0,0]:.0f}...{result2[0,-1]:.0f}]")
    if not jnp.allclose(result2, expected):
        print("\n  ⚠️  注意: 这是列优先语义，与标准行优先不同")
        print("  💡 推荐场景: 数据本身是交错排列（如 KV cache）")

    # 方案3: 索引映射
    print("\n" + "=" * 80)
    print("方案3: 列优先 reshape + 索引映射")
    print("=" * 80)
    result3 = pl.pallas_call(
        lambda kv, o: strided_load_with_index_mapping(kv, o, start=start, step=step),
        out_shape=jax.ShapeDtypeStruct((output_size, 128), jnp.float32),
    )(data)
    print(f"  结果匹配: {jnp.allclose(result3, expected)}")
    print(f"  第0行: [{result3[0,0]:.0f}...{result3[0,-1]:.0f}]")
    print("\n  💡 推荐场景: 需要 reshape 后的数据做其他操作")

    # 总结
    print("\n" + "=" * 80)
    print("选择建议")
    print("=" * 80)
    print(
        """
┌─────────────────────────────────────────────────────────────────┐
│ 需求                                    推荐方案                │
├─────────────────────────────────────────────────────────────────┤
│ 默认选择                               方案1（直接 2D 切片）     │
│ 行优先 + 简单                          方案1（直接 2D 切片）     │
│ 列优先 + 交错数据                      方案2（列优先 reshape）   │
│ 需要 (*,128) 形状 + 行优先语义          方案3（索引映射）        │
│ Flash Attention 式的 KV cache          方案2（列优先 reshape）   │
└─────────────────────────────────────────────────────────────────┘

核心要点：
1. 方案1 最简单高效，适合大部分场景
2. 方案2 用于 zero-cost reshape，但注意列优先语义
3. 方案3 用于需要 reshape 后形状的高级场景
    """
    )
