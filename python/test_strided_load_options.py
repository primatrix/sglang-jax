import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def strided_load_after_col_major_reshape(kv_ref, o_ref, *, start, step):
    """方案1：列优先 reshape + strided load（flash_attention.py 方式）"""
    r, l = kv_ref.shape  # 4, 256
    folds = l // 128  # 2

    # 列优先 reshape: (4, 256) -> (8, 128)
    reshaped = kv_ref.reshape(r * folds, 128)

    # Strided load
    o_ref[...] = reshaped[start::step]


def strided_load_after_row_major_reshape(kv_ref, o_ref, *, start, step):
    """方案2：先行优先 reshape 到 VMEM，再 strided load"""
    r, l = kv_ref.shape  # 4, 256
    folds = l // 128  # 2

    # 行优先 reshape 到临时 buffer（涉及数据复制）
    temp = jnp.zeros((r * folds, 128), dtype=kv_ref.dtype)
    for i in range(r):
        for j in range(folds):
            temp = temp.at[i * folds + j, :].set(kv_ref[i, j * 128 : (j + 1) * 128])

    # Strided load
    o_ref[...] = temp[start::step]


def strided_load_without_reshape(kv_ref, o_ref, *, start, step):
    """方案3：不 reshape，直接计算 2D strided access"""
    r, l = kv_ref.shape  # 4, 256
    folds = l // 128  # 2

    # 将 1D stride 映射回 2D 索引（行优先）
    # 对于行优先：index_1d = i * folds + j
    # 反推：i = index_1d // folds, j = index_1d % folds

    output_rows = (r * folds - start + step - 1) // step

    for out_idx in range(output_rows):
        src_idx_1d = start + out_idx * step
        if src_idx_1d < r * folds:
            src_i = src_idx_1d // folds
            src_j = src_idx_1d % folds
            o_ref[out_idx, :] = kv_ref[src_i, src_j * 128 : (src_j + 1) * 128]


def strided_load_col_major_corrected(kv_ref, o_ref, *, start, step):
    """方案4：列优先 reshape，但调整 stride 逻辑匹配行优先语义"""
    r, l = kv_ref.shape  # 4, 256
    folds = l // 128  # 2

    # 列优先 reshape: (4, 256) -> (8, 128)
    # 映射：原始[i, j*128:(j+1)*128] -> reshaped[j*r + i, :]

    # 如果想要行优先的 stride 语义：从位置 start 开始，每隔 step 取一个
    # 需要将行优先的索引转换为列优先布局中的索引

    reshaped = kv_ref.reshape(r * folds, 128)
    output_rows = (r * folds - start + step - 1) // step

    for out_idx in range(output_rows):
        # 行优先索引
        row_major_idx = start + out_idx * step

        # 转换为列优先索引
        # 行优先: [0,1,2,3,4,5,6,7] -> [行0前, 行0后, 行1前, 行1后, ...]
        # 列优先: [0,1,2,3,4,5,6,7] -> [行0前, 行1前, 行2前, 行3前, 行0后, ...]
        # 映射关系: row_major[i*folds+j] = col_major[j*r+i]
        # 所以: col_major_idx 需要找到满足 j*r+i = target 的位置

        src_i = row_major_idx // folds
        src_j = row_major_idx % folds
        col_major_idx = src_j * r + src_i

        if col_major_idx < r * folds:
            o_ref[out_idx, :] = reshaped[col_major_idx, :]


def strided_load_direct_slice(kv_ref, o_ref, *, start, step):
    """方案5：直接用 2D 切片（最简单）"""
    r, l = kv_ref.shape  # 4, 256
    folds = l // 128  # 2

    # 直接访问原始数据，避免 reshape
    output_rows = (r * folds - start + step - 1) // step

    for out_idx in range(output_rows):
        idx_1d = start + out_idx * step
        src_row = idx_1d // folds
        src_col_start = (idx_1d % folds) * 128
        o_ref[out_idx, :] = kv_ref[src_row, src_col_start : src_col_start + 128]


if __name__ == "__main__":
    print("=" * 80)
    print("Strided Load 方案对比")
    print("=" * 80)

    # 创建测试数据
    tmp_kv = jnp.array(
        [
            jnp.arange(256) + 0,
            jnp.arange(256) + 1000,
            jnp.arange(256) + 2000,
            jnp.arange(256) + 3000,
        ],
        dtype=jnp.float32,
    )

    print("\n原始数据 (4, 256):")
    print("  行0: [0...255]")
    print("  行1: [1000...1255]")
    print("  行2: [2000...2255]")
    print("  行3: [3000...3255]")

    # 参数：从索引 1 开始，步长 2
    start, step = 1, 2
    expected_output_size = (8 - start + step - 1) // step  # 4 rows

    print(f"\nStrided Load 参数: start={start}, step={step}")
    print(f"期望输出: {expected_output_size} 行")

    # 行优先 reshape 后的预期结果
    row_major = tmp_kv.reshape(8, 128)
    expected = row_major[start::step]
    print(f"\n标准行优先结果（参考）:")
    print(
        f"  行0: [{expected[0,0]:.0f}...{expected[0,-1]:.0f}] <- 原始 reshape 后的行{start}"
    )
    print(
        f"  行1: [{expected[1,0]:.0f}...{expected[1,-1]:.0f}] <- 原始 reshape 后的行{start+step}"
    )

    # 方案1：列优先 reshape + strided load
    print("\n方案1: 列优先 reshape + strided load (flash_attention.py 方式)")
    print("-" * 60)
    result1 = pl.pallas_call(
        lambda kv, o: strided_load_after_col_major_reshape(
            kv, o, start=start, step=step
        ),
        out_shape=jax.ShapeDtypeStruct((expected_output_size, 128), jnp.float32),
    )(tmp_kv)
    print(f"  行0: [{result1[0,0]:.0f}...{result1[0,-1]:.0f}]")
    print(f"  行1: [{result1[1,0]:.0f}...{result1[1,-1]:.0f}]")
    print(f"  匹配预期: {jnp.allclose(result1, expected)}")

    # 方案3：不 reshape，直接计算 2D strided access
    print("\n方案3: 不 reshape，直接 2D strided access")
    print("-" * 60)
    result3 = pl.pallas_call(
        lambda kv, o: strided_load_without_reshape(kv, o, start=start, step=step),
        out_shape=jax.ShapeDtypeStruct((expected_output_size, 128), jnp.float32),
    )(tmp_kv)
    print(f"  行0: [{result3[0,0]:.0f}...{result3[0,-1]:.0f}]")
    print(f"  行1: [{result3[1,0]:.0f}...{result3[1,-1]:.0f}]")
    print(f"  匹配预期: {jnp.allclose(result3, expected)}")

    # 方案4：列优先 + 索引转换
    print("\n方案4: 列优先 reshape + 索引转换")
    print("-" * 60)
    result4 = pl.pallas_call(
        lambda kv, o: strided_load_col_major_corrected(kv, o, start=start, step=step),
        out_shape=jax.ShapeDtypeStruct((expected_output_size, 128), jnp.float32),
    )(tmp_kv)
    print(f"  行0: [{result4[0,0]:.0f}...{result4[0,-1]:.0f}]")
    print(f"  行1: [{result4[1,0]:.0f}...{result4[1,-1]:.0f}]")
    print(f"  匹配预期: {jnp.allclose(result4, expected)}")

    # 方案5：直接切片
    print("\n方案5: 直接 2D 切片（推荐）")
    print("-" * 60)
    result5 = pl.pallas_call(
        lambda kv, o: strided_load_direct_slice(kv, o, start=start, step=step),
        out_shape=jax.ShapeDtypeStruct((expected_output_size, 128), jnp.float32),
    )(tmp_kv)
    print(f"  行0: [{result5[0,0]:.0f}...{result5[0,-1]:.0f}]")
    print(f"  行1: [{result5[1,0]:.0f}...{result5[1,-1]:.0f}]")
    print(f"  匹配预期: {jnp.allclose(result5, expected)}")

    print("\n" + "=" * 80)
    print("推荐方案：")
    print("=" * 80)
    print()
    print("💡 方案5（直接 2D 切片）- 最佳选择")
    print("   ✅ 不需要 reshape")
    print("   ✅ 代码简洁清晰")
    print("   ✅ 避免数据复制")
    print("   ✅ 直接在原始数据上做 strided access")
    print()
    print("代码示例：")
    print("   idx_1d = start + out_idx * step")
    print("   src_row = idx_1d // folds")
    print("   src_col_start = (idx_1d % folds) * 128")
    print("   output[out_idx] = kv_ref[src_row, src_col_start:src_col_start+128]")
    print()
    print("如果你的场景需要 (*,128) 形状用于后续操作：")
    print("   可以用方案4（列优先 + 索引转换）来获得行优先的 stride 语义")
