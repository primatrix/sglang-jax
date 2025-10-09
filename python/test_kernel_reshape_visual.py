import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl


def reshape_row_major(kv_ref, o_ref):
    """行优先 reshape：标准方式"""
    r, l = kv_ref.shape
    folds = l // 128
    for i in range(r):
        for j in range(folds):
            o_ref[i * folds + j, :] = kv_ref[i, j * 128 : (j + 1) * 128]


def reshape_col_major(kv_ref, o_ref):
    """列优先 reshape：TPU reference 方式"""
    reshaped_ref = kv_ref.reshape(8, 128)
    o_ref[...] = reshaped_ref[...]


def visualize_reshape(data, name):
    """可视化数据分布"""
    print(f"\n{name}:")
    print("-" * 60)
    for i in range(8):
        start_val = data[i, 0]
        end_val = data[i, -1]
        print(f"  行{i}: [{start_val:6.0f} ... {end_val:6.0f}]", end="")

        # 显示这是从原始数据的哪里来的
        if 0 <= start_val < 256:
            src_row = 0
            src_col_start = int(start_val)
        elif 1000 <= start_val < 1256:
            src_row = 1
            src_col_start = int(start_val - 1000)
        elif 2000 <= start_val < 2256:
            src_row = 2
            src_col_start = int(start_val - 2000)
        elif 3000 <= start_val < 3256:
            src_row = 3
            src_col_start = int(start_val - 3000)
        else:
            src_row = -1
            src_col_start = -1

        if src_row >= 0:
            src_col_end = src_col_start + 127
            print(f"  ← 原始[{src_row}, {src_col_start}:{src_col_end+1}]")


if __name__ == "__main__":
    print("=" * 80)
    print("Pallas Kernel Reshape 数据布局可视化")
    print("=" * 80)

    # 创建测试数据
    tmp_kv = jnp.array(
        [
            jnp.arange(256) + 0,  # 行0: [0 ... 255]
            jnp.arange(256) + 1000,  # 行1: [1000 ... 1255]
            jnp.arange(256) + 2000,  # 行2: [2000 ... 2255]
            jnp.arange(256) + 3000,  # 行3: [3000 ... 3255]
        ],
        dtype=jnp.float32,
    )

    print("\n原始数据形状: (4, 256)")
    print("-" * 60)
    print("  行0: [    0 ...   255]")
    print("  行1: [ 1000 ...  1255]")
    print("  行2: [ 2000 ...  2255]")
    print("  行3: [ 3000 ...  3255]")

    # 行优先 reshape
    result_row_major = pl.pallas_call(
        reshape_row_major, out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32)
    )(tmp_kv)

    visualize_reshape(result_row_major, "行优先 Reshape (4, 256) → (8, 128)")

    # 列优先 reshape
    result_col_major = pl.pallas_call(
        reshape_col_major, out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32)
    )(tmp_kv)

    visualize_reshape(
        result_col_major, "列优先 Reshape (4, 256) → (8, 128) - TPU Reference"
    )

    print("\n" + "=" * 80)
    print("关键区别：")
    print("=" * 80)
    print()
    print("【行优先】- 标准 NumPy/JAX 行为")
    print("  沿着最后一维连续展开，类似 C 语言数组")
    print("  适用场景: 需要保持内存连续性的标准数据处理")
    print()
    print("【列优先】- TPU Pallas Reference Reshape")
    print("  优先处理第一维，类似 Fortran 数组")
    print("  适用场景: 处理交错数据（如 Flash Attention 的 KV cache）")
    print()
    print("实现方法：")
    print("  行优先: for i,j: o_ref[i*folds+j,:] = kv_ref[i, j*128:(j+1)*128]")
    print("  列优先: o_ref[...] = kv_ref.reshape(new_shape)[...]")
    print()
    print("性能考虑：")
    print("  列优先（reference reshape）是 zero-cost 的视图变换")
    print("  行优先（手动循环）涉及实际数据复制，但通常也很快")
