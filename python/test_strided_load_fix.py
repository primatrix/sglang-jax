"""
验证 flash_attention.py 中 strided_load 的修复
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def strided_load_original(ref, o_ref, *, start, step):
    """原始的（有 bug 的）版本"""
    r, l = ref.shape
    folds = l // 128
    reshaped = ref.reshape(r * folds, 128)  # 列优先 reshape
    step_adj = step * folds

    # 错误: 使用 i * 2
    vec = jnp.concat(
        [reshaped[start + i * 2 :: step_adj] for i in range(folds)], axis=1
    )
    o_ref[...] = vec


def strided_load_fixed(ref, o_ref, *, start, step):
    """修复后的版本"""
    r, l = ref.shape
    folds = l // 128
    reshaped = ref.reshape(r * folds, 128)  # 列优先 reshape
    step_adj = step * folds

    # 修复: 使用 i * r
    vec = jnp.concat(
        [reshaped[start + i * r :: step_adj] for i in range(folds)], axis=1
    )
    o_ref[...] = vec


def strided_load_direct(ref, o_ref, *, start, step):
    """参考实现：直接在原始 2D 数据上操作，拼接 K 和 V"""
    r, l = ref.shape
    folds = l // 128
    step_adj = step * folds

    # 直接访问原始数据，拼接所有 folds
    output_rows = (r * folds - start + step_adj - 1) // step_adj
    for out_idx in range(output_rows):
        idx_1d = start + out_idx * step_adj
        src_i = idx_1d // folds
        # 拼接所有 folds
        parts = []
        for fold_idx in range(folds):
            parts.append(ref[src_i, fold_idx * 128 : (fold_idx + 1) * 128])
        o_ref[out_idx, :] = jnp.concatenate(parts)


if __name__ == "__main__":
    print("=" * 80)
    print("验证 Flash Attention strided_load 修复")
    print("=" * 80)

    # 创建测试数据
    # 模拟 KV cache 数据：交错的 K 和 V
    r = 4  # 比如有 4 行数据
    head_dim = 256

    data = jnp.array(
        [
            jnp.concatenate(
                [jnp.arange(128) + 0, jnp.arange(128) + 1000]
            ),  # 行0: K=[0...127], V=[1000...1127]
            jnp.concatenate(
                [jnp.arange(128) + 200, jnp.arange(128) + 1200]
            ),  # 行1: K=[200...327], V=[1200...1327]
            jnp.concatenate(
                [jnp.arange(128) + 400, jnp.arange(128) + 1400]
            ),  # 行2: K=[400...527], V=[1400...1527]
            jnp.concatenate(
                [jnp.arange(128) + 600, jnp.arange(128) + 1600]
            ),  # 行3: K=[600...727], V=[1600...1727]
        ],
        dtype=jnp.float32,
    )

    print(f"\n原始数据形状: {data.shape}")
    print("数据布局（模拟交错的 KV）:")
    print(f"  行0: K=[0...127], V=[1000...1127]")
    print(f"  行1: K=[200...327], V=[1200...1327]")
    print(f"  行2: K=[400...527], V=[1400...1527]")
    print(f"  行3: K=[600...727], V=[1600...1727]")

    # strided load 参数
    start = 0
    step = 2
    folds = head_dim // 128
    step_adj = step * folds
    expected_rows = (r * folds - start + step_adj - 1) // step_adj

    print(f"\nStrided load 参数:")
    print(f"  start={start}, step={step}, step_adj={step_adj}")
    print(f"  预期输出行数: {expected_rows}")

    # 参考实现
    print("\n" + "=" * 80)
    print("参考实现（直接 2D 访问）")
    print("=" * 80)
    ref_result = pl.pallas_call(
        lambda x, o: strided_load_direct(x, o, start=start, step=step),
        out_shape=jax.ShapeDtypeStruct((expected_rows, head_dim), jnp.float32),
    )(data)
    print(f"输出形状: {ref_result.shape}")
    print(
        f"第0行: K=[{ref_result[0,0]:.0f}...{ref_result[0,127]:.0f}], V=[{ref_result[0,128]:.0f}...{ref_result[0,255]:.0f}]"
    )
    if expected_rows > 1:
        print(
            f"第1行: K=[{ref_result[1,0]:.0f}...{ref_result[1,127]:.0f}], V=[{ref_result[1,128]:.0f}...{ref_result[1,255]:.0f}]"
        )

    # 原始版本（有bug）
    print("\n" + "=" * 80)
    print("原始版本（有 bug：使用 i*2）")
    print("=" * 80)
    try:
        orig_result = pl.pallas_call(
            lambda x, o: strided_load_original(x, o, start=start, step=step),
            out_shape=jax.ShapeDtypeStruct((expected_rows, head_dim), jnp.float32),
        )(data)
        print(f"输出形状: {orig_result.shape}")
        print(
            f"第0行: K=[{orig_result[0,0]:.0f}...{orig_result[0,127]:.0f}], V=[{orig_result[0,128]:.0f}...{orig_result[0,255]:.0f}]"
        )
        if expected_rows > 1:
            print(
                f"第1行: K=[{orig_result[1,0]:.0f}...{orig_result[1,127]:.0f}], V=[{orig_result[1,128]:.0f}...{orig_result[1,255]:.0f}]"
            )
        print(f"匹配参考结果: {jnp.allclose(orig_result, ref_result)}")
    except Exception as e:
        print(f"错误: {e}")

    # 修复版本
    print("\n" + "=" * 80)
    print("修复版本（使用 i*r）")
    print("=" * 80)
    fixed_result = pl.pallas_call(
        lambda x, o: strided_load_fixed(x, o, start=start, step=step),
        out_shape=jax.ShapeDtypeStruct((expected_rows, head_dim), jnp.float32),
    )(data)
    print(f"输出形状: {fixed_result.shape}")
    print(
        f"第0行: K=[{fixed_result[0,0]:.0f}...{fixed_result[0,127]:.0f}], V=[{fixed_result[0,128]:.0f}...{fixed_result[0,255]:.0f}]"
    )
    if expected_rows > 1:
        print(
            f"第1行: K=[{fixed_result[1,0]:.0f}...{fixed_result[1,127]:.0f}], V=[{fixed_result[1,128]:.0f}...{fixed_result[1,255]:.0f}]"
        )
    print(f"匹配参考结果: {jnp.allclose(fixed_result, ref_result)}")

    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    if jnp.allclose(fixed_result, ref_result):
        print("✅ 修复成功！修复后的版本与参考实现匹配")
    else:
        print("❌ 修复失败！结果仍然不匹配")
        print("\n差异:")
        diff = jnp.abs(fixed_result - ref_result)
        print(f"最大差异: {jnp.max(diff)}")
        print(f"平均差异: {jnp.mean(diff)}")
