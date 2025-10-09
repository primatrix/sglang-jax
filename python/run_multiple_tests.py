"""
运行多个 flash attention 测试
"""

import sys
import unittest

from sgl_jax.test.test_flashattention import TestAttention

if __name__ == "__main__":
    # 创建测试套件
    suite = unittest.TestSuite()

    # 运行多个不同的测试
    test_names = [
        "test_mha_decode_accuracy_page_size_1",
        "test_mha_decode_accuracy_page_size_8",
        "test_mha_prefill_accuracy_page_size_1",
        "test_gqa_decode_accuracy_page_size_64",
    ]

    for test_name in test_names:
        suite.addTest(TestAttention(test_name))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 打印总结
    print(f"\n{'='*70}")
    print(f"测试总结:")
    print(f"  运行: {result.testsRun}")
    print(f"  成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  失败: {len(result.failures)}")
    print(f"  错误: {len(result.errors)}")
    print(f"{'='*70}")

    # 返回测试结果
    sys.exit(0 if result.wasSuccessful() else 1)
