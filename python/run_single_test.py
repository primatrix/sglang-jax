"""
运行单个 flash attention 测试
"""

import sys
import unittest

from sgl_jax.test.test_flashattention import TestAttention

if __name__ == "__main__":
    # 创建测试套件
    suite = unittest.TestSuite()

    # 只运行一个简单的测试
    suite.addTest(TestAttention("test_mha_decode_accuracy_page_size_1"))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 返回测试结果
    sys.exit(0 if result.wasSuccessful() else 1)
