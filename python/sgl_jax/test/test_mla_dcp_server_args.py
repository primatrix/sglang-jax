import unittest

from sgl_jax.srt.server_args import ServerArgs


class TestMLADCPServerArgs(unittest.TestCase):
    def test_supported_configuration(self):
        args = ServerArgs(
            model_path="dummy",
            tp_size=8,
            decode_context_parallel_size=2,
            page_size=128,
            attention_backend="fa",
            moe_backend="epmoe",
        )
        args.check_server_args()

    def test_fused_moe_is_rejected_before_model_initialization(self):
        args = ServerArgs(
            model_path="dummy",
            tp_size=8,
            decode_context_parallel_size=2,
            page_size=128,
            attention_backend="fa",
            moe_backend="fused",
        )
        with self.assertRaisesRegex(ValueError, "requires --moe-backend epmoe"):
            args.check_server_args()

    def test_page_size_must_be_divisible_by_dcp(self):
        args = ServerArgs(
            model_path="dummy",
            tp_size=8,
            decode_context_parallel_size=4,
            page_size=130,
            attention_backend="fa",
            moe_backend="epmoe",
        )
        with self.assertRaisesRegex(ValueError, "page_size divisible by dcp_size"):
            args.check_server_args()


if __name__ == "__main__":
    unittest.main()
