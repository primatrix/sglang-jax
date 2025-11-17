import asyncio
import itertools
import unittest
from random import random, uniform

import requests

from sgl_jax.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    QWEN3_MOE_30B,
    CustomTestCase,
    is_in_ci,
    kill_process_tree,
    popen_launch_server,
    run_bench_serving,
    write_github_step_summary,
)


class TestBenchServing(CustomTestCase):
    # Class-level variables to store the server process and configuration
    _server_process = None
    _server_base_url = None
    _server_model = None
    _server_args = None

    @classmethod
    def setUpClass(cls):
        """Launch server once for all performance tests in this class."""
        # We'll launch different servers based on which tests are being run
        # For now, we'll detect based on the test methods being run
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        """Kill the server after all tests are done."""
        if cls._server_process is not None:
            kill_process_tree(cls._server_process.pid)
            cls._server_process = None
        super().tearDownClass()

    @classmethod
    def _launch_server_if_needed(cls, model, server_args):
        """Launch server if not already running with the same configuration."""
        if (
            cls._server_process is None
            or cls._server_model != model
            or cls._server_args != tuple(server_args)
        ):
            # Kill existing server if configuration changed
            if cls._server_process is not None:
                kill_process_tree(cls._server_process.pid)

            # Launch new server
            cls._server_model = model
            cls._server_args = tuple(server_args)
            cls._server_base_url = DEFAULT_URL_FOR_TEST
            cls._server_process = popen_launch_server(
                model,
                cls._server_base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=server_args,
            )

        return cls._server_process, cls._server_base_url

    def test_input_throughput_default(self):
        server_args = [
            "--trust-remote-code",
            "--skip-server-warmup",
            "--device",
            "tpu",
            "--random-seed",
            "3",
            "--chunked-prefill-size",
            "2048",
            "--download-dir",
            "/dev/shm/",
            "--dtype",
            "bfloat16",
            "--mem-fraction-static",
            "0.8",
            "--max-running-requests",
            "256",
            "--page-size",
            "128",
            "--disable-radix-cache",
        ]
        process, base_url = self._launch_server_if_needed(DEFAULT_MODEL_NAME_FOR_TEST, server_args)

        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=500,
            request_rate=float("inf"),
            other_server_args=server_args,
            random_input_len=1024,
            random_output_len=1,
            max_concurrency=256,
            random_range_ratio=1,
            base_url=base_url,
            external_process=process,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_input_throughput_default\n"
                f"Input throughput: {res['input_throughput']:.2f} token/s\n"
            )
            self.assertGreater(res["input_throughput"], 28299)

    def test_output_throughput_default(self):
        server_args = [
            "--trust-remote-code",
            "--skip-server-warmup",
            "--device",
            "tpu",
            "--random-seed",
            "3",
            "--chunked-prefill-size",
            "2048",
            "--download-dir",
            "/dev/shm/",
            "--dtype",
            "bfloat16",
            "--mem-fraction-static",
            "0.8",
            "--max-running-requests",
            "256",
            "--page-size",
            "128",
            "--disable-radix-cache",
        ]
        process, base_url = self._launch_server_if_needed(DEFAULT_MODEL_NAME_FOR_TEST, server_args)

        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=500,
            request_rate=float("inf"),
            other_server_args=server_args,
            random_input_len=1,
            random_output_len=1024,
            max_concurrency=256,
            random_range_ratio=1,
            base_url=base_url,
            external_process=process,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_output_throughput_default\n"
                f"Output throughput: {res['output_throughput']:.2f} token/s\n"
            )
            self.assertGreater(res["output_throughput"], 2345)

    def test_input_throughput_default_tp_4(self):
        server_args = [
            "--trust-remote-code",
            "--tp-size",
            "4",
            "--skip-server-warmup",
            "--device",
            "tpu",
            "--random-seed",
            "3",
            "--chunked-prefill-size",
            "2048",
            "--download-dir",
            "/dev/shm/",
            "--dtype",
            "bfloat16",
            "--mem-fraction-static",
            "0.8",
            "--max-running-requests",
            "256",
            "--page-size",
            "128",
            "--disable-radix-cache",
        ]
        process, base_url = self._launch_server_if_needed(DEFAULT_MODEL_NAME_FOR_TEST, server_args)

        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=500,
            request_rate=float("inf"),
            other_server_args=server_args,
            random_input_len=1024,
            random_output_len=1,
            max_concurrency=256,
            random_range_ratio=1,
            base_url=base_url,
            external_process=process,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_input_throughput_default_tp_4\n"
                f"Input throughput: {res['input_throughput']:.2f} token/s\n"
            )
            self.assertGreater(res["input_throughput"], 64960)

    def test_output_throughput_default_tp_4(self):
        server_args = [
            "--trust-remote-code",
            "--tp-size",
            "4",
            "--skip-server-warmup",
            "--device",
            "tpu",
            "--random-seed",
            "3",
            "--chunked-prefill-size",
            "2048",
            "--download-dir",
            "/dev/shm/",
            "--dtype",
            "bfloat16",
            "--mem-fraction-static",
            "0.8",
            "--max-running-requests",
            "256",
            "--page-size",
            "128",
            "--disable-radix-cache",
        ]
        process, base_url = self._launch_server_if_needed(DEFAULT_MODEL_NAME_FOR_TEST, server_args)

        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=500,
            request_rate=float("inf"),
            other_server_args=server_args,
            random_input_len=1,
            random_output_len=1024,
            max_concurrency=256,
            random_range_ratio=1,
            base_url=base_url,
            external_process=process,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_output_throughput_default_tp_4\n"
                f"Output throughput: {res['output_throughput']:.2f} token/s\n"
            )
            self.assertGreater(res["output_throughput"], 9866)

    def test_moe_input_throughput_default(self):
        server_args = [
            "--trust-remote-code",
            "--tp-size",
            "4",
            "--skip-server-warmup",
            "--device",
            "tpu",
            "--random-seed",
            "3",
            "--chunked-prefill-size",
            "2048",
            "--download-dir",
            "/dev/shm/",
            "--dtype",
            "bfloat16",
            "--mem-fraction-static",
            "0.8",
            "--max-running-requests",
            "256",
            "--page-size",
            "128",
            "--disable-radix-cache",
        ]
        process, base_url = self._launch_server_if_needed(QWEN3_MOE_30B, server_args)

        res = run_bench_serving(
            model=QWEN3_MOE_30B,
            num_prompts=500,
            request_rate=float("inf"),
            other_server_args=server_args,
            random_input_len=1024,
            random_output_len=1,
            max_concurrency=256,
            random_range_ratio=1,
            base_url=base_url,
            external_process=process,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_moe_input_throughput_default\n"
                f"Input throughput: {res['input_throughput']:.2f} token/s\n"
            )
            self.assertGreater(res["input_throughput"], 14168)

    def test_moe_output_throughput_default(self):
        server_args = [
            "--trust-remote-code",
            "--tp-size",
            "4",
            "--skip-server-warmup",
            "--device",
            "tpu",
            "--random-seed",
            "3",
            "--chunked-prefill-size",
            "2048",
            "--download-dir",
            "/dev/shm/",
            "--dtype",
            "bfloat16",
            "--mem-fraction-static",
            "0.8",
            "--max-running-requests",
            "256",
            "--page-size",
            "128",
            "--disable-radix-cache",
        ]
        process, base_url = self._launch_server_if_needed(QWEN3_MOE_30B, server_args)

        res = run_bench_serving(
            model=QWEN3_MOE_30B,
            num_prompts=500,
            request_rate=float("inf"),
            other_server_args=server_args,
            random_input_len=1,
            random_output_len=1024,
            max_concurrency=256,
            random_range_ratio=1,
            base_url=base_url,
            external_process=process,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_moe_output_throughput_default\n"
                f"Output throughput: {res['output_throughput']:.2f} token/s\n"
            )
            self.assertGreater(res["output_throughput"], 2835)

    def test_ttft_default(self):
        server_args = [
            "--trust-remote-code",
            "--skip-server-warmup",
            "--device",
            "tpu",
            "--random-seed",
            "3",
            "--chunked-prefill-size",
            "2048",
            "--download-dir",
            "/dev/shm/",
            "--dtype",
            "bfloat16",
            "--mem-fraction-static",
            "0.8",
            "--max-running-requests",
            "256",
            "--page-size",
            "128",
            "--disable-radix-cache",
        ]
        process, base_url = self._launch_server_if_needed(DEFAULT_MODEL_NAME_FOR_TEST, server_args)

        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=1,
            request_rate=float("inf"),
            other_server_args=server_args,
            random_input_len=1024,
            random_output_len=1,
            max_concurrency=256,
            random_range_ratio=1,
            base_url=base_url,
            external_process=process,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_online_ttft_default\n"
                f"median_ttft_ms: {res['median_ttft_ms']:.2f} ms\n"
            )
            self.assertLess(res["median_ttft_ms"], 52)

    def test_itl_default(self):
        server_args = [
            "--trust-remote-code",
            "--skip-server-warmup",
            "--device",
            "tpu",
            "--random-seed",
            "3",
            "--chunked-prefill-size",
            "2048",
            "--download-dir",
            "/dev/shm/",
            "--dtype",
            "bfloat16",
            "--mem-fraction-static",
            "0.8",
            "--max-running-requests",
            "256",
            "--page-size",
            "128",
            "--disable-radix-cache",
        ]
        process, base_url = self._launch_server_if_needed(DEFAULT_MODEL_NAME_FOR_TEST, server_args)

        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=1,
            request_rate=float("inf"),
            other_server_args=server_args,
            random_input_len=1024,
            random_output_len=1024,
            max_concurrency=256,
            random_range_ratio=1,
            base_url=base_url,
            external_process=process,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_online_itl_default\n" f"median_itl_ms: {res['median_itl_ms']:.2f} ms\n"
            )
            self.assertLess(res["median_itl_ms"], 16)

    def test_ttft_default_tp_4(self):
        server_args = [
            "--trust-remote-code",
            "--skip-server-warmup",
            "--tp-size",
            "4",
            "--device",
            "tpu",
            "--random-seed",
            "3",
            "--chunked-prefill-size",
            "2048",
            "--download-dir",
            "/dev/shm/",
            "--dtype",
            "bfloat16",
            "--mem-fraction-static",
            "0.8",
            "--max-running-requests",
            "256",
            "--page-size",
            "128",
            "--disable-radix-cache",
        ]
        process, base_url = self._launch_server_if_needed(DEFAULT_MODEL_NAME_FOR_TEST, server_args)

        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=1,
            request_rate=float("inf"),
            other_server_args=server_args,
            random_input_len=1024,
            random_output_len=1,
            max_concurrency=256,
            random_range_ratio=1,
            base_url=base_url,
            external_process=process,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_online_ttft_default_tp_4\n"
                f"median_ttft_ms: {res['median_ttft_ms']:.2f} ms\n"
            )
            self.assertLess(res["median_ttft_ms"], 38)

    def test_itl_default_tp_4(self):
        server_args = [
            "--trust-remote-code",
            "--skip-server-warmup",
            "--tp-size",
            "4",
            "--device",
            "tpu",
            "--random-seed",
            "3",
            "--chunked-prefill-size",
            "2048",
            "--download-dir",
            "/dev/shm/",
            "--dtype",
            "bfloat16",
            "--mem-fraction-static",
            "0.8",
            "--max-running-requests",
            "256",
            "--page-size",
            "128",
            "--disable-radix-cache",
        ]
        process, base_url = self._launch_server_if_needed(DEFAULT_MODEL_NAME_FOR_TEST, server_args)

        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=1,
            request_rate=float("inf"),
            other_server_args=server_args,
            random_input_len=1024,
            random_output_len=1024,
            max_concurrency=256,
            random_range_ratio=1,
            base_url=base_url,
            external_process=process,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_online_itl_default_tp_4\n"
                f"median_itl_ms: {res['median_itl_ms']:.2f} ms\n"
            )
            self.assertLess(res["median_itl_ms"], 8)

    def test_moe_ttft_default(self):
        server_args = [
            "--trust-remote-code",
            "--skip-server-warmup",
            "--tp-size",
            "4",
            "--device",
            "tpu",
            "--random-seed",
            "3",
            "--chunked-prefill-size",
            "2048",
            "--download-dir",
            "/dev/shm/",
            "--dtype",
            "bfloat16",
            "--mem-fraction-static",
            "0.8",
            "--max-running-requests",
            "256",
            "--page-size",
            "128",
            "--disable-radix-cache",
        ]
        process, base_url = self._launch_server_if_needed(QWEN3_MOE_30B, server_args)

        res = run_bench_serving(
            model=QWEN3_MOE_30B,
            num_prompts=1,
            request_rate=float("inf"),
            other_server_args=server_args,
            random_input_len=1024,
            random_output_len=1,
            max_concurrency=256,
            random_range_ratio=1,
            base_url=base_url,
            external_process=process,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_moe_online_ttft_default\n"
                f"median_ttft_ms: {res['median_ttft_ms']:.2f} ms\n"
            )
            self.assertLess(res["median_ttft_ms"], 106)

    def test_moe_itl_default(self):
        server_args = [
            "--trust-remote-code",
            "--skip-server-warmup",
            "--tp-size",
            "4",
            "--device",
            "tpu",
            "--random-seed",
            "3",
            "--chunked-prefill-size",
            "2048",
            "--download-dir",
            "/dev/shm/",
            "--dtype",
            "bfloat16",
            "--mem-fraction-static",
            "0.8",
            "--max-running-requests",
            "256",
            "--page-size",
            "128",
            "--disable-radix-cache",
        ]
        process, base_url = self._launch_server_if_needed(QWEN3_MOE_30B, server_args)

        res = run_bench_serving(
            model=QWEN3_MOE_30B,
            num_prompts=1,
            request_rate=float("inf"),
            other_server_args=server_args,
            random_input_len=1024,
            random_output_len=1024,
            max_concurrency=256,
            random_range_ratio=1,
            base_url=base_url,
            external_process=process,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_moe_online_itl_default\n"
                f"median_itl_ms: {res['median_itl_ms']:.2f} ms\n"
            )
            self.assertLess(res["median_itl_ms"], 27)


if __name__ == "__main__":
    unittest.main()
