import re
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestDPGrammar(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            check_cache_miss=False,
            other_args=[
                "--trust-remote-code",
                "--skip-server-warmup",
                "--random-seed",
                "3",
                "--max-prefill-tokens",
                "1024",
                "--max-total-tokens",
                "4096",
                "--download-dir",
                "/dev/shm",
                "--dtype",
                "bfloat16",
                "--tp-size",
                "4",
                "--dp-size",
                "2",
                "--max-running-requests",
                "4",
                "--page-size",
                "64",
                "--attention-backend",
                "fa",
                "--grammar-backend",
                "llguidance",
                "--load-format",
                "dummy",
                "--disable-precompile",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _generate(self, payload: dict):
        response = requests.post(
            self.base_url + "/generate",
            json=payload,
            timeout=600,
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertNotIn("error", body, body)
        return body

    def test_mixed_unconstrained_and_grammar_requests_on_different_dp_ranks(self):
        barrier = threading.Barrier(2)

        def unconstrained_request():
            barrier.wait()
            return self._generate(
                {
                    "text": "Say one short English word.",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 8,
                    },
                    "dp_rank": 0,
                }
            )

        def grammar_request():
            barrier.wait()
            return self._generate(
                {
                    "text": "Say Hello.",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 8,
                        "ebnf": 'root ::= "Hello"',
                    },
                    "dp_rank": 1,
                }
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            unconstrained_future = executor.submit(unconstrained_request)
            grammar_future = executor.submit(grammar_request)
            unconstrained = unconstrained_future.result(timeout=600)
            constrained = grammar_future.result(timeout=600)

        self.assertTrue(unconstrained["text"], unconstrained)
        self.assertRegex(constrained["text"].strip(), re.compile(r"^Hello$"))


if __name__ == "__main__":
    unittest.main()
