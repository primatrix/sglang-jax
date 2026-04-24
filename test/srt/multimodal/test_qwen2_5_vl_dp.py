import base64
import io
import os
import unittest

import requests
from PIL import Image

from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

headers = {"Content-Type": "application/json"}


def _solid_color_png_data_uri(color: tuple[int, int, int]) -> str:
    image = Image.new("RGB", (64, 64), color)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class TestQwen25VLDP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = os.environ.get(
            "SGLANG_JAX_QWEN2_5_VL_MODEL",
            "/models/Qwen2.5-VL-3B-Instruct",
        )
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            check_cache_miss=False,
            multimodal=True,
            other_args=[
                "--multimodal",
                "--trust-remote-code",
                "--skip-server-warmup",
                "--random-seed",
                "3",
                "--max-prefill-tokens",
                "2048",
                "--max-total-tokens",
                "8192",
                "--download-dir",
                "/models",
                "--dtype",
                "bfloat16",
                "--tp-size",
                "4",
                "--dp-size",
                "2",
                "--nnodes",
                "1",
                "--dist-init-addr",
                "0.0.0.0:10011",
                "--max-running-requests",
                "2",
                "--page-size",
                "64",
                "--attention-backend",
                "fa",
                "--disable-precompile",
                "--watchdog-timeout",
                "1200",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
                "SGLANG_HEALTH_CHECK_TIMEOUT": "1200",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_red_image_chat_completion(self):
        data = {
            "model": os.path.basename(self.model.rstrip("/")),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": _solid_color_png_data_uri((255, 0, 0))},
                        },
                        {
                            "type": "text",
                            "text": "What color is this image? Answer with one word.",
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 8,
        }
        response = requests.post(
            self.base_url + "/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=1200,
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        content = result["choices"][0]["message"]["content"].lower()
        self.assertIn("red", content)


if __name__ == "__main__":
    unittest.main()
