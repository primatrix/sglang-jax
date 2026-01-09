import unittest

import PIL.Image
import numpy as np

from sgl_jax.srt.multimodal.manager.io_struct import GenerateMMReqInput
from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.server_args import PortArgs
from sgl_jax.srt.multimodal.manager.multimodal_tokenizer import MultimodalTokenizer

from diffusers.utils import load_image

class TestTokenizer(unittest.IsolatedAsyncioTestCase):

    async def test_image_preprocess(self):
        server_args = MultimodalServerArgs(
            model_path="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers", multimodal=True, skip_tokenizer_init=True
        )
        port_args = PortArgs.init_new(server_args)
        tokenizer = MultimodalTokenizer(server_args, port_args)
        input_ref = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"

        obj = await tokenizer._tokenize_one_request(GenerateMMReqInput(input_reference=input_ref))
        self.assertIsNotNone(obj.preprocessed_image)

        image = load_image(input_ref)
        max_area = 720 * 1280
        aspect_ratio = image.height / image.width
        mod_value = 8 * 2
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize((width, height), PIL.Image.Resampling.LANCZOS)

        from diffusers.video_processor import VideoProcessor

        video_processor = VideoProcessor(vae_scale_factor=8, do_convert_rgb=True)
        want = video_processor.preprocess(image, height=image.height, width=image.width)

        np.testing.assert_array_equal(obj.preprocessed_image, want.cpu().numpy())


if __name__ == "__main__":
    unittest.main()
