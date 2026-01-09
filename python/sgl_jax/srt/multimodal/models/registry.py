from sgl_jax.srt.multimodal.models.wan.wan_preprocess import wan21_image_preprocess


class PreprocessorConfigRegistry:

    _IMAGE_PREPROCESS_FUNCS = {
        "Wan2.1-I2V-14B-480P-Diffusers": wan21_image_preprocess,
        "Wan2.1-I2V-14B-720P-Diffusers": wan21_image_preprocess,
    }

    _AUDIO_PREPROCESS_FUNCS = {}

    _VIDEO_PREPROCESS_FUNCS = {}

    @classmethod
    def register_image_preprocessor(cls, model_name: str, func):
        cls._IMAGE_PREPROCESS_FUNCS[model_name] = func

    @classmethod
    def register_audio_preprocessor(cls, model_name: str, func):
        cls._AUDIO_PREPROCESS_FUNCS[model_name] = func

    @classmethod
    def register_video_preprocessor(cls, model_name: str, func):
        cls._VIDEO_PREPROCESS_FUNCS[model_name] = func

    @classmethod
    def get_image_preprocess_func(cls, model_name: str):
        return cls._IMAGE_PREPROCESS_FUNCS.get(model_name)

    @classmethod
    def get_audio_preprocess_func(cls, model_name: str):
        return cls._AUDIO_PREPROCESS_FUNCS.get(model_name)

    @classmethod
    def get_video_preprocess_func(cls, model_name: str):
        return cls._VIDEO_PREPROCESS_FUNCS.get(model_name)
