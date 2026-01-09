import os
from omegaconf import DictConfig, OmegaConf


def load_stage_configs_from_yaml(config_path: str) -> list:
    """Load stage configurations from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        List of stage configuration dictionaries from the file's stage_args
    """
    config_data = OmegaConf.load(config_path)

    return config_data.stage_args

def load_tokenizer_configs_from_yaml(config_path: str) -> DictConfig:
    """Load tokenizer configurations from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Tokenizer args
    """
    config_data = OmegaConf.load(config_path)

    return config_data.tokenizer_args

def extract_model_name(model_path: str) -> str:
    """Extract the model name from a model path.

    Handles both:
    - Local paths: /models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers -> Wan2.1-T2V-1.3B-Diffusers
    - HF repo IDs: Wan-AI/Wan2.1-T2V-1.3B-Diffusers -> Wan2.1-T2V-1.3B-Diffusers
    """
    # Remove trailing slashes
    model_path = model_path.rstrip("/")

    # Get the basename (last component of path)
    basename = os.path.basename(model_path)

    return basename

# if __name__ == '__main__':
# config = load_stage_configs_from_yaml("/Users/icdi/Desktop/inf/sglang-jax/python/sgl_jax/srt/multimodal/models/static_configs/wan2_1_stage_config.yaml")
# print(config[0].runtime.num_tpus)
