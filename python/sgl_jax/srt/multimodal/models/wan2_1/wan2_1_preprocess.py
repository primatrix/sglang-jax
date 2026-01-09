import PIL.Image
import numpy as np


def wan21_image_preprocess(image: PIL.Image.Image, params: dict) -> np.ndarray:
    if params is None:
        params = {}

    # Calculate dimensions based on aspect ratio if not provided
    height, width = params.get("height"), params.get("width")
    if height is None or width is None:
        # Default max area for 720P
        max_area = 720 * 1280
        aspect_ratio = image.height / image.width

        # Calculate dimensions maintaining aspect ratio
        mod_value = 16  # Must be divisible by 16
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value

    # Resize image to target dimensions
    image = image.resize((width, height), PIL.Image.Resampling.LANCZOS)

    # adapter from diffusers.image_processor
    vae_scale_factor = 8
    width, height = (x - x % vae_scale_factor for x in (width, height))
    image = image.resize((width, height), PIL.Image.Resampling.LANCZOS)

    # pil_to_numpy
    images = [np.array(image).astype(np.float32) / 255.0]
    images = np.stack(images, axis=0)  # (N, H, W, C)
    if images.ndim == 3:
        images = images[..., None]
    images = images.transpose(0, 3, 1, 2)  # (N, C, H, W)

    # normalize
    images = 2.0 * images - 1.0

    return images
