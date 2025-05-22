from PIL import Image
from typing import Union
import base64
import io
import numpy as np


def image_to_image_url(image: Union[Image.Image, np.ndarray]):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{image_base64}"


def image_url_to_image(image_url: str) -> Image.Image:
    image_base64 = image_url.replace("data:image/jpeg;base64,", "")
    image_data = base64.b64decode(image_base64.encode())
    buffer = io.BytesIO(image_data)
    image = Image.open(buffer)
    return image
