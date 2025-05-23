from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils.modeling import get_balanced_memory
from PIL import Image
from torch.nn import Module
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
    image_url = f"data:image/jpeg;base64,{image_base64}"
    return image_url


def image_url_to_image(image_url: str) -> Image.Image:
    image_base64 = image_url.replace("data:image/jpeg;base64,", "")
    image_data = base64.b64decode(image_base64.encode())
    buffer = io.BytesIO(image_data)
    image = Image.open(buffer)
    return image


def auto_dispatch_model(model: Module, no_split_module_classes: list[str]) -> Module:
    max_memory = get_balanced_memory(model, no_split_module_classes=no_split_module_classes)
    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=no_split_module_classes,
    )
    model = dispatch_model(model, device_map=device_map)
    return model
