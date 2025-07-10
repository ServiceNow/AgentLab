from accelerate import Accelerator
from accelerate.utils.modeling import load_checkpoint_in_model
from agentlab.llm.llm_utils import AIMessage, Discussion
from dataclasses import dataclass
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration
from transformers.models.mllama.image_processing_mllama import get_all_supported_aspect_ratios
from typing import Optional
from .base import VLModel, VLModelArgs
from ..utils import auto_dispatch_model, image_url_to_image


class LlamaModel(VLModel):
    def __init__(
        self,
        model_path: str,
        torch_dtype: str,
        accelerator_config: dict,
        reproducibility_config: dict,
        max_length: int,
        max_new_tokens: int,
    ):
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.accelerator = Accelerator(**accelerator_config)
        self.reproducibility_config = reproducibility_config
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens

    def __call__(self, messages: Discussion) -> AIMessage:
        input_messages = []
        input_images = []
        for message in messages:
            input_message = {"role": message["role"], "content": []}
            if isinstance(message["content"], str):
                input_message["content"].append({"type": "text", "text": message["content"]})
            else:
                for item in message["content"]:
                    if item["type"] == "text":
                        input_message["content"].append(item)
                    elif item["type"] == "image_url":
                        input_message["content"].append({"type": "image"})
                        input_images.append(image_url_to_image(item["image_url"]["url"]))
                    else:
                        raise ValueError(f"Unsupported type: {item['type']}")
            input_messages.append(input_message)
        input_text = self.processor.apply_chat_template(
            input_messages, add_generation_prompt=True, tokenize=False
        )
        input = self.processor(
            images=input_images,
            text=input_text,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.model.device)
        with self.accelerator.autocast():
            output = self.model.generate(
                **input,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                **self.reproducibility_config,
            )
        output_text = self.processor.tokenizer.batch_decode(
            output[:, input["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return AIMessage([{"type": "text", "text": output_text}])

    def adapt_location(self, image: Image.Image, x: int, y: int) -> tuple[int, int]:
        input = self.processor(images=[image], text="")
        _, _, _, _, tile_height, tile_width = input["pixel_values"].shape
        num_tiles_height, num_tiles_width = get_all_supported_aspect_ratios(
            self.processor.image_processor.max_image_tiles
        )[input["aspect_ratio_ids"].item() - 1]
        height = tile_height * num_tiles_height
        width = tile_width * num_tiles_width
        x = int(x / width * image.width)
        y = int(y / height * image.height)
        return x, y

    @property
    def stats(self) -> dict:
        return {}


@dataclass
class LlamaModelArgs(VLModelArgs):
    model_path: str
    torch_dtype: str
    accelerator_config: dict
    reproducibility_config: dict
    max_length: int
    max_new_tokens: int
    checkpoint: Optional[str]
    device: Optional[str]

    @property
    def model_name(self) -> str:
        return self.model_path.split("/")[-1].replace("-", "_").replace(".", "")

    def make_model(self) -> LlamaModel:
        if not hasattr(self, "llama_model"):
            llama_model = LlamaModel(
                model_path=self.model_path,
                torch_dtype=self.torch_dtype,
                accelerator_config=self.accelerator_config,
                reproducibility_config=self.reproducibility_config,
                max_length=self.max_length,
                max_new_tokens=self.max_new_tokens,
            )
            if self.checkpoint is not None:
                load_checkpoint_in_model(llama_model.model, checkpoint=self.checkpoint)
            if self.device is None:
                layer_classes = set()
                for layer in llama_model.model.model.language_model.layers:
                    layer_classes.add(layer.__class__)
                for layer in llama_model.model.model.vision_model.transformer.layers:
                    layer_classes.add(layer.__class__)
                for layer in llama_model.model.model.vision_model.global_transformer.layers:
                    layer_classes.add(layer.__class__)
                llama_model.model = auto_dispatch_model(
                    llama_model.model,
                    no_split_module_classes=[layer_class.__name__ for layer_class in layer_classes],
                )
            else:
                llama_model.model = llama_model.model.to(self.device)
            llama_model.model.eval()
            self.llama_model = llama_model
        return self.llama_model

    def prepare(self):
        pass

    def close(self):
        pass

    def set_reproducibility_mode(self):
        self.reproducibility_config = {"do_sample": False}
