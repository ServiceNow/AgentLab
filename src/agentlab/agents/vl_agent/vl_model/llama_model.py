from accelerate.utils.modeling import load_checkpoint_in_model
from agentlab.llm.llm_utils import AIMessage, Discussion
from dataclasses import dataclass
from transformers import AutoProcessor, MllamaForConditionalGeneration
from typing import Optional
from .base import VLModel, VLModelArgs
from ..utils import image_url_to_image
import fnmatch
import os


class LlamaModel(VLModel):
    def __init__(
        self,
        model_path: str,
        torch_dtype: str,
        checkpoint_dir: str,
        max_length: int,
        max_new_tokens: int,
        reproducibility_config: dict,
    ):
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype
        )
        if checkpoint_dir is not None:
            checkpoint_file = None
            for item in os.listdir(checkpoint_dir):
                if fnmatch.fnmatch(item, "pytorch_model*.bin") or fnmatch.fnmatch(
                    item, "model*.safetensors"
                ):
                    checkpoint_file = os.path.join(checkpoint_dir, item)
                    break
            load_checkpoint_in_model(self.model, checkpoint_file)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.reproducibility_config = reproducibility_config

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

    def get_stats(self) -> dict:
        return {}


@dataclass
class LlamaModelArgs(VLModelArgs):
    model_path: str
    torch_dtype: str
    checkpoint_dir: Optional[str]
    max_length: int
    max_new_tokens: int
    reproducibility_config: dict

    @property
    def model_name(self) -> str:
        return self.model_path.split("/")[-1].replace("-", "_").replace(".", "")

    def make_model(self) -> LlamaModel:
        return LlamaModel(
            model_path=self.model_path,
            torch_dtype=self.torch_dtype,
            checkpoint_dir=self.checkpoint_dir,
            max_length=self.max_length,
            max_new_tokens=self.max_new_tokens,
            reproducibility_config=self.reproducibility_config,
        )

    def prepare(self):
        pass

    def close(self):
        pass

    def set_reproducibility_mode(self):
        self.reproducibility_config = {"do_sample": False}
