import torch
from PIL import Image

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor
)

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

from ..utils.base_model import BaseImageTextModel


class QwenModel(BaseImageTextModel):
    def __init__(self, model_id: str):
        super().__init__(model_id)

        if "qwen2.5" in model_id.lower():
            ModelClass = Qwen2_5_VLForConditionalGeneration

        elif "qwen2" in model_id.lower():
            ModelClass = Qwen2VLForConditionalGeneration

        else:
            raise ValueError(f"Model '{model_id}' not supported.")

        self.model = ModelClass.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            cache_dir="./models"
        )

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
            use_fast=True,
            cache_dir="./models/processor"
        )

    def model_infer(
            self,
            image: Image.Image,
            system_prompt: str,
            user_prompt: str,
            max_new_tokens: int = 128
        ) -> str:
        if process_vision_info is None:
            raise RuntimeError("qwen_vl_utils.process_vision_info is required for QwenVL")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_prompt}]}
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
            return self.processor.tokenizer.decode(trimmed[0], skip_special_tokens=True)