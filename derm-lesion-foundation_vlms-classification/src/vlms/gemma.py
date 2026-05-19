import torch
from PIL import Image

from transformers import (
    Gemma3ForConditionalGeneration,
    AutoModelForImageTextToText,
    AutoProcessor
)

from ..utils.base_model import BaseImageTextModel


class GemmaModel(BaseImageTextModel):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        if "medgemma" in model_id.lower():
            self.variant = "medgemma"
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir="./models"
            )
        elif "gemma" in model_id.lower():
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir="./models"
            ).eval()
            
        else:
            raise ValueError(f"Unsupported Gemma variant for model_id: {model_id}")

        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir="./models/processor")
        self.device = self.model.device if hasattr(self.model, "device") else self.device

    def model_infer(self, image: Image.Image, system_prompt: str, user_prompt: str, max_new_tokens: int = 128) -> str:
        if self.variant == "medgemma":
            prompts = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}, {"type": "image", "image": image}]}
            ]
        elif self.variant == "gemma3":
            prompts = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_prompt}]}
            ]
        else:
            raise RuntimeError("Unknown variant for GemmaModel.")

        inputs = self.processor.apply_chat_template(
            prompts,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]
        pad_token_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id
            )[0][input_len:]

        return self.processor.decode(output, skip_special_tokens=True)