import torch
from PIL import Image

from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
    Llama4ForConditionalGeneration
)

from ..utils.base_model import BaseImageTextModel


class LlamaModel(BaseImageTextModel):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        if "llama-4" in model_id.lower() or "scout" in model_id.lower() or "maverick" in model_id.lower():
            self.variant = "llama4"
            self.model = Llama4ForConditionalGeneration.from_pretrained(
                model_id,
                attn_implementation="flex_attention",
                device_map="auto",
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                cache_dir="./models"
            )
            
        else:
            self.variant = "llama3"
            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto",
                cache_dir="./models"
            )

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir="./models/processor"
        
        )
        self.device = self.model.device if hasattr(self.model, "device") else self.device
        self.model.eval()

    def model_infer(
            self,
            image: Image.Image,
            system_prompt: str,
            user_prompt: str,
            max_new_tokens: int = 128
        ) -> str:
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt.strip()
            })

        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt.strip()}
            ]
        })

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)

        input_len = inputs["input_ids"].shape[-1]
        pad_token_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id
            )

        return self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)
