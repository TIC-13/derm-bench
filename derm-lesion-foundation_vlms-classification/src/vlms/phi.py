import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor
)

from ..utils.base_model import BaseImageTextModel


class PhiModel(BaseImageTextModel):
    def __init__(self, model_id: str):
        super().__init__(model_id)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            _attn_implementation='eager',
            cache_dir="./models"
        )

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            num_crops=8,
            cache_dir="./models/processor"
        )

    def model_infer(
            self,
            image: Image.Image,
            system_prompt: str,
            user_prompt: str,
            max_new_tokens: int = 128
        ) -> str:
        placeholder = "<|image_1|>\n"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": placeholder + user_prompt}
        ]

        prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            prompt,
            [image],
            return_tensors="pt"
        ).to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                use_cache=False
            )


        output_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
