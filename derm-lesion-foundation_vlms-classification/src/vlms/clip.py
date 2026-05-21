from typing import List

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from .base_model import BaseImageTextModel


class CLIPImageTextModel(BaseImageTextModel):
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32"):
        super().__init__(model_id)

        self.model = CLIPModel.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            cache_dir="./models"
        ).eval()

        self.processor = CLIPProcessor.from_pretrained(model_id, cache_dir="./models/processor")
        self.device = next(self.model.parameters()).device

    def model_infer(
        self,
        image: Image.Image,
        system_prompt: str,
        user_prompt: str,
        labels: List[str] = ['benign', 'malignant'],
        max_new_tokens: int = 128
    ) -> str:
        inputs = self.processor(
            text=labels,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        predicted_index = probs.argmax(dim=1).item()
        return labels[predicted_index]