from typing import List

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

from ..utils.base_model import BaseImageTextModel


class MedSigLIP(BaseImageTextModel):
    def __init__(
            self,
            model_id: str = "google/medsiglip-448"
        ) -> None:
        super().__init__(model_id)

        self.model = AutoModel.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            cache_dir="./models"
        ).eval()

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir="./models/processor"
        )
        
        self.device = next(self.model.parameters()).device

    def model_infer(
        self,
        image: Image.Image,
        system_prompt: str,
        user_prompt: str,
        labels: List[str] = ['benign', 'malignant'],
        max_new_tokens: int = 128
    ) -> str:
        # MedSigLIP requires 448x448
        image = image.resize((448, 448))

        inputs = self.processor(
            text=labels,
            images=image,
            padding="max_length",
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = torch.softmax(logits_per_image, dim=1)

        predicted_index = probs.argmax(dim=1).item()
        return labels[predicted_index]

