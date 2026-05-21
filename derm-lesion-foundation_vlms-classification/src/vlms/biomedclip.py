from typing import List

import torch
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer

from .base_model import BaseImageTextModel


class BiomedCLIP(BaseImageTextModel):
    def __init__(self, model_id: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"):
        super().__init__(model_id)

        self.model, self.preprocess = create_model_from_pretrained(
            model_id,
            cache_dir="./models"
        )

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = get_tokenizer(model_id, cache_dir="./models/tokenizer")

    def model_infer(
        self,
        image: Image.Image,
        system_prompt: str,
        user_prompt: str,
        labels: List[str] = ['benign', 'malignant'],
        max_new_tokens: int = 128
    ) -> str:
        image = torch.stack([self.preprocess(image)]).to(self.device)
        texts = self.tokenizer([l for l in labels], context_length=max_new_tokens).to(self.device)

        with torch.no_grad():
            image_features, text_features, logit_scale = self.model(image, texts)
            logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
            sorted_indices = torch.argsort(logits, dim=-1, descending=True)

            logits = logits.cpu().numpy()
            sorted_indices = sorted_indices.cpu().numpy()

        top_index = sorted_indices[0][0]
        return labels[top_index]
