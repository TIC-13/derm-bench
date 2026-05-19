from typing import List
from io import BytesIO

import numpy as np
from PIL import Image
import tensorflow as tf
from keras.layers import TFSMLayer
import torch
from torchvision import transforms as T
from transformers import AutoImageProcessor, AutoModel

from src.architectures.backbones.dinov3_convnext_tiny import Dinov3ConvTinyModel
from src.architectures.backbones.dinov3_vith16 import Dinov3Vith16l
from src.architectures.backbones.dinov3_7b16 import Dinov3_7B16

DINOV3_MODEL_CONFIGS = {
    "dinov3_convnext_tiny": {"class": Dinov3ConvTinyModel, "embed_dim": 768},
    "dinov3_vith16plus": {"class": Dinov3Vith16l, "embed_dim": 1280},
    "dinov3_vit7b16": {"class": Dinov3_7B16, "embed_dim": 4096},
}

class DermEmbeddingExtractor:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_name == "google/derm-foundation":
            self.backend = "tf"
            try:
                tf.config.set_visible_devices([], "GPU")

            except Exception:
                pass
            
            tf.config.optimizer.set_jit(False)

            with tf.device("/CPU:0"):
                self.model = TFSMLayer("./models/derm_foundation", call_endpoint="serving_default")

        elif model_name == "facebook/dinov2-giant":
            self.backend = "transformers"
            self.processor = AutoImageProcessor.from_pretrained(model_name, cache_dir="./models")
            self.model = AutoModel.from_pretrained(model_name, cache_dir="./models").to(self.device)
            self.model.eval()

        elif model_name in DINOV3_MODEL_CONFIGS:
            self.backend = "torch_local"

            cfg = DINOV3_MODEL_CONFIGS[model_name]
            self.model = cfg["class"]().to(self.device)
            self.model.eval()
            self.embed_dim = cfg["embed_dim"]

            self.transform = T.Compose([
                T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        else:
            raise ValueError(f"Unsupported model_name: {model_name}")


    def extract_embedding(self, image: Image.Image) -> np.ndarray:
        if self.backend == "tf":
            buf = BytesIO()
            image.convert('RGB').save(buf, 'PNG')
            example = tf.train.Example(features=tf.train.Features(
                feature={'image/encoded': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[buf.getvalue()])
                )}
            )).SerializeToString()
            output = self.model(tf.constant([example]))

            return output['embedding'].numpy().flatten()

        elif self.backend == "transformers":
            image = image.convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            return outputs.pooler_output.squeeze().cpu().numpy()

        else:
            x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

            with torch.no_grad():
                y = self.model(x)

            return y.squeeze(0).cpu().numpy()

    def extract_batch_embeddings(self, images: List[Image.Image]) -> np.ndarray:
        if self.backend == "tf":
            raise NotImplementedError("Batch processing is not supported for the TF model.")
        
        if self.backend == "transformers":
            images = [img.convert("RGB") for img in images]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            return outputs.pooler_output.cpu().numpy()
        
        images = [img.convert("RGB") for img in images]
        xs = [self.transform(img) for img in images]
        batch = torch.stack(xs, dim=0).to(self.device)

        with torch.no_grad():
            ys = self.model(batch)

        return ys.cpu().numpy()