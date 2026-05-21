import json
import base64
import io
from typing import List, Union

import requests
import numpy as np
from PIL import Image

from .base_model import BaseImageTextModel


class OllamaInferRequest(BaseImageTextModel):
    def __init__(
            self,
            model_id: str,
            temperature: float = 0.0,
            api_url: str = "http://192.168.155.4:11434/"
        ) -> None:
        super().__init__(model_id)
        self.temperature = temperature
        self.vlm_model = model_id
        self.api_url = api_url

    def model_infer(
            self,
            image: Union[Image.Image, np.ndarray],
            system_prompt: str = None,
            user_prompt: str = None,
            labels: List[str] = [],
            max_new_tokens: int = 128
        ) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        final_prompt = user_prompt
        if system_prompt:
            final_prompt = f"[SYSTEM PROMPT]\n{system_prompt}\n\n[USER PROMPT]\n{user_prompt}"

        payload = {
            "model": self.vlm_model,
            "prompt": final_prompt,
            "images": [img_b64],
            "options": {
                "temperature": self.temperature
            }
        }

        payload = {
            "model": self.vlm_model,
            "prompt": final_prompt,
            "images": [img_b64],
            "stream": False,
            "options": {
                "temperature": self.temperature
            }
        }

        resp = requests.post(f"{self.api_url}api/generate", json=payload, stream=True)

        output = ""
        for line in resp.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    output += data["response"]

        return output.strip()