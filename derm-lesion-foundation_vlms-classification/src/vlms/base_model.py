import os
import re
import gc
from abc import ABC, abstractmethod
import sys
from typing import List, Tuple

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import yaml

from src.metrics.metrics_helper import MetricsHelper


class BaseImageTextModel(ABC):
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.df = None
        self.df_dataset_path = None

    @abstractmethod
    def model_infer(
            self,
            image: Image.Image,
            system_prompt: str,
            user_prompt: str,
            labels: List[str],
            max_new_tokens: int = 128
        ) -> str:
        """Run inference on an image using the vision-language model.

        Args:
            image: Input image.
            system_prompt: System-level prompt.
            user_prompt: User-level prompt.
            labels: List of possible labels.
            max_new_tokens: Maximum number of generated tokens.

        Returns:
            Model prediction text.
        """
        pass

    def load_image(
            self,
            image_path: str
        ) -> Image.Image:
        """Load an image from disk.

        Args:
            image_path: Path to the image file.

        Returns:
            Loaded RGB image.
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            return Image.open(image_path).convert("RGB")
        except Exception as e:
            
            raise RuntimeError(f"Error loading image '{image_path}': {e}")

    def _load_and_prepare_csv(self, csv_file_name: str) -> pd.DataFrame:
        csv_path = os.path.join(self.dataset_path, csv_file_name)
        df = pd.read_csv(csv_path)

        if 'partition' in df.columns:
            df = df[df["partition"] == "test"].drop(columns=["partition"])

        return df
    
    @staticmethod
    def response_filter(pred_raw: str) -> str:
        """Normalize and filter the raw model prediction.

        Args:
            pred_raw: Raw prediction text.

        Returns:
            Normalized prediction label.
        """
        pred = re.sub(r"[^a-zA-Z ]+", "", pred_raw).strip().lower()
        has_malignant = "malignant" in pred
        has_benign = "benign" in pred

        if has_malignant:
            return "malignant"
        if has_benign:
            return "benign"
        
        return "error"

    def _inference_row(
        self,
        row: pd.Series,
        system_prompt: str,
        user_prompt: str,
        labels: List[str]
    ) -> dict | None:
        try:
            image_file, label, final_prompt = self._extract_info(row, user_prompt)
            if label is None:
                return None

            image_path = os.path.join(self.image_folder, image_file)

            if not os.path.exists(image_path):
                return None

            image = self.load_image(image_path)
            
            pred_raw = self.model_infer(image, system_prompt, final_prompt, max_new_tokens=100)

            del image
            gc.collect()
            torch.cuda.empty_cache()

            pred = self.response_filter(pred_raw)

            return {"image": image_file, "prediction": pred, "label": label.lower()}

        except Exception as e:
            print(f"Error processing row: {e}")
            return None


    def _extract_info(self, row: pd.Series, user_prompt: str) -> tuple[str, str | None, str]:
        dataset = self.dataset_path.lower()

        image_file = row.get("img_id")
        label = row.get("benign_malignant")
        
        if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_file += ".jpg"

        if image_file is None:
            raise ValueError(f"Image file is missing for row: {row}")

        if 'isic' in dataset:
            if "patient_info" in self.config_path:
                user_prompt = f'''
                Patient and image metadata:

                - Approximate age: {row.get("age_approx")}
                - Sex: {row.get("sex")}
                - Lesion location: {row.get("anatom_site_general")}
                '''
                if 'isic24' in dataset:
                    user_prompt += f'\n- Lesion diameter: {row.get("clin_size_long_diam_mm")}'

        elif 'hc' in dataset: 
            if "patient_info" in self.config_path:
                user_prompt = f'''
                Patient and image metadata:

                • Lesion location: {row.get("bodyPart_pt")}
                '''


        elif 'pad' in dataset:
            if "patient_info" in self.config_path:
                user_prompt = f"""
                Patient Medical Record

                ── Demographics ──
                • Age: {row.get("age")}
                • Sex: {row.get("gender")}
                • Lesion Location: {row.get("region")}

                ── Lifestyle & Exposure ──
                • Smoke: {row.get("smoke")}
                • Drink: {row.get("drink")}
                • Pesticide Exposure: {row.get("pesticide")}

                ── Symptoms Reported ──
                • Itching: {row.get("itch")}
                • Growth: {row.get("grew")}
                • Pain: {row.get("hurt")}
                • Changes in shape/color: {row.get("changed")}
                • Bleeding: {row.get("bleed")}
                • Elevation: {row.get("elevation")}

                ── Living Conditions ──
                • Piped Water: {row.get("has_piped_water")}
                • Sewage System: {row.get("has_sewage_system")}

                ── Medical History ──
                • Skin Cancer History: {row.get("skin_cancer_history")}
                • Other Cancer History: {row.get("cancer_history")}

                ── Family Background ──
                • Father's origin: {row.get("background_father")}
                • Mother's origin: {row.get("background_mother")}
                """


        if pd.isna(label) or str(label).strip() == "":
            return image_file, None, user_prompt
        
        else:
            return image_file, label, user_prompt

    def extract_prompts(self) -> Tuple[str, str, List[str]]:
        """Load prompts and labels from the YAML configuration.

        Returns:
            System prompt, user prompt, and label list.
        """
        try:
            with open(self.config_path, "r") as f:
                cfg = yaml.safe_load(f)

        except yaml.YAMLError as e:
            print(f"Error loading the YAML configuration: {e}")
            sys.exit(1)

        system_prompt = cfg.get("system_prompt", "")
        user_prompt = cfg.get("user_prompt", "")
        labels = cfg.get("labels", "")

        return system_prompt, user_prompt, labels

    def run(
            self,
            dataset_path: str,
            config_path: str,
            output_path: str
        ) -> None:
        """Run inference and evaluation on a dataset.

        Args:
            dataset_path: Path to the dataset directory.
            config_path: Path to the YAML configuration file.
            output_path: Path where the evaluation report will be saved.
        """
        self.dataset_path = dataset_path
        self.image_folder = os.path.join(self.dataset_path, "images")
        self.config_path = config_path

        self.metrics_helper = MetricsHelper(output_path)

        if self.df is None or self.df_dataset_path != dataset_path:
            self.df = self._load_and_prepare_csv(csv_file_name="test_metadata.csv")
            self.df_dataset_path = dataset_path

        system_prompt, user_prompt, labels = self.extract_prompts()

        predictions = []
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing images"):
            result = self._inference_row(row, system_prompt, user_prompt, labels)

            if result:
                predictions.append(result)

        self.metrics_helper.save_csv_and_metrics(predictions)