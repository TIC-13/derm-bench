import os
from typing import List

from src.eval.base_model import BaseImageTextModel
from src.vlms.gemma import GemmaModel
from src.vlms.qwen import QwenModel
from src.vlms.phi import PhiModel
from src.vlms.llama import LlamaModel
from src.vlms.clip import CLIPImageTextModel
from src.vlms.medsiglip import MedSigLIP
from src.vlms.biomedclip import BiomedCLIP
from src.vlms.ollama_base_request import OllamaInferRequest


class EvaluationPipeline:
    @staticmethod
    def load_model(model_id: str) -> BaseImageTextModel:
        """Load the appropriate model wrapper from a model identifier.

        Args:
            model_id: Model identifier or Hugging Face model name.

        Returns:
            Initialized image-text model wrapper.
        """
        if model_id.startswith("Qwen/"):
            return QwenModel(model_id)
        elif model_id.startswith("google/medgemma"):
            return GemmaModel(model_id)
        elif model_id.startswith("microsoft/Phi"):
            return PhiModel(model_id)
        elif model_id.startswith("meta-llama/Llama"):
            return LlamaModel(model_id)
        elif model_id == "google/medsiglip-448":
            return MedSigLIP(model_id)
        elif model_id == "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224":
            return BiomedCLIP(model_id)
        elif model_id in ["openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14"]:
            return CLIPImageTextModel(model_id)
        else:
            return OllamaInferRequest(model_id)

    def evaluation(
            self,
            models: List[str],
            datasets: List[str],
            configs: List[str],
            dataset_path: str = "../datasets"
        ) -> None:
        """Run evaluations for all model, dataset, and config combinations.

        Args:
            models: Model identifiers to evaluate.
            datasets: Dataset names to use.
            configs: YAML config names without the `.yaml` extension.
            dataset_path: Base directory containing the datasets.
        """
        for model in models:

            model_vlm = self.load_model(model)

            for dataset in datasets:
                for config in configs:
                    print(
                        f"\n[INFO] Running execution with the following parameters: "
                        f"(model:{model}, dataset:{dataset}, config_yaml:{config}).\n"
                    )

                    current_dataset_path = f"{dataset_path}/{dataset}"
                    config_path = f"./configuration_yaml/{config}.yaml"

                    output_dir = f"./results/reports/{config}/{model}/{dataset}"
                    output_txt_path = f"{output_dir}/binary.txt"

                    if os.path.exists(output_dir):
                        print(
                            f"[INFO] Output already exists at {output_dir}. "
                            f"Skipping this run."
                        )
                        continue

                    os.makedirs(output_dir, exist_ok=True)

                    model_vlm.run(
                        dataset_path=current_dataset_path,
                        config_path=config_path,
                        output_path=output_txt_path
                    )

                    print("\n[INFO] Execution ended without errors.\n")