from typing import List

import torch
import torch.nn as nn


class Dinov3BaseModel(nn.Module):
    def __init__(
        self,
        pretrained: bool,
        model_name: str,
        local_weights: str,
        hubconf_folder_path: str,
        in_features: int,
    ):
        super().__init__()
        self.pretrained = pretrained
        self.model_name = model_name
        self.in_features = in_features

        self.model = torch.hub.load(
            repo_or_dir=hubconf_folder_path,
            model=model_name,
            source="local",
            weights=local_weights,
        )

    def replace_classifier(self, num_classes: int):
        self.model.head = nn.Linear(self.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _get_classifier_parameters(self) -> List[nn.Parameter]:
        return list(self.model.head.parameters())

    def get_classifier_layer(self) -> nn.Module:
        return self.model.head

    def get_backbone_modules(self) -> nn.Module:
        return nn.Sequential(self.model.downsample_layers, self.model.stages, self.model.norm)

    def get_full_model(self) -> nn.Module:
        return self.model

    def freeze_backbone(self):
        for param in self.get_backbone_modules().parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.get_backbone_modules().parameters():
            param.requires_grad = True

    def summary(self):
        print(self)

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
