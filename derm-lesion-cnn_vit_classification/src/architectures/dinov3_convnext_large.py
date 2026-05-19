import torch
import torch.nn as nn


class Dinov3ConvLargeModel(nn.Module):
    def __init__(
            self,
            pretrained: bool = True,
            model_name: str = "dinov3_convnext_large",
            local_weights: str = "./models/dinov3_convnext_large.pth",
            hubconf_folder_path: str = "./src/models/dinov3",
            in_features: int = 1536
        ) -> None:
         
        super().__init__()
        self.pretrained = pretrained
        self.model_name = model_name
        self.in_features = in_features

        self.model = torch.hub.load(
            repo_or_dir=hubconf_folder_path,
            model=model_name,
            source="local",
            weights=local_weights
        )

    def replace_classifier(self, num_classes: int) -> None:
        self.model.head = nn.Linear(self.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def get_backbone_modules(self) -> nn.Module:
        return nn.Sequential(self.model.downsample_layers, self.model.stages, self.model.norm)

    def freeze_backbone(self) -> None:
        for param in self.get_backbone_modules().parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.get_backbone_modules().parameters():
            param.requires_grad = True