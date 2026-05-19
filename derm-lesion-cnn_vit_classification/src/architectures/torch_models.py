import torch

from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet152, ResNet152_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b7, EfficientNet_B7_Weights,
    vit_l_16, ViT_L_16_Weights,
    vit_b_32, ViT_B_32_Weights,
    densenet121, DenseNet121_Weights,
    densenet161, DenseNet161_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
    convnext_base, ConvNeXt_Base_Weights,
    convnext_large, ConvNeXt_Large_Weights,
)

from src.architectures.dinov3_convnext_tiny import Dinov3ConvTinyModel
from src.architectures.dinov3_convnext_large import Dinov3ConvLargeModel


class ClassifierLoader:
    def __init__(self, model_name, class_names):
        self.model_name = model_name
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.model = self._load_model()

    def _load_model(self):

        if self.model_name == "dinov3_convnext_tiny_backbone_freezed":
            model = Dinov3ConvTinyModel()
            model.replace_classifier(num_classes=self.num_classes)
            model.freeze_backbone()

        elif self.model_name == "dinov3_convnext_tiny":
            model = Dinov3ConvTinyModel()
            model.replace_classifier(num_classes=self.num_classes)

        elif self.model_name == "dinov3_convnext_large_backbone_freezed":
            model = Dinov3ConvLargeModel()
            model.replace_classifier(num_classes=self.num_classes)
            model.freeze_backbone()

        elif self.model_name == "dinov3_convnext_large":
            model = Dinov3ConvLargeModel()
            model.replace_classifier(num_classes=self.num_classes)
            
        elif self.model_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT
            model = resnet18(weights=weights)
            in_features = model.fc.in_features
            model.fc = torch.nn.Linear(in_features, self.num_classes)

        elif self.model_name == "resnet152":
            weights = ResNet152_Weights.DEFAULT
            model = resnet152(weights=weights)
            in_features = model.fc.in_features
            model.fc = torch.nn.Linear(in_features, self.num_classes)

        elif self.model_name == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT
            model = efficientnet_b0(weights=weights)
            in_features = model.classifier[1].in_features
            model.classifier[1] = torch.nn.Linear(in_features, self.num_classes)

        elif self.model_name == "efficientnet_b7":
            weights = EfficientNet_B7_Weights.DEFAULT
            model = efficientnet_b7(weights=weights)
            in_features = model.classifier[1].in_features
            model.classifier[1] = torch.nn.Linear(in_features, self.num_classes)

        elif self.model_name == "densenet121":
            weights = DenseNet121_Weights.DEFAULT
            model = densenet121(weights=weights)
            in_features = model.classifier.in_features
            model.classifier = torch.nn.Linear(in_features, self.num_classes)

        elif self.model_name == "densenet161":
            weights = DenseNet161_Weights.DEFAULT
            model = densenet161(weights=weights)
            in_features = model.classifier.in_features
            model.classifier = torch.nn.Linear(in_features, self.num_classes)

        elif self.model_name == "vit_l_16":
            weights = ViT_L_16_Weights.DEFAULT
            model = vit_l_16(weights=weights)
            in_features = model.heads.head.in_features
            model.heads.head = torch.nn.Linear(in_features, self.num_classes)

        elif self.model_name == "vit_b_32":
            weights = ViT_B_32_Weights.DEFAULT
            model = vit_b_32(weights=weights)
            in_features = model.heads.head.in_features
            model.heads.head = torch.nn.Linear(in_features, self.num_classes)

        elif self.model_name == "convnext_tiny":
            weights = ConvNeXt_Tiny_Weights.DEFAULT
            model = convnext_tiny(weights=weights)
            in_features = model.classifier[2].in_features
            model.classifier[2] = torch.nn.Linear(in_features, self.num_classes)

        elif self.model_name == "convnext_base":
            weights = ConvNeXt_Base_Weights.DEFAULT
            model = convnext_base(weights=weights)
            in_features = model.classifier[2].in_features
            model.classifier[2] = torch.nn.Linear(in_features, self.num_classes)

        elif self.model_name == "convnext_large":
            weights = ConvNeXt_Large_Weights.DEFAULT
            model = convnext_large(weights=weights)
            in_features = model.classifier[2].in_features
            model.classifier[2] = torch.nn.Linear(in_features, self.num_classes)

        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

        return model

    def get_model(self) -> torch.nn.Module:
        return self.model
    
    def get_model_name(self) -> str:
        return self.model_name

    def get_num_classes(self) -> int:
        return self.num_classes
