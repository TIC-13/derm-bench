from src.architectures.backbones.dinov3_base import Dinov3BaseModel


class Dinov3Vith16l(Dinov3BaseModel):
    def __init__(
        self,
        pretrained: bool = True,
        model_name: str = "dinov3_vith16plus",
        local_weights: str = "./models/dinov3_weights/dinov3_vith16plus.pth",
        hubconf_folder_path: str = "./src/models/dinov3",
        in_features: int = 1280,
    ):
        super().__init__(
            pretrained=pretrained,
            model_name=model_name,
            local_weights=local_weights,
            hubconf_folder_path=hubconf_folder_path,
            in_features=in_features,
        )
