from torchvision import transforms as T
from typing import Any, Dict, List
from PIL import Image, ImageFilter
import random


class PILGaussianBlur:
    def __init__(self, sigma_range=(0.1, 2.0)) -> None:
        self.sigma_range = sigma_range

    def __call__(self, img: Image.Image) -> Image.Image:
        sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))

class SafeAugmentor:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.enabled = bool(cfg.get("enabled", False))
        self.per_image = int(cfg.get("per_image", 5))

        self.rotation_degrees = float(cfg.get("rotation_degrees", 15))
        self.resized_crop_size = int(cfg.get("resized_crop_size", 224))
        self.resized_crop_scale = tuple(cfg.get("resized_crop_scale", [0.8, 1.0]))
        self.resized_crop_ratio = tuple(cfg.get("resized_crop_ratio", [0.9, 1.1]))
        self.brightness = float(cfg.get("brightness", 0.2))
        self.blur_sigma = tuple(cfg.get("blur_sigma", [0.1, 2.0]))
        self.hflip_p = float(cfg.get("hflip_p", 0.5))
        self.vflip_p = float(cfg.get("vflip_p", 0.5))

        a = cfg.get("affine", {})
        self.affine_enabled = bool(a.get("enabled", True))
        self.affine_translate = tuple(a.get("translate", [0.02, 0.02]))
        self.affine_scale = tuple(a.get("scale", [1.0, 1.0]))
        self.affine_shear = float(a.get("shear", 0.0))
        self.affine_fill_white = bool(a.get("fill_white", True))
        self.affine_fill = (255, 255, 255) if self.affine_fill_white else 0

        ops = [
            T.RandomHorizontalFlip(p=self.hflip_p),
            T.RandomVerticalFlip(p=self.vflip_p),
        ]
        if self.affine_enabled:
            ops.append(
                T.RandomAffine(
                    degrees=0,
                    translate=self.affine_translate,
                    scale=self.affine_scale,
                    shear=self.affine_shear,
                    interpolation=T.InterpolationMode.BICUBIC,
                    fill=self.affine_fill,
                )
            )
        ops.extend([
            T.RandomRotation(self.rotation_degrees, interpolation=T.InterpolationMode.BICUBIC, expand=False),
            T.RandomResizedCrop(size=self.resized_crop_size,
                                scale=self.resized_crop_scale,
                                ratio=self.resized_crop_ratio,
                                interpolation=T.InterpolationMode.BICUBIC),
            T.ColorJitter(brightness=self.brightness, contrast=0.0, saturation=0.0, hue=0.0),
            PILGaussianBlur(self.blur_sigma),
        ])
        self._pipeline = T.Compose(ops)

    def _is_enabled(self) -> bool:
        return self.enabled and self.per_image > 0

    def generate(self, image: Image.Image) -> List[Image.Image]:
        if not self._is_enabled():
            return []
        return [self._pipeline(image) for _ in range(self.per_image)]