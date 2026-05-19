import argparse

import torch

from src.metrics.wrong_classifications_runner import WrongClassificationsRunner


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./results/runs/dinov3_convnext_large/HAM10000/best_model.pth",
    )
    parser.add_argument("--hubconf_folder_path", type=str, default="./src/models/dinov3")
    parser.add_argument("--local_weights", type=str, default="./models/dinov3_convnext_large.pth")
    parser.add_argument("--model_name", type=str, default="dinov3_convnext_large")
    parser.add_argument("--in_features", type=int, default=1536)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--data_root", type=str, default="../datasets/HAM10000")
    parser.add_argument("--images_folder", type=str, default="images")
    parser.add_argument("--test_csv", type=str, default="test_metadata.csv")
    parser.add_argument("--label_column", type=str, default="benign_malignant")
    parser.add_argument("--image_column", type=str, default="img_id")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--input_h", type=int, default=224)
    parser.add_argument("--input_w", type=int, default=224)

    parser.add_argument("--out_root", type=str, default="./results/wrong_classifications")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_per_bucket", type=int, default=0)

    parser.add_argument("--mean", type=float, nargs=3, default=[0.485, 0.456, 0.406])
    parser.add_argument("--std", type=float, nargs=3, default=[0.229, 0.224, 0.225])

    args = parser.parse_args()
    WrongClassificationsRunner().run(args)
