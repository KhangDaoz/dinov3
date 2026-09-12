import argparse
from PIL import Image

from pathlib import Path

from transformers import AutoImageProcessor, AutoModel
import torch
import yaml
from dataset import CUBirds

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "cub_e2a.yaml"

def load_processor(config_path=CONFIG_PATH):
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return AutoImageProcessor.from_pretrained(config["model_name"], revision=config["model_revision"], token=config.get("token", None))


def load_dinov3(device=None, config_path=CONFIG_PATH):
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    model_name = config["model_name"]
    processor = load_processor(config_path)

    model = AutoModel.from_pretrained(
        model_name, revision=config["model_revision"], token=config.get("token", None)
    )
    model.eval()

    # freeze the model parameters
    for param in model.parameters():
        param.requires_grad = False

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return model.to(device), processor, device


def extract_last_hidden_state(model, processor, images, device):
    """Process a PIL image batch and return final-layer DINOv3 tokens."""
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    tokens = outputs.last_hidden_state
    if tokens.ndim != 3 or not torch.isfinite(tokens).all().item():
        raise RuntimeError(f"Output tokens không hợp lệ: shape={tuple(tokens.shape)}")
    return tokens, inputs


def main():
    parser = argparse.ArgumentParser(description="Kiểm tra DINOv3 với một ảnh CUB.")
    parser.add_argument("--root", default=str(PROJECT_ROOT / "data"),
                        help="Thư mục chứa CUB_200_2011")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="File config model")
    args = parser.parse_args()

    dataset = CUBirds(args.root, mode="train")
    if not len(dataset):
        raise RuntimeError("Tập train CUB rỗng")
    image_path = dataset.im_paths[0]

    model, processor, device = load_dinov3(config_path=args.config)
    print(f"[OK] Đã load DINOv3 trên {device}")

    with Image.open(image_path) as image:
        inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model(**inputs)
        tokens = outputs.last_hidden_state

    num_registers = model.config.num_register_tokens
    height, width = inputs["pixel_values"].shape[-2:]
    patch_size = model.config.patch_size
    patch_h, patch_w = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
    num_patches = (height // patch_h) * (width // patch_w)
    expected_shape = (1, 1 + num_registers + num_patches, model.config.hidden_size)
    if tuple(tokens.shape) != expected_shape:
        raise RuntimeError(f"Kích thước tokens: {tuple(tokens.shape)}, cần {expected_shape}")
    if not torch.isfinite(tokens).all().item():
        raise RuntimeError("Output tokens chứa NaN/Inf")

    # Thứ tự output DINOv3: CLS, register tokens, patch tokens.
    features = tokens[:, 0, :]
    register_tokens = tokens[:, 1:1 + num_registers, :]
    patch_tokens = tokens[:, 1 + num_registers:, :]

    if features.ndim != 2 or features.shape[0] != 1 or features.shape[1] == 0:
        raise RuntimeError(f"Kích thước đặc trưng không hợp lệ: {tuple(features.shape)}")
    if not torch.isfinite(features).all().item():
        raise RuntimeError("Đặc trưng chứa NaN/Inf")

    print(f"[OK] Đã trích xuất đặc trưng ảnh: {image_path}")
    print(f"     label={dataset.ys[0]}")
    print(f"     features.shape={tuple(features.shape)}")
    print(f"     8 giá trị đầu: {features[0, :8].cpu().tolist()}")
    print(f"     input.shape={tuple(inputs['pixel_values'].shape)}")
    print(f"     all_tokens.shape={tuple(tokens.shape)}")
    print(f"     cls_token.shape={tuple(features.shape)}")
    print(f"     register_tokens.shape={tuple(register_tokens.shape)}")
    print(f"     patch_tokens.shape={tuple(patch_tokens.shape)}")
    print(f"     patch grid={height // patch_h} x {width // patch_w}")
    print(f"     patch đầu, 8 giá trị đầu: {patch_tokens[0, 0, :8].cpu().tolist()}")
    print("[OK] Số lượng tokens đúng và toàn bộ output không chứa NaN/Inf")


if __name__ == "__main__":
    main()
