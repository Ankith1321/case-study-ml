from pathlib import Path
import re

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.models import UNetSmall  # your model :contentReference[oaicite:4]{index=4}


def load_and_preprocess_image(image_path: str, img_h=360, img_w=640):
    """Match your dataset preprocessing: RGB, resize, normalize to [0,1], CHW tensor."""  # :contentReference[oaicite:5]{index=5}
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    img_norm = img_resized.astype(np.float32) / 255.0

    x = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW
    return img_resized, x


def infer_mask(model, x, device, thr=0.5):
    with torch.no_grad():
        logits = model(x.to(device))
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()  # HxW
        pred = (prob > thr).astype(np.uint8)
    return pred


def overlay_mask(img_rgb, mask01, color=(255, 0, 0), alpha=0.55):
    """Overlay predicted lane mask on top of the RGB image."""
    out = img_rgb.copy()
    color_layer = np.zeros_like(out)
    color_layer[mask01 == 1] = np.array(color, dtype=np.uint8)

    out = (out * (1 - alpha) + color_layer * alpha).astype(np.uint8)
    return out


def main():
    # 1) Set your input image path (PC path)
    image_path = r"C:\Users\thanu\road-lane-detection\data\processed\tusimple\images\clips__0313-1__6780__20.jpg"

    # 2) Where your checkpoints are saved (matches train.py) :contentReference[oaicite:6]{index=6}
    ckpt_dir = Path("models/checkpoints")

    # 3) Must match training/dataset image size (360, 640) :contentReference[oaicite:7]{index=7} :contentReference[oaicite:8]{index=8}
    img_h, img_w = 360, 640

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load input image once
    img_resized_rgb, x = load_and_preprocess_image(image_path, img_h=img_h, img_w=img_w)

    # Collect epoch checkpoints
    epoch_ckpts = sorted(
        ckpt_dir.glob("unet_tusimple_epoch*.pt"),
        key=lambda p: int(re.search(r"epoch(\d+)", p.stem).group(1))
    )

    if not epoch_ckpts:
        raise FileNotFoundError(f"No epoch checkpoints found in: {ckpt_dir}")

    # Also include best checkpoint (optional)
    best_ckpt = ckpt_dir / "unet_tusimple_best.pt"

    # Run inference for each epoch checkpoint
    overlays = []
    titles = []

    for ckpt_path in epoch_ckpts:
        ckpt = torch.load(ckpt_path, map_location=device)
        model = UNetSmall().to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        pred = infer_mask(model, x, device, thr=0.5)
        overlays.append(overlay_mask(img_resized_rgb, pred))
        titles.append(ckpt_path.stem)

    # If best exists, infer it too
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        model = UNetSmall().to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        pred = infer_mask(model, x, device, thr=0.5)
        overlays.append(overlay_mask(img_resized_rgb, pred))
        titles.append("unet_tusimple_best")

    # Plot results as a grid
    n = len(overlays)
    cols = 4
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(4 * cols, 3.2 * rows))
    for i, (img, t) in enumerate(zip(overlays, titles), start=1):
        plt.subplot(rows, cols, i)
        plt.imshow(img)
        plt.title(t, fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
