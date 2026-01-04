from pathlib import Path
import re

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.dataset import TuSimpleSegmentationDataset  # uses your val split & preprocessing :contentReference[oaicite:3]{index=3}
from src.models import UNetSmall                    # your UNet :contentReference[oaicite:4]{index=4}

from src.evaluate import (
    iou,
    dice,
    pixel_accuracy,
    precision_score,
    recall_score,
    f1_score,
)



def get_epoch_num(ckpt_path: Path) -> int:
    m = re.search(r"epoch(\d+)", ckpt_path.stem)
    if not m:
        raise ValueError(f"Cannot parse epoch number from: {ckpt_path.name}")
    return int(m.group(1))


def evaluate_checkpoint(model, val_loader, device):
    """
    Computes validation loss + metrics over the whole validation set.
    Uses the same decision rule as training: sigmoid + threshold 0.5. :contentReference[oaicite:6]{index=6}
    """
    loss_fn = nn.BCEWithLogitsLoss()

    model.eval()
    losses = []
    ious, dices = [], []
    accs, precs, recs, f1s = [], [], [], []

    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, leave=False, desc="Val"):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(imgs)
            loss = loss_fn(logits, masks)
            losses.append(loss.item())

            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).float()

            # Convert to numpy 0/1 per image to reuse your evaluate.py metrics :contentReference[oaicite:7]{index=7}
            pred_np = pred.detach().cpu().numpy().astype(np.uint8)   # (B,1,H,W)
            gt_np   = masks.detach().cpu().numpy().astype(np.uint8)  # (B,1,H,W)

            for b in range(pred_np.shape[0]):
                p = pred_np[b, 0]
                g = gt_np[b, 0]

                ious.append(iou(p.astype(bool), g.astype(bool)))
                dices.append(dice(p.astype(bool), g.astype(bool)))
                accs.append(pixel_accuracy(p, g))
                precs.append(precision_score(p, g))
                recs.append(recall_score(p, g))
                f1s.append(f1_score(p, g))

    return {
        "val_loss": float(np.mean(losses)),
        "val_iou": float(np.mean(ious)),
        "val_dice": float(np.mean(dices)),
        "val_acc": float(np.mean(accs)),
        "val_prec": float(np.mean(precs)),
        "val_rec": float(np.mean(recs)),
        "val_f1": float(np.mean(f1s)),
    }


def main():
    # Paths consistent with your code :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}
    processed_root = "data/processed/tusimple"
    ckpt_dir = Path("models/checkpoints")

    # Must match your training size :contentReference[oaicite:10]{index=10} :contentReference[oaicite:11]{index=11}
    img_size = (360, 640)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Validation loader (same dataset class as training) :contentReference[oaicite:12]{index=12}
    val_ds = TuSimpleSegmentationDataset(processed_root, "val", img_size=img_size)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    # Collect epoch checkpoints
    ckpts = sorted(ckpt_dir.glob("unet_tusimple_epoch*.pt"), key=get_epoch_num)
    if not ckpts:
        raise FileNotFoundError(f"No epoch checkpoints found in: {ckpt_dir}")

    epochs = []
    val_loss = []
    val_iou = []
    val_dice = []
    val_prec = []
    val_rec = []
    val_f1 = []

    for ckpt_path in ckpts:
        epoch = get_epoch_num(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)

        model = UNetSmall().to(device)
        model.load_state_dict(ckpt["model"])  # your checkpoint format :contentReference[oaicite:13]{index=13}

        metrics = evaluate_checkpoint(model, val_loader, device)

        epochs.append(epoch)
        val_loss.append(metrics["val_loss"])
        val_iou.append(metrics["val_iou"])
        val_dice.append(metrics["val_dice"])
        val_prec.append(metrics["val_prec"])
        val_rec.append(metrics["val_rec"])
        val_f1.append(metrics["val_f1"])

        print(f"Epoch {epoch:02d} | "
              f"loss={metrics['val_loss']:.4f} "
              f"iou={metrics['val_iou']:.4f} "
              f"dice={metrics['val_dice']:.4f} "
              f"prec={metrics['val_prec']:.4f} "
              f"rec={metrics['val_rec']:.4f} "
              f"f1={metrics['val_f1']:.4f}")

    # ---- Plots ----

    # 1) Validation loss curve (optimization / convergence)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, val_loss, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss (BCEWithLogits)")
    plt.title("Validation Loss vs Epoch")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2) IoU + Dice curves (segmentation quality)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, val_iou, marker="o", label="IoU (Jaccard)")
    plt.plot(epochs, val_dice, marker="o", label="Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation IoU and Dice vs Epoch")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3) Precision / Recall / F1 (error profile)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, val_prec, marker="o", label="Precision")
    plt.plot(epochs, val_rec, marker="o", label="Recall")
    plt.plot(epochs, val_f1, marker="o", label="F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Precision, Recall, F1 vs Epoch")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
