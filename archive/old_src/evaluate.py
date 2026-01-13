import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path

from src.models import UNetSmall
from src.dataset import TuSimpleSegmentationDataset

# -----------------------------
# 1. CONFIG
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 8
THRESH = 0.35  # lower threshold for under-confident UNet

CKPT_PATH = Path("models/checkpoints/unet_tusimple_best.pt")

# -----------------------------
# 2. DATASET & LOADER
# -----------------------------
val_ds = TuSimpleSegmentationDataset(split="val")

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,   # Windows-safe
    pin_memory=True
)

# -----------------------------
# 3. MODEL
# -----------------------------
model = UNetSmall().to(DEVICE)

if not CKPT_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()

print(f"[INFO] Loaded checkpoint from epoch {ckpt.get('epoch', 'N/A')}")

# -----------------------------
# 4. METRIC FUNCTIONS
# -----------------------------
def dice_score(pred, target, eps=1.0):
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    inter = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    return ((2 * inter + eps) / (union + eps)).mean().item()

def iou_score(pred, target, eps=1e-6):
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    inter = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - inter
    return ((inter + eps) / (union + eps)).mean().item()

def recall_score(pred, target, eps=1e-6):
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    tp = (pred * target).sum(dim=1)
    fn = ((1 - pred) * target).sum(dim=1)
    return ((tp + eps) / (tp + fn + eps)).mean().item()

# -----------------------------
# 5. VALIDATION LOOP
# -----------------------------
dice_total = 0.0
iou_total = 0.0
recall_total = 0.0

with torch.no_grad():
    for imgs, masks in val_loader:
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        logits = model(imgs)
        probs = torch.sigmoid(logits)
        preds = (probs > THRESH).float()

        dice_total += dice_score(preds, masks)
        iou_total += iou_score(preds, masks)
        recall_total += recall_score(preds, masks)

num_batches = len(val_loader)

print("\n===== VALIDATION RESULTS =====")
print(f"Dice Score : {dice_total / num_batches:.4f}")
print(f"IoU Score  : {iou_total / num_batches:.4f}")
print(f"Recall    : {recall_total / num_batches:.4f}")
print("==============================\n")

# -----------------------------
# 6. VISUAL VALIDATION (KEY)
# -----------------------------
# Show a few samples to confirm learning
model.eval()

NUM_SAMPLES = 3
indices = [0, len(val_ds)//2, len(val_ds)-1]

for idx in indices[:NUM_SAMPLES]:
    x, y = val_ds[idx]
    x = x.to(DEVICE)

    with torch.no_grad():
        out = model(x.unsqueeze(0))
        prob = torch.sigmoid(out).squeeze().cpu().numpy()

    img = x.cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img, 0, 1)

    gt = y.squeeze().numpy()
    pred = (prob > THRESH).astype(np.uint8)

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 4, 1)
    plt.title("Input Image")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.title("Ground Truth")
    plt.imshow(gt, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.title("Prediction")
    plt.imshow(pred, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.title("Probability Map")
    plt.imshow(prob, cmap="magma")
    plt.colorbar()

    plt.tight_layout()
    plt.show()
