import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from src.models import UNetSmall
from src.dataset import TuSimpleSegmentationDataset

# -----------------------------
# 1. CONFIG
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_H = 360
IMG_W = 640
BATCH_SIZE = 8
EPOCHS = 15
LR = 3e-4
WEIGHT_DECAY = 1e-4

CKPT_DIR = Path("models/checkpoints")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 2. DATASETS & LOADERS
# -----------------------------
train_ds = TuSimpleSegmentationDataset(split="train")
val_ds   = TuSimpleSegmentationDataset(split="val")

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# -----------------------------
# 3. MODEL
# -----------------------------
model = UNetSmall().to(DEVICE)

# -----------------------------
# 4. LOSS FUNCTIONS
# -----------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        inter = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

def compute_pos_weight(dataset, max_batches=200):
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    pos, neg = 0.0, 0.0

    for i, (_, y) in enumerate(loader):
        if i >= max_batches:
            break
        y = y.float()
        pos += y.sum().item()
        neg += (1 - y).sum().item()

    return neg / (pos + 1e-8)

POS_W = compute_pos_weight(train_ds)
print(f"[INFO] POS_W (class imbalance): {POS_W:.2f}")

bce  = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([POS_W], device=DEVICE)
)
dice = DiceLoss()

def loss_fn(logits, targets):
    return 0.5 * bce(logits, targets) + 0.5 * dice(logits, targets)

# -----------------------------
# 5. OPTIMIZER & SCHEDULER
# -----------------------------
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
    verbose=True
)

# -----------------------------
# 6. TRAIN / VALIDATE
# -----------------------------
best_val_dice = 0.0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs  = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # -------------------------
    # VALIDATION
    # -------------------------
    model.eval()
    val_dice = 0.0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs  = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(imgs)
            probs = torch.sigmoid(logits)

            probs = probs.view(probs.size(0), -1)
            masks = masks.view(masks.size(0), -1)

            inter = (probs * masks).sum(dim=1)
            union = probs.sum(dim=1) + masks.sum(dim=1)
            dice_score = (2 * inter + 1.0) / (union + 1.0)

            val_dice += dice_score.mean().item()

    val_dice /= len(val_loader)
    scheduler.step(val_dice)

    print(
        f"[Epoch {epoch+1:02d}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Dice: {val_dice:.4f}"
    )

    # -------------------------
    # CHECKPOINT
    # -------------------------
    if val_dice > best_val_dice:
        best_val_dice = val_dice
        ckpt_path = CKPT_DIR / "unet_tusimple_best.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "val_dice": best_val_dice,
            },
            ckpt_path
        )
        print(f"[SAVED] Best model updated → {ckpt_path}")

print("[DONE] Training complete.")
