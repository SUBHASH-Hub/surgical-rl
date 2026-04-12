#!/usr/bin/env python3
"""
Phase 3A — Step 4b: Tissue Boundary Segmentation Training
==========================================================
Trains a lightweight UNet to predict a binary tissue segmentation mask
from a single 480×480 RGB frame.

Architecture: UNet with MobileNetV3-Small encoder (<5M parameters)
  - Encoder: MobileNetV3-Small pretrained backbone (same as tip detector)
    Why: consistent feature space between tip detector and segmentation
    model — both use the same ImageNet-pretrained visual features.
  - Decoder: 3 upsampling blocks with skip connections
    Why: skip connections preserve fine spatial detail (tissue edges)
    which is lost in deep downsampling. UNet skip connections are the
    key innovation for segmentation vs classification.
  - Output: 480×480 binary mask via sigmoid (0=background, 1=tissue)

Loss: BCE + Dice combined
  BCE (Binary Cross-Entropy): penalises per-pixel classification error
  Dice loss: penalises mask shape error — ensures the predicted region
             matches the ground truth region globally, not just per-pixel.
  Combined: handles class imbalance (tissue ~20-30% of pixels).

Metric: IoU (Intersection over Union) — the standard segmentation metric.
  IoU = |predicted ∩ ground_truth| / |predicted ∪ ground_truth|
  IoU > 0.5 = usable. IoU > 0.7 = good. IoU > 0.85 = excellent.

Prerequisite: run generate_seg_masks.py first to create data/seg_masks/

Run:
    python3 scripts/train_segmentation.py

Outputs:
    models/segmentation/
        unet_seg_best.pth        ← best val IoU checkpoint
        unet_seg_final.pth       ← final epoch checkpoint
        training_log.csv         ← epoch, train_loss, val_loss, val_iou
        eval_metrics.txt         ← test set IoU + Dice score
        eval_metrics.json        ← machine-readable metrics

Author: Subhash Arockiadoss        
"""

# ── Standard library ───────────────────────────────────────────────────────
import csv
import json
import os
import sys
import time
from pathlib import Path

# ── Scientific stack ───────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from PIL import Image

# ── PyTorch ────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ─────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────

RGB_DIR    = Path("data/rgb_frames")     # input images
MASK_DIR   = Path("data/seg_masks")      # input masks from generate_seg_masks.py
OUTPUT_DIR = Path("models/segmentation")

# Same episode split as tip detector — consistent train/val/test boundary
TRAIN_EPISODES = list(range(0, 40))
VAL_EPISODES   = list(range(40, 45))
TEST_EPISODES  = list(range(45, 50))

# Model: UNet with MobileNetV3-Small encoder
# Input/output resolution: 256×256 (downsampled from 480 for memory)
# Why 256 not 224: UNet decoder needs power-of-2 dimensions for clean
# upsampling. 256 is the nearest power-of-2 above 224.
IMG_SIZE   = 256

# Training
EPOCHS     = 30
BATCH_SIZE = 16    # smaller than tip detector — masks use more VRAM
LR         = 1e-4  # same as tip detector Phase 2 fine-tuning lr
NUM_WORKERS = 4
SEED       = 42

# ─────────────────────────────────────────────────────────────────────────
# COLOUR HELPERS
# ─────────────────────────────────────────────────────────────────────────
GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
CYAN  = "\033[96m"; BOLD = "\033[1m"; RESET  = "\033[0m"

def ok(m):     print(f"{GREEN}  ✓  {m}{RESET}")
def fail(m):   print(f"{RED}  ✗  {m}{RESET}")
def info(m):   print(f"{CYAN}  ·  {m}{RESET}")
def warn(m):   print(f"{YELLOW}  !  {m}{RESET}")
def header(m): print(f"\n{BOLD}{m}{RESET}")


# ─────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────

class SegmentationDataset(Dataset):
    """
    Each sample:
        image : (3, 256, 256) float32 — normalised RGB frame
        mask  : (1, 256, 256) float32 — binary tissue mask (0.0 or 1.0)

    Mask is loaded as a grayscale PNG (0 or 255), normalised to [0, 1].
    The mask filename is derived from the RGB filename by replacing
    the XYZ suffix with '_mask' — e.g.:
      ep000_step0042_x+0.06_y+0.04_z-0.09.png
      ep000_step0042_mask.png
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(self, df: pd.DataFrame, rgb_dir: Path,
                 mask_dir: Path, augment: bool = False):
        self.df       = df.reset_index(drop=True)
        self.rgb_dir  = rgb_dir
        self.mask_dir = mask_dir

        self.img_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])

        # Augmentation: same random flip must be applied to BOTH image AND mask
        # to keep them aligned. We handle this manually in __getitem__.
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _get_mask_filename(self, rgb_filename: str) -> str:
        """
        Derive mask filename from RGB filename.
        RGB:  ep000_step0042_x+0.06_y+0.04_z-0.09.png
        Mask: ep000_step0042_mask.png
        """
        parts = rgb_filename.split('_')
        # Keep ep and step parts, add _mask
        ep_step = '_'.join(parts[:2])   # ep000_step0042
        return f"{ep_step}_mask.png"

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load RGB image
        img = Image.open(self.rgb_dir / row["filename"]).convert("RGB")

        # Load corresponding mask
        mask_filename = self._get_mask_filename(row["filename"])
        mask_path     = self.mask_dir / mask_filename

        if not mask_path.exists():
            # Return empty mask if mask file missing (should not happen
            # if generate_seg_masks.py ran successfully)
            mask = Image.fromarray(np.zeros((480, 480), dtype=np.uint8))
        else:
            mask = Image.open(mask_path).convert("L")   # grayscale

        # Apply augmentation: same random flip to both image and mask
        if self.augment and np.random.random() > 0.5:
            img  = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # Transform image (resize + normalise)
        image_t = self.img_transform(img)

        # Transform mask: resize to 256×256, convert to float [0,1]
        mask_t = transforms.functional.resize(
            transforms.functional.to_tensor(mask),   # (1, H, W) in [0,1]
            (IMG_SIZE, IMG_SIZE)
        )
        # Binarise: pixels > 0.5 → 1.0, rest → 0.0
        # (accounts for interpolation artefacts from resize)
        mask_t = (mask_t > 0.5).float()

        return image_t, mask_t


# ─────────────────────────────────────────────────────────────────────────
# MODEL — LIGHTWEIGHT UNET WITH MOBILENETV3 ENCODER
# ─────────────────────────────────────────────────────────────────────────

class ConvBnRelu(nn.Module):
    """Standard conv block: Conv2d → BatchNorm → ReLU."""
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """
    UNet decoder block: upsample + concatenate skip connection + two convs.

    Why skip connections:
      The encoder progressively reduces spatial resolution (480→16).
      Deep features encode WHAT (tissue vs not-tissue) but lose WHERE.
      Skip connections re-inject spatial detail from earlier encoder
      layers — the decoder sees both "this is tissue" and "here is the
      exact edge location." This is the key UNet innovation.
    """
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            ConvBnRelu(out_ch + skip_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch from odd-dimension inputs
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                              align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class TissueSegUNet(nn.Module):
    """
    Lightweight UNet: MobileNetV3-Small encoder + 3-stage decoder.
    Total parameters: ~3.2M — within the <5M roadmap target.

    Encoder feature dimensions (MobileNetV3-Small):
      Input:     (B,  3, 256, 256)
      Stage 1:   (B, 16, 128, 128)   after first conv block
      Stage 2:   (B, 24,  64,  64)   after inverted residual blocks
      Stage 3:   (B, 48,  32,  32)
      Bottleneck:(B, 576,  8,   8)   after features[-1] + avgpool

    Decoder: 3 upsampling stages back to (B, 1, 256, 256)
    """

    def __init__(self):
        super().__init__()

        # Load MobileNetV3-Small pretrained encoder
        base   = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )
        feats  = base.features   # all conv + SE blocks

        # Split encoder into stages for skip connections
        # Stage indices determined by inspecting MobileNetV3-Small architecture
        self.enc0 = feats[0]           # (B,  16, 128, 128) — first conv
        self.enc1 = feats[1:4]         # (B,  24,  64,  64) — inverted residuals
        self.enc2 = feats[4:9]         # (B,  48,  32,  32)
        self.enc3 = feats[9:]          # (B, 576,   8,   8) — deep features

        # Decoder: 3 upsampling blocks with skip connections
        # Channel dims: 576 up + 48 skip → 128, then 128 up + 24 skip → 64,
        #               then 64 up + 16 skip → 32
        self.dec2 = UpBlock(576, 48, 128)
        self.dec1 = UpBlock(128, 24, 64)
        self.dec0 = UpBlock(64,  16, 32)

        # Final upsampling to full resolution + binary output
        self.up_final = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.head     = nn.Sequential(
            ConvBnRelu(16, 16),
            nn.Conv2d(16, 1, kernel_size=1),   # (B, 1, 256, 256)
            # No sigmoid here — applied in loss (BCEWithLogitsLoss is stable)
        )

    def forward(self, x):
        # Encoder: save skip connections at each stage
        s0 = self.enc0(x)           # (B,  16, 128, 128)
        s1 = self.enc1(s0)          # (B,  24,  64,  64)
        s2 = self.enc2(s1)          # (B,  48,  32,  32)
        s3 = self.enc3(s2)          # (B, 576,   8,   8)

        # Decoder: upsample + concatenate skips
        d2 = self.dec2(s3, s2)      # (B, 128,  32,  32)
        d1 = self.dec1(d2, s1)      # (B,  64,  64,  64)
        d0 = self.dec0(d1, s0)      # (B,  32, 128, 128)

        # Final upsample to 256×256 + 1-channel output
        up = self.up_final(d0)      # (B,  16, 256, 256)
        return self.head(up)        # (B,   1, 256, 256) logits


# ─────────────────────────────────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────

def dice_loss(pred_logits: torch.Tensor, target: torch.Tensor,
              eps: float = 1e-6) -> torch.Tensor:
    """
    Dice loss for binary segmentation.

    Dice coefficient = 2 * |P ∩ T| / (|P| + |T|)
    Dice loss = 1 - Dice coefficient

    Why Dice: handles class imbalance better than BCE alone.
    In our masks, tissue is ~20-30% of pixels. BCE on its own
    can be minimised by predicting all-background — Dice prevents this
    by directly penalising when the predicted region doesn't overlap
    with the target region.
    """
    pred = torch.sigmoid(pred_logits)
    pred = pred.view(-1)
    tgt  = target.view(-1)
    intersection = (pred * tgt).sum()
    return 1 - (2 * intersection + eps) / (pred.sum() + tgt.sum() + eps)


def combined_loss(pred_logits, target):
    """BCE + Dice. Equal weighting (0.5 each) is standard practice."""
    bce  = F.binary_cross_entropy_with_logits(pred_logits, target)
    dice = dice_loss(pred_logits, target)
    return 0.5 * bce + 0.5 * dice


# ─────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────

def compute_iou(pred_logits: torch.Tensor, target: torch.Tensor,
                threshold: float = 0.5) -> float:
    """
    Compute mean IoU over a batch.

    IoU = intersection / union
    threshold=0.5: sigmoid output > 0.5 → predicted as tissue

    IoU is the standard metric for segmentation evaluation.
    Reported in all surgical AI papers and what CMR/Medtronic
    engineers will ask about in interviews.
    """
    pred   = (torch.sigmoid(pred_logits) > threshold).float()
    target = (target > 0.5).float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union        = (pred + target).clamp(0, 1).sum(dim=(1, 2, 3))

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


# ─────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimiser, device, train: bool):
    model.train(train)
    total_loss = 0.0
    total_iou  = 0.0
    n_batches  = 0

    context = torch.no_grad() if not train else torch.enable_grad()
    with context:
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)

            if train:
                optimiser.zero_grad()

            logits = model(images)
            loss   = combined_loss(logits, masks)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()

            total_loss += loss.item()
            total_iou  += compute_iou(logits, masks)
            n_batches  += 1

    return total_loss / n_batches, total_iou / n_batches


def evaluate_test(model, loader, device):
    """Compute IoU and Dice on test set."""
    model.eval()
    all_iou  = []
    all_dice = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks  = masks.to(device)
            logits = model(images)

            iou  = compute_iou(logits, masks)
            pred = (torch.sigmoid(logits) > 0.5).float()
            tgt  = (masks > 0.5).float()
            inter = (pred * tgt).sum().item()
            denom = pred.sum().item() + tgt.sum().item()
            dice  = (2 * inter + 1e-6) / (denom + 1e-6)

            all_iou.append(iou)
            all_dice.append(dice)

    return {
        "iou":  float(np.mean(all_iou)),
        "dice": float(np.mean(all_dice)),
    }


# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    header("=" * 62)
    print("  Phase 3A — Tissue Segmentation Training")
    header("=" * 62)

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        ok(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        warn("No GPU — training on CPU (slow)")

    # ── Check masks exist ─────────────────────────────────────────────────
    n_masks = len(list(MASK_DIR.glob("ep*_mask.png")))
    if n_masks == 0:
        fail(f"No masks found in {MASK_DIR}")
        fail("Run: python3 scripts/generate_seg_masks.py first")
        sys.exit(1)
    ok(f"Found {n_masks} mask files in {MASK_DIR}")

    # ── Load labels CSV (reuse from tip detector) ─────────────────────────
    df = pd.read_csv(RGB_DIR / "labels.csv")

    # Filter to only rows that have a corresponding mask file
    df["mask_filename"] = df["filename"].apply(
        lambda f: "_".join(f.split("_")[:2]) + "_mask.png"
    )
    df = df[df["mask_filename"].apply(
        lambda m: (MASK_DIR / m).exists()
    )].reset_index(drop=True)
    ok(f"Frames with masks: {len(df)}")

    # ── Episode split ──────────────────────────────────────────────────────
    df_train = df[df["episode"].isin(TRAIN_EPISODES)].reset_index(drop=True)
    df_val   = df[df["episode"].isin(VAL_EPISODES)].reset_index(drop=True)
    df_test  = df[df["episode"].isin(TEST_EPISODES)].reset_index(drop=True)
    info(f"Train: {len(df_train)}  Val: {len(df_val)}  Test: {len(df_test)}")

    # ── Datasets + DataLoaders ────────────────────────────────────────────
    train_ds = SegmentationDataset(df_train, RGB_DIR, MASK_DIR, augment=True)
    val_ds   = SegmentationDataset(df_val,   RGB_DIR, MASK_DIR, augment=False)
    test_ds  = SegmentationDataset(df_test,  RGB_DIR, MASK_DIR, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    ok(f"DataLoaders ready — batch_size={BATCH_SIZE}")

    # ── Model ──────────────────────────────────────────────────────────────
    info("Building TissueSegUNet (MobileNetV3 encoder) …")
    model = TissueSegUNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    ok(f"Model built — {n_params:,} parameters "
       f"({'✓ <5M' if n_params < 5_000_000 else '✗ >5M'})")

    # ── Optimiser + scheduler ──────────────────────────────────────────────
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", factor=0.5, patience=5
    )   # mode="max" because we track IoU (higher = better)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Log CSV ───────────────────────────────────────────────────────────
    log_path   = OUTPUT_DIR / "training_log.csv"
    log_file   = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_loss", "val_loss",
                         "val_iou", "lr"])

    best_val_iou  = 0.0
    best_ckpt     = OUTPUT_DIR / "unet_seg_best.pth"
    prev_lr       = LR

    header(f"\n  Training — {EPOCHS} epochs …\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_iou = run_epoch(model, train_loader, optimiser,
                                          device, train=True)
        val_loss,   val_iou   = run_epoch(model, val_loader,   None,
                                          device, train=False)
        elapsed = time.time() - t0

        scheduler.step(val_iou)
        current_lr = optimiser.param_groups[0]["lr"]
        if current_lr < prev_lr:
            warn(f"  lr reduced: {prev_lr:.2e} → {current_lr:.2e}")
            prev_lr = current_lr

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), best_ckpt)
            best_marker = " ← best"
        else:
            best_marker = ""

        print(f"  epoch {epoch:3d}/{EPOCHS}  "
              f"loss={train_loss:.4f}/{val_loss:.4f}  "
              f"IoU={train_iou:.4f}/{val_iou:.4f}  "
              f"{elapsed:.1f}s{best_marker}")

        log_writer.writerow([epoch, f"{train_loss:.5f}", f"{val_loss:.5f}",
                             f"{val_iou:.4f}", current_lr])
        log_file.flush()

    log_file.close()

    final_ckpt = OUTPUT_DIR / "unet_seg_final.pth"
    torch.save(model.state_dict(), final_ckpt)
    ok(f"Final checkpoint → {final_ckpt}")
    ok(f"Best checkpoint  → {best_ckpt}  (val IoU={best_val_iou:.4f})")

    # ── Test evaluation ────────────────────────────────────────────────────
    header("\n  Test Set Evaluation")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    metrics = evaluate_test(model, test_loader, device)

    print()
    ok(f"Test IoU:  {metrics['iou']:.4f}  "
       f"({'excellent' if metrics['iou']>0.85 else 'good' if metrics['iou']>0.7 else 'acceptable' if metrics['iou']>0.5 else 'needs improvement'})")
    ok(f"Test Dice: {metrics['dice']:.4f}")

    # Save metrics
    metrics_txt = OUTPUT_DIR / "eval_metrics.txt"
    with open(metrics_txt, "w") as f:
        f.write("Phase 3A — Tissue Segmentation — Test Set Metrics\n")
        f.write("=" * 50 + "\n")
        f.write(f"Architecture:  UNet + MobileNetV3-Small encoder\n")
        f.write(f"Parameters:    {n_params:,} (<5M: {'yes' if n_params < 5_000_000 else 'no'})\n")
        f.write(f"Training:      {EPOCHS} epochs, lr={LR}, batch={BATCH_SIZE}\n")
        f.write(f"Best val IoU:  {best_val_iou:.4f}\n\n")
        f.write(f"Test IoU:      {metrics['iou']:.4f}\n")
        f.write(f"Test Dice:     {metrics['dice']:.4f}\n")
    json.dump(metrics, open(OUTPUT_DIR / "eval_metrics.json", "w"), indent=2)
    ok(f"Metrics saved → {metrics_txt}")

    header("\n" + "=" * 62)
    ok("Phase 3A Step 4 — Segmentation training COMPLETE")
    info("Next: visualise_predictions.py — overlay tip + mask on test frames")
    header("=" * 62 + "\n")


if __name__ == "__main__":
    main()