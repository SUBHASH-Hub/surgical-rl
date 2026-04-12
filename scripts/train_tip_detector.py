#!/usr/bin/env python3
"""
Phase 3A — Step 3: Instrument Tip Detector Training
=====================================================
Trains a MobileNetV3-Small backbone + regression head to predict
normalised instrument tip XYZ from a single 480×480 RGB frame.

Architecture choice: MobileNetV3-Small
  - 2.5M parameters — fits GTX 1650 (4GB VRAM) with room to spare
  - Depthwise separable convolutions — 9x fewer MACs than standard conv
  - Used in Moon Surgical ScoPilot for real-time instrument tracking
  - Pretrained on ImageNet — transfers edge/texture features to surgical scene

This script uses TWO-PHASE transfer learning:
  Phase 1 (epochs 1-10):  backbone FROZEN,  head trains, lr=1e-3
  Phase 2 (epochs 11-30): backbone UNFROZEN, full model, lr=1e-4
Reason: freezing first prevents catastrophic forgetting of ImageNet features.

Run from repo root (setup_env.sh already sourced):
    python3 scripts/train_tip_detector.py

Outputs:
    models/tip_detector/
        mobilenetv3_tip_best.pth     ← best validation loss checkpoint
        mobilenetv3_tip_final.pth    ← final epoch checkpoint
        training_log.csv             ← loss per epoch for plotting
        eval_metrics.txt             ← test set metrics (MSE, pixel error)

        
Author: Subhash Arockiadoss        
"""

# ── Standard library ───────────────────────────────────────────────────────
import os
import sys
import csv
import time
import json
from pathlib import Path

# ── Scientific stack ───────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from PIL import Image

# ── PyTorch ────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ─────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# All tunable values live here with explanations.
# If you want to change something, change it here — not buried in the code.
# ─────────────────────────────────────────────────────────────────────────

# --- Paths ---
DATA_DIR   = Path("data/rgb_frames")       # where PNGs + labels.csv live
OUTPUT_DIR = Path("models/tip_detector")   # where checkpoints + logs go

# --- Dataset split (by episode number, NOT random frame shuffle) ---
# Why by episode: consecutive frames are near-duplicates (~1mm tool motion).
# Random shuffle causes data leakage — test frames look almost identical to
# train frames. Episode split = model tested on trajectories it never saw.
TRAIN_EPISODES = list(range(0, 40))   # episodes 0-39  = 80% of data (~5,800 frames)
VAL_EPISODES   = list(range(40, 45))  # episodes 40-44 = 10% of data (~730 frames)
TEST_EPISODES  = list(range(45, 50))  # episodes 45-49 = 10% of data (~730 frames)

# --- Model ---
# MobileNetV3-Small: 2.5M params. MobileNetV3-Large: 5.4M params.
# Small is sufficient for XYZ regression and fits comfortably on GTX 1650.
# Change to "large" if you want to experiment with more capacity.
BACKBONE = "small"   # "small" or "large"

# --- Regression head ---
# Input: 576-dim feature vector from MobileNetV3-Small backbone (after global avg pool)
# Hidden: 128-dim with ReLU + Dropout
# Output: 3 values (tool_x, tool_y, tool_z) passed through tanh → range [-1,1]
# Why tanh: labels are normalised to [-1,1]. Tanh squashes output to same range,
# preventing the model from predicting values like 5.0 or -10.0.
HIDDEN_DIM  = 128
DROPOUT     = 0.3    # regularisation — prevents overfitting on 7284 frames

# --- Training phases ---
# Phase 1: backbone frozen, only head trains. High lr fine for random init.
PHASE1_EPOCHS = 10
PHASE1_LR     = 1e-3   # high lr — head is randomly initialised, needs big steps

# Phase 2: backbone unfrozen, full model trains end-to-end.
# CRITICAL: use 10x lower lr than Phase 1.
# Backbone has good ImageNet features — large gradients would destroy them.
# Small lr = gentle fine-tuning to adapt features to surgical scene.
PHASE2_EPOCHS = 20
PHASE2_LR     = 1e-4   # 10x lower — preserves backbone knowledge

TOTAL_EPOCHS  = PHASE1_EPOCHS + PHASE2_EPOCHS  # 30 total

# --- DataLoader ---
# batch_size=32: fits GTX 1650 VRAM with fp32. Reduce to 16 if OOM error.
BATCH_SIZE   = 32
NUM_WORKERS  = 4       # parallel data loading threads. Match to CPU cores.
PIN_MEMORY   = True    # faster CPU→GPU transfer when using CUDA

# --- Misc ---
SEED             = 42      # reproducibility
LOG_INTERVAL     = 5       # print loss every N batches within an epoch

# --- Resume flag ---
# Set True to skip Phase 1 and jump straight to Phase 2.
# Use this when Phase 1 completed successfully but Phase 2 crashed.
# The best checkpoint saved during Phase 1 will be loaded automatically.
RESUME_FROM_PHASE2 = True

# ─────────────────────────────────────────────────────────────────────────
# COLOUR HELPERS
# ─────────────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
CYAN   = "\033[96m"; BOLD = "\033[1m"; RESET  = "\033[0m"

def ok(msg):     print(f"{GREEN}  ✓  {msg}{RESET}")
def fail(msg):   print(f"{RED}  ✗  {msg}{RESET}")
def info(msg):   print(f"{CYAN}  ·  {msg}{RESET}")
def warn(msg):   print(f"{YELLOW}  !  {msg}{RESET}")
def header(msg): print(f"\n{BOLD}{msg}{RESET}")


# ─────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────

class SurgicalTipDataset(Dataset):
    """
    PyTorch Dataset for instrument tip localisation.

    Each sample is:
        image : (3, 224, 224) float32 tensor  — normalised RGB frame
        label : (3,)          float32 tensor  — [tool_x, tool_y, tool_z]

    The image is resized from 480×480 to 224×224.
    Why 224: MobileNetV3 was pretrained on ImageNet at 224×224.
    Using the same resolution means the pretrained feature maps make
    spatial sense — no rescaling mismatch in the convolutional layers.

    Normalisation values [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]
    are the ImageNet channel means/stds. We use them because the backbone
    weights were computed with data normalised this way — using different
    values would shift the activation distributions and degrade transfer.
    """

    # ImageNet normalisation — MUST match what the pretrained backbone expects
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(self, df: pd.DataFrame, data_dir: Path, augment: bool = False):
        """
        Args:
            df       : subset of labels.csv for this split
            data_dir : directory containing PNG files
            augment  : if True, apply random augmentations (training only)
        """
        self.df       = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.augment  = augment

        # Training transform: resize → optional augment → tensor → normalise
        # Augmentations simulate camera variation that would occur in real
        # endoscopic video: brightness changes (OR lighting), slight rotations
        # (scope roll), horizontal flip (mirrored anatomy).
        # We do NOT augment val/test — they must reflect true distribution.
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                # Brightness/contrast jitter: simulates endoscope light variation
                # Values chosen to be mild — surgical scenes have constrained lighting
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(self.data_dir / row["filename"]).convert("RGB")
        image = self.transform(img)
        # Label: 3D normalised XYZ — values already in [-1,1] from TissueRetractionV2
        label = torch.tensor(
            [row["tool_x"], row["tool_y"], row["tool_z"]],
            dtype=torch.float32
        )
        return image, label


# ─────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────

class InstrumentTipDetector(nn.Module):
    """
    MobileNetV3-Small backbone + custom regression head.

    The original MobileNetV3 classifier head outputs 1000 logits (ImageNet
    classes). We replace it with a regression head that outputs 3 values.

    Backbone feature dim: 576  (MobileNetV3-Small after adaptive avg pool)
    Head: 576 → 128 → 3 → tanh

    Why tanh on output: labels are in [-1,1]. Tanh guarantees the model
    cannot output values outside this range, which stabilises training and
    prevents loss explosion early on when weights are random.
    """

    def __init__(self, backbone: str = "small", hidden_dim: int = 128,
                 dropout: float = 0.3):
        super().__init__()

        # Load pretrained backbone
        # weights=DEFAULT downloads ImageNet-1K weights (~10MB) if not cached.
        # These weights encode 1.2M images of real-world visual features —
        # edges, textures, shapes — that transfer well to surgical scenes.
        if backbone == "small":
            base = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.DEFAULT
            )
            feat_dim = 576   # MobileNetV3-Small classifier input dimension
        else:
            base = models.mobilenet_v3_large(
                weights=models.MobileNet_V3_Large_Weights.DEFAULT
            )
            feat_dim = 960   # MobileNetV3-Large classifier input dimension

        # Keep everything except the classifier (last layer)
        # base.features: all convolutional + SE blocks
        # base.avgpool:  adaptive average pooling → (batch, 576, 1, 1)
        self.backbone = nn.Sequential(base.features, base.avgpool)

        # New regression head — randomly initialised, trained from scratch
        self.head = nn.Sequential(
            nn.Flatten(),                        # (batch, 576, 1, 1) → (batch, 576)
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 3),            # 3 outputs: tool_x, tool_y, tool_z
            nn.Tanh()                            # clamp output to [-1, 1]
        )

    def forward(self, x):
        features = self.backbone(x)   # (batch, 576, 1, 1)
        xyz      = self.head(features)  # (batch, 3)
        return xyz

    def freeze_backbone(self):
        """Freeze all backbone parameters — only head trains."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        info("Backbone FROZEN — only regression head will train")

    def unfreeze_backbone(self):
        """Unfreeze backbone for end-to-end fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        info("Backbone UNFROZEN — full model will fine-tune")

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimiser, device, train: bool):
    """
    Run one epoch. Returns mean loss over all batches.

    criterion = MSELoss: measures mean squared error between predicted XYZ
    and ground-truth XYZ. In normalised space, an MSE of 0.01 corresponds
    to a mean error of 0.1 units = ~10% of the workspace width.
    """
    model.train(train)
    total_loss = 0.0
    n_batches  = 0

    # torch.no_grad() in eval mode: skip gradient computation → 2x faster,
    # less VRAM. Never call this during training — gradients would be lost.
    context = torch.no_grad() if not train else torch.enable_grad()

    with context:
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking=True)  # non_blocking=True
            labels = labels.to(device, non_blocking=True)  # uses pin_memory

            if train:
                optimiser.zero_grad()   # clear gradients from previous batch

            preds = model(images)       # forward pass
            loss  = criterion(preds, labels)  # MSE(predicted_xyz, true_xyz)

            if train:
                loss.backward()         # backprop — compute gradients
                # Gradient clipping: prevents exploding gradients when backbone
                # is unfrozen. Max norm=1.0 is standard for fine-tuning.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()        # update weights

            total_loss += loss.item()
            n_batches  += 1

            if train and (batch_idx % LOG_INTERVAL == 0):
                print(f"    batch {batch_idx:4d}/{len(loader)}  "
                      f"loss={loss.item():.5f}", end="\r")

    return total_loss / n_batches


def evaluate_test(model, loader, device):
    """
    Compute per-axis MAE and pixel error on the test set.

    MAE (Mean Absolute Error) per axis shows which dimension is hardest:
    - X (horizontal): usually easiest — clear visual cue in 2D image
    - Y (vertical):   usually medium
    - Z (depth):      usually hardest — monocular depth ambiguity.
                      A 2D camera cannot directly see depth; the model must
                      infer Z from tool size, foreshortening, shading.

    Pixel error: convert normalised error to pixels.
    Workspace spans ~2 units ([-1,1]). Image is 480px wide.
    So 1 normalised unit = 240 pixels.
    pixel_error = mean_xyz_error × 240
    """
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds  = model(images).cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    preds_np  = torch.cat(all_preds).numpy()
    labels_np = torch.cat(all_labels).numpy()

    mae_total = np.mean(np.abs(preds_np - labels_np))
    mae_x     = np.mean(np.abs(preds_np[:, 0] - labels_np[:, 0]))
    mae_y     = np.mean(np.abs(preds_np[:, 1] - labels_np[:, 1]))
    mae_z     = np.mean(np.abs(preds_np[:, 2] - labels_np[:, 2]))

    # 480px image spans 2.0 normalised units → 240 px per unit
    px_per_unit = 240.0
    pixel_error = mae_total * px_per_unit

    return {
        "mae_total":   float(mae_total),
        "mae_x":       float(mae_x),
        "mae_y":       float(mae_y),
        "mae_z":       float(mae_z),
        "pixel_error": float(pixel_error),
    }


# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    header("=" * 62)
    print("  Phase 3A — Instrument Tip Detector Training")
    header("=" * 62)

    # ── Device ────────────────────────────────────────────────────────────
    # Use CUDA (GTX 1650) if available, otherwise CPU.
    # Training on CPU is ~20x slower — CUDA is strongly preferred here.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        ok(f"GPU: {torch.cuda.get_device_name(0)}  "
           f"VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")
    else:
        warn("No GPU found — training on CPU (will be slow)")
    info(f"Device: {device}")

    # ── Load labels ───────────────────────────────────────────────────────
    info("Loading labels.csv …")
    df = pd.read_csv(DATA_DIR / "labels.csv")
    ok(f"Loaded {len(df)} rows — episodes {df.episode.min()}–{df.episode.max()}")

    # ── Split by episode ───────────────────────────────────────────────────
    df_train = df[df["episode"].isin(TRAIN_EPISODES)].reset_index(drop=True)
    df_val   = df[df["episode"].isin(VAL_EPISODES)].reset_index(drop=True)
    df_test  = df[df["episode"].isin(TEST_EPISODES)].reset_index(drop=True)

    info(f"Train: {len(df_train)} frames  (ep 0–39)")
    info(f"Val:   {len(df_val)} frames  (ep 40–44)")
    info(f"Test:  {len(df_test)} frames  (ep 45–49)")

    # ── Datasets + DataLoaders ────────────────────────────────────────────
    train_ds = SurgicalTipDataset(df_train, DATA_DIR, augment=True)
    val_ds   = SurgicalTipDataset(df_val,   DATA_DIR, augment=False)
    test_ds  = SurgicalTipDataset(df_test,  DATA_DIR, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    ok(f"DataLoaders ready — batch_size={BATCH_SIZE}  workers={NUM_WORKERS}")

    # ── Model ──────────────────────────────────────────────────────────────
    info(f"Building MobileNetV3-{BACKBONE} + regression head …")
    model = InstrumentTipDetector(
        backbone=BACKBONE,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT
    ).to(device)
    ok(f"Model built — total params: "
       f"{sum(p.numel() for p in model.parameters()):,}")

    # ── Loss function ──────────────────────────────────────────────────────
    # MSELoss: penalises large errors more than small ones (squared).
    # Alternative: L1Loss (MAE) — less sensitive to outliers.
    # MSE is standard for 3D coordinate regression.
    criterion = nn.MSELoss()

    # ── Output directory ───────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Training log CSV ───────────────────────────────────────────────────
    log_path  = OUTPUT_DIR / "training_log.csv"
    log_file  = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "phase", "train_loss", "val_loss",
                         "trainable_params", "lr"])

    best_val_loss  = float("inf")
    best_ckpt_path = OUTPUT_DIR / "mobilenetv3_tip_best.pth"

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1 — backbone frozen, head trains only
    # ══════════════════════════════════════════════════════════════════════
    if RESUME_FROM_PHASE2:
        # Phase 1 already completed — load best checkpoint and skip to Phase 2
        warn("RESUME_FROM_PHASE2=True — skipping Phase 1, loading best checkpoint")
        if not best_ckpt_path.exists():
            fail(f"No checkpoint found at {best_ckpt_path} — set RESUME_FROM_PHASE2=False to run from scratch")
            sys.exit(1)
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        # Approximate best val loss from Phase 1 output (epoch 8 = 0.02315)
        best_val_loss = 0.02315
        ok(f"Loaded Phase 1 best checkpoint → {best_ckpt_path}")
    else:
        header(f"\n  Phase 1 — Warmup (epochs 1–{PHASE1_EPOCHS})  backbone FROZEN")
        model.freeze_backbone()
        info(f"Trainable parameters: {model.count_trainable_params():,}  "
             f"(head only — ~{HIDDEN_DIM * 576 + HIDDEN_DIM * 3:,} weights)")

        optimiser = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=PHASE1_LR
        )

        for epoch in range(1, PHASE1_EPOCHS + 1):
            t0 = time.time()
            train_loss = run_epoch(model, train_loader, criterion, optimiser,
                                   device, train=True)
            val_loss   = run_epoch(model, val_loader,   criterion, None,
                                   device, train=False)
            elapsed = time.time() - t0

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_ckpt_path)
                best_marker = " ← best"
            else:
                best_marker = ""

            print(f"  epoch {epoch:3d}/{TOTAL_EPOCHS}  "
                  f"[P1-frozen]  "
                  f"train={train_loss:.5f}  val={val_loss:.5f}  "
                  f"{elapsed:.1f}s{best_marker}")

            log_writer.writerow([epoch, "frozen", f"{train_loss:.6f}",
                                 f"{val_loss:.6f}", model.count_trainable_params(),
                                 PHASE1_LR])
            log_file.flush()

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2 — backbone unfrozen, full fine-tuning
    # ══════════════════════════════════════════════════════════════════════
    header(f"\n  Phase 2 — Fine-tune (epochs {PHASE1_EPOCHS+1}–{TOTAL_EPOCHS})"
           f"  backbone UNFROZEN")
    model.unfreeze_backbone()
    info(f"Trainable parameters: {model.count_trainable_params():,}  "
         f"(full model)")

    # Re-create optimiser with new lr covering ALL parameters
    optimiser = torch.optim.Adam(model.parameters(), lr=PHASE2_LR)

    # ReduceLROnPlateau: if val_loss doesn't improve for 5 epochs, halve lr.
    # Prevents oscillating around a minimum — common in fine-tuning.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=5
    )

    prev_lr = PHASE2_LR   # track lr so we can print when it reduces

    for epoch in range(PHASE1_EPOCHS + 1, TOTAL_EPOCHS + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, criterion, optimiser,
                               device, train=True)
        val_loss   = run_epoch(model, val_loader,   criterion, None,
                               device, train=False)
        elapsed = time.time() - t0

        scheduler.step(val_loss)
        current_lr = optimiser.param_groups[0]["lr"]
        if current_lr < prev_lr:
            warn(f"  lr reduced: {prev_lr:.2e} → {current_lr:.2e}")
            prev_lr = current_lr

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_ckpt_path)
            best_marker = " ← best"
        else:
            best_marker = ""

        print(f"  epoch {epoch:3d}/{TOTAL_EPOCHS}  "
              f"[P2-tuning]  "
              f"train={train_loss:.5f}  val={val_loss:.5f}  "
              f"lr={current_lr:.2e}  {elapsed:.1f}s{best_marker}")

        log_writer.writerow([epoch, "finetune", f"{train_loss:.6f}",
                             f"{val_loss:.6f}", model.count_trainable_params(),
                             current_lr])
        log_file.flush()

    log_file.close()

    # ── Save final checkpoint ──────────────────────────────────────────────
    final_ckpt_path = OUTPUT_DIR / "mobilenetv3_tip_final.pth"
    torch.save(model.state_dict(), final_ckpt_path)
    ok(f"Final checkpoint saved → {final_ckpt_path}")
    ok(f"Best checkpoint saved  → {best_ckpt_path}  (val_loss={best_val_loss:.5f})")

    # ── Test set evaluation ────────────────────────────────────────────────
    header("\n  Test Set Evaluation (loading best checkpoint)")
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    metrics = evaluate_test(model, test_loader, device)

    print()
    ok(f"Mean XYZ error (MAE):    {metrics['mae_total']:.4f} normalised units")
    ok(f"Pixel localisation error: {metrics['pixel_error']:.1f} px  "
       f"(on 480×480 image)")
    ok(f"  X-axis MAE: {metrics['mae_x']:.4f}")
    ok(f"  Y-axis MAE: {metrics['mae_y']:.4f}")
    ok(f"  Z-axis MAE: {metrics['mae_z']:.4f}  ← depth, expect highest error")

    # Save metrics to file (Phase 3A deliverable)
    metrics_path = OUTPUT_DIR / "eval_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("Phase 3A — Instrument Tip Detector — Test Set Metrics\n")
        f.write("=" * 54 + "\n")
        f.write(f"Dataset:         7284 frames, 50 episodes\n")
        f.write(f"Backbone:        MobileNetV3-{BACKBONE}\n")
        f.write(f"Training epochs: {TOTAL_EPOCHS} "
                f"({PHASE1_EPOCHS} frozen + {PHASE2_EPOCHS} fine-tune)\n")
        f.write(f"Best val loss:   {best_val_loss:.5f} (MSE)\n\n")
        f.write(f"Test MAE (total): {metrics['mae_total']:.4f} normalised units\n")
        f.write(f"Test MAE X:       {metrics['mae_x']:.4f}\n")
        f.write(f"Test MAE Y:       {metrics['mae_y']:.4f}\n")
        f.write(f"Test MAE Z:       {metrics['mae_z']:.4f}\n")
        f.write(f"Pixel error:      {metrics['pixel_error']:.1f} px  "
                f"(on 480x480 image)\n")
        json.dump(metrics, open(OUTPUT_DIR / "eval_metrics.json", "w"), indent=2)

    ok(f"Metrics saved → {metrics_path}")

    header("\n" + "=" * 62)
    ok("Phase 3A Step 3 — Tip detector training COMPLETE")
    info("Next: train_segmentation.py — tissue boundary UNet")
    header("=" * 62 + "\n")


if __name__ == "__main__":
    main()