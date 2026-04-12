#!/usr/bin/env python3
"""
Phase 3A — Step 5: Visualise Predictions
==========================================
Loads both trained models and overlays their predictions on 10 test frames.
Saves demo images for GitHub README and Phase 3A documentation.

What this script produces for each of 10 test frames:
  - GREEN CROSSHAIR  : predicted instrument tip position (MobileNetV3)
  - RED DOT          : ground-truth instrument tip position (from labels.csv)
  - CYAN OVERLAY     : predicted tissue boundary mask (UNet, semi-transparent)
  - WHITE CONTOUR    : tissue boundary edge (UNet mask outline)
  - Text annotations : predicted XYZ, ground-truth XYZ, pixel error

Output layout per image (side by side, 3 panels):
  Panel 1: Original RGB frame
  Panel 2: Tip detection overlay
  Panel 3: Segmentation overlay

Saves:
  docs/assets/predictions/
    frame_01_pred.png  through  frame_10_pred.png   ← individual overlays
    prediction_grid.png                              ← 2×5 grid for README

Run from repo root:
    python3 scripts/visualise_predictions.py

Author: Subhash Arockiadoss
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

sys.path.insert(0, '.')

# ─────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────

RGB_DIR       = Path("data/rgb_frames")
MASK_DIR      = Path("data/seg_masks")
TIP_CKPT      = Path("models/tip_detector/mobilenetv3_tip_best.pth")
SEG_CKPT      = Path("models/segmentation/unet_seg_best.pth")
OUTPUT_DIR    = Path("docs/assets/predictions")
N_FRAMES      = 10       # number of test frames to visualise
IMG_SIZE      = 480      # original image size
SEG_SIZE      = 256      # UNet input size
TIP_SIZE      = 224      # MobileNetV3 input size
TEST_EPISODES = list(range(45, 50))   # same test split as training

# Colour scheme (RGB tuples)
COLOUR_PRED_TIP    = (0,   255, 0)    # green — predicted tip
COLOUR_GT_TIP      = (255, 0,   0)    # red   — ground truth tip
COLOUR_SEG_OVERLAY = (0,   200, 255)  # cyan  — tissue mask overlay
COLOUR_CONTOUR     = (255, 255, 255)  # white — tissue boundary edge
CROSSHAIR_RADIUS   = 12              # pixels for crosshair arms
DOT_RADIUS         = 6               # pixels for GT dot

# ─────────────────────────────────────────────────────────────────────────
# COLOUR HELPERS
# ─────────────────────────────────────────────────────────────────────────
GREEN = "\033[92m"; CYAN = "\033[96m"; BOLD = "\033[1m"; RESET = "\033[0m"
def ok(m):     print(f"{GREEN}  ✓  {m}{RESET}")
def info(m):   print(f"{CYAN}  ·  {m}{RESET}")
def header(m): print(f"\n{BOLD}{m}{RESET}")


# ─────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS (must match training scripts exactly)
# ─────────────────────────────────────────────────────────────────────────

class InstrumentTipDetector(nn.Module):
    """MobileNetV3-Small + regression head. Matches train_tip_detector.py."""
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.backbone = nn.Sequential(base.features, base.avgpool)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 3),
            nn.Tanh()
        )
    def forward(self, x):
        return self.head(self.backbone(x))


class ConvBnRelu(nn.Module):
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
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            ConvBnRelu(out_ch + skip_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class TissueSegUNet(nn.Module):
    """UNet with MobileNetV3-Small encoder. Matches train_segmentation.py."""
    def __init__(self):
        super().__init__()
        base  = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        feats = base.features
        self.enc0 = feats[0]
        self.enc1 = feats[1:4]
        self.enc2 = feats[4:9]
        self.enc3 = feats[9:]
        self.dec2 = UpBlock(576, 48, 128)
        self.dec1 = UpBlock(128, 24, 64)
        self.dec0 = UpBlock(64,  16, 32)
        self.up_final = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.head = nn.Sequential(
            ConvBnRelu(16, 16),
            nn.Conv2d(16, 1, kernel_size=1),
        )
    def forward(self, x):
        s0 = self.enc0(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        d2 = self.dec2(s3, s2)
        d1 = self.dec1(d2, s1)
        d0 = self.dec0(d1, s0)
        return self.head(self.up_final(d0))


# ─────────────────────────────────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

tip_transform = transforms.Compose([
    transforms.Resize((TIP_SIZE, TIP_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

seg_transform = transforms.Compose([
    transforms.Resize((SEG_SIZE, SEG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ─────────────────────────────────────────────────────────────────────────
# COORDINATE CONVERSION
# ─────────────────────────────────────────────────────────────────────────

def ndc_to_pixel(ndc_x: float, ndc_y: float,
                 img_w: int = IMG_SIZE, img_h: int = IMG_SIZE):
    """
    Convert normalised [-1,1] XY coordinates to pixel coordinates.

    The tip detector predicts tool_x and tool_y in the same normalised
    space as the labels.csv values. These are workspace coordinates,
    not image coordinates. To display them on the image we apply a
    linear mapping calibrated to the observed workspace bounds.

    Workspace XY range from labels.csv: approximately [-1, +1]
    This maps linearly to image pixel space [0, 480].

    u = (tool_x + 1) / 2 * img_w
    v = (1 - tool_y) / 2 * img_h   ← Y flipped: +Y in workspace = up = low V
    """
    u = int((ndc_x + 1) / 2 * img_w)
    v = int((1 - ndc_y) / 2 * img_h)
    u = max(0, min(img_w - 1, u))
    v = max(0, min(img_h - 1, v))
    return u, v


# ─────────────────────────────────────────────────────────────────────────
# DRAWING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────

def draw_crosshair(draw: ImageDraw.Draw, u: int, v: int,
                   colour: tuple, radius: int = CROSSHAIR_RADIUS,
                   width: int = 2):
    """Draw a + crosshair at pixel (u, v)."""
    draw.line([(u - radius, v), (u + radius, v)], fill=colour, width=width)
    draw.line([(u, v - radius), (u, v + radius)], fill=colour, width=width)
    draw.ellipse([(u-3, v-3), (u+3, v+3)], fill=colour)


def draw_filled_dot(draw: ImageDraw.Draw, u: int, v: int,
                    colour: tuple, radius: int = DOT_RADIUS):
    """Draw a filled circle at pixel (u, v)."""
    draw.ellipse([(u-radius, v-radius), (u+radius, v+radius)], fill=colour)


def overlay_segmentation(img_rgb: np.ndarray,
                          mask: np.ndarray,
                          alpha: float = 0.35) -> np.ndarray:
    """
    Blend a binary mask over the RGB image as a semi-transparent cyan overlay.
    Also draws the mask contour as a white edge.

    alpha: transparency of overlay (0=invisible, 1=opaque)
    """
    result = img_rgb.copy().astype(np.float32)

    # Cyan overlay on tissue pixels
    tissue = mask > 127
    overlay_colour = np.array(COLOUR_SEG_OVERLAY, dtype=np.float32)
    for c in range(3):
        result[:, :, c] = np.where(
            tissue,
            result[:, :, c] * (1 - alpha) + overlay_colour[c] * alpha,
            result[:, :, c]
        )

    result = np.clip(result, 0, 255).astype(np.uint8)

    # Draw white contour on tissue boundary using simple edge detection
    # Erode mask and XOR with original to get 1-pixel boundary
    try:
        import cv2
        kernel  = np.ones((3, 3), np.uint8)
        eroded  = cv2.erode(mask, kernel, iterations=1)
        contour = mask - eroded   # boundary pixels
        result[contour > 0] = COLOUR_CONTOUR
    except ImportError:
        pass   # skip contour if cv2 not available

    return result


def add_text_annotation(img_pil: Image.Image,
                        pred_xyz: tuple, gt_xyz: tuple,
                        pixel_err: float) -> Image.Image:
    """Add prediction vs ground truth text to bottom of image."""
    draw = ImageDraw.Draw(img_pil)
    w, h = img_pil.size

    # Semi-transparent black background strip at bottom
    strip_h = 52
    strip   = Image.new("RGBA", (w, strip_h), (0, 0, 0, 180))
    img_pil = img_pil.convert("RGBA")
    img_pil.paste(strip, (0, h - strip_h), strip)
    img_pil = img_pil.convert("RGB")
    draw    = ImageDraw.Draw(img_pil)

    # Text lines
    px, py, pz = pred_xyz
    gx, gy, gz = gt_xyz
    line1 = f"Pred XYZ: ({px:+.3f}, {py:+.3f}, {pz:+.3f})"
    line2 = f"GT   XYZ: ({gx:+.3f}, {gy:+.3f}, {gz:+.3f})  err={pixel_err:.1f}px"

    draw.text((8, h - strip_h + 6),  line1, fill=(0, 255, 100))
    draw.text((8, h - strip_h + 28), line2, fill=(255, 100, 100))

    return img_pil


# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────

def main():
    header("=" * 62)
    print("  Phase 3A — Step 5: Visualise Predictions")
    header("=" * 62)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info(f"Device: {device}")

    # ── Load tip detector ─────────────────────────────────────────────────
    info("Loading tip detector …")
    tip_model = InstrumentTipDetector().to(device)
    tip_model.load_state_dict(
        torch.load(TIP_CKPT, map_location=device))
    tip_model.eval()
    ok(f"Tip detector loaded ← {TIP_CKPT}")

    # ── Load segmentation model ───────────────────────────────────────────
    info("Loading segmentation UNet …")
    seg_model = TissueSegUNet().to(device)
    seg_model.load_state_dict(
        torch.load(SEG_CKPT, map_location=device))
    seg_model.eval()
    ok(f"Segmentation UNet loaded ← {SEG_CKPT}")

    # ── Load labels CSV and pick 10 test frames ───────────────────────────
    info("Loading labels.csv …")
    df = pd.read_csv(RGB_DIR / "labels.csv")

    # Use test episodes only (ep 45-49) — frames never seen during training
    df_test = df[df["episode"].isin(TEST_EPISODES)].reset_index(drop=True)

    # Pick 10 frames spread across different episodes and steps
    # Sample 2 frames per episode for variety
    selected_rows = []
    for ep in TEST_EPISODES:
        ep_rows = df_test[df_test["episode"] == ep]
        if len(ep_rows) >= 2:
            # Pick one from first half, one from second half of episode
            n = len(ep_rows)
            selected_rows.append(ep_rows.iloc[n // 4])
            selected_rows.append(ep_rows.iloc[3 * n // 4])
    selected = pd.DataFrame(selected_rows).head(N_FRAMES).reset_index(drop=True)
    ok(f"Selected {len(selected)} test frames from episodes {TEST_EPISODES}")

    # ── Process each frame ────────────────────────────────────────────────
    header(f"\n  Processing {len(selected)} frames …\n")

    px_per_unit = 240.0   # 480px image spans 2 normalised units
    individual_images = []

    for i, row in selected.iterrows():
        frame_path = RGB_DIR / row["filename"]
        info(f"Frame {i+1:2d}/{len(selected)}: {row['filename'][:50]}")

        # Load original RGB image
        img_rgb = np.array(Image.open(frame_path).convert("RGB"))
        img_pil = Image.fromarray(img_rgb)

        # ── Run tip detector ───────────────────────────────────────────
        with torch.no_grad():
            tip_input  = tip_transform(img_pil).unsqueeze(0).to(device)
            pred_xyz   = tip_model(tip_input).squeeze().cpu().numpy()
        pred_x, pred_y, pred_z = float(pred_xyz[0]), float(pred_xyz[1]), float(pred_xyz[2])

        # Ground truth from CSV
        gt_x = float(row["tool_x"])
        gt_y = float(row["tool_y"])
        gt_z = float(row["tool_z"])

        # Pixel error (2D — X and Y only, Z is depth)
        mae_xy    = (abs(pred_x - gt_x) + abs(pred_y - gt_y)) / 2
        pixel_err = mae_xy * px_per_unit

        # Convert normalised coords to pixel coords for display
        pred_u, pred_v = ndc_to_pixel(pred_x, pred_y)
        gt_u,   gt_v   = ndc_to_pixel(gt_x,   gt_y)

        # ── Run segmentation model ─────────────────────────────────────
        with torch.no_grad():
            seg_input = seg_transform(img_pil).unsqueeze(0).to(device)
            seg_logit = seg_model(seg_input)
            seg_prob  = torch.sigmoid(seg_logit).squeeze().cpu().numpy()

        # Resize predicted mask from 256×256 back to 480×480
        seg_mask_256 = (seg_prob > 0.5).astype(np.uint8) * 255
        seg_mask_480 = np.array(
            Image.fromarray(seg_mask_256).resize(
                (IMG_SIZE, IMG_SIZE), Image.NEAREST))

        # ── Build 3-panel visualisation ────────────────────────────────
        panel_w = IMG_SIZE
        panel_h = IMG_SIZE
        combined = Image.new("RGB", (panel_w * 3 + 4, panel_h), (30, 30, 30))

        # Panel 1: Original frame
        combined.paste(img_pil, (0, 0))

        # Panel 2: Tip detection overlay
        tip_overlay = img_rgb.copy()
        tip_pil     = Image.fromarray(tip_overlay)
        draw_tip    = ImageDraw.Draw(tip_pil)
        draw_crosshair(draw_tip, pred_u, pred_v, COLOUR_PRED_TIP)
        draw_filled_dot(draw_tip, gt_u, gt_v, COLOUR_GT_TIP)
        combined.paste(tip_pil, (panel_w + 2, 0))

        # Panel 3: Segmentation overlay
        seg_vis = overlay_segmentation(img_rgb, seg_mask_480)
        # Also draw predicted tip on seg panel
        seg_pil  = Image.fromarray(seg_vis)
        draw_seg = ImageDraw.Draw(seg_pil)
        draw_crosshair(draw_seg, pred_u, pred_v, COLOUR_PRED_TIP, radius=8)
        combined.paste(seg_pil, (panel_w * 2 + 4, 0))

        # Add text annotation bar at bottom of full combined image
        combined = add_text_annotation(
            combined,
            pred_xyz=(pred_x, pred_y, pred_z),
            gt_xyz=(gt_x, gt_y, gt_z),
            pixel_err=pixel_err
        )

        # Add column headers at top
        header_bar = Image.new("RGB", (panel_w * 3 + 4, 24), (50, 50, 50))
        draw_hdr   = ImageDraw.Draw(header_bar)
        draw_hdr.text((panel_w // 2 - 40,        4), "Original",       fill=(220, 220, 220))
        draw_hdr.text((panel_w + 2 + panel_w // 2 - 55, 4), "Tip detection",  fill=(0, 255, 100))
        draw_hdr.text((panel_w * 2 + 4 + panel_w // 2 - 60, 4), "Segmentation",   fill=(0, 200, 255))

        final_frame = Image.new("RGB", (panel_w * 3 + 4, panel_h + 24 + 52), (20, 20, 20))
        final_frame.paste(header_bar, (0, 0))
        final_frame.paste(combined, (0, 24))

        # Save individual frame
        out_path = OUTPUT_DIR / f"frame_{i+1:02d}_pred.png"
        final_frame.save(out_path)
        individual_images.append(final_frame)

        ok(f"  Saved → {out_path}  "
           f"pred=({pred_x:+.3f},{pred_y:+.3f})  "
           f"gt=({gt_x:+.3f},{gt_y:+.3f})  "
           f"err={pixel_err:.1f}px")

    # ── Build 2×5 grid for GitHub README ─────────────────────────────────
    info("Building prediction grid for README …")

    # Thumbnail each frame to fit in grid
    thumb_w = 640
    thumb_h = individual_images[0].size[1] * thumb_w // individual_images[0].size[0]
    thumbs  = [img.resize((thumb_w, thumb_h), Image.LANCZOS)
               for img in individual_images]

    rows, cols = 2, 5
    grid_w = cols * thumb_w + (cols - 1) * 4
    grid_h = rows * thumb_h + (rows - 1) * 4
    grid   = Image.new("RGB", (grid_w, grid_h), (20, 20, 20))

    for idx, thumb in enumerate(thumbs):
        row = idx // cols
        col = idx  % cols
        x   = col * (thumb_w + 4)
        y   = row * (thumb_h + 4)
        grid.paste(thumb, (x, y))

    grid_path = OUTPUT_DIR / "prediction_grid.png"
    grid.save(grid_path)
    ok(f"Grid saved → {grid_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    header("\n" + "=" * 62)
    ok("Phase 3A Step 5 — Visualisation COMPLETE")
    ok(f"Individual frames: {OUTPUT_DIR}/frame_01_pred.png … frame_10_pred.png")
    ok(f"README grid:       {grid_path}")
    print()
    info("Files saved to docs/assets/predictions/ — add to GitHub README:")
    info("  ![Predictions](docs/assets/predictions/prediction_grid.png)")
    print()
    info("Phase 3A is now COMPLETE. All 5 sub-steps done:")
    info("  3A-i   Camera verification      ✓")
    info("  3A-ii  RGB frame collection     ✓  7284 frames")
    info("  3A-iii Tip detector training    ✓  5.1px MAE")
    info("  3A-iv  Segmentation training    ✓  IoU=1.000 (sim)")
    info("  3A-v   Visualisation overlays   ✓  10 demo frames")
    header("=" * 62 + "\n")


if __name__ == "__main__":
    main()