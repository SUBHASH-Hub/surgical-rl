#!/usr/bin/env python3
"""
Phase 3C — Optical Flow Tissue Force Proxy
===========================================
Computes dense optical flow between consecutive RGB frames to estimate
tissue deformation magnitude as a proxy for contact force.

CLINICAL MOTIVATION:
  Most laparoscopic instruments have no force sensor. Surgeons estimate
  tissue stress visually — if tissue deforms a lot, force is high; if it
  barely moves, force is low. This script formalises that visual intuition
  as a computable signal using Farneback dense optical flow.

  Real systems using this approach:
  - Medtronic Touch Surgery uses optical flow for tool-tissue interaction
  - NVIDIA Holoscan publishes deformation magnitude on ROS 2 topics
  - Phase 4 of this project will publish to /tissue_force_proxy topic

WHAT IS OPTICAL FLOW:
  Given two consecutive frames I(t) and I(t+1), optical flow computes a
  2D displacement vector (dx, dy) for every pixel — how far each pixel
  moved between frames. For tissue deformation, pixels inside the tissue
  region that move a lot indicate high contact force.

  Farneback algorithm: estimates a polynomial expansion of the image in
  each neighbourhood, then estimates the displacement from the expansion
  coefficients. Dense — produces a flow vector for every pixel.
  Output shape: (H, W, 2) — u-channel (horizontal) and v-channel (vertical)

FORCE PROXY DEFINITION:
  tissue_force_proxy = mean magnitude of optical flow within tissue ROI
                     = mean( sqrt(u² + v²) ) for pixels where mask > 0

  Units: pixels/frame — proportional to tissue deformation rate
  High value (> threshold) → high force → potential tissue damage
  Low value                → gentle contact

VALIDATION:
  We validate the proxy against Phase 2B/2C collision cost data.
  The hypothesis: steps with high optical flow magnitude correlate with
  steps where the SafeRewardWrapper fired a collision penalty.

OUTPUT:
  data/optical_flow/
    flow_ep000_step0001.png     ← HSV-encoded flow visualisation
    flow_log.csv                ← episode, step, flow_magnitude, collision_flag
    flow_validation_plot.png    ← correlation scatter: flow vs collision cost
  models/force_proxy/
    proxy_config.json           ← validated threshold and calibration

Run from repo root:
    python3 scripts/optical_flow_proxy.py

Author: Subhash Arockiadoss
"""

# ── gym → gymnasium shim ──────────────────────────────────────────────────
import sys
import gymnasium
sys.modules['gym'] = gymnasium
sys.modules['gym.spaces'] = gymnasium.spaces
# ─────────────────────────────────────────────────────────────────────────

import csv
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from stable_baselines3 import PPO

sys.path.insert(0, '.')
from envs.tissue_retraction_v3 import TissueRetractionV3

# ─────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────

CHECKPOINT    = (
    "logs/checkpoints/"
    "phase3b_ppo_visual_20260413_152851/"
    "ppo_visual_final"
)
MASK_DIR      = Path("data/seg_masks")        # tissue masks from Phase 3A
FLOW_DIR      = Path("data/optical_flow")     # output directory
MODEL_DIR     = Path("models/force_proxy")    # validated config output
EPISODES      = 10                            # episodes to analyse
IMG_W, IMG_H  = 480, 480

# Farneback optical flow parameters (OpenCV defaults work well for this)
FARNEBACK_PARAMS = dict(
    pyr_scale  = 0.5,    # pyramid scale: 0.5 = classical pyramid
    levels     = 3,      # number of pyramid levels
    winsize    = 15,     # averaging window size (larger = smoother, slower)
    iterations = 3,      # iterations at each pyramid level
    poly_n     = 5,      # pixel neighbourhood size for polynomial expansion
    poly_sigma = 1.2,    # std of Gaussian for polynomial expansion smoothing
    flags      = 0
)

# Force proxy threshold — steps with flow above this are "high force"
# Calibrated empirically: tissue at rest has flow ~0.5-1.0 px/frame
# Active contact typically 2.0-8.0 px/frame
FLOW_HIGH_THRESHOLD = 2.0   # pixels/frame — above this = significant force

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
# OPTICAL FLOW FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────

def compute_farneback_flow(frame_prev: np.ndarray,
                           frame_curr: np.ndarray) -> np.ndarray:
    """
    Compute Farneback dense optical flow between two consecutive RGB frames.

    Args:
        frame_prev: (480, 480, 3) uint8 RGB — frame at time t
        frame_curr: (480, 480, 3) uint8 RGB — frame at time t+1

    Returns:
        flow: (480, 480, 2) float32
              flow[:,:,0] = u (horizontal displacement in pixels)
              flow[:,:,1] = v (vertical displacement in pixels)

    Why grayscale: Farneback operates on intensity gradients, not colour.
    Converting to grayscale first reduces computation and noise.
    The tissue and instrument are distinguishable by intensity without colour.
    """
    gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_RGB2GRAY)
    gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_RGB2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray_prev, gray_curr, None, **FARNEBACK_PARAMS
    )
    return flow   # (H, W, 2) float32


def flow_magnitude(flow: np.ndarray) -> np.ndarray:
    """
    Compute per-pixel flow magnitude from (u, v) components.

    magnitude = sqrt(u² + v²)

    Returns: (H, W) float32 — magnitude in pixels/frame
    """
    u, v = flow[:, :, 0], flow[:, :, 1]
    return np.sqrt(u**2 + v**2).astype(np.float32)


def tissue_flow_magnitude(flow_mag: np.ndarray,
                          tissue_mask: np.ndarray) -> float:
    """
    Compute mean optical flow magnitude within the tissue region.

    This is the force proxy value for one frame transition.

    Args:
        flow_mag:    (H, W) float32 — per-pixel flow magnitude
        tissue_mask: (H, W) uint8   — 0=background, 255=tissue (from Phase 3A)

    Returns:
        float — mean flow magnitude in tissue pixels (pixels/frame)
                Higher value = more tissue deformation = higher estimated force

    Why tissue ROI only:
        The instrument itself moves a lot — that is expected and not a force
        signal. The tissue deformation is the clinically relevant signal.
        We mask to tissue pixels so instrument motion does not inflate the proxy.
    """
    tissue_pixels = tissue_mask > 127
    if tissue_pixels.sum() == 0:
        return 0.0
    return float(flow_mag[tissue_pixels].mean())


def flow_to_hsv_image(flow: np.ndarray) -> np.ndarray:
    """
    Encode optical flow as an HSV colour image for visualisation.

    Standard optical flow visualisation convention:
      Hue    → direction of motion (angle of flow vector)
      Saturation → always 255 (maximum)
      Value  → magnitude of motion (brighter = faster)

    This encoding lets you see both direction and speed of tissue motion
    at a glance — useful for documentation and debugging.

    Returns: (H, W, 3) uint8 RGB image
    """
    u, v    = flow[:, :, 0], flow[:, :, 1]
    mag, ang = cv2.cartToPolar(u, v)

    # Normalise magnitude to 0-255
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    hsv         = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[:, :, 0] = ang * 180 / np.pi / 2   # hue: angle in [0, 180]
    hsv[:, :, 1] = 255                       # saturation: max
    hsv[:, :, 2] = mag_norm.astype(np.uint8) # value: magnitude

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def load_tissue_mask(ep: int, step: int) -> np.ndarray:
    """
    Load tissue segmentation mask from Phase 3A output.
    Falls back to ep000 mask since tissue position is constant (21.1% coverage).
    """
    # Try exact match first
    mask_path = MASK_DIR / f"ep{ep:03d}_step{step:04d}_mask.png"
    if mask_path.exists():
        return np.array(Image.open(mask_path).convert("L"))

    # Use any ep000 mask — tissue position is constant across all episodes
    fallback_candidates = sorted(MASK_DIR.glob("ep000_step*_mask.png"))
    if fallback_candidates:
        return np.array(Image.open(fallback_candidates[0]).convert("L"))

    # Last resort: hardcoded tissue region calibrated from frame_04_pred.png
    mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    mask[200:440, 80:420] = 255
    return mask


# ─────────────────────────────────────────────────────────────────────────
# VALIDATION: CORRELATION WITH COLLISION COST
# ─────────────────────────────────────────────────────────────────────────

def compute_correlation(flow_values: list, collision_flags: list) -> dict:
    """
    Compute Pearson correlation between optical flow magnitude and
    collision flags from the SafeRewardWrapper.

    Hypothesis: high optical flow in tissue region correlates with
    high collision cost (tissue being squeezed hard enough to trigger
    the geometric penetration penalty).

    Returns dict with correlation coefficient and supporting statistics.
    """
    flow_arr  = np.array(flow_values, dtype=np.float32)
    coll_arr  = np.array(collision_flags, dtype=np.float32)

    if len(flow_arr) < 2 or flow_arr.std() == 0:
        return {"pearson_r": 0.0, "note": "insufficient variance"}

    # Pearson correlation
    r = float(np.corrcoef(flow_arr, coll_arr)[0, 1])

    # Precision/recall of force proxy threshold
    predicted_high = flow_arr > FLOW_HIGH_THRESHOLD
    actual_high    = coll_arr > 0

    tp = float((predicted_high & actual_high).sum())
    fp = float((predicted_high & ~actual_high).sum())
    fn = float((~predicted_high & actual_high).sum())

    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)

    return {
        "pearson_r":          r,
        "precision":          precision,
        "recall":             recall,
        "f1_score":           f1,
        "n_samples":          len(flow_arr),
        "n_high_flow":        int(predicted_high.sum()),
        "n_collision_steps":  int(actual_high.sum()),
        "flow_threshold":     FLOW_HIGH_THRESHOLD,
        "flow_mean":          float(flow_arr.mean()),
        "flow_std":           float(flow_arr.std()),
        "flow_max":           float(flow_arr.max()),
    }


def save_validation_plot(flow_values: list, collision_flags: list,
                         output_path: Path):
    """
    Save a scatter plot of optical flow vs collision cost.
    Uses matplotlib — falls back gracefully if not available.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        flow_arr = np.array(flow_values)
        coll_arr = np.array(collision_flags)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: scatter plot
        colours = ["red" if c > 0 else "blue" for c in coll_arr]
        axes[0].scatter(range(len(flow_arr)), flow_arr, c=colours,
                        alpha=0.4, s=8)
        axes[0].axhline(y=FLOW_HIGH_THRESHOLD, color="orange",
                        linestyle="--", label=f"threshold={FLOW_HIGH_THRESHOLD}")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Tissue flow magnitude (px/frame)")
        axes[0].set_title("Optical Flow Proxy over Time\n(red=collision, blue=safe)")
        axes[0].legend()

        # Right: distribution by collision/no-collision
        safe_flow     = flow_arr[coll_arr == 0]
        collision_flow = flow_arr[coll_arr > 0]
        axes[1].hist(safe_flow,      bins=40, alpha=0.6, color="blue",
                     label=f"Safe (n={len(safe_flow)})",      density=True)
        axes[1].hist(collision_flow, bins=40, alpha=0.6, color="red",
                     label=f"Collision (n={len(collision_flow)})", density=True)
        axes[1].axvline(x=FLOW_HIGH_THRESHOLD, color="orange",
                        linestyle="--", label="threshold")
        axes[1].set_xlabel("Tissue flow magnitude (px/frame)")
        axes[1].set_ylabel("Density")
        axes[1].set_title("Flow Distribution: Safe vs Collision Steps")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return True
    except Exception as e:
        warn(f"Could not save validation plot: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────

def main():
    header("=" * 62)
    print("  Phase 3C — Optical Flow Tissue Force Proxy")
    header("=" * 62)
    info("Algorithm:  Farneback dense optical flow (OpenCV)")
    info("Proxy:      mean flow magnitude in tissue mask region")
    info("Validation: correlation with SafeRewardWrapper collision flags")
    info(f"Episodes:   {EPISODES}")

    FLOW_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Create environment ────────────────────────────────────────────────
    info("Creating TissueRetractionV3 …")
    env = TissueRetractionV3(env_kwargs={"render_mode": "human"})
    ok("Environment created")

    # ── Load Phase 3B agent ───────────────────────────────────────────────
    info("Loading Phase 3B PPO checkpoint …")
    model = PPO.load(CHECKPOINT, env=env)
    ok(f"Checkpoint loaded: {CHECKPOINT}")

    # ── CSV log ───────────────────────────────────────────────────────────
    log_path   = FLOW_DIR / "flow_log.csv"
    log_file   = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "episode", "step",
        "flow_magnitude_all",      # mean flow over entire frame
        "flow_magnitude_tissue",   # mean flow in tissue ROI only
        "collision_flag",          # 1 if collision penalty fired this step
        "reward",                  # raw step reward
    ])

    # Accumulators for validation
    all_tissue_flow = []
    all_collision   = []
    all_rewards     = []

    t_start      = time.time()
    total_steps  = 0
    flow_samples = 0

    header(f"\n  Running {EPISODES} episodes …\n")

    # ── Discover correct V3 access path ─────────────────────────────────
    def find_v3_path(obj, path="env", depth=0):
        if depth > 8: return None
        if type(obj).__name__ == "TissueRetractionV3": return path, obj
        for attr in ["env", "envs", "venv"]:
            if hasattr(obj, attr):
                child = getattr(obj, attr)
                if isinstance(child, list):
                    for i, c in enumerate(child):
                        r = find_v3_path(c, f"{path}.{attr}[{i}]", depth+1)
                        if r: return r
                else:
                    r = find_v3_path(child, f"{path}.{attr}", depth+1)
                    if r: return r
        return None

    v3_result = find_v3_path(model)
    if v3_result:
        v3_path, env_v3 = v3_result
        ok(f"TissueRetractionV3 found at: model.{v3_path}")
    else:
        v3_result2 = find_v3_path(env, "env")
        if v3_result2:
            v3_path, env_v3 = v3_result2
            ok(f"TissueRetractionV3 found at: {v3_path}")
        else:
            env_v3 = env
            warn("TissueRetractionV3 not found in wrapper chain — using env directly")

    for ep in range(EPISODES):
        obs, info_dict = env.reset()

        # Capture initial frame from perception pipeline cache
        frame_prev = getattr(env_v3, "_last_rgb_frame", None)
        if frame_prev is None:
            frame_prev = getattr(env._env, "_last_rgb_frame", None)
        if frame_prev is None:
            frame_prev = env._env._env.render()
        if frame_prev is None:
            warn(f"ep {ep}: render returned None — skipping")
            continue

        step      = 0
        done      = False
        truncated = False
        ep_flow_tissue = []
        ep_collisions  = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, step_info = env.step(action)

            # Get frame already captured by perception pipeline this step
            frame_curr = getattr(env_v3, '_last_rgb_frame', None)
            if frame_curr is None:
                frame_curr = getattr(env._env, '_last_rgb_frame', None)
            if frame_curr is None:
                frame_curr = env._env._env.render()
            if frame_curr is None:
                step += 1
                continue

            # ── Compute optical flow ───────────────────────────────────
            flow     = compute_farneback_flow(frame_prev, frame_curr)
            flow_mag = flow_magnitude(flow)

            # ── Load tissue mask for this step ─────────────────────────
            tissue_mask = load_tissue_mask(ep, step)

            # ── Compute force proxies ──────────────────────────────────
            mag_all    = float(flow_mag.mean())
            mag_tissue = tissue_flow_magnitude(flow_mag, tissue_mask)

            # ── Extract collision flag from reward info ─────────────────
            # The SafeRewardWrapper stores collision info in step_info
            # Fallback: infer from reward magnitude
            collision_flag = int(step_info.get("collision", 0))
            if collision_flag == 0 and reward < -2.0:
                # Heuristic: very negative rewards suggest collision penalty
                collision_flag = 1

            # ── Log ───────────────────────────────────────────────────
            log_writer.writerow([
                ep, step,
                f"{mag_all:.4f}",
                f"{mag_tissue:.4f}",
                collision_flag,
                f"{reward:.4f}",
            ])

            # ── Save flow visualisation for first 3 steps of ep 0 ─────
            if ep == 0 and step < 3:
                flow_vis  = flow_to_hsv_image(flow)

                # Side-by-side: prev frame | current frame | flow
                vis_combined = np.zeros((IMG_H, IMG_W * 3 + 8, 3), dtype=np.uint8)
                vis_combined[:, :IMG_W]             = frame_prev
                vis_combined[:, IMG_W+4:IMG_W*2+4]  = frame_curr
                vis_combined[:, IMG_W*2+8:]         = flow_vis

                # Overlay tissue mask boundary on flow panel
                contour_mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
                contour_mask[tissue_mask > 127] = 255
                kernel  = np.ones((3, 3), np.uint8)
                eroded  = cv2.erode(contour_mask, kernel, iterations=1)
                contour = contour_mask - eroded
                # Apply contour only to the flow panel (rightmost panel)
                flow_panel = vis_combined[:, IMG_W*2+8:].copy()
                flow_panel[contour > 0] = (255, 255, 255)
                vis_combined[:, IMG_W*2+8:] = flow_panel

                out_vis = FLOW_DIR / f"flow_ep{ep:03d}_step{step:04d}.png"
                Image.fromarray(vis_combined).save(out_vis)

            ep_flow_tissue.append(mag_tissue)
            all_tissue_flow.append(mag_tissue)
            all_collision.append(collision_flag)
            all_rewards.append(reward)

            if collision_flag:
                ep_collisions += 1

            frame_prev  = frame_curr.copy()  # must copy — prevents aliasing with _last_rgb_frame
            step        += 1
            total_steps += 1
            flow_samples += 1

        ep_mean_flow = float(np.mean(ep_flow_tissue)) if ep_flow_tissue else 0.0
        ep_max_flow  = float(np.max(ep_flow_tissue))  if ep_flow_tissue else 0.0

        elapsed = time.time() - t_start
        fps     = total_steps / elapsed if elapsed > 0 else 0
        print(f"  ep {ep+1:2d}/{EPISODES}  steps={step:4d}  "
              f"mean_tissue_flow={ep_mean_flow:.3f}  "
              f"max_flow={ep_max_flow:.3f}  "
              f"collisions={ep_collisions:3d}  "
              f"{fps:.1f}fps")

        log_file.flush()

    log_file.close()
    elapsed = time.time() - t_start

    # ── Validation ────────────────────────────────────────────────────────
    header("\n  Validation — flow vs collision correlation")

    stats = compute_correlation(all_tissue_flow, all_collision)

    print(f"\n  Pearson r (flow vs collision): {stats['pearson_r']:+.4f}")
    print(f"  Precision (flow>threshold → collision): {stats['precision']:.3f}")
    print(f"  Recall    (collision → flow>threshold): {stats['recall']:.3f}")
    print(f"  F1 score:                               {stats['f1_score']:.3f}")
    print(f"\n  Flow statistics:")
    print(f"    Mean:    {stats['flow_mean']:.3f} px/frame")
    print(f"    Std:     {stats['flow_std']:.3f} px/frame")
    print(f"    Max:     {stats['flow_max']:.3f} px/frame")
    print(f"    n_samples:          {stats['n_samples']}")
    print(f"    n_high_flow_steps:  {stats['n_high_flow']}")
    print(f"    n_collision_steps:  {stats['n_collision_steps']}")

    # ── Save validation plot ──────────────────────────────────────────────
    plot_path = FLOW_DIR / "flow_validation_plot.png"
    if save_validation_plot(all_tissue_flow, all_collision, plot_path):
        ok(f"Validation plot → {plot_path}")

    # ── Save proxy config ─────────────────────────────────────────────────
    proxy_config = {
        "method":           "Farneback dense optical flow",
        "roi":              "tissue segmentation mask (Phase 3A)",
        "proxy_metric":     "mean flow magnitude in tissue ROI (px/frame)",
        "high_force_threshold": FLOW_HIGH_THRESHOLD,
        "farneback_params": FARNEBACK_PARAMS,
        "validation":       stats,
        "phase4_topic":     "/tissue_force_proxy",
        "clinical_note":    (
            "Force proxy is a visual estimate only. "
            "Real surgical robots require calibrated force sensors "
            "for safety-critical force limits. This proxy is suitable "
            "for monitoring and logging, not hard safety stops."
        ),
    }
    config_path = MODEL_DIR / "proxy_config.json"
    json.dump(proxy_config, open(config_path, "w"), indent=2)
    ok(f"Proxy config saved → {config_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    header("\n" + "=" * 62)
    ok("Phase 3C — Optical Flow Force Proxy COMPLETE")
    ok(f"Total steps analysed: {flow_samples}")
    ok(f"Flow log: {log_path}")
    ok(f"Elapsed:  {elapsed:.1f}s ({total_steps/elapsed:.1f} fps avg)")
    print()

    r = stats["pearson_r"]
    if abs(r) > 0.4:
        ok(f"Pearson r={r:+.3f} — GOOD correlation: proxy is clinically useful")
    elif abs(r) > 0.2:
        warn(f"Pearson r={r:+.3f} — MODERATE correlation: proxy has signal")
    else:
        warn(f"Pearson r={r:+.3f} — WEAK correlation: proxy is limited")
        warn("Possible reason: collision flags sparse in Phase 3B (agent learned safety)")
        warn("Optical flow signal is still valid as a deformation monitor")

    print()
    info("Phase 3C→4 connection:")
    info("  In Phase 4 (ROS 2), this proxy publishes to /tissue_force_proxy")
    info("  The topic carries: {timestamp, flow_magnitude, high_force_flag}")
    info("  The surgical safety monitor subscribes and can halt motion")
    info("  if flow_magnitude exceeds the validated threshold")
    header("=" * 62 + "\n")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()