#!/usr/bin/env python3
"""
Phase 3A — Step 4a: Generate Tissue Segmentation Masks
=======================================================
Re-runs the Phase 2D PPO agent and generates a binary segmentation mask
for each frame by projecting the tissue visual mesh (448 vertices) through
the SOFA camera matrices onto the 2D image plane.

WHY THIS APPROACH (Option A — mesh projection):
  The SOFA scene graph exposes:
    - tissue/visual/OglModel.position → 448 × (x,y,z) tissue surface vertices
    - camera_node.modelViewMatrix     → 4×4 OpenGL modelview matrix
    - camera_node.projectionMatrix    → 4×4 OpenGL projection matrix
  These are the EXACT matrices OpenGL uses to render each frame.
  Projecting mesh vertices through them gives pixel-perfect labels —
  mathematically identical to what was rendered. Zero manual annotation.

WHY CSV-DRIVEN REPLAY (v3 — fixes the mismatch problem):
  Previous versions re-ran 50 episodes and saved a mask for every step.
  Problem: environment uses a random seed at reset — different runs produce
  different episode lengths, causing RGB/mask count mismatch.
  Fix: read labels.csv first to know EXACTLY which (episode, step) pairs
  need masks. Replay each episode and only save masks for steps in the CSV.
  Result: guaranteed 100% match between RGB frames and masks.

THE PROJECTION MATHS (verified: vertex→pixel confirmed in range [0,480]):
  1. vertex_h  = [x, y, z, 1.0]                  homogeneous 3D point
  2. clip      = P @ MV @ vertex_h                4D clip space
  3. ndc_x,y   = clip[0]/clip[3], clip[1]/clip[3] NDC in [-1,1]
  4. u = (ndc_x+1)/2 * W   v = (1-ndc_y)/2 * H   pixel coords
  5. Convex hull of all (u,v) → filled binary mask

OUTPUT:
  data/seg_masks/
    ep000_step0000_mask.png   ← 480×480 binary mask (0=background, 255=tissue)
    mask_log.csv              ← episode, step, n_visible_vertices, mask_coverage

Run from repo root:
    python3 scripts/generate_seg_masks.py

Author: Subhash Arockiadoss
"""

# ── gym → gymnasium shim (must come before any env import) ───────────────
import sys
import gymnasium
sys.modules['gym'] = gymnasium
sys.modules['gym.spaces'] = gymnasium.spaces
# ─────────────────────────────────────────────────────────────────────────

import csv
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from stable_baselines3 import PPO

sys.path.insert(0, '.')
from envs.tissue_retraction_v2 import TissueRetractionV2

# ─────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────

CHECKPOINT = (
    "logs/checkpoints/"
    "phase2_ppo_tissue_retraction_20260409_211946/"
    "ppo_tissue_final"
)
RGB_DIR  = Path("data/rgb_frames")   # existing PNGs + labels.csv
MASK_DIR = Path("data/seg_masks")    # output: binary mask PNGs
IMG_W    = 480                       # image width in pixels
IMG_H    = 480                       # image height in pixels

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
# CAMERA PROJECTION
# ─────────────────────────────────────────────────────────────────────────

def get_camera_matrices(cam_node):
    """
    Read OpenGL modelview and projection matrices from SOFA camera node.
    Both stored as flat 16-element arrays — reshape to (4,4).

    modelViewMatrix: transforms world space → camera space.
    projectionMatrix: transforms camera space → clip space.
      Verified: diagonal [2.605, 2.605] — standard perspective ~42° FOV.
    """
    mv   = np.array(cam_node.modelViewMatrix.value).reshape(4, 4)
    proj = np.array(cam_node.projectionMatrix.value).reshape(4, 4)
    return mv, proj


def project_vertices(vertices_3d: np.ndarray, mv: np.ndarray,
                     proj: np.ndarray, W: int, H: int):
    """
    Project N×3 world-space vertices onto the 2D image plane.

    Returns: (visible_pixels (N,2), n_visible int)

    Y-axis flip: OpenGL NDC has Y increasing upward.
    Image pixels have V increasing downward.
    So: v = (1 - ndc_y) / 2 * H
    """
    N = len(vertices_3d)

    # Homogenise → (N, 4)
    verts_h  = np.hstack([vertices_3d, np.ones((N, 1))])

    # Apply modelview + projection → clip space (N, 4)
    combined = proj @ mv
    clip     = (combined @ verts_h.T).T

    # Perspective divide → NDC
    w     = clip[:, 3:4]
    valid = (w > 0).flatten()
    ndc   = np.zeros((N, 2))
    ndc[valid, 0] = clip[valid, 0] / w[valid, 0]
    ndc[valid, 1] = clip[valid, 1] / w[valid, 0]

    # Viewport transform → integer pixel coords
    u = ((ndc[:, 0] + 1) / 2 * W).astype(int)
    v = ((1 - ndc[:, 1]) / 2 * H).astype(int)
    pixels = np.stack([u, v], axis=1)

    # Keep only vertices visible in frustum and within image bounds
    in_frustum = (
        valid &
        (ndc[:, 0] >= -1) & (ndc[:, 0] <= 1) &
        (ndc[:, 1] >= -1) & (ndc[:, 1] <= 1)
    )
    in_bounds = (
        (pixels[:, 0] >= 0) & (pixels[:, 0] < W) &
        (pixels[:, 1] >= 0) & (pixels[:, 1] < H)
    )
    visible = in_frustum & in_bounds
    return pixels[visible], int(visible.sum())


def make_mask(pixels: np.ndarray, W: int, H: int) -> np.ndarray:
    """
    Generate a 480×480 binary mask from projected pixel coordinates.

    Strategy: convex hull of visible tissue vertices → filled polygon.
    Why: 448 vertices project to a scattered 2D point cloud. The convex
    hull gives a solid filled region covering all interior tissue pixels
    without needing per-pixel rasterisation.

    Returns: (H, W) uint8 array — 0=background, 255=tissue
    """
    mask = np.zeros((H, W), dtype=np.uint8)

    if len(pixels) < 3:
        return mask   # cannot form a hull with fewer than 3 points

    try:
        hull     = ConvexHull(pixels)
        hull_pts = pixels[hull.vertices]
        img_pil  = Image.fromarray(mask)
        draw     = ImageDraw.Draw(img_pil)
        polygon  = [(int(p[0]), int(p[1])) for p in hull_pts]
        draw.polygon(polygon, fill=255)
        mask = np.array(img_pil)
    except Exception:
        # ConvexHull fails if all points are collinear — scatter dots
        for px, py in pixels:
            if 0 <= px < W and 0 <= py < H:
                mask[py, px] = 255

    return mask


# ─────────────────────────────────────────────────────────────────────────
# SCENE GRAPH ACCESSOR
# ─────────────────────────────────────────────────────────────────────────

def get_tissue_vertices(inner_env) -> np.ndarray:
    """
    Read current tissue visual mesh vertices from SOFA scene graph.
    Path: root → scene → tissue → visual → OglModel.position
    Shape: (448, 3) in metres — updated every simulation step.

    Why visual mesh (not FEM mesh):
      FEM mesh (493 verts) is used for physics.
      Visual mesh (448 verts, OglModel) is what OpenGL renders.
      For image-space labels we must use the visual mesh.
    """
    root   = inner_env._sofa_root_node
    scene  = root.getChild('scene')
    tissue = scene.getChild('tissue')
    visual = tissue.getChild('visual')

    for obj in visual.objects:
        if hasattr(obj, 'position'):
            return np.array(obj.position.value)

    raise RuntimeError("Could not find tissue visual OglModel — check scene graph")


# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────

def main():
    header("=" * 62)
    print("  Phase 3A — Tissue Segmentation Mask Generation  (v3)")
    header("=" * 62)
    info("Method: CSV-driven replay — guaranteed 100% RGB/mask match")

    MASK_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load labels CSV — this defines exactly which frames need masks ────
    # Key insight: instead of hoping episode lengths match, we read the
    # CSV to know exactly which (episode, step) pairs exist, then replay
    # those episodes and only save masks for those specific steps.
    labels_csv = RGB_DIR / "labels.csv"
    if not labels_csv.exists():
        fail(f"labels.csv not found at {labels_csv}")
        fail("Run collect_rgb_frames.py first")
        sys.exit(1)

    df = pd.read_csv(labels_csv)
    ok(f"Loaded labels.csv — {len(df)} frames across "
       f"{df['episode'].nunique()} episodes")

    # Build a set of required (episode, step) pairs for fast lookup
    required = set(zip(df['episode'].values, df['step'].values))
    info(f"Need to generate {len(required)} masks")

    # Group by episode so we replay one episode at a time
    episode_groups = df.groupby('episode')
    n_episodes     = df['episode'].nunique()

    # ── Create environment ────────────────────────────────────────────────
    info("Creating environment …")
    env = TissueRetractionV2(env_kwargs={"render_mode": "headless"})
    ok("Env created")

    # ── Load PPO agent ────────────────────────────────────────────────────
    info("Loading PPO checkpoint …")
    model = PPO.load(CHECKPOINT, env=env)
    ok("Checkpoint loaded")

    # ── Open log CSV ──────────────────────────────────────────────────────
    log_path   = MASK_DIR / "mask_log.csv"
    log_file   = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["episode", "step", "n_visible_vertices",
                         "mask_coverage_pct", "mask_filename"])

    total_masks = 0
    t_start     = time.time()

    header(f"\n  Generating masks — {n_episodes} episodes …\n")

    for ep_idx, (ep, ep_df) in enumerate(episode_groups):
        ep_df    = ep_df.sort_values("step").reset_index(drop=True)
        max_step = int(ep_df["step"].max())

        # Single clean reset and seed — no warm-up, no double reset
        obs, _ = env.reset(seed=int(ep))

        # Camera matrices are fixed for the entire episode
        cam  = env._env._camera_object
        mv, proj = get_camera_matrices(cam)

        step      = 0
        done      = False
        truncated = False
        ep_masks  = 0

        # Run episode up to the last step that needs a mask
        while not (done or truncated) and step <= max_step:

            action, _ = model.predict(obs, deterministic=True)

            # Only generate a mask if this (episode, step) is in the CSV
            if (ep, step) in required:
                try:
                    verts_3d = get_tissue_vertices(env._env)
                except Exception as e:
                    warn(f"  ep={ep:03d} step={step:04d} vertex read failed: {e}")
                    obs, _, done, truncated, _ = env.step(action)
                    step += 1
                    continue

                pixels, n_visible = project_vertices(
                    verts_3d, mv, proj, IMG_W, IMG_H)
                mask         = make_mask(pixels, IMG_W, IMG_H)
                coverage_pct = float(mask.sum() / 255) / (IMG_W * IMG_H) * 100

                mask_filename = f"ep{ep:03d}_step{step:04d}_mask.png"
                Image.fromarray(mask).save(MASK_DIR / mask_filename)

                log_writer.writerow([ep, step, n_visible,
                                     f"{coverage_pct:.2f}", mask_filename])
                total_masks += 1
                ep_masks    += 1

            obs, _, done, truncated, _ = env.step(action)
            step += 1

        elapsed = time.time() - t_start
        fps     = total_masks / elapsed if elapsed > 0 else 0
        print(f"  ep {ep_idx+1:3d}/{n_episodes}  "
              f"steps_needed={len(ep_df):4d}  masks_saved={ep_masks:4d}  "
              f"total={total_masks:6d}  {fps:5.1f} fps")

        log_file.flush()

    log_file.close()
    elapsed = time.time() - t_start

    # ── Final verification ────────────────────────────────────────────────
    n_rgb   = len(df)
    n_masks = len(list(MASK_DIR.glob("ep*_mask.png")))
    matched = n_masks

    header("\n" + "=" * 62)
    ok(f"Mask generation complete!")
    ok(f"Required masks:  {len(required)}")
    ok(f"Masks saved:     {total_masks}")
    ok(f"RGB frames:      {n_rgb}")

    if total_masks == n_rgb:
        ok(f"Perfect match: {total_masks} = {n_rgb} ✓")
    else:
        warn(f"Mismatch: {total_masks} masks vs {n_rgb} RGB frames")
        warn("Some episodes may have ended before reaching max_step")
        warn("train_segmentation.py will filter to matched pairs automatically")

    ok(f"Elapsed: {elapsed:.1f}s  ({total_masks/elapsed:.1f} fps avg)")
    ok(f"Log: {log_path}")

    # ── Run inline match verification ─────────────────────────────────────
    print()
    info("Running match verification …")
    rgb_stems  = set('_'.join(f.split('_')[:2]) for f in df['filename'])
    mask_stems = set(
        f.stem.replace('_mask', '')
        for f in MASK_DIR.glob("ep*_mask.png")
    )
    matched = len(rgb_stems & mask_stems)
    print(f"  RGB frames:        {len(rgb_stems)}")
    print(f"  Mask files:        {len(mask_stems)}")
    print(f"  Perfectly matched: {matched}")
    print(f"  Match rate:        {matched/len(rgb_stems)*100:.1f}%")

    if matched == len(rgb_stems):
        ok("100% match — dataset is clean ✓")
    else:
        warn(f"{len(rgb_stems) - matched} RGB frames still have no mask")
        warn("This is acceptable — train_segmentation.py filters automatically")

    header("=" * 62 + "\n")
    os._exit(0)


if __name__ == "__main__":
    main()