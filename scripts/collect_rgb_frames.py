#!/usr/bin/env python3
"""
Phase 3A — RGB Frame Collection
=================================
Runs the Phase 2D PPO agent for 50 episodes and captures one RGB frame per
step from the SOFA endoscopic camera.

Confirmed capture method: env._env.render() → (480, 480, 3) uint8
Confirmed via: test_camera_capture.py  (Strategy A)

Outputs
-------
data/rgb_frames/
    ep{EEE}_step{SSS}_x{X:.4f}_y{Y:.4f}_z{Z:.4f}.png   ← one PNG per step
    labels.csv                                            ← ground-truth labels

labels.csv columns
------------------
frame_id, episode, step, tool_x, tool_y, tool_z,
goal_x, goal_y, goal_z, phase

Usage
-----
    source ~/setup_env.sh
    python3 scripts/collect_rgb_frames.py

    # Optional overrides:
    python3 scripts/collect_rgb_frames.py --episodes 10 --out data/rgb_frames_debug

Author: Subhash Arockiadoss    
"""

# ── gym → gymnasium shim  (must come before any env import) ───────────────
import sys
import gymnasium
sys.modules['gym'] = gymnasium
sys.modules['gym.spaces'] = gymnasium.spaces
# ─────────────────────────────────────────────────────────────────────────

import argparse
import csv
import os
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
from stable_baselines3 import PPO

sys.path.insert(0, '.')
from envs.tissue_retraction_v2 import TissueRetractionV2

# ── Colour helpers ────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):    print(f"{GREEN}  ✓  {msg}{RESET}")
def fail(msg):  print(f"{RED}  ✗  {msg}{RESET}")
def info(msg):  print(f"{CYAN}  ·  {msg}{RESET}")
def warn(msg):  print(f"{YELLOW}  !  {msg}{RESET}")
def header(msg): print(f"\n{BOLD}{msg}{RESET}")

# ── Defaults ──────────────────────────────────────────────────────────────
CHECKPOINT = (
    "logs/checkpoints/"
    "phase2_ppo_tissue_retraction_20260409_211946/"
    "ppo_tissue_final"
)
DEFAULT_OUT      = "data/rgb_frames"
DEFAULT_EPISODES = 50
WARMUP_STEPS     = 3    # steps before first capture to let SOFA render scene


# ─────────────────────────────────────────────────────────────────────────
def capture_frame(env) -> np.ndarray:
    """
    Capture one RGB frame.
    Confirmed method: env._env.render() → (480, 480, 3) uint8
    """
    frame = env._env.render()
    if frame is None:
        raise RuntimeError(
            "env._env.render() returned None. "
            "SOFA camera may not have initialised — check render_mode='headless'."
        )
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def extract_labels(obs: np.ndarray) -> dict:
    """
    Unpack the 7D state observation into named ground-truth labels.
    TissueRetractionV2 obs layout: [tool_x, tool_y, tool_z,
                                     goal_x, goal_y, goal_z, phase_flag]
    """
    return {
        "tool_x": float(obs[0]),
        "tool_y": float(obs[1]),
        "tool_z": float(obs[2]),
        "goal_x": float(obs[3]),
        "goal_y": float(obs[4]),
        "goal_z": float(obs[5]),
        "phase":  float(obs[6]),
    }


def make_filename(episode: int, step: int, labels: dict) -> str:
    """Build PNG filename encoding episode, step, and tool XYZ."""
    return (
        f"ep{episode:03d}_step{step:04d}"
        f"_x{labels['tool_x']:+.4f}"
        f"_y{labels['tool_y']:+.4f}"
        f"_z{labels['tool_z']:+.4f}"
        ".png"
    )


# ─────────────────────────────────────────────────────────────────────────
def collect(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "labels.csv"
    checkpoint = args.checkpoint

    header("=" * 62)
    print(f"  Phase 3A — RGB Frame Collection")
    header("=" * 62)
    info(f"Checkpoint  : {checkpoint}")
    info(f"Episodes    : {args.episodes}")
    info(f"Output dir  : {out_dir}")
    info(f"Labels CSV  : {csv_path}")

    # ── 1. Create env ─────────────────────────────────────────────────────
    info("Creating TissueRetractionV2 (render_mode='headless') …")
    try:
        env = TissueRetractionV2(env_kwargs={"render_mode": "headless"})
        ok("Env created")
    except Exception as e:
        fail(f"Env creation failed: {e}")
        traceback.print_exc()
        os._exit(1)

    # ── 2. Load Phase 2D checkpoint ───────────────────────────────────────
    info(f"Loading PPO checkpoint …")
    try:
        model = PPO.load(checkpoint, env=env)
        ok(f"Checkpoint loaded: {checkpoint}")
    except Exception as e:
        fail(f"PPO.load() failed: {e}")
        traceback.print_exc()
        os._exit(1)

    # ── 3. Open CSV ───────────────────────────────────────────────────────
    csv_file   = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame_id", "episode", "step",
        "tool_x", "tool_y", "tool_z",
        "goal_x", "goal_y", "goal_z",
        "phase",
        "filename",
    ])

    # ── 4. Collection loop ────────────────────────────────────────────────
    total_frames  = 0
    episode_stats = []   # (ep, n_steps, done_reason)

    header(f"\n  Starting collection — {args.episodes} episodes …\n")
    t_start = time.time()

    for ep in range(args.episodes):
        obs, _ = env.reset()

        # Warm-up: run a few no-op steps so SOFA renders the scene fully
        for _ in range(WARMUP_STEPS):
            env.step(env.action_space.sample() * 0.0)

        # Re-fetch obs after warm-up (env state unchanged, just renderer ready)
        obs, _ = env.reset()

        ep_frames = 0
        step      = 0
        done      = False
        truncated = False

        while not (done or truncated):
            # --- Policy action ---
            action, _ = model.predict(obs, deterministic=True)

            # --- Capture frame BEFORE the step (current state) ---
            try:
                frame = capture_frame(env)
            except Exception as e:
                warn(f"  ep={ep:03d} step={step:04d} — frame capture failed: {e}")
                # Still step the env to keep simulation consistent
                obs, reward, done, truncated, info_dict = env.step(action)
                step += 1
                continue

            # --- Extract ground-truth labels from current obs ---
            labels   = extract_labels(obs)
            filename = make_filename(ep, step, labels)
            frame_id = f"ep{ep:03d}_step{step:04d}"

            # --- Save PNG ---
            Image.fromarray(frame).save(out_dir / filename)

            # --- Write CSV row ---
            csv_writer.writerow([
                frame_id, ep, step,
                labels["tool_x"], labels["tool_y"], labels["tool_z"],
                labels["goal_x"], labels["goal_y"], labels["goal_z"],
                labels["phase"],
                filename,
            ])

            # --- Step the environment ---
            obs, reward, done, truncated, info_dict = env.step(action)

            step      += 1
            ep_frames += 1
            total_frames += 1

        episode_stats.append((ep, ep_frames, "done" if done else "truncated"))

        # Progress line every episode
        elapsed = time.time() - t_start
        fps     = total_frames / elapsed if elapsed > 0 else 0
        print(
            f"  ep {ep+1:3d}/{args.episodes}  "
            f"steps={ep_frames:4d}  "
            f"total_frames={total_frames:6d}  "
            f"{fps:5.1f} fps  "
            f"{'done' if done else 'truncated'}"
        )

        # Flush CSV every episode so data is safe even if we crash
        csv_file.flush()

    # ── 5. Summary ────────────────────────────────────────────────────────
    csv_file.close()
    elapsed = time.time() - t_start

    step_counts  = [s for _, s, _ in episode_stats]
    mean_steps   = np.mean(step_counts)
    min_steps    = np.min(step_counts)
    max_steps    = np.max(step_counts)

    header("\n" + "=" * 62)
    ok(f"Collection complete!")
    ok(f"Total frames saved : {total_frames}")
    ok(f"Episodes           : {args.episodes}")
    ok(f"Steps  mean/min/max: {mean_steps:.1f} / {min_steps} / {max_steps}")
    ok(f"Output directory   : {out_dir}/")
    ok(f"Labels CSV         : {csv_path}")
    ok(f"Elapsed time       : {elapsed:.1f}s  "
       f"({total_frames/elapsed:.1f} fps avg)")
    print()
    info("Next step — 3B: Train instrument tip detector on these frames.")
    header("=" * 62 + "\n")

    # Use os._exit to avoid SOFA/SofaPython3 GIL crash on interpreter shutdown
    # (same behaviour as Phase 2 training scripts — all data already flushed)
    os._exit(0)


# ─────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Phase 3A — collect RGB frames")
    p.add_argument(
        "--checkpoint", default=CHECKPOINT,
        help="Path to PPO checkpoint (without .zip)"
    )
    p.add_argument(
        "--episodes", type=int, default=DEFAULT_EPISODES,
        help="Number of episodes to collect (default: 50)"
    )
    p.add_argument(
        "--out", default=DEFAULT_OUT,
        help="Output directory for frames + labels.csv"
    )
    return p.parse_args()


if __name__ == "__main__":
    collect(parse_args())