"""
watch_agent.py — Watch the Phase 2D PPO agent run in HUMAN mode
================================================================
Loads the trained Phase 2D checkpoint and runs it in the SOFA
GUI so you can watch the agent perform tissue retraction.

Usage:
    source setup_env.sh
    python scripts/watch_agent.py

    # Run more episodes:
    python scripts/watch_agent.py --episodes 5

    # Slow it down to watch clearly:
    python scripts/watch_agent.py --slow

Author: Subhash Arockiadoss
"""

import os
import sys
import argparse
import time
import numpy as np
from pathlib import Path

import gymnasium
sys.modules['gym'] = gymnasium
sys.modules['gym.spaces'] = gymnasium.spaces
sys.path.insert(0, '.')

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, default=3,
                    help='Number of episodes to watch (default: 3)')
parser.add_argument('--slow', action='store_true',
                    help='Add delay between steps so you can watch clearly')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='Path to checkpoint without .zip extension')
args = parser.parse_args()

# ── Find checkpoint ───────────────────────────────────────────────────────────
def find_checkpoint():
    checkpoint_dir = Path("logs/checkpoints")
    if not checkpoint_dir.exists():
        return None
    runs = sorted([d for d in checkpoint_dir.iterdir() if d.is_dir()],
                  key=lambda d: d.name)
    for run_dir in reversed(runs):
        candidate = run_dir / "ppo_tissue_final"
        if Path(str(candidate) + ".zip").exists():
            return str(candidate)
    return None

checkpoint_path = args.checkpoint or find_checkpoint()
if checkpoint_path is None:
    print("ERROR: No checkpoint found in logs/checkpoints/")
    sys.exit(1)

print(f"\nCheckpoint:  {checkpoint_path}")
print(f"Episodes:    {args.episodes}")
print(f"Slow mode:   {args.slow}")

# ── Imports ───────────────────────────────────────────────────────────────────
from stable_baselines3 import PPO
from envs import TissueRetractionV2
from envs.safe_reward import SafeRewardWrapper

# ── Import RenderMode from sofa_env ──────────────────────────────────────────
LAPGYM_PATH = Path.home() / "surgical_robot_lapgym_ws" / "lap_gym" / "sofa_env"
if str(LAPGYM_PATH) not in sys.path:
    sys.path.insert(0, str(LAPGYM_PATH))

from sofa_env.scenes.tissue_retraction.tissue_retraction_env import RenderMode

# ── Build environment with HUMAN render mode ──────────────────────────────────
# TissueRetractionV2 accepts env_kwargs dict that passes through to SOFA env
# render_mode goes inside env_kwargs — this is what the signature showed us
print("\nInitialising SOFA GUI — window will open...")

base_env = TissueRetractionV2(env_kwargs={"render_mode": RenderMode.HUMAN})
env = SafeRewardWrapper(
    base_env,
    lambda_force=0.5,
    lambda_collision=0.5,
    force_threshold=0.5,
    step_penalty=0.01
)

# Reset once to verify observation shape before loading model
result = env.reset()
obs = result[0] if isinstance(result, tuple) else result
print(f"Observation shape: {obs.shape}")

if obs.shape != (7,):
    print(f"ERROR: Expected (7,) observation, got {obs.shape}")
    print("The PPO model needs 7D observation to run.")
    os._exit(1)

print("7D observation confirmed — correct for Phase 2D model")

# ── Load trained PPO model ────────────────────────────────────────────────────
print("\nLoading trained PPO policy...")
model = PPO.load(checkpoint_path)
n_params = sum(p.numel() for p in model.policy.parameters())
print(f"Policy loaded — {n_params:,} parameters")
print("Watching agent now...\n")

# ── Run episodes ──────────────────────────────────────────────────────────────
for episode in range(args.episodes):
    print(f"{'='*55}")
    print(f"  Episode {episode + 1} of {args.episodes}")
    print(f"{'='*55}")

    result = env.reset()
    obs = result[0] if isinstance(result, tuple) else result

    ep_steps     = 0
    ep_reward    = 0.0
    goal_reached = False

    while True:
        # Get action from trained policy — deterministic = no exploration noise
        action, _ = model.predict(obs, deterministic=True)

        # Step the environment
        result = env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result

        ep_steps  += 1
        ep_reward += reward

        # Check goal reached
        if info.get('goal_reached', False) or info.get('is_success', False):
            goal_reached = True

        # Print progress every 10 steps
        if ep_steps % 10 == 0:
            phase_id = info.get('phase', info.get('current_phase', 0))
            phase = "GRASPING" if phase_id == 0 else "RETRACTING"
            print(f"  Step {ep_steps:4d} | "
                  f"reward = {ep_reward:8.2f} | "
                  f"phase = {phase:11s} | "
                  f"goal = {goal_reached}")

        # Slow mode — 50ms delay makes motion clearly visible in GUI
        if args.slow:
            time.sleep(0.05)

        if done:
            break

    # ── Episode summary ───────────────────────────────────────────────────────
    status = "GOAL REACHED" if goal_reached else "Truncated (no goal)"
    faster = 247 - ep_steps
    print(f"\n  Result:   {status}")
    print(f"  Steps:    {ep_steps}  "
          f"({'%d faster than' % faster if faster > 0 else '%d slower than' % abs(faster)} "
          f"scripted baseline of 247)")
    print(f"  Reward:   {ep_reward:.2f}  "
          f"({'better' if ep_reward > -165.54 else 'worse'} than baseline of -165.54)")

    if episode < args.episodes - 1:
        print("\n  Pausing 3 seconds before next episode...")
        time.sleep(3)

print(f"\n{'='*55}")
print(f"  All {args.episodes} episodes complete")
print(f"{'='*55}\n")

os._exit(0)