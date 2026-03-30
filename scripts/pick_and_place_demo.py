"""
pick_and_place_demo.py

Phase 1 — LapGym Baseline Demo: Pick and Place (Peg Transfer)

Clinical context:
    Peg transfer is the foundational task from the Fundamentals of
    Laparoscopic Surgery (FLS) curriculum — the standardised training
    program all surgical residents must complete. It tests precision
    instrument manipulation: pick a ring from one peg, transfer to another.

    Key difference from tissue retraction:
    - Object is deformable (rope/torus) but task is position-based
    - Articulated gripper with open/close jaw control
    - Finer timestep (0.01s vs 0.1s) — more precise physics needed
    - Demonstrates rigid peg + deformable ring interaction

Author: Subhash Arockiadoss
"""

import gymnasium
import sys
sys.modules['gym'] = gymnasium
sys.modules['gym.spaces'] = gymnasium.spaces

import numpy as np
from pathlib import Path
from collections import deque
import time
import os
import json

LAPGYM_PATH = Path.home() / "surgical_robot_lapgym_ws" / "lap_gym" / "sofa_env"
if str(LAPGYM_PATH) not in sys.path:
    sys.path.insert(0, str(LAPGYM_PATH))

from sofa_env.scenes.pick_and_place.pick_and_place_env import (
    PickAndPlaceEnv,
    ObservationType,
    ActionType,
    RenderMode,
)


def run_pick_and_place_demo():
    print("=" * 60)
    print("Surgical RL — Pick and Place Demo")
    print("Task: Peg Transfer (FLS Curriculum Task)")
    print("Mode: Random actions (observation verification)")
    print("=" * 60)

    env = PickAndPlaceEnv(
        observation_type=ObservationType.STATE,
        action_type=ActionType.CONTINUOUS,
        render_mode=RenderMode.HEADLESS,
        image_shape=(124, 124),
        frame_skip=3,
        time_step=0.01,
    )

    print("\nInitialising SOFA simulation...")
    observation = env.reset()
    print(f"Environment reset complete.")
    print(f"Observation shape: {observation.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # Run with random actions — just verify the environment works
    # This is NOT an RL agent — purely verification
    fps_list = deque(maxlen=100)
    step_count = 0
    total_reward = 0.0
    max_steps = 200   # limit random run to 200 steps

    print("\nRunning with random actions (max 200 steps)...")
    print(f"{'Step':>6} | {'Reward':>8} | {'Total R':>8} | {'FPS':>6}")
    print("-" * 40)

    done = False
    while not done and step_count < max_steps:
        step_start = time.time()

        # Random action — agent does not know what it is doing
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        fps = 1.0 / (time.time() - step_start + 1e-9)
        fps_list.append(fps)
        step_count += 1
        total_reward += reward

        if step_count % 20 == 0:
            print(
                f"{step_count:>6} | {reward:>8.4f} | "
                f"{total_reward:>8.4f} | {np.mean(fps_list):>6.1f}"
            )

    print("\n" + "=" * 60)
    print("Pick and Place Demo Complete")
    print(f"  Steps run:        {step_count}")
    print(f"  Total reward:     {total_reward:.4f}")
    print(f"  Mean FPS:         {np.mean(fps_list):.1f}")
    print(f"  Goal reached:     {info.get('goal_reached', 'N/A')}")
    print(f"  Obs shape:        {observation.shape}")
    print(f"  Action space dim: {env.action_space.shape}")
    print("=" * 60)

    print("\nKey observations vs tissue retraction:")
    print(f"  time_step=0.01 (vs 0.1) — 10x finer physics timestep")
    print(f"  Articulated gripper with jaw control")
    print(f"  Deformable torus/rope object on rigid pegs")

    try:
        env.close()
    except Exception:
        pass

    os._exit(0)


if __name__ == "__main__":
    run_pick_and_place_demo()