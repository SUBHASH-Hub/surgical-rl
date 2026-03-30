"""
search_for_point_demo.py

Phase 1 — LapGym Baseline Demo: Search for Point (Instrument Navigation)

Clinical context:
    Before any surgical manipulation, the robot must navigate its
    instrument tip to a target point within the surgical field.
    This task isolates pure navigation — reaching a 3D target position
    without any grasping or tissue interaction.

    Clinical relevance:
    - Models instrument placement before dissection or retraction
    - Pure position control — simplest surgical subtask
    - Useful as Phase 2 warm-up: agent learns workspace navigation
      before learning the more complex tissue retraction task

    Key difference from tissue retraction:
    - No tissue — no deformable physics, no compliance matrix
    - Much faster simulation (no FEM solver)
    - Single objective: reach target point
    - Useful baseline for understanding pure motion planning vs
      tissue interaction tasks

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

LAPGYM_PATH = Path.home() / "surgical_robot_lapgym_ws" / "lap_gym" / "sofa_env"
if str(LAPGYM_PATH) not in sys.path:
    sys.path.insert(0, str(LAPGYM_PATH))

from sofa_env.scenes.search_for_point.search_for_point_env import (
    SearchForPointEnv,
    ObservationType,
    ActionType,
    RenderMode,
)


def run_search_for_point_demo():
    print("=" * 60)
    print("Surgical RL — Search for Point Demo")
    print("Task: Instrument Navigation to Target")
    print("Mode: Random actions (observation verification)")
    print("=" * 60)

    env = SearchForPointEnv(
        observation_type=ObservationType.STATE,
        action_type=ActionType.CONTINUOUS,
        # render_mode=RenderMode.HEADLESS,
        render_mode=RenderMode.HUMAN,
    )

    print("\nInitialising SOFA simulation...")
    observation = env.reset()
    print(f"Environment reset complete.")
    print(f"Observation shape: {observation.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    fps_list = deque(maxlen=100)
    step_count = 0
    total_reward = 0.0
    max_steps = 200

    print("\nRunning with random actions (max 200 steps)...")
    print(f"{'Step':>6} | {'Reward':>8} | {'Total R':>8} | {'FPS':>6}")
    print("-" * 40)

    done = False
    while not done and step_count < max_steps:
        step_start = time.time()

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
    print("Search for Point Demo Complete")
    print(f"  Steps run:        {step_count}")
    print(f"  Total reward:     {total_reward:.4f}")
    print(f"  Mean FPS:         {np.mean(fps_list):.1f}")
    print(f"  Goal reached:     {info.get('goal_reached', 'N/A')}")
    print(f"  Obs shape:        {observation.shape}")
    print(f"  Action space dim: {env.action_space.shape}")
    print("=" * 60)

    print("\nKey observations vs tissue retraction:")
    print(f"  No FEM tissue — no compliance matrix computation")
    print(f"  Faster initialisation and simulation")
    print(f"  Pure navigation task — simpler reward structure")

    try:
        env.close()
    except Exception:
        pass

    os._exit(0)


if __name__ == "__main__":
    run_search_for_point_demo()