"""
baseline_demo.py

Phase 1 — Step 8: First LapGym Surgical Simulation Baseline

Clinical context:
    This script runs the TissueRetractionEnv with a scripted waypoint
    trajectory — no RL agent, no learning. The purpose is to verify
    the full simulation stack works end-to-end and to observe the
    physics of deformable tissue interaction before training begins.

    The task models Calot's triangle exposure in laparoscopic
    cholecystectomy: grasp the gallbladder fundus, retract to expose
    the operative field.

Author: Subhash Arockiadoss
"""
# Compatibility shim: sofa_env uses old 'gym' API, we have 'gymnasium'
# This makes 'import gym' resolve to gymnasium transparently
# Add CSV Logging
import gymnasium
import sys
sys.modules['gym'] = gymnasium
sys.modules['gym.spaces'] = gymnasium.spaces
import csv   # For csv logging 
import json  # For saving episode summary
import numpy as np
from pathlib import Path
from collections import deque
import time
import os                      # For clean exit

# ---------------------------------------------------------------------------
# Path setup — tell Python where to find sofa_env
# This is handled by setup_env.sh in production, but we add it here
# explicitly for clarity and IDE support
# ---------------------------------------------------------------------------
LAPGYM_PATH = Path.home() / "surgical_robot_lapgym_ws" / "lap_gym" / "sofa_env"
if str(LAPGYM_PATH) not in sys.path:
    sys.path.insert(0, str(LAPGYM_PATH))

from sofa_env.scenes.tissue_retraction.tissue_retraction_env import (
    TissueRetractionEnv,
    ObservationType,
    ActionType,
    RenderMode,
    CollisionPunishmentMode,
)
from sofa_env.scenes.tissue_retraction.sofa_objects.end_effector import (
    add_waypoints_to_end_effector,
)


def run_baseline_demo():
    """
    Run the tissue retraction environment with a scripted waypoint trajectory.

    This is NOT an RL agent — it is a deterministic demonstration that
    verifies the simulation stack works correctly and lets us observe
    tissue deformation physics before training begins.
    """

    print("=" * 60)
    print("Surgical RL — Phase 1 Baseline Demo")
    print("Task: Tissue Retraction (Calot's Triangle Exposure)")
    print("Mode: Scripted waypoints (no RL agent)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Environment configuration
    # ------------------------------------------------------------------
    env = TissueRetractionEnv(
        # Use STATE observations for baseline — 3 numbers (XYZ position)
        # We switch to RGB in Phase 3 when we add the perception pipeline
        observation_type=ObservationType.STATE,

        # Continuous velocity control — same as real surgical robot teleoperation
        action_type=ActionType.CONTINUOUS,

        # HUMAN mode opens the SOFA GUI window so we can see the simulation
        #render_mode=RenderMode.HUMAN,
        render_mode=RenderMode.HEADLESS,

        # Simulation parameters
        image_shape=(480, 480),
        frame_skip=3,           # 3 physics steps per env step
        time_step=0.1,          # 100ms per physics step
        maximum_robot_velocity=3.0,  # mm/s — conservative for safety

        # Collision punishment mode
        collision_punishment_mode=CollisionPunishmentMode.CONTACTDISTANCE,

        # Reward weights — baseline (no force constraint yet, added in Phase 2)
        reward_amount_dict={
            "one_time_reward_grasped": 1.0,
            "one_time_reward_goal": 1.0,
            "time_step_cost_scale_in_grasp_phase": 1.2,
            "target_visible_scaling": 0.0,   # disabled — no vision yet
            "control_cost_factor": 0.0,       # disabled for baseline
            "workspace_violation_cost": 0.1,
            "collision_cost_factor": 2.0,
        },

        # Scene configuration
        create_scene_kwargs={
            "show_floor": True,
            "texture_objects": False,
            "workspace_height": 0.09,
            "workspace_width": 0.075,
            "workspace_depth": 0.09,
            "camera_field_of_view_vertical": 42,
        },
    )

    # ------------------------------------------------------------------
    # Reset environment — initialises SOFA simulation
    # ------------------------------------------------------------------
    print("\nInitialising SOFA simulation...")
    observation = env.reset()
    print(f"Environment reset complete.")
    print(f"Observation shape: {observation.shape}")
    print(f"Action space: {env.action_space}")

    # ------------------------------------------------------------------
    # Set up scripted waypoints
    # This mimics the path a surgeon would guide the instrument:
    # 1. Move to safe height above tissue
    # 2. Approach grasping position
    # 3. Retract to target end position
    # ------------------------------------------------------------------
    end_effector = env.get_gripper()

    waypoints = [
        [0, 0.007, 0],           # lift to safe height
        [0.05, 0.007, 0.0],      # move laterally
        [0, 0.06, 0],            # approach from above
        list(env._grasping_position),   # descend to grasp point
        list(env._end_position),        # retract to target
    ]

    add_waypoints_to_end_effector(waypoints, end_effector)

    # ------------------------------------------------------------------
    # Create logs directory and open CSV file
    # This happens BEFORE the simulation loop starts
    # so we are ready to write from step 1
    # ------------------------------------------------------------------

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)   # creates logs/ folder if not present

    # Auto-generate unique filename using timestamp
    # Format: baseline_run_20260325_143022.csv
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = logs_dir / f"baseline_run_{timestamp}.csv"
    json_path = logs_dir / f"baseline_results_{timestamp}.json"
 
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
 
    # Write the header row — column names
    csv_writer.writerow([
        'step',          # which step number (1, 2, 3...)
        'reward',        # reward received at this step
        'total_reward',  # cumulative reward so far
        'phase',         # GRASPING or RETRACTING
        'obs_x',         # normalised X position of instrument [-1, 1]
        'obs_y',         # normalised Y position of instrument [-1, 1]
        'obs_z',         # normalised Z position of instrument [-1, 1]
        'fps',           # simulation speed at this step
        'in_collision',  # 1 if instrument collided with wrong tissue, 0 if not
        'goal_reached',  # 1 if task complete, 0 if still running
    ])
 

    # ------------------------------------------------------------------
    # Run simulation loop
    # ------------------------------------------------------------------
    no_action = np.zeros(3, dtype=np.float32)
    done = False
    step_count = 0
    total_reward = 0.0
    fps_list = deque(maxlen=100)

    print("\nRunning simulation...")
    print(f"{'Step':>6} | {'Reward':>8} | {'Total R':>8} | {'Phase':>10} | {'FPS':>6}")
    print("-" * 55)

    while not done:
        step_start = time.time()

        obs, reward, done, info = env.step(no_action)

        step_time = time.time() - step_start
        fps = 1.0 / (step_time + 1e-9)
        fps_list.append(fps)

        step_count += 1
        total_reward += reward
        
        # Define phase_name every step — used by both print and CSV writer
        phase_name = "GRASPING" if info["phase"] == 0 else "RETRACTING"

        # Print progress every 10 steps
        if step_count % 10 == 0:
            print(
                f"{step_count:>6} | {reward:>8.4f} | {total_reward:>8.4f} | "
                f"{phase_name:>10} | {np.mean(fps_list):>6.1f}"
            )

 
        # ------------------------------------------------------------------
        # Write one row to CSV for every single step
        # This runs every iteration of the loop — every step is recorded
        # ------------------------------------------------------------------
        csv_writer.writerow([
            step_count,
            round(reward, 6),
            round(total_reward, 6),
            phase_name,
            round(float(obs[0]), 6),    # normalised X
            round(float(obs[1]), 6),    # normalised Y
            round(float(obs[2]), 6),    # normalised Z
            round(fps, 2),
            # in_collision: 1 if collision cost was negative this step
            1 if (info.get('collision_cost') is not None and
                  info.get('collision_cost', 0) < 0) else 0,
            # goal_reached: 1 only on the final step
            1 if info.get('goal_reached', False) else 0,
        ])
 

        # Hold final pose when done
        if done:
            env.end_effector.set_pose(
                np.append(env._end_position, [0.0, 0.0, 0.0, 1.0])
            )

    # ------------------------------------------------------------------
    # Close the CSV file after the loop finishes
    # Always close files after writing — this flushes remaining data
    # and releases the file handle
    # ------------------------------------------------------------------
    csv_file.close()
    print(f"\nData logged to: {csv_path}")
    print(f"Summary saved to: {json_path}")
    print(f"Total rows written: {step_count}")
 

    # ------------------------------------------------------------------
    # Episode summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Episode Complete")
    print(f"  Total steps:          {step_count}")
    print(f"  Total reward:         {total_reward:.4f}")
    print(f"  Steps in grasping:    {info['steps_in_grasping_phase']}")
    print(f"  Steps in retraction:  {info['steps_in_retraction_phase']}")
    print(f"  Steps in collision:   {info['steps_in_collision']}")
    print(f"  Mean FPS:             {np.mean(fps_list):.1f}")
    print(f"  Goal reached:         {info['goal_reached']}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Clean exit — bypasses SOFA/Python GIL conflict on shutdown
    # env.close() triggers a known SOFA v25 crash on exit
    # os._exit(0) terminates immediately and cleanly
    # Save results before exiting
    # -----------------------------------------------------------------

    results = {
        'total_steps': step_count,
        'total_reward': round(total_reward, 4),
        'steps_in_grasping': info['steps_in_grasping_phase'],
        'steps_in_retraction': info['steps_in_retraction_phase'],
        'steps_in_collision': info['steps_in_collision'],
        'goal_reached': info['goal_reached'],
    }
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    try:
        env.close()
    except Exception:
        pass

    os._exit(0)

if __name__ == "__main__":
    run_baseline_demo()