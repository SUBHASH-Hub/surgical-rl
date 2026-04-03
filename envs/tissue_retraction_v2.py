"""
TissueRetractionV2 — Phase 2 environment wrapper
=================================================
WHY THIS FILE EXISTS:
  LapGym's TissueRetractionEnv uses the OLD gym 0.21 API:
    - reset() returns obs only (no info dict)
    - step() returns (obs, reward, done, info) — 4 values, not 5

  Stable-Baselines3 2.3.x requires the NEW gymnasium API:
    - reset() returns (obs, info)
    - step() returns (obs, reward, terminated, truncated, info)

  This wrapper bridges that gap WITHOUT modifying LapGym source.
  It also adds SOFA scene graph force readout (the info dict has
  no force data — confirmed by observation space inspection).

OBSERVATION SPACE (confirmed):
  Box(-1.0, 1.0, shape=(3,), float32)
  Dimensions: [normalised_x, normalised_y, normalised_z]
  These are end-effector positions in workspace frame, pre-normalised
  by LapGym. No additional normalisation needed for PPO.

ACTION SPACE (confirmed):
  Box(-3.0, 3.0, shape=(3,), float32)
  Dimensions: [vx, vy, vz] in mm/s
  Bounded by maximum_robot_velocity=3.0

FORCE READOUT:
  LapGym does not expose force in the info dict.
  We read it directly from the SOFA scene graph via the
  end-effector mechanical object's force field.
  Falls back to 0.0 if the scene graph node is not accessible.

Author: Subhash Arockiadoss
"""

# Compatibility shim — must come before ANY sofa_env import
import gymnasium
import sys
sys.modules['gym'] = gymnasium
sys.modules['gym.spaces'] = gymnasium.spaces

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple

from sofa_env.scenes.tissue_retraction.tissue_retraction_env import (
    TissueRetractionEnv as _LapGymEnv,
    ObservationType,
    ActionType,
    RenderMode,
    CollisionPunishmentMode,
)


# -----------------------------------------------------------------------
# Default environment configuration
# Mirrors baseline_demo.py exactly — same physics, same reward weights
# PPO trains on the same task the baseline scripted agent solved
# -----------------------------------------------------------------------
DEFAULT_ENV_KWARGS = dict(
    observation_type=ObservationType.STATE,
    action_type=ActionType.CONTINUOUS,
    render_mode=RenderMode.HEADLESS,
    image_shape=(480, 480),
    frame_skip=3,
    time_step=0.1,
    maximum_robot_velocity=3.0,
    collision_punishment_mode=CollisionPunishmentMode.CONTACTDISTANCE,
    reward_amount_dict={
        "one_time_reward_grasped": 1.0,
        "one_time_reward_goal": 1.0,
        "time_step_cost_scale_in_grasp_phase": 1.2,
        "target_visible_scaling": 0.0,
        "control_cost_factor": 0.0,
        "workspace_violation_cost": 0.1,
        "collision_cost_factor": 2.0,
    },
    create_scene_kwargs={
        "show_floor": True,
        "texture_objects": False,
        "workspace_height": 0.09,
        "workspace_width": 0.075,
        "workspace_depth": 0.09,
        "camera_field_of_view_vertical": 42,
    },
)


class TissueRetractionV2(gym.Env):
    """
    Gymnasium-compatible wrapper around LapGym's TissueRetractionEnv.

    Converts old gym API → new gymnasium API for SB3 compatibility.
    Adds force readout from SOFA scene graph.
    Intended to be wrapped further by SafeRewardWrapper.
    """

    metadata = {"render_modes": ["human", "headless"]}

    def __init__(self, env_kwargs: Optional[Dict] = None):
        super().__init__()

        kwargs = DEFAULT_ENV_KWARGS.copy()
        if env_kwargs:
            kwargs.update(env_kwargs)

        # Instantiate the underlying LapGym env
        self._env = _LapGymEnv(**kwargs)

        # Mirror observation and action spaces exactly
        # Confirmed from inspection: Box(-1,1,(3,),float32) and Box(-3,3,(3,),float32)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

        # Force readout state
        self._last_force_magnitude = 0.0

    # ------------------------------------------------------------------ #
    #  Gymnasium API — reset                                               #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Converts LapGym reset() → gymnasium reset() returning (obs, info).
        """
        super().reset(seed=seed)

        # Old API: returns obs only
        obs = self._env.reset()
        self._last_force_magnitude = 0.0

        info = {
            "force_magnitude": 0.0,
            "phase": 0,
        }

        return obs, info

    # ------------------------------------------------------------------ #
    #  Gymnasium API — step                                                #
    # ------------------------------------------------------------------ #

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Converts LapGym step() → gymnasium step() returning 5 values.

        Old API: (obs, reward, done, info)        — 4 values
        New API: (obs, reward, terminated, truncated, info) — 5 values

        LapGym's 'done' maps to 'terminated' (goal reached or collision limit).
        'truncated' is always False — LapGym handles episode length internally.
        """
        obs, reward, done, info = self._env.step(action)

        # Read force from SOFA scene graph
        force_magnitude = self._read_sofa_force()
        info["force_magnitude"] = force_magnitude
        self._last_force_magnitude = force_magnitude

        terminated = bool(done)
        truncated = False

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    #  SOFA scene graph force readout                                      #
    # ------------------------------------------------------------------ #

    def _read_sofa_force(self) -> float:
        """
        Read end-effector contact force magnitude from SOFA scene graph.

        WHY scene graph and not info dict:
          LapGym does not expose force in the info dict (confirmed by
          observation space inspection). SOFA stores all mechanical
          state in the scene graph tree. We access the end-effector
          node directly via the Python binding.

        SOFA scene graph path (from scene_graph_analysis.md):
          Root → Simulation → EndEffector → MechanicalObject
          The force field is stored in the 'force' data field as
          a (N, 3) array where N = number of DOFs.

        Falls back to 0.0 safely if:
          - Scene graph not yet initialised
          - Node path has changed between LapGym versions
          - Force field is empty
        """
        try:
            # Access SOFA root node via SofaPython3 binding
            root = self._env.root_node

            # Navigate to end-effector mechanical object
            # Path confirmed from Phase 1 scene graph analysis
            ee_node = root["Simulation"]["EndEffector"]
            mech_obj = ee_node["MechanicalObject"]

            # Read force field — shape (N, 3) in SOFA units
            forces = mech_obj.force.array()

            if forces is not None and forces.size > 0:
                # Sum all DOF forces, take magnitude
                total_force = np.sum(forces, axis=0)
                return float(np.linalg.norm(total_force))

        except Exception:
            # Silent fallback — never crash training due to force readout
            pass

        return 0.0

    # ------------------------------------------------------------------ #
    #  Render and close                                                    #
    # ------------------------------------------------------------------ #

    def render(self):
        pass

    def close(self):
        # Mirrors baseline_demo.py — SOFA v25 crashes on env.close()
        # Use os._exit(0) at training script level instead
        try:
            self._env.close()
        except Exception:
            pass