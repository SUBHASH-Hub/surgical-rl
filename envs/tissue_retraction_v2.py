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

OBSERVATION SPACE (Phase 2B, confirmed):
  Box(-1.0, 1.0, shape=(7,), float32)
  Dimensions: [tool_x, tool_y, tool_z,
               goal_x, goal_y, goal_z,
               phase_flag]
  tool_xyz: end-effector position, normalised by workspace half-dims
  goal_xyz: grasping or end position, normalised by _WS_HALF
  phase_flag: 0.0=GRASPING, 1.0=RETRACTING

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
    # Workspace dimensions for normalising goal position to [-1, 1]
    # Matches create_scene_kwargs in DEFAULT_ENV_KWARGS
    # workspace_width=0.075, workspace_height=0.090, workspace_depth=0.090
    _WS_HALF = np.array([0.0375, 0.045, 0.045], dtype=np.float32)

    metadata = {"render_modes": ["human", "headless"]}

    def __init__(self, env_kwargs: Optional[Dict] = None):
        super().__init__()

        kwargs = DEFAULT_ENV_KWARGS.copy()
        if env_kwargs:
            kwargs.update(env_kwargs)

        # Instantiate the underlying LapGym env
        self._env = _LapGymEnv(**kwargs)
        
        
        # Action space unchanged
        self.action_space = self._env.action_space

        # 7D observation: [tool_xyz(3), goal_xyz(3), phase(1)]
        # WHY 7D: Phase 2A proved agent cannot learn task without seeing
        # goal position. Tool position alone gives no directional signal.
        # Goal position allows agent to compute distance and direction.
        # Phase flag tells agent which target to aim for.
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32,
        )

        # Cache goal position — updated each reset
        self._current_goal_norm = np.zeros(3, dtype=np.float32)
        self._current_phase = 0

        # Force readout state
        self._last_force_magnitude = 0.0
        # Episode step counter and limit
        # WHY 500: baseline solves in ~247 steps. 500 gives PPO 2x
        # the baseline time to explore before forced reset.
        # Without this limit, random policy never terminates — no learning.
        self._episode_steps = 0
        self._max_episode_steps = 500

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
        Returns 7D observation: [tool_xyz, goal_xyz_normalised, phase]
        Goal position is _grasping_position normalised to [-1,1] using
        workspace half-dimensions confirmed from inspection.
        """
        super().reset(seed=seed)

        # Old API: returns obs only
        obs_3d = self._env.reset()
        self._last_force_magnitude = 0.0
        self._episode_steps = 0

        # Get and normalise grasping position (Phase 0 target)
        grasping_raw = np.array(
            self._env._grasping_position, dtype=np.float32
        )
        self._current_goal_norm = np.clip(
            grasping_raw / self._WS_HALF, -1.0, 1.0
        )
        self._current_phase = 0.0

        obs_7d = np.concatenate([
            obs_3d.astype(np.float32),
            self._current_goal_norm,
            [self._current_phase],
        ]).astype(np.float32)

        info = {
            "force_magnitude": 0.0,
            "phase": 0,
            "goal_norm": self._current_goal_norm.copy(),
        }

        return obs_7d, info

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
        
        Returns 7D observation.
        Updates goal target when phase switches from GRASPING to RETRACTING.
        """
        obs_3d, reward, done, info = self._env.step(action)

        # Read force from SOFA scene graph
        force_magnitude = self._read_sofa_force()
        info["force_magnitude"] = force_magnitude
        self._last_force_magnitude = force_magnitude
        
        # Replace with Episode step count and limit 
        self._episode_steps += 1
        terminated = bool(done)
        truncated = self._episode_steps >= self._max_episode_steps

        # Update goal when phase switches
        # Phase 0 = GRASPING → target is _grasping_position
        # Phase 1 = RETRACTING → target is _end_position
        current_phase = int(info.get("phase", 0))
        if current_phase != int(self._current_phase):
            if current_phase == 1:
                end_raw = np.array(
                    self._env._end_position, dtype=np.float32
                )
                self._current_goal_norm = np.clip(
                    end_raw / self._WS_HALF, -1.0, 1.0
                )
            self._current_phase = float(current_phase)

        obs_7d = np.concatenate([
            obs_3d.astype(np.float32),
            self._current_goal_norm,
            [self._current_phase],
        ]).astype(np.float32)

        info["goal_norm"] = self._current_goal_norm.copy()

        return obs_7d, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    #  SOFA scene graph force readout                                      #
    # ------------------------------------------------------------------ #

    def _read_sofa_force(self) -> float:
        """
        Force proxy via tissue intrusion cost.

        FINDING (08-09 April 2026):
        LapGym uses geometric collision detection, not SOFA contact
        forces. Direct force readout from MechanicalObject.force
        returns zero because constraint forces are stored internally
        by the BlockGaussSeidelConstraintSolver.

        The collision_cost field goes negative ONLY when the end-effector
        position is simultaneously inside the tissue bounding box on all
        three axes — which rarely occurs in normal approach trajectories
        (agent approaches from above, Y stays outside tissue Y range).

        ACCEPTED PROXY:
        force_magnitude returns 0.0 during training.
        The SafeRewardWrapper's r_collision term provides equivalent
        safety enforcement through curriculum-scaled geometric penalty.
        Tissue intrusion distance (CONTACTDISTANCE mode) correlates
        with contact force via Young's modulus = 27040 Pa.

        FUTURE WORK:
        Access BlockGaussSeidelConstraintSolver lambda values directly,
        or instrument the tissue FEM node for internal stress readout.
        """
        try:
            reward_info = self._env.reward_info
            collision_cost = reward_info.get("collision_cost", 0.0)
            if collision_cost is not None and float(collision_cost) < 0.0:
                scaling = float(self._env._reward_scaling_factor)
                return abs(float(collision_cost)) / scaling
        except Exception:
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