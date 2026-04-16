"""
Phase 3B — TissueRetractionV3
================================
Extends TissueRetractionV2 by replacing the ground-truth 7D state
observation with a 132D multimodal visual observation from the
perception pipeline.

Observation change:
  V2 (Phase 2): [tool_xyz(3), goal_xyz(3), phase(1)] = 7D   ← ground truth
  V3 (Phase 3): [features(128), xyz(3), phase(1)]    = 132D ← visual

The 132D observation is built at every step by:
  1. Capturing the RGB frame from the SOFA endoscopic camera
  2. Running it through the trained MobileNetV3 perception pipeline
  3. Concatenating visual features + estimated XYZ + phase flag

Why keep goal_xyz from the observation in V3:
  In Phase 2D, goal_xyz was part of the 7D obs so the agent knew
  exactly where the target is. In V3, the agent does NOT have
  direct access to goal coordinates — it must learn to reach the
  goal from visual information alone (the tissue deformation and
  instrument position visible in the camera). This is more
  clinically realistic — a real surgical robot cannot read the
  simulator's goal position directly.

  The phase flag (0.0 GRASPING / 1.0 RETRACTING) is retained
  because phase transitions are externally observable — the
  gripper state changes visually and is a legitimate sensor input.

Action space: unchanged from V2 — 3D velocity [vx, vy, vz]
Reward:       unchanged from V2 — SafeRewardWrapper wraps this env

Usage:
    env = TissueRetractionV3()
    obs, info = env.reset()
    # obs.shape == (132,)
    action = policy.predict(obs)
    obs, reward, done, truncated, info = env.step(action)

Author: Subhash Arockiadoss    
"""

from typing import Optional, Dict, Tuple

import numpy as np
import gymnasium as gym

# Import V2 as the physics base — reuse all SOFA setup unchanged
from envs.tissue_retraction_v2 import TissueRetractionV2

# Import perception pipeline
from envs.perception_pipeline import PerceptionPipeline, OBS_DIM


class TissueRetractionV3(gym.Env):
    """
    TissueRetractionV3 — visual multimodal observation wrapper.

    Wraps TissueRetractionV2 physics environment and replaces the
    7D state observation with a 132D visual observation.

    The underlying SOFA physics, reward function, action space, and
    episode termination conditions are all unchanged from V2.
    Only the observation space changes.
    """

    # 132D observation: 128 visual features + 3 xyz + 1 phase
    OBS_DIM = OBS_DIM   # 132

    def __init__(self, env_kwargs: Optional[Dict] = None,
                 perception_device: str = None):
        """
        Args:
            env_kwargs:         passed to TissueRetractionV2
                                render_mode must be 'headless' or 'human'
                                (not 'none') — camera is required for V3
            perception_device:  'cuda', 'cpu', or None (auto-detect)
        """
        super().__init__()

        # Default to headless render — camera required for visual obs
        if env_kwargs is None:
            env_kwargs = {"render_mode": "headless"}
        elif "render_mode" not in env_kwargs:
            env_kwargs["render_mode"] = "headless"

        # Build underlying physics environment
        # render_mode must NOT be 'none' — camera is needed every step
        self._env = TissueRetractionV2(env_kwargs=env_kwargs)

        # Build perception pipeline (loads MobileNetV3 checkpoint)
        self.perception = PerceptionPipeline(device=perception_device)

        # Action space: unchanged from V2
        # 3D velocity [vx, vy, vz] in [-1, 1] normalised
        self.action_space = self._env.action_space

        # Observation space: 132D continuous float32 in [-inf, inf]
        # Why not bounded: visual features from ReLU can be any non-negative value.
        # XYZ from Tanh is in [-1,1] but we don't constrain the full vector.
        self.observation_space = gym.spaces.Box(
            low  = -np.inf,
            high =  np.inf,
            shape = (self.OBS_DIM,),
            dtype = np.float32
        )

        # Cache the last raw state obs for phase flag extraction
        # Phase flag is at index 6 of the 7D V2 observation
        self._last_state_obs = None

    # ── Gym interface ──────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment and return first 132D visual observation.

        The first RGB frame is captured after reset so the camera has
        rendered the initial scene state.
        """
        # Reset underlying physics env — gets 7D state obs
        state_obs, info = self._env.reset(seed=seed, options=options)
        self._last_state_obs = state_obs

        # Capture first RGB frame and build 132D observation
        obs_132d = self._build_visual_obs(state_obs)
        return obs_132d, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Step physics, capture frame, return 132D visual observation.

        Capture order:
          1. Step physics with action
          2. Get new 7D state obs (for phase flag and reward computation)
          3. Capture RGB frame (now shows the post-step scene)
          4. Build 132D visual observation from frame + phase flag

        This keeps the visual observation aligned with the physics state
        after the action — the agent sees the consequence of its action.
        """
        # Step underlying physics environment
        state_obs, reward, done, truncated, info = self._env.step(action)
        self._last_state_obs = state_obs

        # Build 132D visual observation from post-step frame
        obs_132d = self._build_visual_obs(state_obs)

        return obs_132d, reward, done, truncated, info

    def render(self):
        """Forward render call to underlying env."""
        return self._env.render()

    def close(self):
        """Close underlying env."""
        self._env.close()

    # ── Internal ──────────────────────────────────────────────────────────

    def _build_visual_obs(self, state_obs: np.ndarray) -> np.ndarray:
        """
        Capture RGB frame and build 132D visual observation.

        Args:
            state_obs: 7D state observation from V2
                       layout: [tool_x, tool_y, tool_z,
                                goal_x, goal_y, goal_z, phase_flag]

        Returns:
            132D float32 array: [features(128), xyz(3), phase(1)]

        The phase flag is extracted from the 7D state obs (index 6).
        This is the only element from the ground-truth state that is
        retained — it represents the gripper state, which is externally
        observable and a legitimate sensor input.
        """
        # Extract phase flag from 7D state (index 6: 0.0 or 1.0)
        phase_flag = float(state_obs[6])

        # Capture RGB frame from SOFA camera
        # env._env._env is the raw SofaEnv (V3 wraps V2 wraps SofaEnv)
        try:
            rgb_frame = self._env._env.render()
        except Exception:
            # Fallback: if render fails, return zero visual features
            # This should not happen with render_mode='headless'
            return np.zeros(self.OBS_DIM, dtype=np.float32)

        if rgb_frame is None:
            return np.zeros(self.OBS_DIM, dtype=np.float32)

        # Store frame for external access (optical flow proxy)
        self._last_rgb_frame = rgb_frame.copy()  # copy — prevents same-object aliasing in optical flow
        # Run through perception pipeline → 132D observation
        obs_132d = self.perception.get_observation(rgb_frame, phase_flag)
        return obs_132d

    @property
    def unwrapped(self):
        """Access to underlying SofaEnv for diagnostics."""
        return self._env._env
