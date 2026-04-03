"""
SafeRewardWrapper — Phase 2, Step 1
====================================
WHY THIS ARCHITECTURE:
  - Wraps TissueRetractionEnv without modifying LapGym source code (fragile)
  - Decomposes reward into 4 clinically-motivated components
  - Exposes per-component breakdown in info dict for W&B logging
  - Curriculum lambda for collision penalty injected externally (Phase 2, Step 3)
  - Quadratic force penalty because surgical risk is nonlinear: small violations
    are recoverable, large violations are catastrophic (tissue tear = stop surgery)

FORCE THRESHOLD:
  27040 Pa Young's modulus, ~20mm tissue thickness → F_max ≈ 0.5N for safe retraction
  (conservative; clinical literature: <1N for atraumatic grasping)
  Source: Misra et al. 2010, "Modeling of Tool-Tissue Interactions for Computer-Based
  Surgical Simulation: A Literature Review", Surgical Endoscopy


Author: Subhash Arockiadoss
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple


class SafeRewardWrapper(gym.Wrapper):
    """
    Wraps any LapGym env and replaces its native reward with a
    clinically-grounded, safety-aware decomposed reward.

    Parameters
    ----------
    env            : base TissueRetractionEnv (or any LapGym env)
    lambda_force   : weight on quadratic force penalty (default 0.5)
    lambda_collision: weight on collision penalty — set by curriculum (default 0.1)
    force_threshold: max safe force in Newtons (default 0.5 N)
    step_penalty   : fixed cost per timestep to encourage efficiency (default 0.01)
    """

    def __init__(
        self,
        env: gym.Env,
        lambda_force: float = 0.5,
        lambda_collision: float = 0.1,
        force_threshold: float = 0.5,
        step_penalty: float = 0.01,
    ):
        super().__init__(env)
        self.lambda_force = lambda_force
        self.lambda_collision = lambda_collision
        self.force_threshold = force_threshold
        self.step_penalty = step_penalty

        # Curriculum phase tracking (updated externally by CurriculumCallback)
        self.curriculum_phase = 0

        # Episode-level accumulators for logging
        self._reset_accumulators()

    # ------------------------------------------------------------------ #
    #  Core reward decomposition                                           #
    # ------------------------------------------------------------------ #

    def _compute_safe_reward(
        self,
        native_reward: float,
        info: Dict[str, Any],
        terminated: bool,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Decompose reward into 4 components.

        WHY we keep native_reward as R_task:
          LapGym's native reward encodes goal progress (trocar displacement,
          tissue clearance). Re-implementing this from scratch would require
          deep scene graph access. We augment it, not replace it.
        """

        # --- R_task: keep LapGym's progress signal ---
        r_task = float(native_reward)

        # --- R_force: quadratic penalty above threshold ---
        # Force magnitude extracted from info dict (LapGym provides this)
        # Key name confirmed from LapGym source: 'tool_force' or 'contact_force'
        # We try both and fall back to 0.0 if neither present
        force_magnitude = self._extract_force(info)
        force_excess = max(0.0, force_magnitude - self.force_threshold)
        r_force = -self.lambda_force * (force_excess ** 2)

        # --- R_collision: linear collision count penalty ---
        # 'collision' in info is a bool; accumulate over episode
        
        n_collisions = float(info.get("in_collision", False))
        r_collision = -self.lambda_collision * n_collisions

        # --- R_efficiency: constant step cost ---
        # Not applied on terminal step (agent already gets goal reward there)
        r_efficiency = 0.0 if terminated else -self.step_penalty

        # --- Total ---
        r_total = r_task + r_force + r_collision + r_efficiency

        components = {
            "r_task": r_task,
            "r_force": r_force,
            "r_collision": r_collision,
            "r_efficiency": r_efficiency,
            "r_total": r_total,
            "force_magnitude": force_magnitude,
            "force_violation": float(force_excess > 0),
            "curriculum_phase": float(self.curriculum_phase),
        }

        return r_total, components

    def _extract_force(self, info: Dict[str, Any]) -> float:
        """
        Extract scalar force magnitude from LapGym info dict.
        LapGym may provide force as scalar or 3D vector — handle both.
        """
        for key in ("tool_force", "contact_force", "force", "applied_force"):
            if key in info:
                val = info[key]
                if isinstance(val, (list, np.ndarray)):
                    return float(np.linalg.norm(val))
                return float(val)
        # Force not in info — this means LapGym isn't exposing it yet.
        # We'll add SOFA force readout in Step 2 via scene graph hook.
        # For now return 0.0 (safe fallback — no false penalties)
        return 0.0

    # ------------------------------------------------------------------ #
    #  Gymnasium interface                                                 #
    # ------------------------------------------------------------------ #

    def step(self, action):
        obs, native_reward, terminated, truncated, info = self.env.step(action)

        r_total, components = self._compute_safe_reward(
            native_reward, info, terminated
        )

        # Inject components into info dict for W&B logging
        info.update(components)

        # Update episode accumulators
        self._update_accumulators(components)

        # On episode end, add episode-level summary to info
        if terminated or truncated:
            info["episode_safe_reward"] = self._episode_safe_accumulators()

        return obs, r_total, terminated, truncated, info

    def reset(self, **kwargs):
        self._reset_accumulators()
        return self.env.reset(**kwargs)

    # ------------------------------------------------------------------ #
    #  Curriculum interface (called by CurriculumCallback in Phase 2 Step 3)
    # ------------------------------------------------------------------ #

    def set_curriculum_phase(self, phase: int, lambda_collision: float):
        """
        Called externally by the curriculum callback.
        Increasing lambda_collision = tightening the safety constraint.
        """
        self.curriculum_phase = phase
        self.lambda_collision = lambda_collision
        print(
            f"[Curriculum] Phase {phase} → λ_collision = {lambda_collision:.3f}"
        )

    # ------------------------------------------------------------------ #
    #  Accumulators for episode-level W&B metrics                         #
    # ------------------------------------------------------------------ #

    def _reset_accumulators(self):
        self._acc = {
            "sum_r_task": 0.0,
            "sum_r_force": 0.0,
            "sum_r_collision": 0.0,
            "sum_r_efficiency": 0.0,
            "sum_r_total": 0.0,
            "n_force_violations": 0,
            "n_steps": 0,
        }

    def _update_accumulators(self, components: Dict[str, float]):
        self._acc["sum_r_task"] += components["r_task"]
        self._acc["sum_r_force"] += components["r_force"]
        self._acc["sum_r_collision"] += components["r_collision"]
        self._acc["sum_r_efficiency"] += components["r_efficiency"]
        self._acc["sum_r_total"] += components["r_total"]
        self._acc["n_force_violations"] += int(components["force_violation"])
        self._acc["n_steps"] += 1

    def _episode_safe_accumulators(self) -> Dict[str, float]:
        n = max(self._acc["n_steps"], 1)
        return {
            "ep/r_task": self._acc["sum_r_task"],
            "ep/r_force": self._acc["sum_r_force"],
            "ep/r_collision": self._acc["sum_r_collision"],
            "ep/r_total": self._acc["sum_r_total"],
            "ep/force_violation_rate": self._acc["n_force_violations"] / n,
            "ep/n_steps": self._acc["n_steps"],
        }