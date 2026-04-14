#!/usr/bin/env python3
"""
Phase 3B — PPO Training on Visual Observations
================================================
Retrains the PPO agent on the 132D multimodal visual observation from
TissueRetractionV3. Compares performance against Phase 2D ground-truth baseline.

Observation:  132D [visual_features(128), estimated_xyz(3), phase(1)]
              vs Phase 2D: 7D [tool_xyz(3), goal_xyz(3), phase(1)]

Expected result:
  Lower absolute performance than Phase 2D — perception adds noise to the
  XYZ estimate (5.1px error) and the agent no longer has access to goal_xyz.
  This is the clinically honest result: a real surgical robot cannot read
  ground-truth coordinates from a simulator.

  The value of Phase 3B is demonstrating the system works end-to-end
  with visual observations — not that it beats ground-truth performance.

PPO config: same as Phase 2D (750k steps, MlpPolicy, 256×256)
  Why same config: fair comparison — only the observation changes.
  The reward function, curriculum, action space are all unchanged.

W&B logging: surgical-rl-phase3 project
  Comparison tag: phase3b-visual-obs

Run from repo root:
    python3 scripts/train_ppo_visual.py

Outputs:
    logs/checkpoints/phase3b_ppo_visual_{timestamp}/
        ppo_visual_final.zip    ← final checkpoint
        ppo_visual_best.zip     ← best mean reward checkpoint
    docs/phase3/phase3b_results.md  ← auto-generated results

Author: Subhash Arockiadoss    
"""

# ── gym → gymnasium shim ──────────────────────────────────────────────────
import sys
import gymnasium
sys.modules['gym'] = gymnasium
sys.modules['gym.spaces'] = gymnasium.spaces
# ─────────────────────────────────────────────────────────────────────────

import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback
)
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, '.')
from envs.tissue_retraction_v3 import TissueRetractionV3

# ─────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# All values match Phase 2D exactly — only observation space changes.
# This ensures a fair comparison: any performance difference is due to
# the visual observation, not a hyperparameter change.
# ─────────────────────────────────────────────────────────────────────────

# --- Training ---
TOTAL_TIMESTEPS = 750_000   # matches Phase 2D — fair comparison
SEED            = 42

# --- PPO hyperparameters (identical to Phase 2D) ---
PPO_CONFIG = {
    "policy":        "MlpPolicy",   # standard MLP — obs is now 132D vector
    "learning_rate": 3e-4,
    "n_steps":       2048,
    "batch_size":    64,
    "n_epochs":      10,
    "gamma":         0.99,
    "gae_lambda":    0.95,
    "clip_range":    0.2,
    "ent_coef":      0.0,
    "vf_coef":       0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": {
        "net_arch": [256, 256],  # same 256×256 architecture as Phase 2D
    },
    "verbose": 1,
    "seed":    SEED,
}

# --- Curriculum (identical to Phase 2D step-based) ---
# λ_collision increases in two steps to avoid catastrophic shock
# Phase 0: 0–149,999 steps, λ=0.1  (explore freely)
# Phase 1: 150,000–349,999 steps, λ=0.3 (build collision awareness)
# Phase 2: 350,000+ steps, λ=0.5 (enforce safety)
CURRICULUM = [
    (0,       0.1),
    (150_000, 0.3),
    (350_000, 0.5),
]

# --- Checkpoint ---
TIMESTAMP  = datetime.now().strftime("%Y%m%d_%H%M%S")
CKPT_DIR   = Path(f"logs/checkpoints/phase3b_ppo_visual_{TIMESTAMP}")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# --- W&B ---
# Set to True to enable W&B logging (requires wandb login)
USE_WANDB  = True

# ─────────────────────────────────────────────────────────────────────────
# CURRICULUM CALLBACK
# ─────────────────────────────────────────────────────────────────────────

class CurriculumCallback(BaseCallback):
    """
    Step-based curriculum: increases λ_collision at fixed timestep triggers.
    Identical logic to Phase 2D CurriculumCallback in train_ppo.py.

    Why step-based not episode-based:
      Step-based triggers are reproducible regardless of episode length.
      Episode-based would fire at different actual timesteps depending
      on how quickly the agent reaches the goal.
    """

    def __init__(self, curriculum: list, verbose: int = 1):
        super().__init__(verbose)
        self.curriculum  = sorted(curriculum, key=lambda x: x[0])
        self.phase_idx   = 0
        self.current_lam = curriculum[0][1]

    def _on_step(self) -> bool:
        n_steps = self.num_timesteps

        # Check if we should advance to next curriculum phase
        while (self.phase_idx + 1 < len(self.curriculum) and
               n_steps >= self.curriculum[self.phase_idx + 1][0]):
            self.phase_idx  += 1
            new_lam          = self.curriculum[self.phase_idx][1]
            old_lam          = self.current_lam
            self.current_lam = new_lam

            # Update λ in the SafeRewardWrapper
            try:
                # Access through DummyVecEnv → Monitor → V3 → V2 → SafeRewardWrapper
                inner_env = self.training_env.envs[0]
                while hasattr(inner_env, 'env'):
                    inner_env = inner_env.env
                if hasattr(inner_env, 'lambda_collision'):
                    inner_env.lambda_collision = new_lam
                    if self.verbose:
                        print(f"\n  [Curriculum] step={n_steps:,}  "
                              f"λ_collision: {old_lam} → {new_lam}")
            except Exception as e:
                if self.verbose:
                    print(f"\n  [Curriculum] λ update failed: {e}")

        return True


# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 62)
    print("  Phase 3B — PPO Training on Visual Observations")
    print("=" * 62)
    print(f"  Observation: 132D [features(128), xyz(3), phase(1)]")
    print(f"  Timesteps:   {TOTAL_TIMESTEPS:,}")
    print(f"  Checkpoint:  {CKPT_DIR}")
    print("=" * 62 + "\n")

    # ── W&B initialisation ────────────────────────────────────────────────
    if USE_WANDB:
        try:
            import wandb
            wandb.init(
                project="surgical-rl-phase3",
                name=f"phase3b-visual-obs-{TIMESTAMP}",
                config={
                    "phase":        "3B",
                    "obs_type":     "visual_multimodal_132D",
                    "obs_dim":      132,
                    "backbone":     "MobileNetV3-Small",
                    "total_steps":  TOTAL_TIMESTEPS,
                    "curriculum":   CURRICULUM,
                    **PPO_CONFIG,
                },
                tags=["phase3b", "visual-obs", "ppo"],
            )
            print("  W&B initialised ✓")
        except Exception as e:
            print(f"  W&B init failed (continuing without): {e}")

    # ── Create environment ────────────────────────────────────────────────
    print("  Creating TissueRetractionV3 (visual observation) …")
    env = TissueRetractionV3(env_kwargs={"render_mode": "headless"})
    env = Monitor(env, str(CKPT_DIR / "monitor_train"))
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space:      {env.action_space}")
    print()

    # ── Build PPO model ───────────────────────────────────────────────────
    # MlpPolicy on 132D input: creates a 256×256 MLP automatically
    # The policy sees: [visual_features(128) | xyz(3) | phase(1)]
    # and learns to map this to 3D velocity actions
    print("  Building PPO model …")
    model = PPO(env=env, **PPO_CONFIG)

    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  Policy parameters: {total_params:,}")
    print(f"  (Phase 2D had 136,711 params — larger obs → larger network)")
    print()

    # ── Callbacks ─────────────────────────────────────────────────────────
    callbacks = [
        # Save checkpoint every 50k steps
        CheckpointCallback(
            save_freq=50_000,
            save_path=str(CKPT_DIR),
            name_prefix="ppo_visual",
            verbose=1,
        ),
        # Step-based curriculum — matches Phase 2D exactly
        CurriculumCallback(curriculum=CURRICULUM, verbose=1),
    ]

    # ── Training ──────────────────────────────────────────────────────────
    print(f"  Starting training — {TOTAL_TIMESTEPS:,} steps …\n")
    t_start = time.time()

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=True,
    )

    elapsed = time.time() - t_start
    print(f"\n  Training complete — {elapsed/60:.1f} minutes")

    # ── Save final checkpoint ──────────────────────────────────────────────
    final_ckpt = CKPT_DIR / "ppo_visual_final"
    model.save(str(final_ckpt))
    print(f"  Final checkpoint saved → {final_ckpt}.zip")

    # ── Save config for reproducibility ───────────────────────────────────
    import json
    config = {
        "phase":            "3B",
        "timestamp":        TIMESTAMP,
        "total_timesteps":  TOTAL_TIMESTEPS,
        "obs_dim":          132,
        "obs_layout":       "features(128) + xyz(3) + phase(1)",
        "backbone":         "MobileNetV3-Small",
        "tip_ckpt":         "models/tip_detector/mobilenetv3_tip_best.pth",
        "curriculum":       CURRICULUM,
        "ppo_config":       {k: v for k, v in PPO_CONFIG.items()
                            if k != "policy_kwargs"},
        "elapsed_seconds":  elapsed,
        "checkpoint_dir":   str(CKPT_DIR),
    }
    config_path = CKPT_DIR / "training_config.json"
    json.dump(config, open(config_path, "w"), indent=2)
    print(f"  Config saved → {config_path}")

    if USE_WANDB:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

    print("\n" + "=" * 62)
    print("  Phase 3B training complete.")
    print(f"  Run eval_agent_visual.py to compare vs Phase 2D baseline.")
    print("=" * 62 + "\n")

    os._exit(0)


if __name__ == "__main__":
    main()