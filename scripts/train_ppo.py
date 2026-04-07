"""
train_ppo.py — Phase 2 PPO Training Entry Point
================================================
Connects every Phase 2 component into a single training run:
  - TissueRetractionV2: gymnasium-compatible SOFA env
  - SafeRewardWrapper: decomposed safety-aware reward
  - PPO (SB3): policy optimisation with clipped surrogate objective
  - CurriculumCallback: tightens collision penalty across 3 phases
  - WandbCallback: logs all reward components and curriculum state
  - CheckpointCallback: saves model every 50k steps

Clinical context:
  We are training an agent to perform autonomous tissue retraction
  for Calot's triangle exposure in laparoscopic cholecystectomy.
  Phase 2 target: under 200 steps, under 20 collision steps,
  reward above -100, force violations under 5%.

Usage:
  source setup_env.sh
  python3 scripts/train_ppo.py
  python3 scripts/train_ppo.py --config configs/phase2_baseline.yaml

Author: Subhash Arockiadoss
"""

# -----------------------------------------------------------------------
# Compatibility shim — MUST be before any sofa_env import
# -----------------------------------------------------------------------
import gymnasium
import sys
sys.modules['gym'] = gymnasium
sys.modules['gym.spaces'] = gymnasium.spaces

import os
import argparse
import yaml
import numpy as np
import torch.nn as nn
from pathlib import Path
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

import wandb
from wandb.integration.sb3 import WandbCallback

sys.path.insert(0, str(Path(__file__).parent.parent))
from envs.tissue_retraction_v2 import TissueRetractionV2
from envs.safe_reward import SafeRewardWrapper


# ======================================================================
# Curriculum Callback
# ======================================================================

class CurriculumCallback(BaseCallback):
    """
    Advances curriculum phases based on episode count.

    WHY a callback and not inside the env:
      The curriculum schedule is a training decision, not an env decision.
      The env applies the penalty; the callback decides when to tighten it.
      This separation means you can change the curriculum without
      touching the reward wrapper code.

    HOW it works:
      After each episode, checks if episode count has crossed a phase
      threshold. If yes, calls env.set_curriculum_phase() to update
      lambda_collision in the SafeRewardWrapper.
    """

    def __init__(self, curriculum_cfg: list, env: SafeRewardWrapper, verbose=0):
        super().__init__(verbose)
        self.curriculum_cfg = curriculum_cfg
        self.env = env
        self.current_phase = 0
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Check if any episode finished this step
        dones = self.locals.get("dones", [])
        self.episode_count += sum(dones)

        # Check if we should advance to next phase
        for phase_cfg in self.curriculum_cfg:
            phase = phase_cfg["phase"]
            trigger = phase_cfg["trigger_episodes"]
            lambda_c = phase_cfg["lambda_collision"]

            if (self.episode_count >= trigger and
                    phase > self.current_phase):
                self.env.set_curriculum_phase(phase, lambda_c)
                self.current_phase = phase

                if self.verbose:
                    print(f"\n[Curriculum] Advanced to Phase {phase} "
                          f"at episode {self.episode_count} "
                          f"(λ_collision={lambda_c})")

                # Log phase transition to W&B
                if wandb.run is not None:
                    wandb.log({
                        "curriculum/phase": phase,
                        "curriculum/lambda_collision": lambda_c,
                        "curriculum/episode": self.episode_count,
                    })

        return True  # True = continue training


# ======================================================================
# Safe Reward Logging Callback
# ======================================================================

class SafeRewardLoggerCallback(BaseCallback):
    """
    Logs decomposed reward components to W&B at episode end.

    WHY a separate callback for this:
      WandbCallback logs SB3's built-in metrics (policy loss, value loss).
      Our custom metrics (r_task, r_force, r_collision, force_violation_rate)
      live in the info dict. This callback extracts them and logs separately.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._episode_rewards = []
        self._episode_lengths = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, (info, done) in enumerate(zip(infos, dones)):
            if not done:
                continue

            # When done=True, SB3's VecEnv puts terminal info
            # under info["terminal_observation"] and the original
            # info under info itself — check both locations
            ep_data = None

            # Location 1: directly in info (before Monitor strips)
            if "episode_safe_reward" in info:
                ep_data = info["episode_safe_reward"]

            # Location 2: SB3 VecEnv stores final info here
            if ep_data is None and "final_info" in info:
                final = info["final_info"]
                if final and "episode_safe_reward" in final:
                    ep_data = final["episode_safe_reward"]

            # Location 3: read directly from the env wrapper
            if ep_data is None:
                try:
                    # Unwrap: Monitor → SafeRewardWrapper
                    safe_env = self.training_env.envs[0].env
                    if hasattr(safe_env, '_last_episode_data'):
                        ep_data = safe_env._last_episode_data
                except Exception:
                    pass

            if ep_data is None:

                continue

            if wandb.run is not None:

                wandb.log({
                    "safe_reward/r_task": ep_data.get("ep/r_task", 0),
                    "safe_reward/r_force": ep_data.get("ep/r_force", 0),
                    "safe_reward/r_collision": ep_data.get("ep/r_collision", 0),
                    "safe_reward/r_total": ep_data.get("ep/r_total", 0),
                    "safe_reward/force_violation_rate": ep_data.get("ep/force_violation_rate", 0),
                    "safe_reward/episode_length": ep_data.get("ep/n_steps", 0),
                })

            if self.verbose:
                print(
                    f"  ep_len={ep_data.get('ep/n_steps',0):4d} | "
                    f"r_total={ep_data.get('ep/r_total',0):8.3f} | "
                    f"r_task={ep_data.get('ep/r_task',0):8.3f} | "
                    f"r_coll={ep_data.get('ep/r_collision',0):7.3f} | "
                    f"force_viol={ep_data.get('ep/force_violation_rate',0):.3f}"
                )

        return True


# ======================================================================
# Environment builder
# ======================================================================

def build_env(exp_cfg: dict) -> SafeRewardWrapper:
    """
    Builds the composed env stack from experiment config.
    Returns SafeRewardWrapper(TissueRetractionV2()).
    Wraps with Monitor for SB3 episode tracking.
    """
    wrapper_kwargs = exp_cfg["environment"]["wrapper_kwargs"]

    base_env = TissueRetractionV2()
    env = SafeRewardWrapper(
        base_env,
        lambda_force=wrapper_kwargs["lambda_force"],
        lambda_collision=wrapper_kwargs["lambda_collision"],
        force_threshold=wrapper_kwargs["force_threshold"],
        step_penalty=wrapper_kwargs["step_penalty"],
    )

    # Monitor wraps the env to track episode rewards and lengths
    # Required by SB3's EvalCallback and logging infrastructure
    env = Monitor(env)

    return env


# ======================================================================
# Model builder
# ======================================================================

def build_model(ppo_cfg: dict, env, run_name: str) -> PPO:
    """
    Instantiates PPO with hyperparameters from ppo_config.yaml.
    Handles policy_kwargs separately because activation_fn
    requires a Python class, not a string.
    """
    policy_kwargs = dict(
        net_arch=ppo_cfg["policy_kwargs"]["net_arch"],
        activation_fn=nn.ReLU,
        # WHY ReLU not Tanh: avoids vanishing gradients,
        # faster convergence on continuous control tasks
    )

    log_dir = Path(ppo_cfg["tensorboard_log"]) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    model = PPO(
        policy=ppo_cfg["policy"],
        env=env,
        learning_rate=ppo_cfg["learning_rate"],
        n_steps=ppo_cfg["n_steps"],
        batch_size=ppo_cfg["batch_size"],
        n_epochs=ppo_cfg["n_epochs"],
        gamma=ppo_cfg["gamma"],
        gae_lambda=ppo_cfg["gae_lambda"],
        clip_range=ppo_cfg["clip_range"],
        ent_coef=ppo_cfg["ent_coef"],
        vf_coef=ppo_cfg["vf_coef"],
        max_grad_norm=ppo_cfg["max_grad_norm"],
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(log_dir),
        verbose=ppo_cfg["verbose"],
        seed=42,
    )

    return model


# ======================================================================
# Main training function
# ======================================================================

def train(args):
    # ------------------------------------------------------------------
    # Load configs
    # ------------------------------------------------------------------
    with open("agents/ppo_config.yaml") as f:
        ppo_cfg = yaml.safe_load(f)

    with open(args.config) as f:
        exp_cfg = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Run name — used for W&B, checkpoints, tensorboard
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{exp_cfg['experiment']['name']}_{timestamp}"

    print("=" * 60)
    print(f"Surgical RL — Phase 2 PPO Training")
    print(f"Run: {run_name}")
    print(f"Targets: steps<{exp_cfg['targets']['max_steps']}, "
          f"reward>{exp_cfg['targets']['min_reward']}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # W&B initialisation
    # ------------------------------------------------------------------
    wandb_cfg = exp_cfg.get("wandb", {})
    run = wandb.init(
        project=wandb_cfg.get("project", "surgical-rl-phase2"),
        name=run_name,
        config={**ppo_cfg, **exp_cfg},
        tags=wandb_cfg.get("tags", []),
        sync_tensorboard=True,
    )

    # ------------------------------------------------------------------
    # Build env and model
    # ------------------------------------------------------------------
    print("\nInitialising environment stack...")
    env = build_env(exp_cfg)

    print("Building PPO model...")
    model = build_model(ppo_cfg, env, run_name)

    print(f"\nPolicy architecture:")
    print(f"  Actor:  3 → 256 → 256 → 3  (ReLU)")
    print(f"  Critic: 3 → 256 → 256 → 1  (ReLU)")
    print(f"  Total parameters: "
          f"{sum(p.numel() for p in model.policy.parameters()):,}")

    # ------------------------------------------------------------------
    # Create checkpoint directory
    # ------------------------------------------------------------------
    checkpoint_dir = Path("logs/checkpoints") / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Build callbacks
    # ------------------------------------------------------------------

    # 1. Checkpoint every 50k steps
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=str(checkpoint_dir),
        name_prefix="ppo_tissue",
        verbose=1,
    )

    # 2. Curriculum — tightens collision penalty across phases
    # Unwrap Monitor to get SafeRewardWrapper
    safe_env = env.env
    curriculum_cb = CurriculumCallback(
        curriculum_cfg=exp_cfg["curriculum"]["phases"],
        env=safe_env,
        verbose=1,
    )

    # 3. Safe reward component logger
    reward_logger_cb = SafeRewardLoggerCallback(verbose=1)

    # 4. W&B callback — logs SB3 built-in metrics
    wandb_cb = WandbCallback(
        gradient_save_freq=1000,
        model_save_path=str(checkpoint_dir),
        verbose=1,
    )

    callbacks = CallbackList([
        checkpoint_cb,
        curriculum_cb,
        reward_logger_cb,
        wandb_cb,
    ])

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    print(f"\nStarting training: {ppo_cfg['total_timesteps']:,} timesteps")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"W&B run: {run.url}")
    print("-" * 60)

    try:
        model.learn(
            total_timesteps=ppo_cfg["total_timesteps"],
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n[Training interrupted by user]")

    # ------------------------------------------------------------------
    # Save final model
    # ------------------------------------------------------------------
    final_model_path = checkpoint_dir / "ppo_tissue_final"
    model.save(str(final_model_path))
    print(f"\nFinal model saved: {final_model_path}")

    wandb.finish()

    # Clean exit — bypasses SOFA/Python GIL crash on shutdown
    os._exit(0)


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2 PPO training for tissue retraction"
    )
    parser.add_argument(
        "--config",
        default="configs/phase2_baseline.yaml",
        help="Path to experiment config YAML",
    )
    args = parser.parse_args()
    train(args)