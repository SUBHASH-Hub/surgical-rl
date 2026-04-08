# Phase 2A — PPO Training Results and Analysis

## Overview

Phase 2A implemented the complete PPO training infrastructure for
autonomous tissue retraction. A 500k-step training run was completed
on 07-08 April 2026. The value function converged perfectly but the
agent did not complete the surgical task.

**Phase 2A status: infrastructure complete, task learning incomplete**
**Phase 2B objective: enrich observation space with goal position**

---

## What Was Built

### New files
envs/safe_reward.py          — SafeRewardWrapper: 4-component reward
envs/tissue_retraction_v2.py — Gymnasium bridge + 500-step limit
agents/ppo_config.yaml       — PPO hyperparameters (256x256 ReLU)
configs/phase2_baseline.yaml — Curriculum + experiment config
scripts/train_ppo.py         — Training entry point with W&B + tmux

### Architecture
PPO (SB3 2.3.2, MlpPolicy)
Actor:  obs(3) → 256 → 256 → action(3)   ReLU
Critic: obs(3) → 256 → 256 → value(1)    ReLU
Total parameters: 134,663
Device: CUDA (GTX 1650)

### Reward decomposition
R_total = R_task + R_force + R_collision + R_efficiency
R_task       : LapGym native reward (distance to goal)
R_force      : -lambda_f * max(0, F - 0.5N)²  quadratic penalty
R_collision  : -lambda_c * in_collision        curriculum scaled
R_efficiency : -0.01 per step

### Curriculum phases
| Phase | Episodes | lambda_collision | Clinical intent |
|-------|----------|----------------- |-----------------|
| 0     | 0–499    | 0.1              | Allow free exploration |
| 1     | 500–999  | 0.3              | Moderate collision awareness |
| 2     | 1000+    | 0.8                | Strict safety enforcement |

---

## Training Run — 07 April 2026

| Parameter | Value |
|-----------|-------|
| W&B run | `phase2_ppo_tissue_retraction_20260407_215855` |
| Total steps | 501,760 |
| Wall-clock time | 8 hours 27 minutes |
| Simulation FPS | 15–16 (GTX 1650 CUDA) |
| PPO updates | 2,440 |
| Episodes completed | ~1,003 |

### Key metrics progression

| Checkpoint | ep_rew_mean | explained_variance | value_loss |
|------------|-------------|-------------------|------------|
| Step 2,048 | -468 | -0.011 (random) | 65.3 |
| Step 6,144 | -450 | 0.712 | 15.6 |
| Step 8,192 | -442 | 0.899 | 9.63 |
| Step 495,616 | -359 | 0.9999 | 0.00656 |
| Step 501,760 | -359 | 0.9999 | 0.00608 |

### Curriculum execution (confirmed from log)
Phase 1 triggered: episode 500  → lambda_collision = 0.30
Phase 2 triggered: episode 1000 → lambda_collision = 0.80
at step ~499,999 (end of training)

---

## Phase 2 Targets — Scorecard

| Metric | Baseline | Target | Phase 2A Result | Status |
|--------|----------|--------|-----------------|--------|
| Episode steps | 247 | < 200 | 500 (always truncated) | Not met |
| Total reward | -165.54 | > -100 | -359 | Not met |
| Collision steps | 49 | < 20 | ~500 per episode | Not met |
| Force violations | not measured | < 5% | 0%* | Met* |
| Value convergence | — | > 0.9 | 0.9999 | Exceeded |
| Curriculum execution | — | 3 phases | All 3 fired | Met |

*Force violations = 0% because SOFA scene graph force hook
returns 0.0 fallback — real force data not yet wired.
To be fixed in Phase 2B.

---

## Root Cause Analysis — Why Targets Were Not Met

### Primary cause: observation blindness

Current observation space:
obs = [x_tool, y_tool, z_tool]   shape=(3,), Box(-1,1)

The agent sees only its own position. It cannot see:
- Where the grasping target is located
- Distance to target
- Current task phase (GRASPING vs RETRACTING)
- Tissue boundary positions

Clinical analogy: a surgeon operating with eyes closed, knowing
only where their hand is but not where the tissue is.

### Secondary cause: sparse reward

LapGym's goal bonus only triggers when BOTH grasping AND
retraction complete. A random policy has near-zero probability
of achieving this in 500 steps. The agent never experienced
a positive terminal reward, so it never learned what leads to
task completion.

### Evidence from log
steps_in_grasping_phase:   300/300 (always in grasping, never transitions)
steps_in_retraction_phase: 0
goal_reached:              False  (every episode)
ep_len_mean:               500    (always hits step limit, never goal)

### What did improve
- r_task improved from -423 → -200 (agent learned to stay
  closer to grasping position than random policy)
- Value function converged perfectly (0.9999)
- Curriculum executed correctly
- No crashes over 8.5 hours of SOFA simulation

---

## Phase 2B Plan — Goal-Conditioned Observation

### The fix
Enrich observation from 3D to 7D:
PHASE 2A (current):
obs = [x_tool, y_tool, z_tool]
PHASE 2B (planned):
obs = [x_tool, y_tool, z_tool,
x_goal, y_goal, z_goal,
phase_flag]

Where:
- `x_goal, y_goal, z_goal` = grasping position (phase 0)
  or end position (phase 1), from `env._grasping_position`
  and `env._end_position`
- `phase_flag` = 0.0 (GRASPING) or 1.0 (RETRACTING)

### Expected impact
The agent can now compute distance-to-goal directly.
Gradient descent will learn "reduce this distance" from the
first few episodes. Expected to reach grasping position
within 50k steps vs never in 500k steps.

### Files to modify
envs/tissue_retraction_v2.py  — update observation_space to Box(7,)
update reset() and step() to return 7D obs
agents/ppo_config.yaml        — net_arch stays [256,256], input 3→7 auto
configs/phase2_baseline.yaml  — update observation notes

### Network changes
Input layer changes from `Linear(3→256)` to `Linear(7→256)`.
All other layers unchanged. Fresh training required (saved
3D policy weights are incompatible with 7D input).

---

## Technical Fixes Applied During Phase 2A

| Issue | Root cause | Fix applied |
|-------|-----------|-------------|
| `done` never True | LapGym has no step limit | Added `_max_episode_steps=500` in TissueRetractionV2 |
| safe_reward not logging | Monitor strips info dict | Added `_last_episode_data` attr on SafeRewardWrapper |
| wandb step warning | step= conflicts with tensorboard sync | Removed `step=self.num_timesteps` from wandb.log |
| gym vs gymnasium | LapGym uses old gym 0.21 API | Shim: sys.modules['gym'] = gymnasium |
| Force key missing | LapGym does not expose force | _extract_force returns 0.0 fallback |

---

## Compatibility Notes

Same as Phase 1 — all Phase 1 compatibility fixes remain active.
See `docs/compatibility_fixes.md`.

New dependency versions:
stable-baselines3 == 2.3.2
wandb             == 0.25.1   (upgraded from 0.17.0 — NumPy 2.0 fix)
gymnasium         == 0.29.1
torch             == 2.11.0+cu128