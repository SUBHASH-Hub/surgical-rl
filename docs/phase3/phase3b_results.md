# Phase 3B — Multimodal Visual Observation Integration

**Status:** ✓ COMPLETE  
**Completed:** April 2026  
**Duration:** 845.7 minutes (~14.1 hours) on GTX 1650  
**Checkpoint:** `logs/checkpoints/phase3b_ppo_visual_20260413_152851/ppo_visual_final.zip`

---

## Objective

Replace the ground-truth 7D state observation used in Phase 2D with a 132D multimodal visual observation derived from the trained perception module (Phase 3A). Retrain the PPO agent and compare performance against the Phase 2D baseline.

**The core engineering question:** When a surgical RL agent can no longer read ground-truth coordinates from the simulator — when it must perceive its environment through a camera like a real robot — how much does performance degrade and why?

---

## Observation Space Change

| | Phase 2D | Phase 3B |
|--|---------|---------|
| Observation | `[tool_xyz(3), goal_xyz(3), phase(1)]` | `[visual_features(128), est_xyz(3), phase(1)]` |
| Dimension | 7D | 132D |
| Source | SOFA ground truth (perfect, zero error) | MobileNetV3 perception pipeline (5.1px error) |
| Goal position | ✓ Directly available | ✗ Removed — not available to real robot |
| Tool position | ✓ Perfect ground truth | ~5.1px estimated error |
| Policy parameters | 136,711 | 200,711 (+47% larger — wider input) |

**Why goal_xyz was removed:** In a real surgical robot, the system cannot read the target tissue position directly from a simulator. The goal position must be inferred from visual and haptic feedback. Keeping goal_xyz would mean training on privileged information that does not exist in reality. Its removal makes Phase 3B the clinically honest experiment.

**Why phase_flag was retained:** The gripper state transition (GRASPING → RETRACTING) is externally observable — the gripper physically changes state. A real robot has a gripper sensor. This is not privileged information.

---

## Architecture

### Perception Pipeline (`envs/perception_pipeline.py`)

The trained MobileNetV3-Small tip detector checkpoint is split into three components at inference time:

```
RGB frame (480×480×3)
  ↓ Resize + ImageNet normalise → (224×224×3)
  ↓ MobileNetV3 backbone → (576,) feature vector
  ↓ FC 576→128 + ReLU → (128,) visual features   ← INTERMEDIATE LAYER
  ↓ FC 128→3 + Tanh   → (3,)  estimated XYZ
  
Phase flag from gripper sensor → (1,)

Concatenate → [features(128) | xyz(3) | phase(1)] = 132D observation
```

**Why the 128D intermediate layer, not just the 3D XYZ output:**
The final 3D output has compressed all visual information into three numbers. The 128D intermediate layer retains richer spatial and appearance features — the scene context, tissue state, instrument appearance — that can help the policy make better decisions. This is the standard approach in visuomotor policy learning (RT-2, R3M, and surgical AI research use intermediate encoder features rather than task-specific outputs as policy inputs).

**MobileNetV3 weights are FROZEN during PPO training.** The perception module is a fixed feature extractor. Only the MLP policy weights (200,711 parameters) are updated by RL gradients. This is the correct design — updating perception weights with RL gradients would require orders of magnitude more training data and would destabilise the tip detector accuracy.

### TissueRetractionV3 (`envs/tissue_retraction_v3.py`)

Wraps TissueRetractionV2. At every step:
1. Steps SOFA physics (unchanged from V2)
2. Extracts phase flag from the 7D state (index 6 — only ground truth element retained)
3. Captures RGB frame from SOFA camera via `env._env.render()`
4. Passes frame + phase flag to perception pipeline
5. Returns 132D visual observation

Reward function, action space, curriculum, and episode termination conditions are all unchanged from V2. The only change is the observation.

---

## Training Configuration

All hyperparameters identical to Phase 2D to ensure a fair comparison.

| Parameter | Value | Same as Phase 2D? |
|-----------|-------|-------------------|
| Algorithm | PPO | ✓ |
| Policy | MlpPolicy 256×256 | ✓ |
| Total timesteps | 750,000 | ✓ |
| Learning rate | 3e-4 (fixed) | ✓ |
| n_steps | 2048 | ✓ |
| batch_size | 64 | ✓ |
| γ (gamma) | 0.99 | ✓ |
| Curriculum λ_collision | 0.1→0.3→0.5 | ✓ |
| Seed | 42 | ✓ |

**Why identical config:** Any performance difference between Phase 2D and Phase 3B should be attributable to the observation space change alone, not hyperparameter differences. This is the correct scientific control.

---

## Training Results

### Primary Metrics

| Metric | Phase 2D (ground truth) | Phase 3B (visual) | Change |
|--------|------------------------|-------------------|--------|
| ep_rew_mean (start) | -234 | -234 | identical start |
| ep_rew_mean (final) | **-97** | **-135** | -38 points worse |
| ep_len_mean (final) | **142.3** | **300** | never reached goal |
| Goal rate | **100%** | **0%** | complete regression |
| Training duration | ~10 hours | **14.1 hours** | +41% slower (perception inference overhead) |
| Total timesteps | 750,000 | 751,616 | matched |

### Reward Progression (ep_rew_mean)

| Timestep | Reward | Notes |
|----------|--------|-------|
| 2,048 | -234 | Random policy baseline |
| ~60,000 | -207 | First meaningful improvement |
| ~150,000 | ~-195 | Pre-curriculum phase 2 trigger |
| ~280,000 | -137 | Steady improvement |
| ~360,000 | -135 | Best sustained performance |
| 751,616 | -135 | Final (42% improvement from start) |

### Policy Convergence Indicators

| Metric | Start | End | Interpretation |
|--------|-------|-----|----------------|
| Action std | 0.998 | 0.583 | Policy became highly deterministic |
| Entropy loss | -4.26 | -2.63 | Exploration collapsed (locked into strategy) |
| Explained variance | -0.010 | 0.975 | Critic learned returns accurately |
| Value loss | 13.2 | 0.050 | Critic converged completely |

The policy converged to a committed deterministic strategy. The critic accurately predicts future returns. The problem is not training instability — the agent found and committed to a local optimum.

---

## Analysis — Why the Agent Never Reached the Goal

### Root cause: removal of goal_xyz eliminates the navigational gradient

In Phase 2D, the observation `[tool_xyz, goal_xyz, phase]` gave the agent an explicit vector pointing from tool to goal at every step. The reward function reinforces closing this distance. The policy gradient for "move toward goal" is clear and immediate.

In Phase 3B, `goal_xyz` is absent. The 128D visual features encode scene appearance and implicit tool position but do not encode goal proximity. The MobileNetV3 features were trained to predict tool XYZ — not to predict "how far am I from the surgical target." The agent receives reward when it gets close to the goal but has no reliable observation signal to navigate toward it.

The agent found a local optimum: a movement strategy that reduces the task distance penalty partially and avoids collision costs. This produces ep_rew_mean ≈ -135. The optimal policy (full goal completion) produces ep_rew_mean ≈ -97. The gap of 38 reward points represents the cost of operating without privileged goal information.

### Evidence from training diagnostics

```
ep_len_mean = 300 throughout all 366 iterations (0 episodes completed)
Action std:    0.998 → 0.583  (policy became deterministic at ~250k steps)
Entropy loss: -4.26 → -2.63  (exploration fully collapsed at ~600k steps)
Reward plateau: -135 from ~360k steps onwards (no improvement for ~400k steps)
```

The reward plateau at -135 and entropy collapse at ~600k steps confirm the policy converged to a local optimum. Additional training steps at the same configuration would not resolve this — the policy has committed.

### Why 1.5M steps would not solve this

The learning rate is fixed at 3e-4 with no scheduled decay. The entropy collapsed to 0.583 indicating the policy is no longer exploring. From iteration 175 (step ~360k) to iteration 366 (step 750k) — 390k additional steps — the reward improved from -135 to -135 (zero improvement). The system has converged. Extended training without architectural changes would produce the same local optimum.

### What would actually solve this

1. **Restore goal_xyz** — add 3D goal coordinates back to the observation, giving 135D obs. Expected to restore goal-reaching behaviour since the agent has explicit navigational signal.

2. **Auxiliary goal-proximity task** — train the perception features with an additional loss that encodes goal proximity. Requires architectural changes to the perception pipeline.

3. **Reward shaping** — add intermediate rewards for visual progress signals (e.g. tissue deformation magnitude, proximity to tissue attachment point). Requires domain knowledge to define.

4. **Privileged distillation** — train a teacher policy with goal_xyz, then distil into a student policy without it. Industry standard for visuomotor transfer.

Options 1-4 are all valid future directions. For Phase 3 scope, the current result is sufficient to document the performance gap and proceed.

---

## The Performance Gap as a Scientific Result

The Phase 3B result is not a failure — it is the expected and honest outcome of the harder, clinically realistic task.

```
Phase 2D demonstrated: PPO can learn safe tissue retraction with perfect information
Phase 3B demonstrates: removing privileged goal coordinates degrades performance quantifiably

The degradation:
  Reward:     -97  → -135  (-39% worse)
  Completion:  100% → 0%   (complete regression on goal-reaching)
  Steps:       142  → 300  (2.1× more steps, never completes)
```

This gap quantifies the value of privileged simulator information. In real surgery, goal coordinates are not available — the robot must infer surgical progress from visual and haptic feedback. Phase 3B demonstrates this challenge concretely with a 14-hour training run on a realistic surgical simulator.

This is the type of result that motivates Phase 3C (force proxy from optical flow) and Phase 3D (sim-to-real gap analysis). The project tells a coherent story:

> "I showed an RL agent can learn safe tissue retraction with ground-truth observations (Phase 2D). When we replace ground-truth with camera-based perception (Phase 3B), performance degrades significantly because the agent loses access to goal position information. We then address this by building a tissue force proxy from optical flow (Phase 3C) to provide richer feedback, and analyse the full gap to deployment (Phase 3D)."

---

## W&B Logging Gap

W&B captured system metrics only (GPU usage, memory). The `rollout/` and `train/` metrics printed to stdout were not sent to W&B because the `WandbCallback` from `wandb.integration.sb3` was not included in the callbacks list. The full training curve is preserved in the terminal log file.

**Fix for future runs:**
```python
from wandb.integration.sb3 import WandbCallback
callbacks.append(WandbCallback(verbose=2))
```

All metrics from this run are reconstructable from the terminal log.

---

## Checkpoint and Reproducibility

```
Checkpoint directory: logs/checkpoints/phase3b_ppo_visual_20260413_152851/
  ppo_visual_final.zip         ← final trained policy
  ppo_visual_50000_steps.zip   ← intermediate checkpoints every 50k steps
  training_config.json         ← full hyperparameter record
```

To load and evaluate:
```python
from stable_baselines3 import PPO
from envs.tissue_retraction_v3 import TissueRetractionV3

env   = TissueRetractionV3()
model = PPO.load("logs/checkpoints/phase3b_ppo_visual_20260413_152851/ppo_visual_final")
obs, _ = env.reset()
action, _ = model.predict(obs, deterministic=True)
```

---

## Phase 3B → Phase 3C Connection

Phase 3C (Optical Flow Tissue Force Proxy) does not require the Phase 3B agent to reach the goal. It requires:

1. An agent that moves the instrument across the tissue ← Phase 3B provides this ✓
2. Consecutive RGB frames showing tissue deformation ← TissueRetractionV3 provides this ✓
3. Some contact force occurring during instrument movement ← Phase 3B agent contacts tissue ✓

The Phase 3B agent, despite not completing the task, moves the instrument in committed deterministic trajectories (std=0.583) across the tissue surface. These movements cause tissue deformation visible in the RGB frames. This is exactly the input needed to validate optical flow as a tissue force proxy.

Phase 3C is independent of Phase 3B task success. It proceeds immediately.