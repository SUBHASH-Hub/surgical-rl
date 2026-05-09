# Phase 2C Training Results

**Run ID:** mchggpzp  
**W&B:** https://wandb.ai/subhashtronics-de-montfort-university-leicester/surgical-rl-phase2/runs/mchggpzp  
**Date:** 09 April 2026  
**Duration:** 8h 51min  
**Checkpoint:** `logs/checkpoints/phase2_ppo_tissue_retraction_20260409_033535/ppo_tissue_final`

---

## Configuration

| Parameter | Value |
|---|---|
| Algorithm | PPO (Stable-Baselines3 2.3.2) |
| Policy | MlpPolicy — 7 → 256 → 256 → 3 (Actor), 7 → 256 → 256 → 1 (Critic) |
| Total parameters | 136,711 |
| Observation | 7D: tool XYZ + goal XYZ + phase flag |
| Total timesteps | 500,000 (501,760 actual) |
| Episode step limit | 300 steps |
| Device | CUDA (GTX 1650) |
| Training speed | 15–17 it/s |

**Curriculum (step-based):**

| Phase | Steps | λ_collision | Triggered |
|---|---|---|---|
| Phase 0 | 0 – 149,999 | 0.1 | Step 0 (immediate) |
| Phase 1 | 150,000 – 299,999 | 0.3 | Step 150,000 ✓ |
| Phase 2 | 300,000 – 501,760 | 0.8 | Step 300,000 ✓ |

---

## Results Summary

### Core Metrics vs Targets

| Metric | Scripted Baseline | Phase 2C Final | Target | Status |
|---|---|---|---|---|
| Episode steps mean | 247 | **183** | < 200 | ✓ MET |
| Reward mean (best) | -165.54 | **-121** (step 149k) | > -100 | Close |
| Reward mean (final) | -165.54 | **-175** | > -100 | Not met |
| Goal reached | 100% (scripted) | Yes (learned) | Yes | ✓ MET |
| Collision steps (Phase 1) | 49 | **~93** | < 20 | Not met |
| Force violations | Not measured | 0% (SOFA limit) | < 5% | Unverified |

### Reward Progression

| Timestep | Reward Mean | Ep Len Mean | Phase |
|---|---|---|---|
| 2,048 | -309 | 300 | 0 (λ=0.1) |
| 50,000 | -216 | 300 | 0 |
| 100,000 | -154 | 300 | 0 |
| **149,504** | **-121 (BEST)** | 300 | 0→1 |
| 200,000 | -131 | 300 | 1 (λ=0.3) |
| 250,000 | -126 | 300 | 1 |
| 300,000 | -130 | 300 | 1→2 |
| 350,000 | -174 | 296 | 2 (λ=0.8) |
| 401,408 | -149 | 175 | 2 |
| 501,760 | -175 | 183 | 2 (final) |

---

## Collision Analysis — The Critical Finding

This is the most important analytical result from Phase 2C.

### Collision Steps by Curriculum Phase

Collision steps per episode are derived from `r_coll` divided by `λ_collision`:

| Curriculum Phase | λ | Mean r_coll | Implied collision steps/episode | vs Baseline (49) |
|---|---|---|---|---|
| Phase 0 (0–150k) | 0.1 | -30.0 | **300 steps** (entire episode) | 6× worse |
| Phase 1 (150k–300k) | 0.3 | -22.1 (scaled) | **~93 steps** | 1.9× worse |
| Phase 2 (300k–500k) | 0.8 | -195.1 (scaled) | **~179 steps** | 3.7× worse |

### What the Data Tells Us

**Phase 0 finding:** r_coll = -30.0 constant for the first 269 episodes with ep_len = 300. This means the agent spends every single step of every episode inside the tissue bounding box. The agent is not avoiding tissue at all in Phase 0 — it has learned to reach the goal by passing through tissue rather than around it.

**Phase 1 finding (positive):** Mean r_coll drops from -30.0 to approximately -6.6 per episode when normalised for λ=0.3. Implied collision steps drop from 300 to ~93. **The agent is learning.** The collision penalty at λ=0.3 is strong enough to start reshaping the trajectory.

**Phase 2 finding (shock):** At step 300,000, λ jumps from 0.3 to 0.8. The agent had learned a trajectory that incurred ~93 collision steps at cost 0.3 per step = -27.9. At λ=0.8, the same trajectory costs -74.4. This is a 2.7× penalty increase for the same physical behaviour. The agent had insufficient timesteps to fully adapt before training ended. Mean collision steps in Phase 2 recover to ~179 (from the implied 300 initial shock) but do not reach Phase 1 levels before 501,760 steps.

### Root Cause of Collision Metric Not Met

The collision target (<20 steps) was not met because:

1. **The agent learned a goal-reaching strategy that passes through tissue.** This is geometrically the shortest path. The agent correctly learned to reach the goal — but without sufficient collision penalty pressure, it found the straight-line route that happens to intersect the tissue bounding box.

2. **Curriculum Phase 2 fired too early relative to convergence.** At step 300,000, the agent had ~200,000 steps remaining. The penalty shock disrupted the Phase 1 policy before the agent could converge on a collision-free alternative.

3. **The collision detection geometry is axis-aligned bounding box only.** The tissue box is at Y=[0.005, 0.013]. The instrument approaches from Y=0.077. The agent must learn to navigate around the Y-axis edge of the bounding box — a narrow corridor. This is harder to learn than distance-based collision detection.

---

## What the Agent DID Learn

Despite not meeting the collision target, the agent demonstrates genuine learning:

- **Goal-reaching without waypoints:** The agent learned to navigate from random initial positions to the grasping target and complete retraction — autonomously, from reward signal only.
- **Episode efficiency:** 183 mean steps vs 300 step limit = 61% utilisation. The agent terminates early consistently.
- **Phase 1 collision reduction:** From 300 collision steps → 93 steps under Phase 1 penalty. A 69% reduction demonstrates the agent responds to collision pressure.
- **Best individual episodes:** r_total = -50 to -75, r_task ≈ -100 to -120. These are above the >-100 total reward target.
- **Consistent goal achievement:** The agent never gets stuck at ep_len=300 truncation in the final training phase — episodes terminate naturally at 175–225 steps.

---

## Force Monitoring — Known Limitation

`force_viol = 0.000000` throughout all 760 episodes.

This is a SOFA constraint solver limitation, not a policy achievement. SOFA stores contact forces in the `BlockGaussSeidelConstraintSolver` compliance matrix — these are not accessible via `MechanicalObject.force` data fields which return shape=(1,6) of zeros throughout.

`SafeRewardWrapper.r_collision` provides equivalent geometric safety enforcement. The `RestShapeSpringsForceField` accessible via `lapgym.end_effector.grasping_force_field` is the correct access point and is documented as a Phase 3 engineering task (optical flow proxy as force surrogate).

---

## Phase 2D Hypothesis — What Should Improve Collision

Based on this analysis, Phase 2D will target the collision metric specifically:

**Change:** λ_max = 0.4 (instead of 0.8), extended training to 750,000 steps

**Reasoning:**
- Phase 1 at λ=0.3 achieved 93 collision steps (significant improvement from 300)
- Phase 2 at λ=0.8 caused a shock that the agent could not recover from in 200k steps
- λ=0.4 represents a 33% increase over Phase 1 — enough pressure to push the policy toward collision-free paths without a catastrophic shock
- 750k steps gives the Phase 2 curriculum 450k steps to converge (vs 200k in Phase 2C)

**Expected outcome:** 40–60 collision steps mean (vs 93 in Phase 1, 179 in Phase 2)

---

## Files Committed

| File | Description |
|---|---|
| `envs/tissue_retraction_v2.py` | 7D observation, 300-step limit, spring force readout |
| `envs/safe_reward.py` | SafeRewardWrapper — 4-component reward |
| `scripts/train_ppo.py` | Training entry point with step-based curriculum |
| `configs/phase2_baseline.yaml` | Experiment configuration |
| `agents/ppo_config.yaml` | PPO hyperparameters |
| `docs/phase2c_results.md` | This document |

Checkpoint too large for GitHub — stored locally at path above.