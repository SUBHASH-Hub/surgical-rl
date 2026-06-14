# Phase 5A — PPO Agent Systematic Evaluation

**Status:** COMPLETE  
**Date:** June 2026  
**Author:** Subhash Arockiadoss  
**Git tag:** `v5.0-phase5a-evaluation`

---

## Overview

Phase 5A runs the Phase 2D PPO checkpoint for 30 deterministic episodes
against TissueRetractionV2 wrapped in SafeRewardWrapper — the same
environment used during training. This converts anecdotal observations
("I ran it 3 times and it worked") into statistically valid paper results.

**Does NOT require the full ROS2 system.**  
Uses `env.step()` directly — same pattern as `scripts/eval_agent.py`.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/phase5a_eval.py` | Runs 30 episodes, saves CSV outputs |
| `scripts/phase5a_analysis.py` | Reads CSV, computes statistics, saves summary |

### Run order

```bash
source ~/surgical_robot_lapgym_ws/activate.sh
cd ~/surgical_robot_lapgym_ws/surgical-rl

# Step 1 — run 30 episodes (~5 minutes)
python scripts/phase5a_eval.py

# Step 2 — statistical analysis (instant)
python scripts/phase5a_analysis.py
```

---

## Configuration

| Parameter | Value |
|-----------|-------|
| PPO checkpoint | `logs/checkpoints/phase2_ppo_tissue_retraction_20260409_211946/ppo_tissue_final` |
| Episodes | 30 |
| Seed | 42 |
| Action mode | Deterministic (`deterministic=True`) |
| Max steps/episode | 300 |
| SafeRewardWrapper lambda_force | 0.5 |
| SafeRewardWrapper lambda_collision | 0.5 |
| SafeRewardWrapper force_threshold | 0.5 |
| SafeRewardWrapper step_penalty | 0.01 |
| Platform | Ubuntu 22.04, GTX 1650 |
| Runtime | 288s (4.8 minutes) |

**Why deterministic=True:** During training PPO samples actions from
a probability distribution. For evaluation, actions are taken as the
mean of that distribution — this removes sampling noise and gives the
cleanest measure of what the policy has learned. All published surgical
RL papers evaluate deterministically.

---

## Output Files

| File | Description |
|------|-------------|
| `results/phase5a_ppo_eval.csv` | Per-episode metrics (30 rows) |
| `results/phase5a_comparison.csv` | PPO vs scripted baseline (Table 1 for paper) |
| `results/phase5a_summary.txt` | Full summary with statistics |

---

## Results

### Core Metrics (N=30)

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Episode steps | **147.7** | 31.0 | 100 | 205 |
| Total reward | **-104.68** | 38.13 | -176.71 | -48.82 |
| Task reward | -57.62 | 23.04 | -101.17 | -25.33 |
| Collision steps/ep | 91.2 | 29.7 | 45 | 147 |
| Inference time (ms) | **0.884** | 0.207 | — | — |

**Goal success rate: 100% (30/30 episodes)**

### PPO vs Scripted Baseline (Phase 1)

| Metric | Scripted Baseline (N=3) | PPO Phase 2D (N=30) | Change |
|--------|------------------------|---------------------|--------|
| Episode steps | 247.0 ± 8.2 | **147.7 ± 31.0** | **-40.2%** |
| Total reward | -165.54 ± 12.3 | **-104.68 ± 38.13** | **+36.8%** |
| Goal rate | 100% | **100% (30/30)** | Same |
| Collision steps | 49.0 ± 4.1 | 91.2 ± 29.7 | +86% |
| Inference time | N/A (scripted) | **0.884 ms** | New metric |

### Statistical Analysis

**Episode steps vs baseline (247.0):**

| Test | Result |
|------|--------|
| One-sample t-test | t = -17.559, p = 5.41e-17 |
| Cohen's d | -3.206 (large effect) |
| 95% CI for mean | [136.2, 159.3] steps |
| Significant? | YES (p << 0.05) |

**Total reward vs baseline (-165.54):**

| Test | Result |
|------|--------|
| One-sample t-test | t = 8.742, p = 1.27e-09 |
| Cohen's d | 1.596 (large effect) |
| 95% CI for mean | [-118.92, -90.45] |
| Significant? | YES (p << 0.05) |

---

## Per-Episode Results

| Ep | Steps | Reward | R_task | Coll | Goal |
|----|-------|--------|--------|------|------|
| 1 | 130 | -80.23 | -42.94 | 72 | Y |
| 2 | 181 | -148.20 | -84.90 | 123 | Y |
| 3 | 157 | -112.27 | -61.21 | 99 | Y |
| 4 | 115 | -65.88 | -34.74 | 60 | Y |
| 5 | 188 | -151.94 | -85.07 | 130 | Y |
| 6 | 125 | -75.74 | -39.50 | 70 | Y |
| 7 | 176 | -139.25 | -78.50 | 118 | Y |
| 8 | 156 | -111.25 | -60.70 | 98 | Y |
| 9 | 136 | -88.63 | -46.78 | 81 | Y |
| 10 | 138 | -94.09 | -50.72 | 84 | Y |
| 11 | 157 | -112.52 | -61.46 | 99 | Y |
| 12 | 103 | -52.99 | -27.97 | 48 | Y |
| 13 | 190 | -157.93 | -90.04 | 132 | Y |
| 14 | 142 | -96.64 | -51.73 | 87 | Y |
| 15 | 184 | -149.36 | -84.53 | 126 | Y |
| 16 | 192 | -167.15 | -98.24 | 134 | Y |
| 17 | 125 | -75.64 | -39.40 | 70 | Y |
| 18 | 205 | -176.71 | -101.17 | 147 | Y |
| 19 | 175 | -140.36 | -80.12 | 117 | Y |
| 20 | 111 | -60.53 | -31.43 | 56 | Y |
| 21 | 156 | -111.21 | -60.66 | 98 | Y |
| 22 | 123 | -75.12 | -39.40 | 69 | Y |
| 23 | 155 | -109.23 | -59.19 | 97 | Y |
| 24 | 133 | -84.77 | -44.45 | 78 | Y |
| 25 | 106 | -56.13 | -29.08 | 52 | Y |
| 26 | 100 | -48.82 | -25.33 | 45 | Y |
| 27 | 192 | -159.42 | -90.51 | 134 | Y |
| 28 | 110 | -59.63 | -31.04 | 55 | Y |
| 29 | 147 | -102.35 | -56.39 | 89 | Y |
| 30 | 124 | -76.48 | -41.25 | 68 | Y |
| **Mean** | **147.7** | **-104.68** | **-57.62** | **91.2** | **100%** |
| **Std** | **31.0** | **38.13** | **23.04** | **29.7** | — |

---

## Paper-Ready Sentences

These sentences are ready to copy directly into the paper:

**Steps:**
> The PPO agent completes tissue retraction in 147.7 ± 31.0 steps
> (N=30, 95% CI [136.2, 159.3]), a 40.2% reduction vs the 247-step
> scripted baseline (t=-17.56, p=5.41e-17, Cohen's d=-3.21).

**Goal rate:**
> The agent achieved 100% task success across all 30 deterministic
> evaluation episodes (30/30), using zero hardcoded waypoints.

**Reward:**
> Mean total reward of -104.68 ± 38.13 (95% CI [-118.92, -90.45]),
> a 36.8% improvement vs the scripted baseline reward of -165.54
> (t=8.74, p=1.27e-09, Cohen's d=1.60).

**Inference:**
> PPO inference latency of 0.884 ± 0.207 ms per step on GTX 1650,
> confirming real-time deployment viability.

**Collision (honest framing):**
> Collision steps per episode exceeded the scripted baseline
> (91.2 ± 29.7 vs 49.0 ± 4.1), as the scripted policy exploits
> hardcoded geometric knowledge of tissue boundaries unavailable
> to the learned policy — motivating the Phase 3 visual
> perception pipeline.

---

## Notes on r_coll = 0.0

`r_coll` reads as 0.0 across all 30 episodes. This is the known SOFA
`BlockGaussSeidelConstraintSolver` limitation documented in Phase 2B
and Phase 2C — contact forces are stored internally and do not populate
`MechanicalObject.force`. The `collision_steps` counter (counting steps
where the instrument is inside the tissue bounding box) is the valid
collision metric and is correctly reported above.

---

## Known Issues Fixed

| Issue | Root cause | Fix applied |
|-------|-----------|-------------|
| `render_mode` keyword error | TissueRetractionV2 does not accept render_mode | Removed — environment initialised via SafeRewardWrapper same as eval_agent.py |
| `phase5a_summary.txt` empty | Unicode em-dash character caused UnicodeEncodeError in SOFA logger | Replaced with `+/-` and `->` in analysis script |
| SOFA SIGABRT on exit | Known SOFA/SofaPython3 GIL destructor race on interpreter shutdown | Expected — all data saved before crash. Same behaviour across all phases. |

---

## Connection to Paper

Phase 5A produces the evidence for paper Section IV-A (Experimental Results):

| Paper claim | Evidence source |
|-------------|----------------|
| 100% goal rate (N=30) | phase5a_ppo_eval.csv — all 30 rows goal_reached=True |
| -40.2% step reduction | phase5a_comparison.csv — 147.7 vs 247.0 |
| p=5.41e-17 significance | phase5a_analysis.py statistical output |
| Cohen's d=3.21 large effect | phase5a_analysis.py statistical output |
| 0.884ms inference latency | phase5a_ppo_eval.csv — inference_ms column |
| Table 1 PPO vs Baseline | phase5a_comparison.csv |

---

## Next Phase

**Phase 5B — Stop Latency Measurement**

Measures Python vs C++ action server stop latency formally:
- 20 measurements each condition (Python servers, C++ servers)
- Metrics: mean ms, std, P50, P95, P99
- Requires full ROS2 system running
- Converts anecdotal "~130ms" into paper result with statistics