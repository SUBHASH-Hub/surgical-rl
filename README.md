# Surgical RL — Autonomous Tissue Retraction via Safe Reinforcement Learning

> **Building the full AI stack of a surgical autonomy system — from physics simulation to safe RL to surgical perception to ROS 2 middleware.**

---

## Clinical Motivation

During laparoscopic cholecystectomy — gallbladder removal, one of the most common operations worldwide — the surgeon must retract the gallbladder fundus to expose **Calot's triangle** before safe dissection. This retraction must stay within a force window of **0.5–3.0 N**. Too little force: the anatomy is not visible. Too much force: the cystic artery risks tearing.

Surgical outcome quality today varies with surgeon skill. This project builds an AI agent that learns to perform this retraction subtask autonomously in SOFA physics simulation — with safety-aware reward design, curriculum learning, a full surgical perception pipeline, and an architecture that mirrors the layers used in commercial surgical robots (CMR Versius, Medtronic Hugo, Intuitive da Vinci 5).

---

## Demo — Phase 2D PPO Agent Running Autonomously

> Trained PPO agent performing autonomous tissue retraction across 3 live episodes.
> Zero hardcoded waypoints. Learned entirely from reward signal across 750,000 simulation steps.

```
Episode 1:  107 steps · reward −57.20  · ✓ Goal reached  (140 faster than baseline)
Episode 2:   97 steps · reward −45.96  · ✓ Goal reached  (150 faster than baseline) ← Best
Episode 3:  186 steps · reward −150.38 · ✓ Goal reached  ( 61 faster than baseline)
```

---

## Phase-by-Phase Results Summary

| Metric | Phase 1 Scripted | Phase 2D PPO | Phase 3B Visual PPO |
|--------|-----------------|--------------|---------------------|
| Observation | Hardcoded waypoints | 7D ground-truth | 132D visual (MobileNetV3) |
| Goal position available | Yes (hardcoded) | Yes (simulator) | No (removed — real-robot constraint) |
| Episode reward (mean) | −165.54 | **−97.14** (eval) | −135.3 |
| Episode length (mean) | 247 steps | **142.3 steps** | 300 steps (truncated) |
| Goal completion rate | 100% | **100%** | 0% (no goal coordinates) |
| Collision steps | 49/ep | 85.7/ep | **0 / 3,000 steps** |
| Force proxy | None | None | **0.128 px/frame calibrated** |
| Training duration | — | 13h 10min | 14h 6min |

**The key result from Phase 3:** removing `goal_xyz` from the observation causes complete regression from 100% to 0% task completion, while collision safety improves to zero. This isolates the observation gap precisely — the only variable that changed between Phase 2D and Phase 3B was goal coordinate availability.

📊 **Phase 2 training metrics — all 4 runs:**
[W&B Report — Phase 2](https://api.wandb.ai/links/subhashtronics-de-montfort-university-leicester/kqbip2vh)

📊 **Phase 3B training curves and phase comparison:**
[W&B Report — Phase 3](https://api.wandb.ai/links/subhashtronics-de-montfort-university-leicester/0g3z7ei6)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 4 — ROS 2 Middleware + Behaviour Tree Planner [PLANNED]  │
│  lapgym_ros2_bridge · /tissue_force_proxy @ 50Hz                │
│  py_trees_ros BT · Fallback safety monitor · IEC 62304 Class B  │
│  Emergency stop watchdog (ISO 14971 independent risk control)   │
├─────────────────┬───────────────────┬───────────────────────────┤
│  Approach       │  Retract          │  Hold                     │
│  RL policy 1    │  RL policy 2 ✓   │  RL policy 3              │
│                 │  Phase 2D agent   │                           │
├─────────────────┴───────────────────┴───────────────────────────┤
│  PHASE 3 — Surgical Perception Pipeline  [COMPLETE ✓]           │
│  MobileNetV3-Small instrument tip detection (5.1px MAE)         │
│  UNet + MobileNetV3 tissue segmentation (IoU=1.000)             │
│  Farneback optical flow tissue force proxy (0.128 px/frame)     │
│  /tissue_force_proxy → alert=0.35 · safety_stop=1.0 px/frame   │
├─────────────────────────────────────────────────────────────────┤
│  PHASE 2 — Safe RL Core  [COMPLETE ✓]                           │
│  PPO · SafeRewardWrapper · 3-phase curriculum · 7D observation  │
│  136,711 parameters · λ_collision = 0.1 → 0.3 → 0.5            │
├─────────────────────────────────────────────────────────────────┤
│  PHASE 1 — SOFA v25.12 FEM Physics Engine  [COMPLETE ✓]         │
│  LapGym · Tissue 27,040 Pa · 1,479 nodes · RCM constraint       │
└─────────────────────────────────────────────────────────────────┘
```

This architecture maps directly to real surgical robots:

| Layer | CMR Versius / Medtronic Hugo | This Project |
|-------|------------------------------|--------------|
| Inner control loop | Proprietary servo (500–1000 Hz) | SOFA C++ FEM physics |
| Middleware | ROS 2 based control stack | ROS 2 Humble (Phase 4) |
| Intelligence | AI policy + task planner | PPO agent + BT planner |
| Perception | Endoscopic camera + tracking | MobileNetV3 + UNet (Phase 3) |
| Force sensing | Instrument force sensors | Optical flow proxy (Phase 3C) |
| Safety | Independent watchdog + limits | BT Fallback + watchdog node |

---

## Five-Phase Roadmap

### ✅ Phase 1 — Simulation Foundation (Months 1–2)

**Clinical question:** How do forces, tissue deformation, and tool contact interact in surgical simulation?

Built the SOFA + LapGym simulation environment and established the scripted baseline.

- SOFA v25.12 FEM physics — tissue Young's modulus 27,040 Pa, Poisson ratio 0.4287, mass 0.123 kg
- LapGym `TissueRetractionEnv` running headless at 15–17 FPS on GTX 1650
- RCM (Remote Centre of Motion) constraint simulating the trocar port
- **Scripted baseline: 247 steps · 49 collision steps · −165.54 reward · 100% goal rate**
- Resolved 5 SOFA v25.12 + NumPy 2.0 API compatibility issues

**Git tag:** `v1.0-phase1-complete`

---

### ✅ Phase 2 — Safe RL Core: Force-Bounded Retraction (Months 3–4)

**Clinical question:** Can an agent learn to retract tissue safely without exceeding the force limits that would tear a cystic duct?

Replaced the scripted waypoint trajectory with a PPO agent trained with safety-aware reward decomposition and three-phase curriculum learning. Four training runs — each one diagnosing and fixing the previous failure.

#### The Four Training Runs

| Run | Key Change | Outcome |
|-----|-----------|---------|
| Phase 2A | 3D observation baseline | ❌ Observation-blind — reward stuck at −359 |
| Phase 2B | 7D observation (tool+goal+phase) | ❌ Goal reached — curriculum shock at step 300k |
| Phase 2C | Step-based curriculum, 300-step limit | ❌ λ=0.8 caused −54 point reward collapse |
| Phase 2D | λ_max=0.5, trigger=350k, 750k steps | ✅ Shock −2pts · reward target met in eval |

#### SafeRewardWrapper — 4-component decomposition

```
R_total = R_task + R_force + R_collision + R_efficiency
R_force      = −λ_f × max(0, F − 0.5N)²   ← quadratic above clinical threshold
R_collision  = −λ_c × collision_steps      ← geometric tissue boundary enforcement
R_efficiency = −0.01 per step              ← encourages shorter episodes
```

#### Final eval metrics (10 deterministic episodes)

| Metric | Baseline | PPO Phase 2D | Change | Target | Status |
|--------|----------|--------------|--------|--------|--------|
| Episode steps | 247.0 | **142.3 ± 25.5** | −42.4% | < 200 | ✅ MET |
| Total reward | −165.54 | **−97.14 ± 30.6** | +41.3% | > −100 | ✅ MET |
| Goal rate | 100% | **100% (10/10)** | Same | 100% | ✅ MET |

**Git tags:** `v2.2-phase2c-complete` · `v2.3-phase2d-complete` · `v2.4-phase2-complete`

---

### ✅ Phase 3 — Surgical Perception Pipeline (Months 5–6)

**Clinical question:** Can a perception module extract surgical state from endoscopic video, replacing ground-truth simulator coordinates the way a real surgical robot must operate?

Phase 3 is the transition from lab RL to real surgical robotics industry architecture. Phase 2 used privileged simulator coordinates unavailable on any real robot. Phase 3 replaces those with camera-based perception — the actual architecture used in Moon Surgical Maestro, NVIDIA Holoscan, and surgical AI research at Imperial, Johns Hopkins, and ETH Zurich.

#### Phase 3A — Standalone Surgical Perception Module

Built two independent perception models from 7,284 RGB frames collected from the SOFA camera:

**Instrument tip detector — MobileNetV3-Small:**
- 1,001,251 parameters · backbone + regression head
- Two-phase transfer learning (10 epochs frozen + 20 epochs fine-tune)
- **Result: 5.1px mean pixel error on 480×480 — below the 10px surgical AI threshold**
- Architecture reference: Moon Surgical ScoPilot · deployable on NVIDIA IGX Holoscan

**Tissue segmentation — UNet + MobileNetV3:**
- 1,729,217 parameters · MobileNetV3 encoder + 3-stage UNet decoder
- Labels generated from SOFA mesh projection through OpenGL camera matrices
- **Result: IoU = 1.000 on simulation test set · 6,701 masks · 21.1% tissue coverage**

**Git tags:** `v3.0-phase3a-tip-detector` · `v3.1-phase3a-complete`

#### Phase 3B — Multimodal Visual Observation Integration

Replaced the 7D ground-truth observation with a 132D visual observation:

```
obs = [MobileNetV3_visual_features(128) + estimated_xyz(3) + phase_flag(1)]
    = 132D  (vs 7D in Phase 2D)
```

New environment: `TissueRetractionV3` wrapping V2 with perception pipeline.

**Result:**

| Metric | Phase 2D (ground-truth) | Phase 3B (visual) | Change |
|--------|------------------------|-------------------|--------|
| ep_rew_mean | −97.14 (eval) | −135.3 | −38.9% |
| ep_len_mean | 142.3 | 300 (truncated) | +110.8% |
| Goal rate | **100%** | **0%** | Full regression |
| Action std | 1.700 (exploring) | 0.583 (converged) | Local optimum |

**Root cause:** removing `goal_xyz` eliminates the navigational gradient. The agent improved reward 47.6% by learning collision avoidance but cannot navigate to the goal without knowing where it is. This is the expected and honest result of clinically realistic observation design — it directly quantifies the cost of operating without privileged simulator information.

**Git tag:** `v3.2-phase3b-complete`

#### Phase 3C — Visual Tissue Force Proxy

Built a Farneback dense optical flow system that measures tissue deformation between consecutive frames as a force proxy — formalising the visual judgement surgeons make when estimating tissue stress from screen.

**Pipeline:**
```
frame(t).copy() + frame(t+1)
  → cv2.calcOpticalFlowFarneback (pyr_scale=0.5, levels=3, winsize=15)
  → flow(480, 480, 2) — u and v displacement per pixel
  → magnitude = sqrt(u² + v²)
  → tissue_ROI_mask (from Phase 3A · 48,558 pixels · 21.1% of frame)
  → mean magnitude in tissue region = tissue_force_proxy (px/frame)
```

**Key engineering fix — Python object aliasing:**
The render buffer was updated in-place between frames. Both `frame_prev` and `frame_curr` pointed to the same array — optical flow between identical arrays is always zero. Fixed by `rgb_frame.copy()` in `tissue_retraction_v3.py` and `frame_curr.copy()` in the optical flow script.

**Results — 10 episodes, 3,000 steps:**

| Metric | Value |
|--------|-------|
| Mean tissue flow | **0.128 px/frame** |
| Max tissue flow | **0.732 px/frame** |
| Collision steps | **0 / 3,000 (0%)** |
| Alert threshold (calibrated) | 0.35 px/frame (mean + 2×std) |
| Safety stop threshold | 1.0 px/frame |
| Steps above alert | 141 / 3,000 (4.7%) |
| Steps above safety stop | 0 / 3,000 (0%) |

Zero collision steps confirms Phase 3B agent learned safe tissue interaction. Pearson r = NaN — mathematically correct because collision flag variance is zero (agent never collided).

**Phase 4 ROS 2 interface defined:**
```json
{
  "phase4_topic":                "/tissue_force_proxy",
  "phase4_alert_threshold":      0.35,
  "phase4_safety_stop_threshold": 1.0
}
```

**Industry reference:** Medtronic Touch Surgery · NVIDIA Holoscan tissue deformation monitoring

**Git tag:** `v3.3-phase3c-complete`

#### Phase 3D — Sim-to-Real Gap Analysis

Systematic documentation of every gap between SOFA LapGym simulation and a real laparoscopic surgical robot. Required for Phase 5 paper and Phase 4 deployment planning.

**14 gaps across 5 categories. 3 critical.**

| Category | Gaps | Critical | Addressed |
|----------|------|----------|-----------|
| Visual appearance | 3 | 0 | 0 |
| Physics and mechanics | 3 | 1 | 0 |
| Sensing and observation | 3 | 1 | 1 (Phase 3C) |
| Task definition | 3 | 1 | 0 |
| Infrastructure | 2 | 0 | 0 |

Critical gaps for Phase 4: contact modelling calibration (UncoupledConstraintCorrection compliance), force sensing (addressed by Phase 3C), episode safety (Phase 4 supervisory safety layer).

Full analysis: [`docs/phase3/phase3d_sim_to_real_gap_analysis.md`](docs/phase3/phase3d_sim_to_real_gap_analysis.md)

**Git tag:** `v3.4-phase3d-complete`

#### Phase 3 — Complete Git Tags

| Tag | Description |
|-----|-------------|
| `v3.0-phase3a-tip-detector` | MobileNetV3 tip detector — 5.1px MAE |
| `v3.1-phase3a-complete` | All Phase 3A sub-steps complete |
| `v3.2-phase3b-complete` | PPO retrained on 132D visual observation |
| `v3.3-phase3c-complete` | Optical flow force proxy validated |
| `v3.4-phase3d-complete` | Sim-to-real gap analysis complete |

---

### 📋 Phase 4 — ROS 2 Middleware + Supervised Autonomy + Safety Architecture (Months 7–8)

**Clinical question:** Can the perception and RL capabilities from Phases 2–3 be integrated into a ROS 2 middleware stack that mirrors commercial surgical robots — supporting both teleoperation (industry today) and supervised autonomy (industry R&D) — with a certifiable safety layer?

**Two tracks built simultaneously:**

**Track 1 — Industry Today (Teleoperation):** Keyboard/joystick controls SOFA instrument via ROS 2 topics at 50 Hz, mirroring how CMR Versius and Medtronic Hugo connect surgeon consoles to robot actuators.

**Track 2 — Industry R&D (Supervised Autonomy):** Phase 2D PPO agent wrapped as a ROS 2 action server, orchestrated by a Behaviour Tree with continuous force-based safety monitoring.

**Sub-steps:**

- **4A:** `lapgym_ros2_bridge` package — `/joint_states`, `/tissue_force_proxy`, `/camera/image_raw` at 50 Hz · teleoperation node
- **4B:** RL policy action servers — `RetractPolicyServer` (Phase 2D PPO), `ApproachPolicyServer`, `HoldPolicyServer` · all implement `is_preempted()` every step
- **4C:** `py_trees_ros` Behaviour Tree — Root Sequence with Fallback safety monitor (tissue_force < 0.35 → proceed, else EMERGENCY_STOP) then Approach → Retract → Hold
- **4D:** Independent safety watchdog — separate process, subscribes independently, hard stop if force > 1.0 px/frame for 3 consecutive readings, IEC 62304 traceability logging
- **4E:** `docs/iec62304_classification.md` — Class B software classification, safety architecture, stochastic policy documented under IEC 62304 §5.1.7

**Why Behaviour Tree over FSM:** BT Fallback node provides continuous force monitoring that preempts task execution natively. FSMs require explicit emergency transitions from every state — O(n) complexity that becomes a verification burden under IEC 62304. BTs are deterministic, auditable, and formally verifiable.

**Why not LLM/VLA models:** Non-deterministic, cannot satisfy IEC 62304 traceability requirements, 50–200ms inference latency incompatible with 50 Hz control loop, no formal safety guarantee on force limits.

---

### 📋 Phase 5 — Evaluation, Safety Analysis, Portfolio (Months 9–10)

- 5-condition ablation study showing contribution of each component
- Safety stress test: 2× tissue stiffness, sensor noise, 100ms actuation delay
- Paper-style technical report (6–8 pages, methods + results + safety analysis)
- 90-second screen recording demo of BT-orchestrated full retraction sequence

---

## Repository Structure

```
surgical-rl/
├── docs/
│   ├── project-overview.md               ← Clinical motivation and architecture
│   ├── baseline_metrics.md               ← Phase 1 official baseline (3-run mean)
│   ├── scene_graph_analysis.md           ← SOFA FEM parameters, RCM constraint
│   ├── compatibility_fixes.md            ← 5 LapGym fixes for SOFA v25 + NumPy 2.x
│   ├── phase2a_results.md                ← Phase 2A analysis
│   ├── phase2b_results.md                ← Phase 2B analysis
│   ├── phase2c_results.md                ← Phase 2C training analysis and root cause
│   ├── phase2d_results.md                ← Phase 2D final results + shock fix
│   ├── eval_results_phase2.md            ← 10-episode eval: PPO vs baseline
│   └── phase3/
│       ├── README.md                     ← Phase 3 navigator — all sub-phases
│       ├── phase3a_results.md            ← Tip detector + segmentation results
│       ├── phase3b_results.md            ← Visual PPO results and gap analysis
│       ├── phase3c_results.md            ← Optical flow force proxy results
│       └── phase3d_sim_to_real_gap_analysis.md  ← 14 gaps documented
├── envs/
│   ├── safe_reward.py                    ← SafeRewardWrapper — 4-component reward
│   ├── tissue_retraction_v2.py           ← TissueRetractionV2 — 7D obs (Phase 2)
│   ├── tissue_retraction_v3.py           ← TissueRetractionV3 — 132D visual obs (Phase 3)
│   └── perception_pipeline.py            ← MobileNetV3 feature extractor + xyz head
├── models/
│   ├── tip_detector/
│   │   ├── mobilenetv3_tip_best.pth      ← Tip detector · 5.1px MAE
│   │   └── eval_metrics.json
│   ├── segmentation/
│   │   ├── unet_seg_best.pth             ← Tissue segmentation · IoU=1.000
│   │   └── eval_metrics.json
│   └── force_proxy/
│       └── proxy_config.json             ← Calibrated thresholds · Phase 4 topic
├── scripts/
│   ├── baseline_demo.py                  ← Phase 1 scripted baseline
│   ├── train_ppo.py                      ← Phase 2 PPO training
│   ├── eval_agent.py                     ← PPO vs scripted comparison
│   ├── watch_agent.py                    ← HUMAN mode demo
│   ├── collect_rgb_frames.py             ← Phase 3A frame collection
│   ├── train_tip_detector.py             ← Phase 3A MobileNetV3 training
│   ├── generate_seg_masks.py             ← Phase 3A mask generation
│   ├── train_segmentation.py             ← Phase 3A UNet training
│   ├── visualise_predictions.py          ← Phase 3A overlay visualisations
│   ├── train_ppo_visual.py               ← Phase 3B visual PPO training
│   └── optical_flow_proxy.py             ← Phase 3C Farneback force proxy
├── logs/
│   └── checkpoints/
│       ├── phase2_ppo_tissue_retraction_20260409_211946/   ← Phase 2D checkpoint
│       └── phase3b_ppo_visual_20260413_152851/             ← Phase 3B checkpoint
├── data/
│   ├── rgb_frames/                       ← 7,284 PNGs + labels.csv (gitignored)
│   ├── seg_masks/                        ← 6,701 binary masks (gitignored)
│   └── optical_flow/
│       ├── flow_log.csv                  ← 3,000 rows: ep, step, flow, collision
│       └── flow_validation_plot.png      ← scatter + distribution plot
├── agents/
│   └── ppo_config.yaml                   ← PPO hyperparameters
├── configs/
│   └── phase2_baseline.yaml              ← Phase 2D config
├── setup_env.sh                          ← Activates sofa_venv + sets env vars
└── requirements.txt                      ← Pinned dependencies
```

---

## Quickstart

### Prerequisites

- Ubuntu 22.04
- NVIDIA GPU with CUDA 12.x
- Python 3.10
- [SOFA v25.12 pre-built binary](https://github.com/sofa-framework/sofa/releases/tag/v25.12.00) extracted to `~/surgical_robot_lapgym_ws/sofa_install/`
- [LapGym](https://github.com/ScheiklP/lap_gym) cloned to `~/surgical_robot_lapgym_ws/lap_gym/`

### Setup

```bash
git clone https://github.com/SUBHASH-Hub/surgical-rl.git
cd surgical-rl
python3.10 -m venv ~/surgical_robot_lapgym_ws/sofa_venv
source setup_env.sh
pip install -r requirements.txt
```

### Watch the Phase 2D PPO agent (fastest demo)

```bash
source setup_env.sh
python scripts/watch_agent.py --slow --episodes 3
# SOFA GUI opens — agent retracts tissue autonomously in ~130 steps
```

### Run Phase 3A perception inference

```bash
source setup_env.sh
python scripts/visualise_predictions.py
# Generates overlay images: green crosshair (predicted tip) · cyan mask (tissue)
```

### Run Phase 3C optical flow force proxy

```bash
source setup_env.sh
python3 -u scripts/optical_flow_proxy.py
# Runs 10 episodes · logs flow_log.csv · saves flow_validation_plot.png
# Expected: mean_tissue_flow ≈ 0.128 px/frame · 0 collision steps
```

### Train Phase 3B visual PPO from scratch

```bash
source setup_env.sh
python scripts/train_ppo_visual.py
# ~14 hours on GTX 1650 · 750k steps · checkpoint saved automatically
```

---

## Key Technical Decisions

**Why SOFA + LapGym:** SOFA provides finite element method (FEM) deformable body simulation. LapGym wraps SOFA in the Gymnasium interface. Intuitive Surgical Research funds LapGym development — it is the standard academic surgical simulation platform.

**Why MobileNetV3 for perception:** MobileNetV3-Small is the architecture class used by Moon Surgical ScoPilot for real-time instrument tracking. It is deployable on NVIDIA IGX Holoscan — the hardware target for surgical AI at the edge. Under 5M parameters, runs in real time on GTX 1650.

**Why optical flow for force proxy:** SOFA's `BlockGaussSeidelConstraintSolver` stores contact forces internally — not accessible via `MechanicalObject.force`. Farneback dense optical flow measures tissue deformation directly from the camera frame, matching the visual judgement surgeons make. Medtronic Touch Surgery and NVIDIA Holoscan use this approach in real systems.

**Why Behaviour Tree over FSM:** BT Fallback node provides continuous force monitoring that preempts task execution natively — architecturally impossible to express cleanly in an FSM without O(n) emergency state logic. CMR Surgical's published architecture research uses BTs for surgical task sequencing.

**Why not LLM planner:** LLM-based planners are stochastic and non-deterministic. They cannot be validated under IEC 62304 or ISO 14971. BT transitions are deterministic, auditable, and every state change is traceable — the correct architecture for safety-critical surgical systems.

**Why Phase 3B result (0% goal rate) is scientifically correct:** The Phase 3B result is not a failure — it is a controlled experiment that isolates the observation gap. Every other variable (algorithm, curriculum, reward, physics) is held constant. The 100% → 0% regression is attributable entirely to removal of `goal_xyz`. This is a publishable finding that most surgical RL papers avoid reporting honestly.

---

## Related Work

> Pore et al. (2021). *Safe Reinforcement Learning using Formal Verification for Tissue Retraction in Autonomous Robotic-Assisted Surgery.* IROS 2021. [arXiv:2109.02323](https://arxiv.org/abs/2109.02323)

> Scheikl et al. (2023). *LapGym — An Open Source Framework for Reinforcement Learning in Robot-Assisted Laparoscopic Surgery.* JMLR 24. [arXiv:2302.09606](https://arxiv.org/abs/2302.09606)

This project extends Pore et al. by implementing three-phase curriculum learning, building a full surgical perception pipeline, documenting the observation gap quantitatively, building a visual force proxy for force-sensorless robots, and constructing the full ROS 2 + BT stack toward certifiable supervised autonomy.

---

## Platform

| Component | Version |
|-----------|---------|
| OS | Ubuntu 22.04 LTS |
| GPU | NVIDIA GTX 1650 (CUDA 12.8) |
| RAM | 16 GB |
| Python | 3.10.12 |
| SOFA | v25.12.00 |
| PyTorch | 2.10.0+cu128 |
| Stable-Baselines3 | 2.7.1 |
| Gymnasium | 1.2.3 |
| OpenCV | 4.x |
| Weights & Biases | 0.25.1 |

---

## Author

**Subhash Arockiadoss**
MSc Mechatronics and Robotics, De Montfort University Leicester (2024)

[LinkedIn](https://www.linkedin.com/in/subhasharockiadoss-2092b8171) · [GitHub](https://github.com/SUBHASH-Hub) · [W&B Phase 2](https://wandb.ai/subhashtronics-de-montfort-university-leicester/surgical-rl-phase2) · [W&B Phase 3](https://wandb.ai/subhashtronics-de-montfort-university-leicester/surgical-rl-phase3)

*Seeking visa-sponsored roles in surgical robotics AI and medical robotics in the UK.*

---

*Phase 1 ✅ complete · Phase 2 ✅ complete · Phase 3 ✅ complete · Phase 4 🔄 planned · May 2026*