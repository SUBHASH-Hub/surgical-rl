# Surgical RL — Autonomous Tissue Retraction via Safe Reinforcement Learning

> **Building the complete AI stack of a surgical autonomy system — from physics simulation to safe RL to surgical perception to ROS 2 middleware with human-in-the-loop control.**

---

## Clinical Motivation

During laparoscopic cholecystectomy — gallbladder removal, one of the most common operations worldwide — the surgeon must retract the gallbladder fundus to expose **Calot's triangle** before safe dissection. This retraction must stay within a force window of **0.5–3.0 N**. Too little force: the anatomy is not visible. Too much force: the cystic artery risks tearing.

Surgical outcome quality today varies with surgeon skill. This project builds an AI agent that learns to perform this retraction subtask autonomously in SOFA physics simulation — with safety-aware reward design, curriculum learning, a full surgical perception pipeline, and a complete ROS 2 architecture that mirrors the layers used in commercial surgical robots (CMR Versius, Medtronic Hugo, Intuitive da Vinci 5).

---

## Demo — Phase 4E: Full System Running Live

> SOFA GUI (left) + Surgeon Console (right). Full autonomous procedure:
> APPROACH → RETRACT (PPO agent) → HOLD, with surgeon S/R/E control active throughout.

**Phase 4 system — 7 ROS 2 nodes, single launch command:**

```bash
source ~/surgical_robot_lapgym_ws/activate.sh
ros2 launch lapgym_ros2_bridge surgical_system.launch.py
```

**Surgeon console key bindings:**

| Key | Action | Effect |
|-----|--------|--------|
| S | Surgeon Stop | Freezes agent mid-trajectory — instrument holds position |
| R | Resume | Agent continues from exact stop step — no reset |
| E | Emergency Stop | Halts all nodes — BT reports FAILED |
| Q | Quit | Closes console (system continues) |

---

## Phase-by-Phase Results Summary

| Metric | Phase 1 Scripted | Phase 2D PPO | Phase 3B Visual PPO | Phase 4 ROS 2 |
|--------|-----------------|--------------|---------------------|---------------|
| Observation | Hardcoded waypoints | 7D ground-truth | 132D visual (MobileNetV3) | 7D (Phase 2D agent) |
| Goal position | Yes (hardcoded) | Yes (simulator) | No (real-robot constraint) | Yes (via action server) |
| Episode reward (mean) | −165.54 | **−97.14** (eval) | −135.3 | −97.14 (same agent) |
| Episode length (mean) | 247 steps | **142.3 steps** | 300 steps (truncated) | 119–134 steps (observed) |
| Goal completion rate | 100% | **100%** | 0% (no goal coordinates) | **100%** (integrated) |
| Collision steps | 49/ep | 85.7/ep | **0 / 3,000 steps** | 0 (watchdog active) |
| Force proxy | None | None | **0.128 px/frame** | Published at 50Hz |
| Safety layer | None | None | Calibrated thresholds | IEC 62304-inspired watchdog |
| Human control | None | None | None | **S/R/E surgeon console** |

📊 **Phase 2 training metrics — all 4 runs:**
[W&B Report — Phase 2](https://api.wandb.ai/links/subhashtronics-de-montfort-university-leicester/kqbip2vh)

📊 **Phase 3B training curves and phase comparison:**
[W&B Report — Phase 3](https://api.wandb.ai/links/subhashtronics-de-montfort-university-leicester/0g3z7ei6)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 4 — ROS 2 Middleware + Supervised Autonomy  [COMPLETE ✓]     │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  LAYER 5 — surgeon_console (Phase 4E)                        │   │
│  │  curses terminal · S/R/E/Q keys · /surgeon_stop · /estop    │   │
│  └────────────────────────┬─────────────────────────────────────┘   │
│                            │ /console_feedback                       │
│  ┌─────────────────────────▼────────────────────────────────────┐   │
│  │  LAYER 4 — surgical_bt_node (Phase 4D)                       │   │
│  │  py_trees_ros · Root → SafetyMonitor → Sequence              │   │
│  │  Approach → Retract → Hold · ForceCondition guard            │   │
│  └────────────────────────┬─────────────────────────────────────┘   │
│                  ROS 2 Action (lapgym_interfaces/Retract)            │
│  ┌──────────────┬──────────▼──────────┬──────────────────────────┐  │
│  │  LAYER 3 — Action Servers (Phase 4B)                          │  │
│  │  approach_policy_server  │  retract_policy_server             │  │
│  │  proportional controller │  Phase 2D PPO agent                │  │
│  │  hold_policy_server      │  zero-action position hold         │  │
│  │  Each: separate rclpy.Context · 10ms spin · dual freeze loop  │  │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  LAYER 2 — safety_watchdog_node (Phase 4C)                   │   │
│  │  Independent process · 50Hz · IEC 62304 Class B              │   │
│  │  ALERT=0.35 px/frame · STOP=1.0 px/frame                     │   │
│  │  Cannot be blocked by application logic                      │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  LAYER 1 — sofa_bridge_node (Phase 4A)                       │   │
│  │  SOFA↔ROS2 bridge · 50Hz · teleop fallback                   │   │
│  │  /tissue_force_proxy · /joint_states · /camera/image_raw     │   │
│  └──────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│  PHASE 3 — Surgical Perception Pipeline  [COMPLETE ✓]               │
│  MobileNetV3-Small instrument tip detection (5.1px MAE)             │
│  UNet + MobileNetV3 tissue segmentation (IoU=1.000)                 │
│  Farneback optical flow tissue force proxy (0.128 px/frame)         │
│  /tissue_force_proxy → alert=0.35 · safety_stop=1.0 px/frame        │
├─────────────────────────────────────────────────────────────────────┤
│  PHASE 2 — Safe RL Core  [COMPLETE ✓]                               │
│  PPO · SafeRewardWrapper · 3-phase curriculum · 7D observation       │
│  136,711 parameters · λ_collision = 0.1 → 0.3 → 0.5                │
├─────────────────────────────────────────────────────────────────────┤
│  PHASE 1 — SOFA v25.12 FEM Physics Engine  [COMPLETE ✓]             │
│  LapGym · Tissue 27,040 Pa · 1,479 nodes · RCM constraint           │
└─────────────────────────────────────────────────────────────────────┘
```

**Industry mapping — how this project mirrors commercial surgical robots:**

| Layer | CMR Versius / Medtronic Hugo | This Project |
|-------|------------------------------|--------------|
| Inner control loop | Proprietary servo (500–1000 Hz) | SOFA C++ FEM physics (~15Hz) |
| Middleware | ROS 2 based control stack | ROS 2 Humble — 7 nodes |
| Intelligence | AI policy + task planner | PPO agent + BT planner |
| Perception | Endoscopic camera + tracking | MobileNetV3 + UNet (Phase 3) |
| Force sensing | Instrument force sensors | Optical flow proxy (Phase 3C) |
| Safety | Independent watchdog + limits | safety_watchdog_node at 50Hz |
| Human control | Surgeon console | surgeon_console S/R/E |

---

## Phase 4 — Complete Node Table

| Node | Role | Hz | Status |
|------|------|----|--------|
| `sofa_bridge_node` | SOFA↔ROS2 bridge, teleop fallback | 50 | ✅ Phase 4A |
| `approach_policy_server` | Proportional controller to grasping zone | ~15 | ✅ Phase 4B |
| `retract_policy_server` | Phase 2D PPO autonomous retraction | ~15 | ✅ Phase 4B |
| `hold_policy_server` | Zero-action position hold | ~10 | ✅ Phase 4B |
| `safety_watchdog_node` | IEC 62304-inspired force monitor | 50 | ✅ Phase 4C |
| `surgical_bt_node` | Behaviour tree orchestrator | 10 | ✅ Phase 4D |
| `surgeon_console` | Human-in-the-loop terminal UI | 10 | ✅ Phase 4E |

**ROS 2 topics:**

| Topic | Type | Publisher | Subscribers |
|-------|------|-----------|-------------|
| `/tissue_force_proxy` | Float32 | sofa_bridge_node | safety_watchdog_node, surgeon_console |
| `/surgeon_stop` | Bool | surgeon_console | approach, retract, hold servers |
| `/emergency_stop` | Bool | surgeon_console | all nodes |
| `/watchdog_status` | String | safety_watchdog_node | surgeon_console |
| `/watchdog_heartbeat` | Bool | safety_watchdog_node | surgeon_console |
| `/console_feedback` | String | surgical_bt_node | surgeon_console |
| `/joint_states` | JointState | sofa_bridge_node | surgical_bt_node |

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

Four training runs — each one diagnosing and fixing the previous failure.

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

#### Phase 3A — Standalone Surgical Perception Module

**Instrument tip detector — MobileNetV3-Small:**
- **Result: 5.1px mean pixel error on 480×480 — below the 10px surgical AI threshold**
- Architecture reference: Moon Surgical ScoPilot · deployable on NVIDIA IGX Holoscan

**Tissue segmentation — UNet + MobileNetV3:**
- **Result: IoU = 1.000 on simulation test set · 6,701 masks · 21.1% tissue coverage**

#### Phase 3B — Visual Observation Integration

| Metric | Phase 2D (ground-truth) | Phase 3B (visual) | Change |
|--------|------------------------|-------------------|--------|
| Goal rate | **100%** | **0%** | Full regression |
| ep_rew_mean | −97.14 (eval) | −135.3 | −38.9% |

**Root cause:** Removing `goal_xyz` eliminates the navigational gradient. This directly quantifies the cost of operating without privileged simulator information — a publishable finding that most surgical RL papers avoid reporting honestly.

#### Phase 3C — Visual Tissue Force Proxy

Farneback dense optical flow measuring tissue deformation as a force proxy — formalising the visual judgement surgeons make when estimating tissue stress from screen.

**Results — 10 episodes, 3,000 steps:**

| Metric | Value |
|--------|-------|
| Mean tissue flow | **0.128 px/frame** |
| Max tissue flow | **0.732 px/frame** |
| Collision steps | **0 / 3,000 (0%)** |
| Alert threshold | 0.35 px/frame |
| Safety stop threshold | 1.0 px/frame |

**Git tags:** `v3.0` through `v3.4-phase3d-complete`

---

### ✅ Phase 4 — ROS 2 Middleware + Supervised Autonomy + Safety Architecture (Months 7–8)

**Clinical question:** Can the RL agent and perception capabilities be integrated into a ROS 2 architecture that mirrors commercial surgical robots — with a certifiable safety layer and human-in-the-loop control?

#### Phase 4A — ROS 2 Bridge (`sofa_bridge_node`)

Mapped SOFA simulation coordinates to the ROS 2 world frame. Solved the coordinate transform between FEM physics engine and robot kinematic model.

- Published `/tissue_force_proxy`, `/joint_states`, `/camera/image_raw` at 50Hz
- Teleoperation node for keyboard/joystick control (industry today — teleoperation track)
- SOFA headless mode with RenderMode.HEADLESS for production, RenderMode.HUMAN for demo

**Git tag:** `v4.0-phase4a-complete`

#### Phase 4B — RL Policy Action Servers

Wrapped the PPO agent as a ROS 2 action server using `lapgym_interfaces/action/Retract`.

**Key engineering decision — Separate ROS 2 Context per server:**

`env.step()` is a ~65ms synchronous SOFA blocking call. During this time the Python GIL is held and no ROS 2 callback can fire. Solution: each server creates an isolated DDS instance via `rclpy.Context()` with its own executor on a background thread at 10ms spin timeout.

```python
# Each server has this pattern
self._stop_context = rclpy.Context()
self._stop_context.init()
self._stop_node = rclpy.create_node('_surgeon_stop_approach',
                                     context=self._stop_context)
self._stop_executor = rclpy.executors.SingleThreadedExecutor(
                         context=self._stop_context)
self._stop_thread = threading.Thread(target=self._spin_stop_node, daemon=True)
```

**Dual freeze loop pattern** — two surgeon stop checks per execute cycle (before and after `env.step()`) bounds stop latency to at most one physics step (~65ms).

**Git tag:** `v4.1-phase4b-complete`

#### Phase 4C — Safety Watchdog (`safety_watchdog_node`)

Independent process per IEC 62304 requirements — cannot be blocked by application logic.

- Runs at **50Hz** — independent of the 15Hz SOFA physics loop
- Monitors `/tissue_force_proxy` against Phase 3C calibrated thresholds
- `ALERT` state at 0.35 px/frame — warns BT to slow approach
- `STOP` state at 1.0 px/frame — publishes `/emergency_stop=True`
- Publishes `/watchdog_status` and `/watchdog_heartbeat` for console display

**IEC 62304 design intent:** Safety-critical functions run in processes independent of the application logic they protect. The watchdog subscribes to `/tissue_force_proxy` independently — even if the BT or action servers hang, the watchdog continues monitoring.

**Git tag:** `v4.2-phase4c-complete`

#### Phase 4D — Behaviour Tree (`surgical_bt_node`)

Orchestrates the full surgical sequence using `py_trees_ros`.

```
Root (Sequence)
└── SafetyMonitor (Fallback)
    ├── SurgicalSequence (Sequence)
    │   ├── Approach  (ActionLeaf → approach_policy)
    │   ├── Retract   (ActionLeaf → retract_policy)
    │   └── Hold      (ActionLeaf → hold_policy)
    └── ForceWatchdog (Condition — /tissue_force_proxy < 0.35)
```

**Why BT over FSM:** BT Fallback node provides continuous force monitoring that preempts task execution natively. FSMs require explicit emergency transitions from every state — O(n) complexity that becomes a verification burden under IEC 62304. BTs are deterministic, auditable, and formally verifiable. Architecture matches Hannaford et al. (2018) — the reference paper for behaviour trees in medical procedures.

**Git tag:** `v4.4-phase4d-complete`

#### Phase 4E — Surgeon Console (`surgeon_console`)

Terminal-based human-in-the-loop control interface replicating the surgeon console layer of real surgical robots.

**Live telemetry display:**
```
============================================================
            SURGICAL ROBOT CONSOLE  v1.0
============================================================
PHASE:    RETRACT          STEP:  117 / 300
DISTANCE: 6.0mm            FORCE: 0.000
WATCHDOG: ● NOMINAL        ESTOP: CLEAR
BT STATE: RUNNING [SURGEON STOPPED]
------------------------------------------------------------
      [S] STOP    [R] RESUME    [E] EMERGENCY    [Q] QUIT
============================================================
Event log:
[13:52:04] [S] STOP  phase=RETRACT step=044 dist=86.7mm
[13:52:11] [R] RESUME phase=RETRACT step=050 dist=79.2mm
```

**Hardest engineering problem:** Stop latency bounded by SOFA `env.step()` execution time (~65ms per step). This is the sim-to-real gap made visible — on a real surgical robot, stop latency is hardware-bounded by PWM torque cutoff (<1ms) regardless of software state. Knowing that gap exists, why it exists, and how to document it is what IEC 62304 Class C software development means in practice.

**Verified integration test:**
```
APPROACH → [S] freeze → [R] resume → goal_reached steps=119
RETRACT  → [S] freeze → [R] resume → goal_reached steps=134 dist=2.6mm
HOLD     → [S] freeze → [R] resume → [E] emergency → ESTOP ACTIVE
```

**Git tag:** `v4.5-phase4e-complete`

#### Phase 4 — Complete Git Tags

| Tag | Description |
|-----|-------------|
| `v4.0-phase4a-complete` | ROS 2 bridge, coordinate mapping, teleop |
| `v4.1-phase4b-complete` | Action servers, separate context pattern |
| `v4.2-phase4c-complete` | Safety watchdog at 50Hz |
| `v4.4-phase4d-complete` | Behaviour tree orchestration |
| `v4.5-phase4e-complete` | Surgeon console S/R/E — system complete |

---

### 📋 Phase 5 — Evaluation, Benchmarking and Research Paper (Planned)

**Quantitative evaluation:**
- Task success rate, stop latency measurement, collision rate with/without safety watchdog
- Comparison: PPO agent vs proportional controller on retraction task
- Ablation: with/without surgeon stop, with/without BT, with/without force watchdog
- Safety stress test: 2× tissue stiffness, sensor noise injection, 100ms actuation delay

**Research paper target:** ISMR 2026 / IROS 2026 workshop / IEEE RA-L

---

## Repository Structure

```
surgical-rl/
├── docs/
│   ├── project-overview.md
│   ├── baseline_metrics.md
│   ├── scene_graph_analysis.md
│   ├── compatibility_fixes.md
│   ├── phase2a_results.md through phase2d_results.md
│   ├── eval_results_phase2.md
│   ├── phase3/
│   │   ├── README.md
│   │   ├── phase3a_results.md
│   │   ├── phase3b_results.md
│   │   ├── phase3c_results.md
│   │   └── phase3d_sim_to_real_gap_analysis.md
│   └── phase4/
│       ├── phase4a_ros2_bridge.md
│       ├── phase4b_action_servers.md
│       ├── phase4c_safety_watchdog.md
│       ├── phase4d_behaviour_tree.md
│       └── phase4e_surgeon_console.md      ← IEC 62304 design decisions
├── envs/
│   ├── safe_reward.py
│   ├── tissue_retraction_v2.py
│   ├── tissue_retraction_v3.py
│   └── perception_pipeline.py
├── ros2_packages/
│   └── lapgym_ros2_bridge/
│       ├── lapgym_ros2_bridge/
│       │   ├── sofa_bridge_node.py
│       │   ├── approach_policy_server.py
│       │   ├── retract_policy_server.py
│       │   ├── hold_policy_server.py
│       │   ├── safety_watchdog_node.py
│       │   ├── surgical_bt_node.py
│       │   ├── action_leaf.py
│       │   └── surgeon_console.py
│       ├── launch/
│       │   └── surgical_system.launch.py
│       └── package.xml
├── lapgym_interfaces/
│   └── action/
│       └── Retract.action
├── models/
│   ├── tip_detector/mobilenetv3_tip_best.pth
│   ├── segmentation/unet_seg_best.pth
│   └── force_proxy/proxy_config.json
├── scripts/
│   ├── baseline_demo.py
│   ├── train_ppo.py
│   ├── eval_agent.py
│   ├── watch_agent.py
│   ├── train_tip_detector.py
│   ├── train_segmentation.py
│   ├── train_ppo_visual.py
│   └── optical_flow_proxy.py
├── logs/
│   └── checkpoints/
│       └── phase2_ppo_tissue_retraction_20260409_211946/ppo_tissue_final
└── requirements.txt
```

---

## Quickstart

### Prerequisites

- Ubuntu 22.04
- NVIDIA GPU with CUDA 12.x
- Python 3.10
- ROS 2 Humble
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

### Watch the Phase 2D PPO agent (fastest demo — no ROS 2 required)

```bash
source setup_env.sh
python scripts/watch_agent.py --slow --episodes 3
# SOFA GUI opens — agent retracts tissue autonomously in ~130 steps
```

### Run the full Phase 4 ROS 2 system

```bash
source ~/surgical_robot_lapgym_ws/activate.sh
cd ~/surgical_robot_lapgym_ws/ros2_ws
colcon build --packages-select lapgym_ros2_bridge lapgym_interfaces
source install/setup.bash
cd ~/surgical_robot_lapgym_ws/surgical-rl
ros2 launch lapgym_ros2_bridge surgical_system.launch.py
# Surgeon console opens in xterm — press S to stop, R to resume, E for emergency
```

### Run Phase 3C optical flow force proxy

```bash
source setup_env.sh
python3 -u scripts/optical_flow_proxy.py
# 10 episodes · logs flow_log.csv · mean_tissue_flow ≈ 0.128 px/frame · 0 collisions
```

---

## Key Technical Decisions

**Why SOFA + LapGym:** SOFA provides finite element method (FEM) deformable body simulation. LapGym wraps SOFA in the Gymnasium interface. Intuitive Surgical Research funds LapGym development — it is the standard academic surgical simulation platform.

**Why separate rclpy.Context per action server:** `env.step()` is a ~65ms synchronous blocking call. The Python GIL prevents any callback from firing during this time. A separate `rclpy.Context()` creates an isolated DDS instance that spins on its own background thread — the only architecture that allows surgeon stop callbacks to fire during SOFA physics computation.

**Why dual freeze loops:** One freeze loop before `env.step()` and one after bounds stop latency to at most one physics step (~65ms). Without the second loop, the agent could run one full step after the surgeon pressed stop.

**Why IEC 62304-inspired safety watchdog as independent process:** IEC 62304 mandates that safety-critical functions run in processes independent of the application logic they protect. The watchdog subscribes to `/tissue_force_proxy` independently — even if the BT or action servers hang, the watchdog continues monitoring.

**Why Behaviour Tree over FSM:** BT Fallback node provides continuous force monitoring that preempts task execution natively — architecturally impossible to express cleanly in an FSM without O(n) emergency state logic. CMR Surgical's published architecture research uses BTs for surgical task sequencing.

**Why optical flow for force proxy:** SOFA's `BlockGaussSeidelConstraintSolver` stores contact forces internally — not accessible via `MechanicalObject.force`. Farneback dense optical flow measures tissue deformation directly from the camera frame, matching the visual judgement surgeons make. Medtronic Touch Surgery and NVIDIA Holoscan use this approach in real systems.

**Why Phase 3B result (0% goal rate) is scientifically correct:** Every variable except `goal_xyz` is held constant. The 100% → 0% regression is attributable entirely to removal of goal coordinate availability. This is a publishable finding that most surgical RL papers avoid reporting honestly.

**Why not LLM planner:** LLM-based planners are stochastic and non-deterministic. They cannot be validated under IEC 62304 or ISO 14971. BT transitions are deterministic, auditable, and every state change is traceable — the correct architecture for safety-critical surgical systems.

---

## Related Work

> Pore et al. (2021). *Safe Reinforcement Learning using Formal Verification for Tissue Retraction in Autonomous Robotic-Assisted Surgery.* IROS 2021. [arXiv:2109.02323](https://arxiv.org/abs/2109.02323)

> Scheikl et al. (2023). *LapGym — An Open Source Framework for Reinforcement Learning in Robot-Assisted Laparoscopic Surgery.* JMLR 24. [arXiv:2302.09606](https://arxiv.org/abs/2302.09606)

> Hannaford et al. (2018). *Behavior Trees as a Representation for Medical Procedures.* ICRA 2018.

> Tagliabue et al. (2021). *Learning from Demonstrations for Autonomous Soft-tissue Retraction.* [arXiv:2110.00336](https://arxiv.org/abs/2110.00336)

This project extends Pore et al. by implementing three-phase curriculum learning, building a full surgical perception pipeline, documenting the observation gap quantitatively, building a visual force proxy for force-sensorless robots, and constructing the full ROS 2 + BT + human-in-the-loop stack toward certifiable supervised autonomy.

---

## Platform

| Component | Version |
|-----------|---------|
| OS | Ubuntu 22.04 LTS |
| GPU | NVIDIA GTX 1650 (CUDA 12.8) |
| RAM | 16 GB |
| Python | 3.10.12 |
| SOFA | v25.12.00 |
| ROS 2 | Humble Hawksbill |
| PyTorch | 2.10.0+cu128 |
| Stable-Baselines3 | 2.7.1 |
| Gymnasium | 1.2.3 |
| OpenCV | 4.x |
| py_trees_ros | 2.x |
| Weights & Biases | 0.25.1 |

---

## Author

**Subhash Arockiadoss**
MSc Mechatronics and Robotics, De Montfort University Leicester (2024)

[LinkedIn](https://www.linkedin.com/in/subhasharockiadoss-2092b8171) · [GitHub](https://github.com/SUBHASH-Hub) · [W&B Phase 2](https://api.wandb.ai/links/subhashtronics-de-montfort-university-leicester/kqbip2vh) · [W&B Phase 3](https://api.wandb.ai/links/subhashtronics-de-montfort-university-leicester/0g3z7ei6)

*Seeking roles in surgical robotics AI and medical robotics — Open to sponsorship.*

---

*Phase 1 ✅ · Phase 2 ✅ · Phase 3 ✅ · Phase 4 ✅ · Phase 5 📋 planned · May 2026*