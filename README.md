# Surgical RL — Autonomous Tissue Retraction via Safe Reinforcement Learning

> **Building the complete AI stack of a surgical autonomy system — from physics simulation to safe RL to surgical perception to ROS 2 middleware with hybrid C++/Python human-in-the-loop control.**

---

## Clinical Motivation

During laparoscopic cholecystectomy — gallbladder removal, one of the most common operations worldwide — the surgeon must retract the gallbladder fundus to expose **Calot's triangle** before safe dissection. This retraction must stay within a force window of **0.5–3.0 N**. Too little force: the anatomy is not visible. Too much force: the cystic artery risks tearing.

Surgical outcome quality today varies with surgeon skill. This project builds an AI agent that learns to perform this retraction subtask autonomously in SOFA physics simulation — with safety-aware reward design, curriculum learning, a full surgical perception pipeline, a complete ROS 2 architecture mirroring commercial surgical robots, IEC 62304 Class C safety documentation, and a hybrid C++/Python control layer that mirrors how real surgical AI systems separate control logic from ML inference.

---

## Demo — Phase 4G: Hybrid C++/Python System Running Live

> Full autonomous surgical procedure with surgeon console control.
> C++ action servers (no GIL) + Python ML/physics layer.
> APPROACH → RETRACT (PPO) → HOLD — single launch command.

```bash
source ~/surgical_robot_lapgym_ws/activate.sh
cd ~/surgical_robot_lapgym_ws/surgical-rl
ros2 launch lapgym_ros2_bridge surgical_system.launch.py
```

**Surgeon console key bindings:**

| Key | Action | Effect |
|-----|--------|--------|
| S | Surgeon Stop | Freezes C++ server mid-step (~130ms latency) |
| R | Resume | Continues from exact stop point — no reset |
| E | Emergency Stop | /emergency_stop=True — all nodes halt, BT FAILED |
| Q | Quit | Closes console |

---

## Phase-by-Phase Results Summary

| Metric | Phase 1 Scripted | Phase 2D PPO | Phase 3B Visual | Phase 4E Python | Phase 4G C++ |
|--------|-----------------|--------------|-----------------|-----------------|---------------|
| Observation | Hardcoded | 7D ground-truth | 132D visual | 7D (Phase 2D) | 7D (Phase 2D) |
| Episode reward | −165.54 | **−97.14** | −135.3 | −97.14 | −97.14 |
| Episode length | 247 steps | **142.3 steps** | 300 (truncated) | 119–134 steps | 72–118 steps |
| Goal rate | 100% | **100%** | 0% | **100%** | **100%** |
| Safety layer | None | None | Calibrated | IEC 62304 watchdog | IEC 62304 watchdog |
| Stop latency | N/A | N/A | N/A | ~1-1.3 seconds | **~130ms (10× faster)** |
| C++ control | No | No | No | No | **Yes** |
| IEC 62304 docs | No | No | No | No | **Yes (Class C)** |

📊 [W&B Phase 2 — all 4 training runs](https://api.wandb.ai/links/subhashtronics-de-montfort-university-leicester/kqbip2vh)
📊 [W&B Phase 3 — visual PPO + perception](https://api.wandb.ai/links/subhashtronics-de-montfort-university-leicester/0g3z7ei6)

---

## System Architecture — Hybrid C++/Python (Phase 4G)

```
┌──────────────────────────────────────────────────────────────────────┐
│  LAYER 5 — Human Interface (Python — curses)                         │
│  surgeon_console · S/R/E/Q · /surgeon_stop · /emergency_stop         │
└──────────────────────────────┬───────────────────────────────────────┘
                                │ /console_feedback
┌──────────────────────────────▼───────────────────────────────────────┐
│  LAYER 4 — Task Orchestration (Python — py_trees)                    │
│  surgical_bt_node · Approach→Retract→Hold · ForceCondition guard     │
└──────────────────────────────┬───────────────────────────────────────┘
                ROS 2 Action: /approach_policy_cpp
                               /retract_policy_cpp
                               /hold_policy_cpp
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 3 — C++ Control Layer (rclcpp — no GIL)                      │
│                                                                      │
│  approach_policy_server_cpp    retract_policy_server_cpp             │
│  Proportional controller       PPO via /ppo_predict + /sofa_step     │
│                                                                      │
│  hold_policy_server_cpp                                              │
│  Zero-action position hold                                           │
│                                                                      │
│  Each: std::atomic<bool> stop flags                                  │
│         MultiThreadedExecutor + 3-4 callback groups                  │
│         Dual freeze loop — stop within 1-2 steps                     │
└───────────────┬─────────────────────────────────────────────────────┘
    /sofa_step  │  /ppo_predict
┌───────────────▼─────────────────────────────────────────────────────┐
│  LAYER 2 — Python ML/Physics Services                                │
│                                                                      │
│  sofa_step_service        ppo_predict_service                        │
│  TissueRetractionV2       PPO.load(checkpoint)                       │
│  env.step() → obs/dist    policy.predict(obs) → action               │
│  SofaPython3 (GIL held)   PyTorch releases GIL → GPU                │
└─────────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — Safety + Bridge (Python)                                  │
│  sofa_bridge_node: /tissue_force_proxy at 50Hz                       │
│  safety_watchdog_node: IEC 62304 Class C — 50Hz independent          │
└──────────────────────────────────────────────────────────────────────┘
├──────────────────────────────────────────────────────────────────────┤
│  PHASE 3 — Surgical Perception  [COMPLETE ✓]                         │
│  MobileNetV3 tip detection (5.1px MAE) · UNet segmentation (IoU=1.0) │
│  Farneback optical flow force proxy (0.128 px/frame)                 │
├──────────────────────────────────────────────────────────────────────┤
│  PHASE 2 — Safe RL Core  [COMPLETE ✓]                                │
│  PPO · SafeRewardWrapper · 3-phase curriculum · 136,711 params        │
├──────────────────────────────────────────────────────────────────────┤
│  PHASE 1 — SOFA v25.12 FEM Physics Engine  [COMPLETE ✓]              │
│  LapGym · Tissue 27,040 Pa · 1,479 nodes · RCM constraint            │
└──────────────────────────────────────────────────────────────────────┘
```

Phase 4 Architecture SVG

![Phase 4G System Architecture](docs/phase4/images/phase4g_architecture.svg)
---

## Why the Hybrid C++/Python Pattern

The original Python action servers had surgeon stop latency of ~1 second. The root cause is the Python GIL — `env.step()` holds the GIL for ~65ms and blocks all callbacks including surgeon stop.

C++ has no GIL. Threads run truly in parallel on separate CPU cores. `std::atomic<bool>` provides safe shared state between parallel threads without mutex overhead.

```
Python (Phase 4E):                C++ (Phase 4G):
env.step() holds GIL ~65ms        sofaStep() on Thread A
→ ALL callbacks blocked            → stop callback fires on Thread B
→ stop delayed 15-20 steps         → std::atomic flag set immediately
→ ~1 second latency                → stop within 1-2 steps = ~130ms
```

PyTorch already releases the GIL before GPU computation, so PPO inference stays fast in Python. SOFA has Python bindings (SofaPython3) — no C++ application interface exists. The hybrid split is architecturally correct and matches how CMR Surgical, Intuitive Surgical, and Moon Surgical build production systems.

**Industry mapping:**

| Layer | CMR Versius / Medtronic Hugo | This Project |
|-------|------------------------------|--------------|
| Inner control | Proprietary servo 500–1000Hz | SOFA C++ FEM ~15Hz |
| Middleware | ROS 2 control stack | ROS 2 Humble — 9 nodes |
| Control logic | C++ rclcpp | C++ rclcpp (Phase 4F/4G) |
| ML/AI | Python + GPU | Python PyTorch (Phase 2) |
| Perception | Endoscopic camera | MobileNetV3 + UNet (Phase 3) |
| Force sensing | Hardware F/T sensor | Optical flow proxy (Phase 3C) |
| Safety | Independent watchdog | safety_watchdog_node 50Hz |
| Human control | Surgeon console | surgeon_console S/R/E |

---

## IEC 62304 Class C Compliance

The system was brought into IEC 62304 Class C compliance framework in Phase 4E and extended by the C++ implementation in Phase 4F/4G.

**Classification rationale:** Autonomous surgical instrument control — safety layer failure could result in undetected dangerous tissue force.

**Design history file:** `docs/iec62304/`

| Document | Key Content |
|---------|-------------|
| SDP-001 | Development plan — lifecycle, tools, problem resolution table |
| SRS-001 | 56 numbered requirements — FR, SR, PR with safety classes |
| SAD-001 | 5-layer architecture, safety independence argument |
| SOUP-001 | 12 SOUP items — Python GIL anomaly, SOFA blocking call documented |
| RMF-001 | 6 risks per ISO 14971 — all mitigated to ALARP |
| TRACEABILITY-001 | Requirements → code → test evidence chain |

The SOUP analysis explicitly documents the Python GIL as a safety-relevant anomaly and the C++ callback group pattern as the mitigation — demonstrating that Phase 4F/4G was not just a performance improvement but a safety-driven architectural decision.

**Git tag:** `v4.7-phase4e-iec62304`

---

## Phase 4 — Complete Node Table

| Node | Language | Role | Hz | Phase |
|------|----------|------|-----|-------|
| `sofa_bridge_node` | Python | SOFA bridge + /tissue_force_proxy | 50 | 4A |
| `sofa_step_service` | Python | env.step() service for C++ servers | on-demand | 4F |
| `ppo_predict_service` | Python | policy.predict() service for C++ retract | on-demand | 4G |
| `approach_policy_server_cpp` | **C++** | Proportional controller | ~15 | 4F |
| `retract_policy_server_cpp` | **C++** | PPO retract via /ppo_predict | ~15 | 4G |
| `hold_policy_server_cpp` | **C++** | Zero-action hold | ~10 | 4G |
| `safety_watchdog_node` | Python | IEC 62304 independent force monitor | 50 | 4C |
| `surgical_bt_node` | Python | Behaviour tree orchestrator | 10 | 4D |
| `surgeon_console` | Python | Human-in-the-loop terminal UI | 10 | 4E |

---

## Five-Phase Roadmap

### ✅ Phase 1 — Simulation Foundation (Months 1–2)

Built the SOFA + LapGym simulation environment and established the scripted baseline.

- SOFA v25.12 FEM physics — tissue Young's modulus 27,040 Pa, Poisson ratio 0.4287
- LapGym TissueRetractionEnv at 15–17 FPS on GTX 1650
- **Scripted baseline: 247 steps · 49 collision steps · −165.54 reward**
- Resolved 5 SOFA v25.12 + NumPy 2.0 API compatibility issues

**Git tag:** `v1.0-phase1-complete`

---

### ✅ Phase 2 — Safe RL Core (Months 3–4)

Four training runs. Each failure root-caused and fixed.

| Run | Key Change | Outcome |
|-----|-----------|---------|
| Phase 2A | 3D observation baseline | ❌ Reward stuck at −359 |
| Phase 2B | 7D observation | ❌ Curriculum shock at step 300k |
| Phase 2C | Step-based curriculum | ❌ λ=0.8 reward collapse |
| Phase 2D | λ_max=0.5, trigger=350k | ✅ Target met |

**Final eval (10 episodes):** steps 142.3 (−42%), reward −97.14 (+41%), 100% goal rate

**Git tags:** `v2.2` through `v2.4-phase2-complete`

---

### ✅ Phase 3 — Surgical Perception Pipeline (Months 5–6)

| Component | Result |
|-----------|--------|
| MobileNetV3 tip detector | 5.1px MAE — below 10px surgical AI threshold |
| UNet tissue segmentation | IoU = 1.000 on simulation test set |
| Farneback force proxy | 0.128 px/frame mean · 0/3,000 collision steps |
| Phase 3B observation gap | 100% → 0% goal rate when goal_xyz removed (publishable) |

**Git tags:** `v3.0` through `v3.4-phase3d-complete`

---

### ✅ Phase 4 — ROS 2 Middleware + Hybrid C++/Python (Months 7–8)

#### Phase 4A–4E — Foundation (Python)
ROS 2 bridge, Python action servers, safety watchdog (IEC 62304), behaviour tree, surgeon console with S/R/E control.

**Stop latency (Python):** ~1-1.3 seconds due to GIL blocking during env.step().

**Git tags:** `v4.0` through `v4.5-phase4e-complete`

#### Phase 4E — IEC 62304 Design History
Six documents covering full software lifecycle per IEC 62304 Class C.

**Git tag:** `v4.7-phase4e-iec62304`

#### Phase 4F — C++ Approach Server
First C++ rclcpp action server. Demonstrates:
- MultiThreadedExecutor + callback groups
- std::atomic<bool> for thread-safe stop flags
- future.wait_for() for service calls without executor conflict
- CMakeLists.txt + ament_cmake build system

**Git tag:** `v4.6-phase4f-cpp`

#### Phase 4G — C++ Hold + Retract + Hybrid Launch
Complete C++ control layer with PPO predict service bridge.

**Verified results:**
```
Approach (C++): steps=72   dist=24.4mm  goal_reached
Retract  (C++): steps=118  dist=29.5mm  goal_reached
Hold     (C++): steps=83   emergency_stop (E key — correct)
Stop latency:   ~130ms (10× improvement over Python)
```

**Git tags:** `v4.8-phase4g-cpp` · `v4.9-phase4g-hybrid-launch`

---

### 📋 Phase 5 — Evaluation and Research Paper (Planned)

- 10-episode systematic evaluation with varied starting positions
- Stop latency measurement: Python vs C++ servers
- Safety ablation: with/without watchdog, with/without surgeon stop
- Research paper target: ISMR 2026 / IROS 2026 / IEEE RA-L

---

## Repository Structure

```
surgical-rl/
├── docs/
│   ├── iec62304/                          ← IEC 62304 Class C design history
│   │   ├── 1_SDP.md                       SDP-001
│   │   ├── 2_SRS.md                       SRS-001 (56 requirements)
│   │   ├── 3_SAD.md                       SAD-001
│   │   ├── 4_SOUP.md                      SOUP-001 (12 items, GIL documented)
│   │   ├── 5_RMF.md                       RMF-001 (6 risks, ISO 14971)
│   │   └── 6_TRACEABILITY.md              TRACEABILITY-001
│   ├── phase3/
│   │   └── phase3a–3d results
│   └── phase4/
│       ├── phase4a–4e docs
│       ├── phase4f_cpp_action_server.md   C++ approach — GIL/atomic explanation
│       └── phase4g_cpp_hold_and_retract.md C++ hold + retract + PPO service
├── ros2_packages/
│   ├── lapgym_interfaces/
│   │   ├── action/Retract.action
│   │   └── srv/
│   │       ├── SofaStep.srv               C++↔Python SOFA bridge
│   │       └── PPOPredict.srv             C++↔Python PPO bridge
│   ├── lapgym_ros2_bridge/
│   │   ├── lapgym_ros2_bridge/
│   │   │   ├── sofa_step_service.py       Python SOFA service
│   │   │   ├── ppo_predict_service.py     Python PPO service
│   │   │   ├── surgical_bt_node.py        BT — updated _cpp action names
│   │   │   └── *.py (all other nodes)
│   │   └── launch/surgical_system.launch.py  Hybrid C++/Python launch
│   └── lapgym_ros2_bridge_cpp/
│       ├── src/
│       │   ├── approach_policy_server.cpp
│       │   ├── hold_policy_server.cpp
│       │   └── retract_policy_server.cpp
│       └── CMakeLists.txt
├── envs/
├── models/
├── scripts/
└── requirements.txt
```

---

## Quickstart

### Prerequisites
- Ubuntu 22.04, NVIDIA GPU CUDA 12.x, Python 3.10, ROS 2 Humble
- SOFA v25.12 at `~/surgical_robot_lapgym_ws/sofa_install/`
- LapGym at `~/surgical_robot_lapgym_ws/lap_gym/`

### Setup
```bash
git clone https://github.com/SUBHASH-Hub/surgical-rl.git
source setup_env.sh
pip install -r requirements.txt
```

### Watch PPO agent (no ROS 2 required)
```bash
source setup_env.sh
python scripts/watch_agent.py --slow --episodes 3
```

### Build and run full hybrid system
```bash
# Build C++ packages (deactivate venv first)
deactivate
source /opt/ros/humble/setup.bash
cd ~/surgical_robot_lapgym_ws/ros2_ws
colcon build --packages-select lapgym_interfaces lapgym_ros2_bridge_cpp lapgym_ros2_bridge
source install/setup.bash

# Launch
source ~/surgical_robot_lapgym_ws/activate.sh
cd ~/surgical_robot_lapgym_ws/surgical-rl
ros2 launch lapgym_ros2_bridge surgical_system.launch.py
```

---

## Key Technical Decisions

**Why C++ for action servers:** Python GIL holds during SOFA env.step() (~65ms) blocking all callbacks including surgeon stop. C++ has no GIL — threads run truly in parallel. std::atomic<bool> provides safe shared stop flag between parallel threads. Stop latency improved from ~1 second to ~130ms.

**Why Python for ML/physics:** PyTorch releases GIL before GPU computation — PPO inference does not block callbacks. SOFA has Python bindings (SofaPython3) — no C++ application interface. py_trees_ros has no C++ equivalent.

**Why MultiThreadedExecutor + callback groups:** sofaStep() and ppoPredict() service calls use future.wait_for() which blocks the execute() thread. Without separate callback groups, the executor cannot process service responses during this block — deadlock. Each service gets its own MutuallyExclusive callback group.

**Why separate rclpy.Context per Python server (Phase 4E):** Same problem, Python-specific solution — isolated DDS instance on background thread. C++ servers replaced this with callback groups in Phase 4F.

**Why IEC 62304 Class C:** Autonomous surgical instrument control — safety watchdog failure could result in undetected dangerous tissue force. Class C mandates independent safety processes, full traceability, and SOUP analysis.

**Why BT over FSM:** Fallback node provides continuous force monitoring natively. FSM requires O(n) emergency transitions from every state. BTs are deterministic, auditable, formally verifiable.

**Why optical flow for force proxy:** SOFA BlockGaussSeidelConstraintSolver does not expose contact forces via MechanicalObject.force. Farneback optical flow measures tissue deformation from camera frame — matching the visual judgement surgeons make clinically.

**Why Phase 3B 0% goal rate is scientifically correct:** Only goal_xyz changed between Phase 2D and 3B. 100% → 0% regression is attributable entirely to removal of navigational gradient. Publishable finding that most surgical RL papers avoid reporting honestly.

---

## Complete Tag History

```
v1.0-phase1-complete       Simulation baseline
v2.4-phase2-complete       PPO safe RL (750k steps)
v3.4-phase3d-complete      Perception pipeline
v4.0-phase4a-complete      ROS2 bridge
v4.1-phase4b-complete      Action servers (Python baseline)
v4.2-phase4c-complete      Safety watchdog (IEC 62304)
v4.4-phase4d-complete      Behaviour tree
v4.5-phase4e-complete      Surgeon console S/R/E
v4.6-phase4f-cpp           C++ approach server (GIL gap filled)
v4.7-phase4e-iec62304      IEC 62304 Class C design history
v4.8-phase4g-cpp           C++ hold + retract servers
v4.9-phase4g-hybrid-launch Hybrid C++/Python launch — BT updated
```

---

## Related Work

> Pore et al. (2021). *Safe RL using Formal Verification for Tissue Retraction.* IROS 2021. [arXiv:2109.02323](https://arxiv.org/abs/2109.02323)

> Scheikl et al. (2023). *LapGym — Open Source Framework for RL in Laparoscopic Surgery.* JMLR 24. [arXiv:2302.09606](https://arxiv.org/abs/2302.09606)

> Hannaford et al. (2018). *Behavior Trees for Medical Procedures.* ICRA 2018.

This project extends Pore et al. by: three-phase curriculum learning, full surgical perception pipeline, quantified observation gap, visual force proxy, ROS 2 + BT + human-in-the-loop stack, IEC 62304 Class C compliance framework, and hybrid C++/Python control architecture demonstrating production-pattern surgical robotics engineering.

---

## Platform

| Component | Version |
|-----------|---------|
| OS | Ubuntu 22.04 LTS |
| GPU | NVIDIA GTX 1650 (CUDA 12.8) |
| Python | 3.10.12 |
| SOFA | v25.12.00 |
| ROS 2 | Humble Hawksbill |
| C++ Standard | C++17 (GCC 11.4.0) |
| PyTorch | 2.10.0+cu128 |
| Stable-Baselines3 | 2.7.1 |
| Gymnasium | 1.2.3 |
| OpenCV | 4.x |
| py_trees_ros | 2.x |

---

## Author

**Subhash Arockiadoss**
MSc Mechatronics and Robotics, De Montfort University Leicester (2024)

[LinkedIn](https://www.linkedin.com/in/subhasharockiadoss-2092b8171) · [GitHub](https://github.com/SUBHASH-Hub) · [W&B Phase 2](https://api.wandb.ai/links/subhashtronics-de-montfort-university-leicester/kqbip2vh) · [W&B Phase 3](https://api.wandb.ai/links/subhashtronics-de-montfort-university-leicester/0g3z7ei6)

*Seeking roles in surgical robotics AI and medical robotics — open to UK/Switzerland/US sponsorship.*

---

*Phase 1 ✅ · Phase 2 ✅ · Phase 3 ✅ · Phase 4 ✅ (hybrid C++/Python + IEC 62304) · Phase 5 📋 · June 2026*