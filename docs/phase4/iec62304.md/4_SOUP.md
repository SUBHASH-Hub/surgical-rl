# SOUP Analysis
## Autonomous Tissue Retraction via Safe Reinforcement Learning
**Document ID:** SOUP-001  
**Version:** 1.0  
**Date:** May 2026  
**Author:** Subhash Arockiadoss  
**Standard:** IEC 62304:2006+Amd1:2015 §8 (SOUP)  
**Status:** Approved

---

## 1. Purpose

Software of Unknown Provenance (SOUP) is any software item used in a
medical device software system that was not developed under a life cycle
process meeting IEC 62304 requirements. This document identifies all SOUP
used in the system, their versions, known anomalies, and the risk controls
applied.

Per IEC 62304 §8.1.2, for each SOUP item used in Class B or C software,
the manufacturer shall:
- Identify the SOUP item and version
- Identify known anomalies relevant to safety
- Describe risk controls applied

---

## 2. SOUP Register

### 2.1 Runtime SOUPs — Class C Safety-Critical Path

These SOUPs are in the execution path of Class C safety-critical functions.

---

#### SOUP-001: Python 3.10.12

| Field | Value |
|-------|-------|
| Name | Python Interpreter |
| Version | 3.10.12 (pinned) |
| Supplier | Python Software Foundation |
| Source | Ubuntu 22.04 system package |
| Purpose | Primary execution runtime for all nodes |
| Safety class of using component | Class C |
| Functional use | Executes all node logic, threading, async callbacks |

**Known anomalies relevant to safety:**

| Anomaly | Relevance | Control |
|---------|-----------|---------|
| Python GIL prevents true multi-threading | **Safety-relevant** — prevents surgeon stop callbacks firing during `env.step()` | Mitigated by separate rclpy.Context with isolated DDS instance per action server (see SR-021) |
| Memory management is automatic (GC) | Low risk in this application | No control required |
| asyncio event loop not re-entrant | Potential issue if async patterns mixed with blocking calls | Avoided by design — no asyncio used in action servers |

**Version control:** Pinned to 3.10.12 via sofa_venv. System Python not used.

---

#### SOUP-002: ROS 2 Humble Hawksbill

| Field | Value |
|-------|-------|
| Name | Robot Operating System 2 |
| Version | Humble Hawksbill LTS (2022) |
| Supplier | Open Robotics / ROS 2 community |
| Source | Ubuntu apt: ros-humble-* packages |
| Purpose | Middleware — inter-node communication, action protocol, DDS |
| Safety class of using component | Class C |
| Sub-packages used | rclpy, rclcpp, std_msgs, sensor_msgs, geometry_msgs, action_msgs, py_trees_ros, ros-humble-py-trees-ros-interfaces |

**Known anomalies relevant to safety:**

| Anomaly | Relevance | Control |
|---------|-----------|---------|
| `rclpy.spin()` is single-threaded by default | **Safety-relevant** — main executor cannot process callbacks during blocking calls | Mitigated by separate rclpy.Context per action server |
| `goal_handle.canceled()` raises RCLError during invalid state transition | **Safety-relevant** — emergency stop during state transition could raise exception | Mitigated by try/except pattern with `goal_handle.abort()` fallback (SR-014) |
| DDS discovery latency at startup | Low risk — 15s startup delay in launch file covers this | TimerAction delay in launch file |
| `rcl_shutdown already called` on node exit | Cosmetic — harmless double-shutdown race | Known and documented, no safety impact |
| SOFA SIGABRT on Python exit | Cosmetic — SOFA C++ destructor issue | Handled by os._exit(0) |

**Version control:** Pinned to Humble LTS via apt. LTS support until May 2027.

---

#### SOUP-003: SOFA Framework v25.12.00

| Field | Value |
|-------|-------|
| Name | Simulation Open Framework Architecture |
| Version | v25.12.00 (pinned) |
| Supplier | INRIA / SOFA Consortium |
| Source | Pre-built binary from github.com/sofa-framework/sofa/releases/tag/v25.12.00 |
| Purpose | FEM physics simulation — tissue deformation, collision, instrument kinematics |
| Safety class of using component | Class C |
| Sub-components used | SofaPython3 bindings, ArticulatedSystemPlugin, FEM Elastic, Collision Detection |

**Known anomalies relevant to safety:**

| Anomaly | Relevance | Control |
|---------|-----------|---------|
| `env.step()` is synchronous blocking ~65ms | **Safety-relevant** — surgeon stop cannot interrupt mid-step | Mitigated by dual freeze loop pattern — stops after each step |
| UncoupledConstraintCorrection compliance warning | Cosmetic physics warning | Verified: simulation runs correctly, confirmed by 100% goal completion rate |
| RequiredPlugin errors for renamed plugins (SofaBoundaryCondition, SofaDeformable, SofaEngine, SofaGeneralRigid) | Cosmetic — plugins renamed in v24 | Confirmed harmless — simulation physics not affected |
| SIGABRT on Python interpreter exit (GIL destructor) | Cosmetic | Handled by os._exit(0) |
| NumPy 2.0 API breaking changes | Fixed at project start | 5 compatibility fixes applied — documented in docs/compatibility_fixes.md |
| BlockGaussSeidelConstraintSolver force not exposed via MechanicalObject.force | **Design-relevant** — direct force reading not available | Mitigated by optical flow force proxy (Phase 3C) |

**Version control:** Binary pinned to v25.12.00 at exact release tag.

---

#### SOUP-004: sofa_env / LapGym

| Field | Value |
|-------|-------|
| Name | LapGym — Laparoscopic Surgery Gymnasium Environment |
| Version | Commit pinned (April 2026 clone) |
| Supplier | Scheikl et al., JMLR 2023. Funded by Intuitive Surgical Research |
| Source | github.com/ScheiklP/lap_gym |
| Purpose | Wraps SOFA in Gymnasium interface — TissueRetractionV2 environment |
| Safety class of using component | Class C |
| Reference | Scheikl et al. "LapGym — An Open Source Framework for Reinforcement Learning in Robot-Assisted Laparoscopic Surgery." JMLR 24, 2023. |

**Known anomalies relevant to safety:**

| Anomaly | Relevance | Control |
|---------|-----------|---------|
| TissueRetractionV2 uses randomised grasping position each episode | Low risk — approach server compensates | ApproachPolicyServer queries world position at runtime |
| `obs[0:3]` (tool) and `obs[3:6]` (goal) in different normalised coordinate spaces — cannot be subtracted directly | Design-relevant — incorrect subtraction gives wrong guidance | Fixed in Phase 4A HUD: used `info['distance_to_grasping_position']` for correct distance |
| Render buffer updated in-place between frames | Fixed — caused optical flow to read identical frames | Fixed by `.copy()` when storing frame reference (Phase 3C) |
| `coverage.types` AttributeError (numba) | Setup issue | Patched `coverage_support.py` — documented in compatibility_fixes.md |

**Version control:** Specific commit hash pinned. No automatic updates.

---

#### SOUP-005: Stable-Baselines3

| Field | Value |
|-------|-------|
| Name | Stable-Baselines3 |
| Version | 2.7.1 (pinned in requirements.txt) |
| Supplier | DLR-RM / Community |
| Source | pip install stable-baselines3==2.7.1 |
| Purpose | PPO policy implementation — training and inference |
| Safety class of using component | Class C (inference path) |

**Known anomalies relevant to safety:**

| Anomaly | Relevance | Control |
|---------|-----------|---------|
| PPO policy is stochastic during training; deterministic=True at inference | **Safety-relevant during training** — not safety-relevant at deployment since deterministic=True | All deployment uses `policy.predict(obs, deterministic=True)` |
| Model checkpoint format may change between versions | Low risk — checkpoint from same version | Checkpoint saved and loaded with same SB3 2.7.1 |
| CUDA unknown error warning on multi-process startup | Cosmetic — multiple processes each init CUDA | No safety impact; policy loads correctly after warning |

**Version control:** Pinned to 2.7.1 in requirements.txt.

---

#### SOUP-006: PyTorch

| Field | Value |
|-------|-------|
| Name | PyTorch |
| Version | 2.10.0+cu128 (pinned) |
| Supplier | Meta AI / PyTorch community |
| Source | pip install torch==2.10.0+cu128 |
| Purpose | Neural network inference for PPO policy |
| Safety class of using component | Class C (inference path) |

**Known anomalies relevant to safety:**

| Anomaly | Relevance | Control |
|---------|-----------|---------|
| CUDA device memory not automatically freed between processes | Low risk — each server owns one model | Each action server owns exactly one model instance |
| Non-determinism on GPU (some CUDA operations) | Low risk — policy determinism controlled by SB3 | deterministic=True in policy.predict() |

**Version control:** Pinned to 2.10.0+cu128.

---

#### SOUP-007: py_trees / py_trees_ros

| Field | Value |
|-------|-------|
| Name | py_trees + py_trees_ros |
| Version | py_trees 2.2.3 (pinned), py_trees_ros (ros-humble apt) |
| Supplier | Daniel Stonier / py_trees community |
| Source | pip install py_trees==2.2.3, apt ros-humble-py-trees-ros |
| Purpose | Behaviour tree implementation — surgical task orchestration |
| Safety class of using component | Class C |

**Known anomalies relevant to safety:**

| Anomaly | Relevance | Control |
|---------|-----------|---------|
| ForceCondition consecutive counter reset bug — `terminate()` called on SUCCESS reset counter | **Safety-relevant** — prevented ForceCondition from reaching 3/3 threshold | Fixed in Phase 4D: terminate() only resets on INVALID (BT reset), not on SUCCESS |
| BT tick is synchronous — if tick takes > 100ms, tick rate drops below 10Hz | Low risk — ticks complete in <10ms | Verified in testing |
| BT is application-layer safety only — not independent per IEC 62304 | **Architectural** — intentional design | Safety watchdog node (SOUP-009) provides independent layer |

**Version control:** py_trees pinned to 2.2.3. py_trees_ros pinned via ros-humble apt.

---

### 2.2 Runtime SOUPs — Class B Path

---

#### SOUP-008: OpenCV

| Field | Value |
|-------|-------|
| Name | OpenCV |
| Version | 4.x (latest in sofa_venv) |
| Supplier | OpenCV community |
| Source | pip install opencv-python |
| Purpose | Farneback optical flow for tissue force proxy; image processing |
| Safety class of using component | Class B (force proxy input to Class C watchdog) |

**Known anomalies:**

| Anomaly | Relevance | Control |
|---------|-----------|---------|
| cv2.calcOpticalFlowFarneback GPU acceleration not used | Performance only | CPU implementation sufficient at 15Hz |
| Object aliasing — both frame pointers could reference same array | **Fixed** — caused flow=0 | Fixed by `.copy()` on frame store |

---

#### SOUP-009: Gymnasium

| Field | Value |
|-------|-------|
| Name | Gymnasium |
| Version | 1.2.3 (pinned) |
| Supplier | Farama Foundation |
| Source | pip install gymnasium==1.2.3 |
| Purpose | RL environment interface — step(), reset(), observation/action spaces |
| Safety class of using component | Class B |

**Known anomalies:**

| Anomaly | Relevance | Control |
|---------|-----------|---------|
| gym → gymnasium shim required | Compatibility | sys.modules['gym'] = gymnasium at top of each server |

---

#### SOUP-010: NumPy

| Field | Value |
|-------|-------|
| Name | NumPy |
| Version | Pinned (compatible with PyTorch 2.10) |
| Supplier | NumPy community |
| Purpose | Array operations, action clipping, distance calculations |
| Safety class of using component | Class B |

**Known anomalies:**

| Anomaly | Relevance | Control |
|---------|-----------|---------|
| NumPy 2.0 API breaking changes | Fixed at project start | 5 compatibility fixes in docs/compatibility_fixes.md |

---

#### SOUP-011: pynput

| Field | Value |
|-------|-------|
| Name | pynput |
| Version | Latest in sofa_venv |
| Supplier | Moses Palmér |
| Purpose | Keyboard capture for teleop node |
| Safety class of using component | Class B (teleop only) |

**Known anomalies:** None safety-relevant.

---

#### SOUP-012: curses (Python stdlib)

| Field | Value |
|-------|-------|
| Name | curses |
| Version | Python 3.10.12 stdlib |
| Supplier | Python Software Foundation |
| Purpose | Terminal UI for surgeon console |
| Safety class of using component | Class B |

**Known anomalies:** `curses.error` raised on terminal resize. Caught by try/except in draw loop.

---

### 2.3 Build-Time SOUPs (not in execution path)

| SOUP | Version | Purpose | Safety Risk |
|------|---------|---------|------------|
| empy | 3.3.4 | ROS2 code generation | None — build only |
| catkin_pkg | Latest | ament build system | None — build only |
| lark | Latest | rosidl parser | None — build only |
| setuptools | 65.5.1 (pinned) | ament compatibility | None — build only |

---

## 3. SOUP Risk Summary

| SOUP | Safety Class | Key Risk | Control Applied |
|------|-------------|----------|-----------------|
| Python 3.10 | C | GIL blocks callbacks | Separate rclpy.Context |
| ROS 2 Humble | C | State transition RCLError | try/except abort() fallback |
| SOFA v25.12 | C | env.step() blocking 65ms | Dual freeze loop pattern |
| LapGym | C | Frame aliasing, coord mismatch | .copy(), use info dict |
| SB3 2.7.1 | C | Stochastic policy | deterministic=True |
| PyTorch 2.10 | C | GPU non-determinism | deterministic=True |
| py_trees 2.2.3 | C | Counter reset bug | Fixed: terminate() logic |
| OpenCV 4.x | B | Frame aliasing | .copy() fix |
| Gymnasium 1.2.3 | B | gym/gymnasium namespace | shim at module top |

All known safety-relevant SOUP anomalies have been mitigated by design.
No unresolved SOUP risks at v4.5-phase4e-complete.