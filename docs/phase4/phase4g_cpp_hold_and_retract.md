# Phase 4G — C++ Hold and Retract Servers

**Author:** Subhash Arockiadoss  
**Status:** Complete  
**Tag:** v4.8-phase4g-cpp  
**Date:** May 2026

---

## Overview

Phase 4G extends the C++ port from Phase 4F by adding two more action
servers — hold and retract — completing the full C++ control layer.

Phase 4G also introduces the PPO predict service pattern: PPO inference
stays in Python (PyTorch releases GIL for GPU), while the C++ retract
server handles all control loop logic, threading, and stop handling.

---

## Complete C++ Architecture After Phase 4G
C++ control layer:                Python ML/physics layer:
──────────────────                ────────────────────────
approach_policy_server_cpp  →     /sofa_step (SofaStepService)
hold_policy_server_cpp      →     /sofa_step (SofaStepService)
retract_policy_server_cpp   →     /ppo_predict (PPOPredictService)
→     /sofa_step (SofaStepService)

C++ handles: action servers, threading, stop flags, control loop  
Python handles: PPO inference (PyTorch + GPU), SOFA FEM physics

---

## What Was Built

### New C++ nodes

**hold_policy_server_cpp** (`/hold_policy_cpp`)  
Sends zero-delta actions every step. No ML dependency. Simplest C++ port.

**retract_policy_server_cpp** (`/retract_policy_cpp`)  
Calls /ppo_predict then /sofa_step per step. Most complex C++ node —
four callback groups for concurrent execution.

### New Python service

**ppo_predict_service.py** (`/ppo_predict`)  
Loads Phase 2D PPO checkpoint. Receives obs[7] from C++ server, calls
`policy.predict(obs, deterministic=True)`, returns action[3].

### New service interface

**PPOPredict.srv**
Request
float32[7] observation
Response
float32[3] action
bool success

---

## Why PPO Stays in Python

PPO inference uses PyTorch. PyTorch explicitly releases the GIL before
launching CUDA kernels on GPU. This means PPO inference does NOT block
other Python threads — unlike SOFA's env.step() which holds the GIL.
policy.predict(obs):
Python calls PyTorch → PyTorch releases GIL
→ CUDA kernel launches on GPU
→ other Python threads can run freely
→ GPU computes inference
→ GIL reacquired → returns action[3]

Moving PPO inference to C++ would require a C++ PyTorch inference
pipeline (LibTorch) — significant complexity with no safety benefit
since the GIL is already released during GPU computation.

---

## Retract Server Data Flow
Per step:
obs[7] ──→ /ppo_predict ──→ action[3] ──→ /sofa_step ──→ new_obs[7]
└──→ dist, terminated
↑                                                              │
└──────────────────── next step ───────────────────────────────┘
Repeat until dist < 30mm (GOAL_THRESHOLD) or terminated

---

## Four Callback Groups in Retract Server

The retract server needs four concurrent callback groups — one more
than the approach server:
Group 1: default            → action server goals/cancels
Group 2: sofa_callback_group_  → /sofa_step responses
Group 3: ppo_callback_group_   → /ppo_predict responses  ← new
Group 4: stop_callback_group_  → /surgeon_stop, /emergency_stop

Both service responses must be processable while execute() loop runs.
Without separate callback groups for each service, future.wait_for()
would deadlock — the executor cannot process the response while
blocked waiting for it.

---

## Hold Server — Why No PPO Service Needed

Hold sends `[0.0, 0.0, 0.0]` every step — pure control logic. No
ML inference required. Three callback groups only:
Group 1: default               → action server
Group 2: client_callback_group_ → /sofa_step responses
Group 3: stop_callback_group_   → /surgeon_stop, /emergency_stop

---

## Verified Results

### Hold server
Goal: max_steps=10
Result: success=true, steps=10, termination=timeout ✅

### Retract server — 3 runs
Run 1: steps=101  dist=29.0mm  goal_reached ✅
Run 2: steps=104  dist=29.9mm  goal_reached ✅
Run 3: steps=154  dist=29.0mm  goal_reached ✅

Step variation (101→154) caused by randomised episode starting
position — not a bug. Larger starting distance requires more steps.

### Observed behaviour — 0.0mm phase

First ~75 steps show dist=0.0mm then suddenly non-zero. This is
correct — `distance_to_end_position` only becomes active after
grasping triggers. Before grasp, SOFA returns 0.0 for end distance.
In full system the BT runs approach first so retract always starts
post-grasp.

---

## Files Added

| File | Description |
|------|-------------|
| `ros2_packages/lapgym_ros2_bridge_cpp/src/hold_policy_server.cpp` | C++ zero-action hold |
| `ros2_packages/lapgym_ros2_bridge_cpp/src/retract_policy_server.cpp` | C++ PPO retract |
| `ros2_packages/lapgym_ros2_bridge_cpp/CMakeLists.txt` | Updated — 3 executables |
| `ros2_packages/lapgym_ros2_bridge/lapgym_ros2_bridge/ppo_predict_service.py` | Python PPO service |
| `ros2_packages/lapgym_interfaces/srv/PPOPredict.srv` | Service definition |

---

## Complete C++ Package Summary After Phase 4F + 4G

| Executable | Action | Controller | Services called |
|-----------|--------|-----------|----------------|
| approach_policy_server_cpp | /approach_policy_cpp | Proportional | /sofa_step |
| hold_policy_server_cpp | /hold_policy_cpp | Zero-action | /sofa_step |
| retract_policy_server_cpp | /retract_policy_cpp | PPO Phase 2D | /ppo_predict + /sofa_step |