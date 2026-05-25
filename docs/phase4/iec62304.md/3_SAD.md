# Software Architecture Document (SAD)
## Autonomous Tissue Retraction via Safe Reinforcement Learning
**Document ID:** SAD-001  
**Version:** 1.0  
**Date:** May 2026  
**Author:** Subhash Arockiadoss  
**Standard:** IEC 62304:2006+Amd1:2015 §5.3  
**Status:** Approved

---

## 1. Purpose

This document describes the software architecture of the surgical autonomy
system — component decomposition, interfaces, safety arguments, and key
design decisions with rationale.

---

## 2. System Decomposition

The system is decomposed into 7 ROS 2 nodes across 5 architectural layers:

```
┌──────────────────────────────────────────────────────────────┐
│  LAYER 5 — HUMAN INTERFACE                                    │
│  surgeon_console.py                                           │
│  Class: SurgeonConsole (rclpy.Node)                          │
│  Safety class: B                                              │
└──────────────────────────┬───────────────────────────────────┘
                 /surgeon_stop (Bool)
                 /emergency_stop (Bool)
                 /console_feedback (String) ←
                 /watchdog_status (String)  ←
                 /tissue_force_proxy (Float32) ←
┌──────────────────────────▼───────────────────────────────────┐
│  LAYER 4 — TASK ORCHESTRATION                                 │
│  surgical_bt_node.py                                          │
│  Class: SurgicalBTNode (rclpy.Node)                          │
│  BT: Root→SafetyMonitor(Parallel)→SurgicalSequence           │
│      Approach→Retract→Hold + ForceWatchdog(Condition)        │
│  Safety class: C                                              │
└──────────────────────────┬───────────────────────────────────┘
              ROS 2 Action: lapgym_interfaces/action/Retract
              /approach_policy, /retract_policy, /hold_policy
┌────────────────┬─────────▼──────────┬────────────────────────┐
│  LAYER 3 — AUTONOMOUS CONTROL (3 nodes)                       │
│                                                               │
│  approach_policy_server.py     Class: ApproachPolicyServer    │
│  retract_policy_server.py      Class: RetractPolicyServer     │
│  hold_policy_server.py         Class: HoldPolicyServer        │
│                                                               │
│  Each node has:                                               │
│  ├── Main rclpy executor (action server)                     │
│  ├── Separate rclpy.Context (stop listener)                   │
│  ├── Background std::thread spinning stop executor            │
│  └── SOFA environment instance (TissueRetractionV2)           │
│  Safety class: C                                              │
└────────────────────────────────────────────────────────────────┘
                 /tissue_force_proxy (Float32) ←
                 /emergency_stop (Bool) ←
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2 — INDEPENDENT SAFETY                                │
│  safety_watchdog_node.py                                     │
│  Class: SafetyWatchdogNode (rclpy.Node)                     │
│  Independent PID, 50Hz timer, no BT dependency              │
│  Safety class: C (IEC 62304 independent layer)               │
└─────────────────────────────────────────────────────────────┘
                 /tissue_force_proxy (Float32) ←
                 /joint_states (JointState) ←
                 /emergency_stop (Bool) ←
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1 — SIMULATION INTERFACE                              │
│  sofa_bridge_node.py                                         │
│  Class: SofaBridgeNode (rclpy.Node)                         │
│  SOFA TissueRetractionV2, 50Hz timer                        │
│  Safety class: B                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Component Descriptions

### 3.1 sofa_bridge_node (Layer 1)

**Responsibility:** Single point of contact between SOFA simulation and ROS 2 graph.

**Key interfaces:**
- Publishes `/tissue_force_proxy` Float32 at 50Hz from Farneback optical flow
- Publishes `/joint_states` JointState at 50Hz with 7D observation
- Publishes `/camera/image_raw` Image at 50Hz
- Subscribes to `/joint_target` for teleoperation commands
- Subscribes to `/emergency_stop` — halts on True

**Implementation notes:**
- TissueRetractionV2 used (not V3) — 7D ground-truth observation, minimal overhead
- Stub mode activates when SOFA unavailable — publishes zeros
- `os._exit(0)` on shutdown — bypasses SOFA SIGABRT destructor issue
- Grasping auto-triggers at 3mm threshold — no explicit command needed

### 3.2 safety_watchdog_node (Layer 2)

**Responsibility:** Independent force monitor — the IEC 62304 independent safety layer.

**Key interfaces:**
- Subscribes to `/tissue_force_proxy` independently
- Publishes `/emergency_stop=True` on STOP condition
- Publishes `/watchdog_status` String: NOMINAL | ALERT | STOP
- Publishes `/watchdog_heartbeat` Bool every 1 second

**Safety architecture argument:**
Per IEC 62304 §5.1 safety-critical functions shall run independent of
application logic. This node runs as a separate operating system process
(separate PID). It has no dependency on surgical_bt_node, any action server,
or surgeon_console. Crash of any other node does not affect watchdog operation.

**Three-tier logic:**
```
force < 0.35          → NOMINAL  → log every 10s
0.35 ≤ force < 1.0    → ALERT   → publish WARNING
force ≥ 1.0 × 3 readings → STOP → publish /emergency_stop=True
```

**3-reading requirement rationale:** At 50Hz, 3 readings = 60ms sustained force.
This prevents false positives from single-frame noise spikes while responding
within the 100ms industry target.

### 3.3 Action Servers (Layer 3)

Three nodes implementing the same safety pattern:

**approach_policy_server:**
- Controller: Proportional. `action = clip(direction × 2.0, -3.0, 3.0)`
- Terminates when instrument within 25mm of grasping target
- No PPO loading — deterministic geometric navigation

**retract_policy_server:**
- Controller: Phase 2D PPO checkpoint (deterministic=True at inference)
- Checkpoint: `logs/checkpoints/phase2_ppo_tissue_retraction_20260409_211946/ppo_tissue_final`
- Terminates on goal_reached (dist < threshold) or timeout 300 steps

**hold_policy_server:**
- Controller: Zero-action `[0.0, 0.0, 0.0]`
- Terminates on timeout 500 steps or cancel
- Always returns success=True (zero actions cannot cause harm)

**Shared safety pattern — separate rclpy.Context:**

```python
# Problem: env.step() blocks main executor ~65ms
# Python GIL prevents any callback during this time
# Solution: isolated DDS instance on background thread

self._stop_context = rclpy.Context()
self._stop_context.init()
self._stop_node = rclpy.create_node(
    '_surgeon_stop_{approach|retract|hold}',
    context=self._stop_context, enable_rosout=False)
self._stop_executor = rclpy.executors.SingleThreadedExecutor(
    context=self._stop_context)
self._stop_thread = threading.Thread(
    target=self._spin_stop_node, daemon=True)

def _spin_stop_node(self):
    while self._stop_context.ok():
        self._stop_executor.spin_once(timeout_sec=0.01)  # 10ms
```

**Background node names (must be unique across all three servers):**
- approach: `_surgeon_stop_approach`
- retract: `_surgeon_stop_retract`
- hold: `_surgeon_stop_hold`

**Dual freeze loop pattern:**
```python
# Loop 1: catches stop pressed between steps
while self._surgeon_stopped and not self._emergency:
    self._stop_event.wait(timeout=0.05)

obs, reward, terminated, truncated, info = self._env.step(action)

# Loop 2: catches stop pressed during the step
while self._surgeon_stopped and not self._emergency:
    self._stop_event.wait(timeout=0.05)
```

### 3.4 surgical_bt_node (Layer 4)

**Responsibility:** Orchestrates Approach → Retract → Hold sequence with force monitoring.

**BT structure:**
```
Root (Sequence)
└── SafetyMonitor (Parallel, SuccessOnSelected=[SurgicalSequence])
    ├── SurgicalSequence (Sequence)
    │   ├── Approach (ActionLeaf → /approach_policy)
    │   ├── Retract  (ActionLeaf → /retract_policy)
    │   └── Hold     (ActionLeaf → /hold_policy)
    └── ForceWatchdog (Condition → /tissue_force_proxy)
```

**ActionLeaf implementation:** Reusable leaf wrapping any ROS 2 action.
On first tick: send_goal(). On subsequent ticks: return RUNNING.
On server response: return SUCCESS or FAILURE. On terminate(): cancel_goal().

**ForceCondition fix (Phase 4D):**
`terminate()` only resets consecutive counter on `NodeStatus.INVALID`
(BT reset/interrupt) — not on SUCCESS. This prevents the counter from
resetting between ticks when force is near threshold.

**Why BT over FSM:**
BT Fallback provides continuous force monitoring natively.
FSM requires explicit emergency transitions from every state — O(n) complexity.
BTs are deterministic, auditable, and formally verifiable per Iovino et al.
2022 survey.

### 3.5 surgeon_console (Layer 5)

**Responsibility:** Human-in-the-loop terminal interface.

**UI refresh:** curses at 10Hz. Processes up to 5 ROS 2 callbacks per refresh
cycle for state freshness.

**Republish timer:** /surgeon_stop=True republished at 10Hz while surgeon_stopped=True.
Prevents resume if BT restarts goal and new server instance misses original message.

---

## 4. Interface Definitions

### 4.1 Custom Action Interface

File: `lapgym_interfaces/action/Retract.action`

```
# Goal
float32 max_steps    # 0 = use server default
bool    render       # open SOFA GUI during execution
---
# Result
bool    success
int32   steps_taken
float32 final_distance   # metres
string  termination      # goal_reached|preempted|timeout|collision|emergency_stop
---
# Feedback (published every step)
float32 distance_to_goal   # metres
float32 distance_mm        # millimetres
int32   step
bool    in_collision
float32 collision_cost
```

### 4.2 Console Feedback Format

Topic: `/console_feedback` String
Format: `PHASE|STEP|MAX_STEPS|DISTANCE|BT_STATE`
Example: `RETRACT|117|300|0.006|RUNNING`

### 4.3 Complete Topic Table

| Topic | Type | Publisher | Subscribers | Rate |
|-------|------|-----------|-------------|------|
| `/tissue_force_proxy` | Float32 | sofa_bridge_node | safety_watchdog, surgical_bt (ForceCondition), surgeon_console | 50Hz |
| `/emergency_stop` | Bool | safety_watchdog, surgeon_console | sofa_bridge_node, all action servers | on event |
| `/surgeon_stop` | Bool | surgeon_console | separate rclpy.Context in each action server | 10Hz (republish) |
| `/joint_states` | JointState | sofa_bridge_node | teleop_keyboard (HUD), surgical_bt | 50Hz |
| `/joint_target` | Vector3 | teleop_keyboard | sofa_bridge_node | 20Hz |
| `/watchdog_status` | String | safety_watchdog_node | surgeon_console | 50Hz |
| `/watchdog_heartbeat` | Bool | safety_watchdog_node | surgeon_console | 1Hz |
| `/console_feedback` | String | surgical_bt_node | surgeon_console | 10Hz |
| `/guidance` | Float32MultiArray | sofa_bridge_node | teleop_keyboard (HUD) | 50Hz |
| `/camera/image_raw` | Image | sofa_bridge_node | (Phase 5 perception) | 50Hz |

---

## 5. Safety Architecture Argument

### Defence in Depth

Two independent safety layers per IEC 62304 §5.1 independence requirement:

```
Layer 1 (Application) — ForceCondition in surgical_bt_node
├── Check rate: 10Hz (BT tick)
├── Threshold: 0.35 px/frame × 3 consecutive = 300ms
├── Action: cancel_goal() on active action server
└── Dependency: shares process with BT

Layer 2 (Independent) — safety_watchdog_node
├── Check rate: 50Hz (own timer)
├── Threshold: 1.0 px/frame × 3 consecutive = 60ms
├── Action: publish /emergency_stop=True → all nodes halt
└── Dependency: NONE — independent process, independent subscription
```

**Safety argument:** Both layers must fail simultaneously for a dangerous
force condition to go undetected. The probability of simultaneous failure
of two independent processes is the product of their individual failure
probabilities — substantially lower than either alone.

### Verified Response Times

| Event | Expected | Measured | Status |
|-------|---------|---------|--------|
| Force threshold to /emergency_stop published | ≤ 100ms | 60ms | ✓ PASS |
| /emergency_stop to bridge_node halt | ≤ 100ms | ~80ms | ✓ PASS |
| Surgeon stop to freeze (one step) | ≤ 65ms | ≤ 65ms | ✓ PASS |

---

## 6. Known Architectural Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| env.step() ~65ms blocking | Surgeon stop max one-step latency | Dual freeze loop, documented |
| SOFA ~15Hz physics | Below 50Hz industry standard | Documented sim-to-real gap |
| Python GIL | No true parallel threads | Separate rclpy.Context |
| Force proxy indirect measurement | Force underestimation possible | Conservative alert threshold 0.35 |
| PPO policy simulation-trained | Sim-to-real gap at deployment | Documented in phase3d_sim_to_real_gap_analysis.md |