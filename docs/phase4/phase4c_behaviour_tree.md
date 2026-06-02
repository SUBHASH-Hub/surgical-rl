# Phase 4C/4G — Behaviour Tree Supervised Autonomy

**Updated: Phase 4G — action names updated to C++ servers**

---

## Overview

Phase 4C implements supervised autonomy using a py_trees Behaviour Tree
that orchestrates all three action servers in sequence with simultaneous
safety monitoring via tissue force.

**Phase 4G update:** The BT action names were changed from Python server
names to C++ server names after the action servers were ported to C++
in Phase 4F and 4G. The BT orchestration logic is unchanged — only the
action server targets were updated.

---

## Why the BT Action Names Were Changed

### The Engineering Reason

When the action servers were ported to C++ in Phase 4F/4G, the new C++
servers registered under different action names (`_cpp` suffix):

```
Python servers (Phase 4B):     C++ servers (Phase 4F/4G):
/approach_policy           →   /approach_policy_cpp
/retract_policy            →   /retract_policy_cpp
/hold_policy               →   /hold_policy_cpp
```

The `_cpp` suffix was kept deliberately — it distinguishes the C++
implementation from the Python baseline and prevents confusion if both
are ever run simultaneously for testing.

### Why Change BT Config Not Executable Name

Two options existed to make the BT use C++ servers:

**Option A:** Rename C++ executables to match old Python names.
Risk: impossible to tell Python and C++ versions apart in logs.

**Option B:** Update BT action name strings to point to `_cpp` variants.
Benefit: one-line change per server, clear intent, both implementations
can coexist for testing.

Option B was chosen. This is the correct production pattern — the BT is
a configuration layer that can point to any compatible action server
without changing the orchestration logic.

---

## What Was Built

### Files in `lapgym_ros2_bridge`

**`action_leaf.py` — `ActionLeaf`**
Reusable py_trees leaf that wraps any ROS 2 action server. Sends goal
on first tick, returns RUNNING while executing, returns SUCCESS or FAILURE
on completion. Calls `cancel_goal()` immediately via `terminate()`.
Language-agnostic — works with both Python and C++ action servers.

**`force_condition.py` — `ForceCondition`**
py_trees condition leaf subscribing to `/tissue_force_proxy`. Returns
FAILURE if force exceeds 0.35 px/frame for 3 consecutive readings.

**`surgical_bt_node.py` — `SurgicalBTNode`**
Builds and ticks the behaviour tree at 10 Hz. Updated in Phase 4G:
action names changed to `_cpp` variants.

---

## Behaviour Tree Structure (Phase 4G — C++ servers)

```
{-} Root               [Sequence]
/_/ SafetyMonitor  [Parallel - SuccessOnSelected(SurgicalSequence)]
    {-} SurgicalSequence  [Sequence]
        --> Approach   [ActionLeaf -> /approach_policy_cpp]  ← C++ rclcpp
        --> Retract    [ActionLeaf -> /retract_policy_cpp]   ← C++ rclcpp
        --> Hold       [ActionLeaf -> /hold_policy_cpp]      ← C++ rclcpp
    --> ForceWatchdog  [ForceCondition -> /tissue_force_proxy]
```

The BT is language-agnostic. The ActionLeaf sends goals via the standard
ROS 2 action protocol — it does not know or care whether the server is
Python or C++. Only the action name string changed.

---

## Why BT Stays Python Despite C++ Servers

The BT orchestration layer stays Python for three reasons:

**1. py_trees_ros has no C++ equivalent** — the library is Python-only.
Reimplementing BT orchestration in C++ would require writing a custom BT
framework — significant effort with no safety or performance benefit since
BT ticks are 10Hz with no blocking calls (GIL not an issue).

**2. BT logic has no blocking calls** — the GIL problem (the reason action
servers moved to C++) does not apply to BT ticks. Each tick is < 1ms.

**3. Correct separation of concerns** — orchestration (what to do) in
Python, execution (how to do it) in C++. This matches how CMR Surgical
and Medtronic separate their planning and control layers.

---

## Safety Architecture

### Two Independent Layers

| Property | ForceCondition (BT — Phase 4C) | SafetyWatchdogNode (Phase 4D) |
|----------|-------------------------------|-------------------------------|
| Language | Python | Python |
| Process | Inside BT (same PID) | Independent PID |
| Check rate | 10 Hz (BT tick) | 50 Hz (own timer) |
| BT dependency | Dies if BT crashes | Survives BT crash |
| Scope | cancel_goal() on active C++ server | /emergency_stop → all halt |
| Standard | Application safety | IEC 62304 independent |

The C++ servers subscribe to `/emergency_stop` via their dedicated
stop_callback_group — the emergency stop fires in parallel with any
ongoing sofaStep() service call, no GIL conflict.

---

## Verified Results

### Phase 4C — Python server baseline
```
Approach: goal_reached steps=258 dist=24.8mm
Retract:  goal_reached steps=98  dist=2.5mm
Hold:     timeout      steps=200
BT ROOT:  SUCCESS  (~98 seconds total)
```

### Phase 4G — C++ servers (hybrid architecture)
```
Approach: goal_reached steps=72  dist=24.4mm  (C++ approach_policy_cpp)
Retract:  goal_reached steps=118 dist=29.5mm  (C++ retract_policy_cpp)
Hold:     emergency_stop steps=83             (E key pressed — correct)
BT ROOT:  FAILED (correct — E key = dangerous event)
```

**Stop latency improvement:**
```
Phase 4C (Python): ~15-20 steps × 65ms = ~1-1.3 seconds
Phase 4G (C++):    ~1-2  steps × 65ms = ~65-130ms
Improvement: 10× faster
```

---

## Running the Full System (Phase 4G)

```bash
source ~/surgical_robot_lapgym_ws/activate.sh
cd ~/surgical_robot_lapgym_ws/surgical-rl
ros2 launch lapgym_ros2_bridge surgical_system.launch.py
```

The BT starts after 20s delay (increased from 15s to allow C++ server
and Python service initialisation before first tick).