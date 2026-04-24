# Phase 4C — Behaviour Tree Supervised Autonomy

## Overview

Phase 4C implements supervised autonomy using a py_trees Behaviour Tree
that orchestrates all three Phase 4B action servers in sequence with
simultaneous safety monitoring via tissue force.

This phase replicates the core architecture of real surgical robot systems
such as Versius (CMR Surgical) and Hugo (Medtronic) — autonomous execution
with human oversight and instant cancellation at any point.

## What Was Built

### New files in `lapgym_ros2_bridge`

**`action_leaf.py` — `ActionLeaf`**
Reusable py_trees leaf that wraps any ROS 2 action server. On first tick
sends a goal. On subsequent ticks returns RUNNING while server executes.
Returns SUCCESS or FAILURE when server responds. If BT cancels the leaf,
`cancel_goal()` is sent to the server immediately via `terminate()`.

**`force_condition.py` — `ForceCondition`**
py_trees condition leaf that subscribes to `/tissue_force_proxy` and
monitors tissue force every BT tick. Returns FAILURE if force exceeds
`FORCE_ALERT_THRESHOLD = 0.35` for 3 consecutive readings. Uses
consecutive readings to avoid false positives from transient spikes.

**`surgical_bt_node.py` — `SurgicalBTNode`**
Main ROS 2 node that builds and ticks the behaviour tree at 10 Hz.
Prints tree structure on startup. Cancels tick timer when root returns
SUCCESS or FAILURE.

## Behaviour Tree Structure
{-} Root               [Sequence]
/_/ SafetyMonitor  [Parallel - SuccessOnSelected(SurgicalSequence)]
{-} SurgicalSequence  [Sequence]
--> Approach      [ActionLeaf -> /approach_policy]
--> Retract       [ActionLeaf -> /retract_policy]
--> Hold          [ActionLeaf -> /hold_policy]
--> ForceWatchdog     [ForceCondition -> /tissue_force_proxy]

### Node type reference

| Symbol | Type | Behaviour |
|---|---|---|
| `{-}` | Sequence | Runs children left to right, stops on first FAILURE |
| `/_/` | Parallel | Runs all children simultaneously |
| `-->` | Leaf | Action or Condition |

### Parallel policy

`SuccessOnSelected(children=[SurgicalSequence])` — the parallel node
succeeds when `SurgicalSequence` succeeds. If `ForceWatchdog` returns
FAILURE, the parallel node fails immediately, which causes the active
`ActionLeaf.terminate()` to fire, which sends `cancel_goal()` to the
active action server.

## Safety Architecture

### ForceCondition vs Phase 4D SafetyWatchdog

| Property | ForceCondition (4C) | SafetyWatchdog (4D) |
|---|---|---|
| Location | Inside BT | Independent process |
| Check rate | 10 Hz (BT tick rate) | 50 Hz |
| BT dependency | Stops if BT crashes | Independent of BT |
| Scope | Cancels active leaf | Publishes /emergency_stop |
| Standard | Software safety | IEC 62304 safety layer |

ForceCondition provides application-level safety. Phase 4D adds the
hardware-level independent safety layer required by IEC 62304.

## Verified Results

### Full autonomous procedure — end to end
Phase 1: Approach
Server:      ApproachPolicyServer
Steps:       258
Start dist:  177.2mm
End dist:    24.8mm
Result:      goal_reached
Phase 2: Retract
Server:      RetractPolicyServer (Phase 2D PPO)
Steps:       98
Start dist:  28.0mm
End dist:    2.5mm
Result:      goal_reached
Phase 3: Hold
Server:      HoldPolicyServer
Steps:       200
Duration:    ~33 seconds
Result:      timeout (expected success)
Total procedure: ~98 seconds
Final BT status: SUCCESS

### Key observations

The approach phase pre-positioned the instrument at 24.8mm. This meant
the PPO retract agent only needed 98 steps instead of 175 steps in the
Phase 4B standalone test — proving the two-server approach+retract
architecture works as designed.

The BT correctly sequenced all three servers automatically. No manual
intervention was required between phases.

## Dependencies installed

```bash
pip install py_trees==2.2.3
pip install netifaces
sudo apt-get install -y ros-humble-py-trees-ros
sudo apt-get install -y ros-humble-py-trees-ros-interfaces
```

## Running Phase 4C (development mode — 5 terminals)

```bash
# Terminal 1 -- bridge (force proxy source)
source ~/surgical_robot_lapgym_ws/activate.sh
cd ~/surgical_robot_lapgym_ws/surgical-rl
ros2 run lapgym_ros2_bridge bridge_node --ros-args -p render_mode:=headless

# Terminal 2 -- approach server
ros2 run lapgym_ros2_bridge approach_policy_server

# Terminal 3 -- retract server
ros2 run lapgym_ros2_bridge retract_policy_server

# Terminal 4 -- hold server
ros2 run lapgym_ros2_bridge hold_policy_server

# Terminal 5 -- behaviour tree (start last)
ros2 run lapgym_ros2_bridge surgical_bt_node
```

Wait for all four servers to print `ready` before starting Terminal 5.

## Launch file — one command start

`launch/surgical_system.launch.py` starts all 5 nodes simultaneously:

```bash
ros2 launch lapgym_ros2_bridge surgical_system.launch.py
```

The BT node is delayed by 15 seconds via `TimerAction` to allow all four
SOFA environments to initialise before the first tick.

**Note on GUI:** The launch file includes `bridge_node` which opens the
SOFA GUI when `render_mode:=human`. However the GUI shows the bridge's
own environment which is idle — the actual surgical procedure runs inside
the action servers' headless environments. The bridge is included for
`/tissue_force_proxy` publishing which the ForceWatchdog monitors.

**Verified launch result (headless):**
```
Approach: goal_reached steps=150 dist=24.5mm
Retract:  goal_reached steps=150 dist=2.9mm
Hold:     timeout      steps=200
BT ROOT:  SUCCESS
```

## Roadmap

| Phase | Description | Status |
|---|---|---|
| 4A | Teleop + HUD guidance | COMPLETE |
| 4B | PPO policy action servers | COMPLETE |
| 4C | Behaviour Tree (launch file pending) | IN PROGRESS |
| 4D | Independent safety watchdog (IEC 62304) | NEXT |
| 4E | Surgeon console terminal (stop/resume) | PLANNED |