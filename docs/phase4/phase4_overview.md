# Phase 4 — ROS 2 Supervised Autonomy: Complete Overview

**Repository:** github.com/SUBHASH-Hub/surgical-rl  
**Author:** Subhash Arockiadoss  
**Platform:** Ubuntu 22.04, ROS 2 Humble, SOFA Framework  
**Status:** Phase 4A–4D complete. Phase 4E (surgeon console) in progress.

---

## What Phase 4 Is

Phase 4 integrates all prior research phases into a deployable supervised
autonomy surgical robot system using ROS 2 as the middleware layer.

Phases 1–3 produced:
- A trained PPO tissue retraction policy (Phase 2)
- A surgical perception pipeline with tissue segmentation (Phase 3)

Phase 4 deploys the Phase 2 PPO policy as a real-time autonomous controller
orchestrated by a Behaviour Tree, monitored by an independent safety watchdog,
and operated via a single launch command — replicating the architecture of
production surgical robot systems such as Versius (CMR Surgical) and Hugo
(Medtronic).

---

## One Command — Complete System

```bash
source ~/surgical_robot_lapgym_ws/activate.sh
cd ~/surgical_robot_lapgym_ws/surgical-rl
ros2 launch lapgym_ros2_bridge surgical_system.launch.py
```

This starts 6 nodes simultaneously. The system performs a complete autonomous
tissue retraction procedure and halts cleanly.

---

## System Architecture
┌─────────────────────────────────────────────────────────┐
│                    One Launch Command                    │
└─────────────────────────────────────────────────────────┘
│
┌──────────────────┼──────────────────┐
│                  │                  │
┌───────▼──────┐  ┌────────▼───────┐  ┌──────▼────────────┐
│ SOFA Bridge  │  │ Behaviour Tree  │  │ Safety Watchdog   │
│ bridge_node  │  │ surgical_bt_node│  │ safety_watchdog   │
│ 50 Hz        │  │ 10 Hz ticks     │  │ 50 Hz independent │
│              │  │                 │  │ IEC 62304 Class B │
└───────┬──────┘  └────────┬────────┘  └──────┬────────────┘
│                  │                  │
│         ┌────────▼────────┐         │
│         │  3 Action Servers│         │
│         │                 │         │
│    ┌────▼──┐ ┌──────┐ ┌───▼──┐      │
│    │Approach│ │Retract│ │ Hold │      │
│    │ Props  │ │ PPO  │ │ Zero │      │
│    └────────┘ └──────┘ └──────┘      │
│                                      │
└────── /tissue_force_proxy ───────────┘
/emergency_stop

---

## Complete Node Table

| Node | Executable | Role | Phase |
|---|---|---|---|
| `sofa_bridge_node` | `bridge_node` | SOFA simulation + `/tissue_force_proxy` | 4A |
| `teleop_keyboard` | `teleop_keyboard` | Human teleoperation + HUD guidance | 4A |
| `approach_policy_server` | `approach_policy_server` | Proportional controller to grasping zone | 4B |
| `retract_policy_server` | `retract_policy_server` | Phase 2D PPO tissue retraction policy | 4B |
| `hold_policy_server` | `hold_policy_server` | Zero-action position hold | 4B |
| `surgical_bt_node` | `surgical_bt_node` | Behaviour Tree orchestrator | 4C |
| `safety_watchdog_node` | `safety_watchdog_node` | Independent force monitor IEC 62304 | 4D |
| `policy_test_client` | `policy_test_client` | One-shot test utility | 4B |

---

## Behaviour Tree Structure
{-} Root
/_/ SafetyMonitor  [Parallel]
{-} SurgicalSequence  [Sequence]
--> Approach   [/approach_policy action server]
--> Retract    [/retract_policy action server]
--> Hold       [/hold_policy   action server]
--> ForceWatchdog [/tissue_force_proxy subscriber]

The BT orchestrates three action servers sequentially. The Parallel node runs
ForceWatchdog simultaneously — if force exceeds threshold the active leaf is
cancelled via `cancel_goal()`.

---

## Defence in Depth — Two Independent Safety Layers
Layer 1 (Application)  — ForceCondition
Process:     Inside BT (surgical_bt_node)
Check rate:  10 Hz
Threshold:   0.35 (ALERT), 3 consecutive readings
Action:      cancel_goal() on active action server
Standard:    Application safety
Layer 2 (Independent)  — SafetyWatchdogNode
Process:     Independent PID (separate from BT)
Check rate:  50 Hz
Threshold:   0.35 (ALERT), 1.0 (STOP, 3 consecutive readings = 60ms)
Action:      publish /emergency_stop=True → all nodes halt
Standard:    IEC 62304 Class B architectural independence

Both layers must fail simultaneously for a dangerous condition to go
undetected. This is defence in depth — the same principle used in
aircraft fly-by-wire systems and certified medical devices.

---

## Verified Surgical Procedure Results

All results from `ros2 launch lapgym_ros2_bridge surgical_system.launch.py`:

| Phase | Server | Steps | Start dist | End dist | Result |
|---|---|---|---|---|---|
| Approach | ApproachPolicyServer | 123–179 | ~120mm | 24mm | goal_reached |
| Retract | RetractPolicyServer (PPO) | 98–155 | ~100mm | 2.4–2.9mm | goal_reached |
| Hold | HoldPolicyServer | 200 | 0mm | 0mm | timeout/SUCCESS |

**BT ROOT: SUCCESS — total procedure ~75 seconds**

Retract PPO consistently achieved sub-3mm final distance.
The approach pre-positioning reduced retract steps from 175 (standalone)
to 98–137 steps — proving the two-server architecture works as designed.

---

## IEC 62304 Emergency Stop — Verified Results

Force injection test during Hold phase:
t=0ms    Force injected: 1.500 (above STOP threshold 1.0)
t=60ms   Watchdog triggered: 3 consecutive readings at 50Hz
t=~80ms  /emergency_stop=True published
t=~100ms bridge_node halted: EMERGENCY STOP received
Watchdog heartbeat continued throughout
Watchdog state=STOP maintained until force resolved

Response time: 60ms from force onset to emergency stop.
Industry target: 100ms. We achieved 60ms. ✓

---

## Phase Breakdown

| Phase | Description | Tag | Key Files |
|---|---|---|---|
| 4A | ROS 2 bridge + HUD teleop guidance | v4.1 | `bridge_node.py`, `teleop_keyboard.py` |
| 4B | PPO policy action servers | v4.2 | `retract_policy_server.py`, `approach_policy_server.py`, `hold_policy_server.py` |
| 4C | Behaviour Tree + launch file | v4.3 | `surgical_bt_node.py`, `action_leaf.py`, `force_condition.py`, `surgical_system.launch.py` |
| 4D | Independent safety watchdog | v4.4 | `safety_watchdog_node.py` |
| 4E | Surgeon console terminal | planned | `surgeon_console.py` |

---

## ROS 2 Topics Map

| Topic | Type | Publisher | Subscribers | Rate |
|---|---|---|---|---|
| `/tissue_force_proxy` | Float32 | bridge_node | safety_watchdog, surgical_bt (ForceCondition) | 50 Hz |
| `/emergency_stop` | Bool | safety_watchdog | bridge_node, all action servers | on event |
| `/joint_target` | Float32MultiArray | teleop_keyboard | bridge_node | 10 Hz |
| `/joint_states` | JointState | bridge_node | teleop_keyboard (HUD) | 50 Hz |
| `/guidance` | Float32MultiArray | bridge_node | teleop_keyboard (HUD) | 50 Hz |
| `/watchdog_status` | String | safety_watchdog | surgeon console (Phase 4E) | 50 Hz |
| `/watchdog_heartbeat` | Bool | safety_watchdog | surgeon console (Phase 4E) | 1 Hz |

---

## Why Phase 2 PPO and Not Phase 3 Perception

Phase 3 produces segmentation masks (pixel output).
Phase 2 PPO consumes a 7-number state vector (position input).
These formats are incompatible without a perception-to-control bridge.

Connecting them requires: stereo depth estimation, coordinate frame
transforms, and policy fine-tuning with noisy observations — a complete
research project. This is documented as future work.

Phase 4 demonstrates the **control infrastructure** — the ROS 2 action
server interface that any future controller (PPO, classical, or
perception-driven) can plug into without changing the orchestration layer.

---

## What Comes Next — Phase 4E Surgeon Console

Phase 4E builds a terminal-based surgeon console that replicates the
physical deadman switch interface of real surgical robot consoles:

- Live dashboard: force, distance, phase, step count, watchdog status
- `S` key: stop — sends cancel_goal() to active server, Hold activates
- `R` key: resume — sends new goal to continue from current position
- `E` key: emergency stop — publishes /emergency_stop directly
- `Q` key: quit — clean system shutdown

This completes the supervised autonomy system with full human oversight.

---

## Detailed Documentation

- [Phase 4A — ROS 2 Bridge + HUD](phase4a_ros2_bridge.md)
- [Phase 4B — PPO Policy Action Servers](phase4b_policy_action_server.md)
- [Phase 4C — Behaviour Tree Supervised Autonomy](phase4c_behaviour_tree.md)
- [Phase 4D — Independent Safety Watchdog](phase4d_safety_watchdog.md)
- [Launch and Operations Guide](phase4_launch_guide.md)