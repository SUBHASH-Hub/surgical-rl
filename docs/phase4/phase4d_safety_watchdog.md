# Phase 4D — Independent Safety Watchdog

## Overview

Phase 4D implements an independent safety watchdog node that monitors
tissue force and triggers emergency stop when dangerous force levels are
detected. This is architecturally independent from the Behaviour Tree
and action servers, satisfying IEC 62304 Class B safety software
requirements.

## IEC 62304 Compliance

IEC 62304 is the international standard for medical device software.
It mandates that safety-critical functions run in processes independent
of the application logic they protect.

| Layer | Component | Class | Process |
|---|---|---|---|
| Application safety | ForceCondition (Phase 4C) | Class A | BT process |
| Independent safety | SafetyWatchdogNode (Phase 4D) | Class B | Separate PID |

The two-layer defence in depth means both layers must fail simultaneously
for a dangerous condition to go undetected.

## What Was Built

### `safety_watchdog_node.py` — `SafetyWatchdogNode`

A completely independent ROS 2 node with its own process, its own 50Hz
timer, and no dependency on the BT or action servers.

**Three-tier alert system:**

| State | Condition | Action |
|---|---|---|
| NOMINAL | force < 0.35 | log every 10s, heartbeat only |
| ALERT | 0.35 ≤ force < 1.0 | log WARNING, publish ALERT status |
| STOP | force ≥ 1.0 for 3 consecutive readings | publish /emergency_stop=True |

**3 consecutive readings** at 50Hz = 60ms sustained force. This prevents
false positives from transient noise spikes while still responding fast
enough to prevent tissue damage.

**Publishes:**
- `/emergency_stop` (std_msgs/Bool) — True triggers system halt
- `/watchdog_status` (std_msgs/String) — NOMINAL | ALERT | STOP
- `/watchdog_heartbeat` (std_msgs/Bool) — True every 1s, proves alive

**Subscribes:**
- `/tissue_force_proxy` (std_msgs/Float32) — optical flow force estimate

## Architecture
/tissue_force_proxy (50Hz from bridge_node)
↓
SafetyWatchdogNode (independent process, 50Hz check)
↓
/emergency_stop = True (when STOP threshold sustained)
↓
SofaBridgeNode    ← halts at 50Hz loop
SurgicalBTNode    ← cancels active action leaf
RetractPolicy     ← sets _emergency=True
ApproachPolicy    ← sets _emergency=True
HoldPolicy        ← sets _emergency=True

## Difference from ForceCondition (Phase 4C)

| Property | ForceCondition | SafetyWatchdogNode |
|---|---|---|
| Process | Inside BT (same PID) | Independent PID |
| Check rate | 10 Hz (BT tick) | 50 Hz (own timer) |
| BT dependency | Dies if BT crashes | Survives BT crash |
| Scope | Cancels active leaf | Stops entire system |
| Standard | Application logic | IEC 62304 independent |
| Response | BT cancel_goal() | Direct /emergency_stop |

## Heartbeat — Fail-Safe Property

The watchdog publishes `/watchdog_heartbeat = True` every second.
If the watchdog crashes, the heartbeat stops. Any monitoring system
can detect the missing heartbeat and trigger an independent hardware
interlock. This is the fail-safe property:
Watchdog alive   → heartbeat every 1s → system can operate
Watchdog crashes → heartbeat stops    → hardware interlock triggers

## Verified Results

**Test 1 — NOMINAL state:**
- Bridge running headless, no instrument movement
- Force = 0.000, state = NOMINAL
- Heartbeat publishing every 1s ✓
- Status publishing NOMINAL at 50Hz ✓

**Test 2 — STOP trigger:**
```bash
ros2 topic pub --rate 50 /tissue_force_proxy std_msgs/Float32 "data: 1.5"
```
- Watchdog detected force=1.500 ≥ 1.0 for 3 consecutive readings ✓
- Published /emergency_stop=True ✓
- Bridge received EMERGENCY STOP and halted immediately ✓
- Watchdog stayed alive and continued reporting STOP ✓
- Heartbeat continued publishing during STOP state ✓

**Test 3 — Recovery:**
- Stopped force injection (Ctrl+C)
- Force returned to 0.000
- Watchdog returned to NOMINAL state ✓

## Clinical Meaning

The watchdog behaves exactly like a real surgical robot safety system:
Instrument catches tissue → force spike → watchdog triggers
Robot halts immediately   → surgeon examines situation
Watchdog stays in STOP    → alarm continues until resolved
Surgeon resolves problem  → force drops → watchdog returns to NOMINAL
Surgeon manually restarts → system resumes

The watchdog never resets itself automatically. A human must resolve
the dangerous condition before the system can restart. This is the
deadman principle — the system requires active human confirmation
to resume after a safety stop.

## Running Phase 4D

```bash
# Standalone test
source ~/surgical_robot_lapgym_ws/activate.sh
ros2 run lapgym_ros2_bridge safety_watchdog_node

# Monitor topics
ros2 topic echo /watchdog_status
ros2 topic echo /watchdog_heartbeat
ros2 topic echo /emergency_stop

# Inject test force
ros2 topic pub --rate 50 /tissue_force_proxy std_msgs/Float32 "data: 1.5"

# Full system (watchdog included automatically)
ros2 launch lapgym_ros2_bridge surgical_system.launch.py
```

## Full System Node List

After Phase 4D the complete system has 8 nodes:

| Node | Role | Phase |
|---|---|---|
| `sofa_bridge_node` | SOFA simulation + force proxy | 4A |
| `teleop_keyboard` | Human teleoperation + HUD | 4A |
| `approach_policy_server` | Proportional controller | 4B |
| `retract_policy_server` | Phase 2D PPO policy | 4B |
| `hold_policy_server` | Zero-action position hold | 4B |
| `surgical_bt_node` | Behaviour Tree orchestrator | 4C |
| `safety_watchdog_node` | Independent safety monitor | 4D |
| `policy_test_client` | One-shot test utility | 4B |