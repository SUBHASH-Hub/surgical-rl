# Phase 4 Launch and Operations Guide

**Updated: Phase 4G — Hybrid C++/Python Architecture**

---

## Architecture Change — Why the Launch File Was Updated

### Phase 4A–4E: Python-only action servers

The original launch file started Python action servers. The Python GIL
caused surgeon stop latency of 15-20 steps (~1 second) because env.step()
held the GIL and blocked all callbacks.

### Phase 4F–4G: Hybrid C++/Python

The action servers were ported to C++ (rclcpp). C++ has no GIL — stop
callbacks fire in parallel with physics execution using std::atomic<bool>
and dedicated callback groups. Stop latency improved to 1-2 steps (~130ms).

Two new Python services were added as bridges:
- `sofa_step_service` — wraps env.step() for C++ servers via /sofa_step
- `ppo_predict_service` — wraps policy.predict() for C++ retract via /ppo_predict

The BT action names were updated from `approach_policy` to
`approach_policy_cpp` to reflect the C++ servers.

---

## Starting the System

```bash
source ~/surgical_robot_lapgym_ws/activate.sh
cd ~/surgical_robot_lapgym_ws/surgical-rl
ros2 launch lapgym_ros2_bridge surgical_system.launch.py
```

Optional argument:
```bash
ros2 launch lapgym_ros2_bridge surgical_system.launch.py render_mode:=human
```

---

## Node Startup Order (Phase 4G)

```
9 nodes start simultaneously:

Python layer (ML/physics):
  [bridge_node-1]            SOFA bridge + /tissue_force_proxy
  [sofa_step_service-2]      SOFA env.step() service (NEW Phase 4F)
  [ppo_predict_service-3]    PPO policy.predict() service (NEW Phase 4G)

C++ control layer (no GIL):
  [approach_policy_server_cpp-4]    C++ proportional controller
  [retract_policy_server_cpp-5]     C++ PPO retract
  [hold_policy_server_cpp-6]        C++ zero-action hold

Safety + orchestration:
  [safety_watchdog_node-7]   IEC 62304 independent watchdog
  [xterm-8]                  Surgeon console in xterm window

BT (delayed 20s):
  [surgical_bt_node-9]       Behaviour tree (was 15s, now 20s for C++ init)
```

---

## Expected Startup Sequence

```
[bridge_node-1]                 process started
[sofa_step_service-2]           process started
[ppo_predict_service-3]         process started
[approach_policy_server_cpp-4]  Waiting for /sofa_step service...
[retract_policy_server_cpp-5]   Waiting for /sofa_step...
[hold_policy_server_cpp-6]      Waiting for /sofa_step service...
[safety_watchdog_node-7]        IEC 62304 independent safety layer ACTIVE
[xterm-8]                       surgeon console opens

... SOFA plugin loading (10-15s) ...

[sofa_step_service-2]           TissueRetractionV2 ready
[sofa_step_service-2]           SofaStepService ready on /sofa_step
[approach_policy_server_cpp-4]  /sofa_step service ready
[approach_policy_server_cpp-4]  ApproachPolicyServerCpp ready on /approach_policy_cpp
[retract_policy_server_cpp-5]   /sofa_step ready
[retract_policy_server_cpp-5]   /ppo_predict ready
[retract_policy_server_cpp-5]   RetractPolicyServerCpp ready on /retract_policy_cpp
[hold_policy_server_cpp-6]      /sofa_step service ready
[hold_policy_server_cpp-6]      HoldPolicyServerCpp ready on /hold_policy_cpp
[ppo_predict_service-3]         PPO checkpoint loaded successfully
[ppo_predict_service-3]         PPOPredictService ready on /ppo_predict

--- 20 second delay (TimerAction) ---

[surgical_bt_node-9]    Building surgical behaviour tree
[surgical_bt_node-9]    [Approach] Action server approach_policy_cpp ready
[surgical_bt_node-9]    [Retract]  Action server retract_policy_cpp ready
[surgical_bt_node-9]    [Hold]     Action server hold_policy_cpp ready
[surgical_bt_node-9]    SurgicalBTNode started -- ticking at 10 Hz
[surgical_bt_node-9]    [Approach] Sending goal to approach_policy_cpp
```

---

## Expected Procedure Output

```
[approach_policy_server_cpp]  Approach (C++) step   5 | Dist: XXX.Xmm
...
[approach_policy_server_cpp]  Approach complete: goal_reached steps=NNN dist=XX.Xmm
[surgical_bt_node]            [Retract] Sending goal to retract_policy_cpp
[retract_policy_server_cpp]   Retract (C++) step   5 | Dist: 0.0mm
...
[retract_policy_server_cpp]   Retract (C++) step  90 | Dist: XX.Xmm
[retract_policy_server_cpp]   Goal reached at step NNN dist=XX.Xmm
[surgical_bt_node]            [Hold] Sending goal to hold_policy_cpp
[hold_policy_server_cpp]      Hold policy active — holding position
...
[hold_policy_server_cpp]      Hold timeout reached at step NNN
[surgical_bt_node]            Surgical procedure complete -- SUCCESS
```

**Note on 0.0mm in retract phase:** The first ~75-90 steps show
dist=0.0mm. This is correct — `distance_to_end_position` only becomes
active after grasping triggers in the SOFA simulation. Distance becomes
non-zero once the instrument has grasped the tissue and retraction begins.

---

## Surgeon Console Controls

| Key | Action | Effect |
|-----|--------|--------|
| S | Surgeon Stop | Freezes active C++ server mid-step |
| R | Resume | Unfreeze — continues from exact stop point |
| E | Emergency Stop | /emergency_stop=True — all nodes halt, BT FAILED |
| Q | Quit | Closes console window (system continues) |

**Stop latency (C++ servers):** 1-2 env.step() cycles = ~65-130ms
This is a 10× improvement over Python servers (15-20 steps = ~1 second).
Remaining latency is SOFA physics computation time — not a GIL issue.

---

## SOFA Warnings — Expected and Safe to Ignore

```
[ERROR] RequiredPlugin(SofaBoundaryCondition) Plugin not found ← renamed v24
[ERROR] RequiredPlugin(SofaEngine)            Plugin not found ← renamed v24
[ERROR] RequiredPlugin(SofaDeformable)        Plugin not found ← renamed v24
[ERROR] RequiredPlugin(SofaGeneralRigid)      Plugin not found ← renamed v24
[WARN]  UncoupledConstraintCorrection Default compliance not set ← cosmetic
[WARN]  BoxROI No rest position yet defined                      ← cosmetic
CUDA unknown error (ppo_predict_service startup)                 ← harmless
```

None of these affect simulation physics or policy execution.

---

## Monitoring Topics

```bash
# Watchdog status
ros2 topic echo /watchdog_status

# Emergency stop channel
ros2 topic echo /emergency_stop

# Heartbeat (proves watchdog alive)
ros2 topic echo /watchdog_heartbeat

# Live force readings
ros2 topic echo /tissue_force_proxy
```

---

## Force Injection Test — IEC 62304 Verification

```bash
# In a second terminal while system is running (hold phase):
source ~/surgical_robot_lapgym_ws/activate.sh
ros2 topic pub --rate 50 /tissue_force_proxy std_msgs/Float32 "data: 1.5"
```

Expected sequence:
```
t=0ms:   Force 1.5 injected
t=60ms:  [safety_watchdog] WATCHDOG EMERGENCY STOP — 3 readings at 50Hz
t=~80ms: /emergency_stop=True published
t=~100ms:[bridge_node] EMERGENCY STOP received
         [C++ servers]  EMERGENCY STOP received (all three simultaneously)
         [surgical_bt]  Surgical procedure FAILED
```

Response time: 60ms. Industry target: 100ms. ✓

---

## Known Behaviours

| Behaviour | Cause | Action |
|-----------|-------|--------|
| SURGEON STOP spam in logs | Republish timer 10Hz while stopped | Normal — prevents race conditions |
| `rcl_shutdown already called` | Double-shutdown race on Ctrl+C | Harmless |
| Bridge keeps running after BT SUCCESS | bridge_node has no stop condition | Ctrl+C to stop |
| SOFA backtrace on shutdown | SOFA signal handler on SIGINT | Normal |
| 0.0mm dist in retract first ~75 steps | Grasping not yet triggered | Normal — see note above |