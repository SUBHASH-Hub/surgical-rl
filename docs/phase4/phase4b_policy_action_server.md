# Phase 4B — PPO Policy Action Server

## Overview

Phase 4B wraps the Phase 2D PPO policy as a ROS 2 action server, implementing
Track 2 of the Phase 4 architecture: Industry R&D (Supervised Autonomy).

The key advance over Phase 2 training is that the policy is now a composable,
interruptible, observable ROS 2 service. Any node on the network can start it,
monitor it, and stop it at any time through the standard ROS 2 action protocol.

## What Was Built

### New package: `lapgym_interfaces`

Location: `ros2_ws/src/lapgym_interfaces/`
Git mirror: `surgical-rl/ros2_packages/lapgym_interfaces/`

A separate `ament_cmake` package that defines the custom action interface.
This must be separate from `lapgym_ros2_bridge` because action interface
generation requires CMake, not Python setuptools.

**`action/Retract.action`** defines three message types:
Goal -- client sends this to start the policy
float32 max_steps    # 0 = use default 300
bool    render       # open SOFA GUI during execution
Result -- server sends this when done
bool    success
int32   steps_taken
float32 final_distance   # metres
string  termination      # goal_reached | preempted | timeout | collision
Feedback -- server sends this every 5 steps
float32 distance_to_goal # metres
float32 distance_mm      # millimetres
int32   step
bool    in_collision
float32 collision_cost

### New nodes in `lapgym_ros2_bridge`

**`retract_policy_server.py` — `RetractPolicyServer`**

Loads the Phase 2D PPO checkpoint and implements the ROS 2 action server
execution loop. Critical design: `is_cancel_requested` is checked at the
very first line of every step — before policy inference, before env.step.
This guarantees preemption latency of at most one simulation step (~20ms).

**`policy_test_client.py` — `PolicyTestClient`**

One-shot test client that sends a single Retract goal, prints live feedback,
and reports the final result. Used to verify the action server works correctly
before connecting to the Behaviour Tree in Phase 4C.

## ROS 2 Communication Patterns

### Three patterns in ROS 2

**Topic (publish/subscribe)** — continuous data stream, fire and forget.
Publisher does not know who is listening. Used for: `/joint_states`,
`/tissue_force_proxy`, `/camera/image_raw`, `/guidance`.

**Service (request/response)** — synchronous, short tasks. Client blocks
until server responds. Good for instant queries, bad for long operations.

**Action (goal/feedback/result)** — asynchronous, long tasks, cancellable.
Client sends goal and continues. Server sends progress updates. Client can
cancel at any time. Used for: surgical policy execution.

### Why actions for surgical policies

A tissue retraction takes 10-20 seconds (175 steps at ~9 Hz). A service
would block the client for 20 seconds. A topic has no concept of a task
with a start, progress, and end. Only an action provides all three:
start (goal), progress (feedback), end (result), and cancellation.

## Architecture
PolicyTestClient                    RetractPolicyServer
│                                      │
│─── send_goal(max_steps=300) ────────>│
│                                      │ loop:
│<── feedback(step=5,  dist=126mm) ───│   1. check is_cancel_requested
│<── feedback(step=10, dist=119mm) ───│   2. policy.predict(obs)
│<── feedback(step=15, dist=112mm) ───│   3. env.step(action)
│         ... 35 more updates ...      │   4. publish feedback every 5
│<── feedback(step=175, dist=3mm) ────│
│<── result(success=True,             │
│           termination=goal_reached, │
│           steps=175)                │

## Preemption Protocol

The most safety-critical feature. Every single step:

```python
if goal_handle.is_cancel_requested:
    goal_handle.canceled()
    return Retract.Result(
        success=False,
        steps_taken=step,
        termination='preempted'
    )
```

This guarantees that a cancel request from any source — surgeon console,
Behaviour Tree safety monitor, Phase 4D watchdog — stops the policy within
one simulation step. At 9 Hz step rate that is approximately 110ms maximum
preemption latency.

In the real surgical workflow:
Surgeon presses STOP → /emergency_stop published →
Behaviour Tree receives → cancel_goal() sent to server →
is_cancel_requested True on next step →
Policy stops, returns PREEMPTED →
HoldPolicyServer activated → instrument holds position

## Difference between Phase 2 and Phase 4B

| Aspect | Phase 2 training | Phase 4B deployment |
|---|---|---|
| Agent learning | Yes — weights update | No — weights frozen |
| Control | Training script | ROS 2 action client |
| Interruption | Cannot stop mid-episode | cancel_goal() any time |
| Visibility | No external monitoring | Feedback every 5 steps |
| Composability | Standalone script | Any ROS 2 node can call it |
| Purpose | Research | Deployment |

The PPO weights are identical. Only the wrapper changed.

## Verification Results

Verified on Ubuntu 22.04, ROS 2 Humble, GTX 1650:
Episode result:
Success:      True
Termination:  goal_reached
Steps taken:  175
Final dist:   3.0mm

- PPO checkpoint loaded correctly ✓
- Phase 2D 7D observation space confirmed: Box(-1, 1, (7,)) ✓
- Action space confirmed: Box(-3, 3, (3,)) ✓
- Distance decreased 126mm → 3mm over 175 steps ✓
- Goal reached within 300 step limit ✓
- Feedback streamed correctly every 5 steps ✓
- Action visible via `ros2 action list` as `/retract_policy` ✓
- Both nodes visible via `ros2 node list` ✓

## Dependencies installed

These packages were required and installed into `sofa_venv`:

```bash
pip install empy==3.3.4   # required by rosidl for action generation
pip install catkin_pkg    # required by ament_cmake build system
pip install lark          # required by rosidl_parser
```

## Running Phase 4B

**Terminal 1 — action server:**
```bash
source ~/surgical_robot_lapgym_ws/activate.sh
cd ~/surgical_robot_lapgym_ws/surgical-rl
ros2 run lapgym_ros2_bridge retract_policy_server
```

**Terminal 2 — test client (one episode):**
```bash
source ~/surgical_robot_lapgym_ws/activate.sh
ros2 run lapgym_ros2_bridge policy_test_client
```

**Verify action:**
```bash
ros2 action list          # shows /retract_policy
ros2 node list            # shows /retract_policy_server /policy_test_client
ros2 action info /retract_policy
```

## Next Steps — Phase 4B remaining

- `ApproachPolicyServer` — navigates instrument to grasping zone
- `HoldPolicyServer` — holds retracted position

These complete the three-server Phase 4B architecture before Phase 4C
Behaviour Tree orchestration.