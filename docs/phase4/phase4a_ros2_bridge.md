# Phase 4A — ROS 2 Bridge and Keyboard Teleoperation

## Overview

Phase 4A establishes the ROS 2 middleware layer connecting the SOFA LapGym 
simulation to the outside world. It implements Track 1 of the Phase 4 
architecture: Industry Today (Teleoperation), mirroring how commercial 
systems such as CMR Versius and Medtronic Hugo connect surgeon consoles 
to robot actuators.

## What Was Built

### Package: `lapgym_ros2_bridge`

Location: `~/surgical_robot_lapgym_ws/ros2_ws/src/lapgym_ros2_bridge/`  
Git mirror: `surgical-rl/ros2_packages/lapgym_ros2_bridge/`  
Git tag: `v4.0-phase4a-ros2-bridge`

### Nodes

**`bridge_node.py` — `SofaBridgeNode`**

Drives the SOFA simulation at 50 Hz and relays simulation state to ROS 2 
topics. Acts as the single point of contact between the SOFA Python 
environment and the ROS 2 graph.

**`teleop_keyboard.py` — `TeleopKeyboardNode`**

Captures keyboard input via pynput and publishes instrument movement 
commands at 20 Hz to `/joint_target`.

## ROS 2 Topic Interface

| Topic | Type | Direction | Rate | Description |
|---|---|---|---|---|
| `/joint_states` | `sensor_msgs/JointState` | Published | 50 Hz | Instrument XYZ position + phase |
| `/tissue_force_proxy` | `std_msgs/Float32` | Published | 50 Hz | Optical flow force estimate (px/frame) |
| `/camera/image_raw` | `sensor_msgs/Image` | Published | 50 Hz | 480×480 RGB frame from SOFA |
| `/joint_target` | `geometry_msgs/Vector3` | Subscribed | 20 Hz | XYZ delta command from teleop |
| `/emergency_stop` | `std_msgs/Bool` | Subscribed | On demand | Safety halt signal |

## What the Keyboard Controls

The keyboard teleop node controls the **rigid laparoscopic instrument** 
visible in the SOFA GUI — the orange rod interacting with the yellow FEM 
tissue pad.

| Key | Axis | Direction | Physical meaning |
|---|---|---|---|
| W | Y | +1.0 | Instrument forward |
| S | Y | -1.0 | Instrument backward |
| D | X | +1.0 | Instrument right |
| A | X | -1.0 | Instrument left |
| Q | Z | +1.0 | Instrument up |
| E | Z | -1.0 | Instrument down (insertion) |
| Space | — | 0.0 | Hold position |
| Esc | — | — | Emergency stop + exit |

### Speed Calculation

Each keypress publishes a delta of 1.0 raw units.  
Bridge applies `DELTA_SCALE = 0.002` metres per unit.  
Timer runs at 50 Hz.  
Maximum instrument velocity = `1.0 × 0.002 × 50 = 0.1 m/s`

This matches the safe operating speed used during Phase 2 PPO training, 
ensuring the teleoperated instrument operates within the same dynamic 
envelope as the trained policy.

## Architecture
Keyboard input (pynput)
│
▼
TeleopKeyboardNode (20 Hz)
│  /joint_target (Vector3 delta)
▼
SofaBridgeNode (50 Hz)
│
├── env.step(action) ──► SOFA FEM simulation
│                              │
│◄─────────────────── obs, info (xyz, rgb, force)
│
├── /joint_states    ──► downstream consumers
├── /tissue_force_proxy ──► Phase 4D watchdog
└── /camera/image_raw  ──► Phase 4B vision pipeline

## Environment Setup

Three-step sourcing required for every terminal working on this project:

```bash
source ~/surgical_robot_lapgym_ws/activate.sh
```

Contents of `activate.sh`:
```bash
source ~/surgical_robot_lapgym_ws/sofa_venv/bin/activate
source /opt/ros/humble/setup.bash
source ~/surgical_robot_lapgym_ws/ros2_ws/install/setup.bash
export PYTHONPATH=/home/ubuntu/surgical_robot_lapgym_ws/sofa_venv/lib/python3.10/site-packages:$PYTHONPATH
export PYTHONPATH=/home/ubuntu/surgical_robot_lapgym_ws/surgical-rl:$PYTHONPATH
echo "[surgical-rl] sofa_venv + ROS 2 Humble ready"
```

**Source order is mandatory.** `sofa_venv` must be activated before ROS 2 
is sourced to prevent Python path conflicts. Never reverse this order.

## Running Phase 4A

**Terminal 1 — Bridge node:**
```bash
source ~/surgical_robot_lapgym_ws/activate.sh
ros2 run lapgym_ros2_bridge bridge_node
```

**Terminal 2 — Keyboard teleop:**
```bash
source ~/surgical_robot_lapgym_ws/activate.sh
ros2 run lapgym_ros2_bridge teleop_keyboard
```

**Terminal 3 — Verify topics (optional):**
```bash
source ~/surgical_robot_lapgym_ws/activate.sh
ros2 topic hz /joint_states          # verify 50 Hz
ros2 topic echo /joint_target        # verify teleop commands
ros2 topic echo /tissue_force_proxy  # verify force proxy
```

## Stub Mode

When SOFA is not available (missing `sofagym` module), `bridge_node` 
automatically runs in stub mode, publishing zero-filled messages on all 
topics. This allows ROS 2 graph development and testing without requiring 
the full SOFA installation to be active.

Stub mode is detected at startup:
[WARN] [sofa_bridge_node]: SOFA not available -- running in stub mode
[INFO] [sofa_bridge_node]: SofaBridgeNode started at 50 Hz

## Known Issues

**matplotlib Axes3D warning** — harmless conflict between system matplotlib 
and venv matplotlib. Does not affect functionality.

**VS Code import warnings** — `rclpy`, `std_msgs`, `sensor_msgs`, 
`geometry_msgs` show as unresolved in VS Code intellisense. This is because 
ROS 2 packages are only added to `sys.path` at runtime when ROS 2 is sourced. 
Code runs correctly. Cosmetic only.

**SOFA SIGABRT on exit** — known C++ GIL destructor issue in SOFA. 
Handled by `os._exit(0)` at the end of both nodes. Not a code bug.

## Verification Results

Phase 4A verified working on:
- Platform: Ubuntu 22.04, GTX 1650, ROS 2 Humble
- Python: 3.10.12 (sofa_venv)
- `/joint_states` rate: 49.99 Hz (target 50 Hz) ✓
- All 6 keyboard axes verified publishing correct Vector3 deltas ✓
- Emergency stop via Esc verified ✓
- Stub mode verified ✓

## Update: TissueRetractionV2 (Phase 4A Final)

Phase 4A bridge was updated from TissueRetractionV3 to TissueRetractionV2
after analysis confirmed V2 is the correct environment for the deployment
layer. V3 is the research environment (Phase 3) and adds unnecessary
perception overhead to the bridge.

### Environment change rationale

| Aspect | TissueRetractionV2 | TissueRetractionV3 |
|---|---|---|
| Observation | 7D ground-truth | 132D visual |
| Perception pipeline | None | MobileNetV3 inference |
| Bridge overhead | Minimal | High |
| Phase index | obs[6] | obs[3] |
| Purpose | Deployment layer | Research layer |

### Observation mapping (V2, 7D)
obs[0:3] → tool_xyz (normalised to [-1, 1])
obs[3:6] → goal_xyz (normalised to [-1, 1])
obs[6]   → phase (0.0=GRASPING, 1.0=RETRACTING)

### Grasping mechanics

Grasping in TissueRetractionV2 is **automatic** — no button press required.
When the instrument tip reaches within 3mm (`grasping_threshold=0.003`) of
the grasping position, the environment automatically triggers the grasp.

The grasping position is randomised each episode. Example values:
Grasping position (world): [-0.0486, 0.0085, 0.0356] metres
Tool start position (obs):  [-0.181,  0.379, -0.802]  normalised
Goal normalised (obs):      [-1.0,    0.189,  0.791]  normalised

The red marker visible in the SOFA GUI is the grasping target. Navigate
the instrument tip to within 3mm of that marker to trigger automatic grasp.

### Teleop reset behaviour (updated)

| Event | Behaviour |
|---|---|
| `terminated=True` (goal reached or collision) | Environment resets |
| `truncated=True` (300 step limit) | Counter resets only, surgeon continues |

This allows continuous teleoperation without interruption at 300 steps,
matching real surgical operation where the surgeon operates until task
completion.

### Compatibility fixes applied

These fixes were required to run the bridge with full SOFA stack:

| Issue | Fix |
|---|---|
| `No module named 'gymnasium'` | PYTHONPATH set in activate.sh |
| `No module named 'sofagym'` | Corrected to envs.tissue_retraction_v2 |
| `No module named 'sofa_env'` | lap_gym/sofa_env added to PYTHONPATH |
| `No module named 'Sofa'` | sofa_install SofaPython3 path added to PYTHONPATH |
| `coverage.types` AttributeError | numba coverage_support.py patched |
| `setuptools --editable` error | Downgraded setuptools to 65.5.1 |
| Model path not found | Run bridge from surgical-rl directory |
| `render_mode` string vs enum | Convert string to RenderMode enum in _init_env |

### Position HUD (Phase 4A Option A — planned)

A live position HUD will be added to the teleop terminal showing:
- Current tool XYZ position
- Goal XYZ position  
- Distance to goal
- Recommended movement direction

This is implemented via a `/guidance` ROS 2 topic subscriber in the
teleop node, giving the surgeon real-time navigation feedback without
requiring the PPO agent (which is Phase 4B).

## Running Phase 4A with SOFA GUI

```bash
# Terminal 1 — bridge with GUI (must run from surgical-rl directory)
source ~/surgical_robot_lapgym_ws/activate.sh
cd ~/surgical_robot_lapgym_ws/surgical-rl
ros2 run lapgym_ros2_bridge bridge_node --ros-args -p render_mode:=human

# Terminal 2 — keyboard teleop
source ~/surgical_robot_lapgym_ws/activate.sh
ros2 run lapgym_ros2_bridge teleop_keyboard

# Terminal 3 — monitor topics
source ~/surgical_robot_lapgym_ws/activate.sh
ros2 topic echo /joint_states
```

## Verified behaviour

- SOFA GUI opens showing FEM tissue and rigid instrument ✓
- Keyboard controls instrument continuously without 300-step reset ✓
- Grasping triggers automatically when tip reaches within 3mm of red marker ✓
- `/joint_states` publishes at 50 Hz with correct 7D V2 observation ✓
- `/tissue_force_proxy` shows force reading during tissue interaction ✓
- Emergency stop via Esc key functional ✓


## Phase 4A HUD Guidance — Teleop Position Display

### What is the HUD

HUD (Heads-Up Display) is a real-time single-line status display in the
teleop terminal showing instrument position, goal position, distance, and
collision status. The surgeon sees all critical navigation information
without looking away from the SOFA GUI.

### What Was Built

A world-space millimetre guidance system added to `teleop_keyboard.py`
and `bridge_node.py` that gives the surgeon exact navigation feedback
to reach the grasping target.

**HUD format:**
[GRASPING] G(-49,+9,+36mm) T(-46,+11,+33mm) D:5mm X:OK Y:OK Z:OK SAFE

| Field | Meaning |
|---|---|
| `[GRASPING]` / `[RETRACTING]` | Current task phase |
| `G(x,y,z mm)` | Goal position in world millimetres |
| `T(x,y,z mm)` | Tool position in world millimetres |
| `D:Nmm` | Distance to goal in world millimetres |
| `X:±N(key)` | Millimetres to move on X axis and which key |
| `Y:±N(key)` | Millimetres to move on Y axis and which key |
| `Z:±N(key)` | Millimetres to move on Z axis and which key |
| `SAFE` / `COL:N` | Collision status and cost |

**Colour coding:**
- Distance red `> 20mm` — navigating
- Distance yellow `5–20mm` — approaching
- Distance green `< 5mm` — grasp imminent (triggers at 3mm)
- Axis yellow — needs correction
- Axis green `OK` — within 3mm of goal on that axis
- Collision red — instrument touching tissue incorrectly
- Safe green — no collision

### Surgeon Workflow

1. Launch bridge with GUI and teleop nodes
2. Read `G(x,y,z)` — goal position in world millimetres
3. Read `T(x,y,z)` — current tool position
4. Follow axis hints — press the indicated key for each axis
5. When all axes show `OK` and distance turns green — grasp triggers automatically
6. Goal `G` switches to retraction end position — follow new hints to retract tissue

### Key Engineering Findings

**Coordinate space mismatch** — `obs[0:3]` (tool position) and `obs[3:6]`
(goal position) from the gymnasium observation are in different normalised
coordinate spaces and cannot be compared directly by subtraction. This
caused incorrect axis guidance in earlier iterations.

**Correct distance metric** — `info['distance_to_grasping_position']` from
the LapGym info dict provides the physically correct world-space distance
in metres. This is what the environment itself uses to trigger grasping at
the 3mm threshold.

**Tool world position access** — the instrument tip position in world
metres is accessed via the SOFA scene graph:
```python
env._env.end_effector.gripper.motion_target_mechanical_object
    .position.array()[0][:3]
```

**Goal switching** — after grasping triggers, the guidance goal switches
from `_grasping_position` to `_end_position` and distance switches from
`distance_to_grasping_position` to `distance_to_end_position`.

**Fixed goal positions (world metres):**
Grasping target:   X=-0.0486  Y=+0.0085  Z=+0.0356  (= -49, +9, +36 mm)
Retraction target: X=-0.0194  Y=+0.0626  Z=-0.0033  (= -19, +63, -3 mm)

### ROS 2 Topic: /guidance

Published by `bridge_node.py` at 50 Hz. Contains 9 values:

| Index | Name | Units | Description |
|---|---|---|---|
| 0 | goal_x | metres | Goal X world position |
| 1 | goal_y | metres | Goal Y world position |
| 2 | goal_z | metres | Goal Z world position |
| 3 | distance | metres | World-space distance to goal |
| 4 | collision | — | Collision cost magnitude |
| 5 | steps_in_collision | count | Steps in collision this episode |
| 6 | tool_wx | metres | Tool X world position |
| 7 | tool_wy | metres | Tool Y world position |
| 8 | tool_wz | metres | Tool Z world position |

### Verification

Verified on Ubuntu 22.04, ROS 2 Humble, GTX 1650:
- HUD stays on single line — no scrolling ✓
- Tool position updates correctly as instrument moves ✓
- Distance decreases correctly as instrument approaches goal ✓
- All axes show OK when within 3mm of goal ✓
- Grasp triggers automatically at 3mm ✓
- Goal switches to retraction target after grasp ✓
- Collision indicator turns red on tissue contact ✓
- Phase changes GRASPING → RETRACTING in HUD ✓


## Next Phase

Phase 4B: PPO policy action servers — wrapping the Phase 2C checkpoint as a ROS 2 action server with `is_preempted()` checking every step.


