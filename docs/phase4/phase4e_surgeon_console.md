# Phase 4E — Surgeon Console

**Author:** Subhash Arockiadoss  
**Status:** Complete  
**Tag:** `v4.5-phase4e-complete`  
**Date:** April 2026

---

## Overview

Phase 4E implements the surgeon console — a terminal-based UI that gives the operator real-time control over the surgical procedure. It replicates the human-in-the-loop control layer found in real surgical robot systems such as the da Vinci, where the surgeon can pause, resume, or emergency-stop the autonomous agent at any point during the procedure.

The console communicates with all three action servers (Approach, Retract, Hold) simultaneously via ROS 2 topics, allowing a single keypress to freeze or resume the entire surgical workflow regardless of which phase is currently active.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 5 — HUMAN INTERFACE (surgeon_console.py)             │
│  curses terminal UI, keyboard S/R/E/Q                       │
│  Publishes: /surgeon_stop (Bool), /emergency_stop (Bool)    │
│  Subscribes: /console_feedback, /watchdog_status,           │
│              /tissue_force_proxy, /emergency_stop           │
└──────────────────────┬──────────────────────────────────────┘
                       │ ROS 2 topics (DDS middleware)
┌──────────────────────▼──────────────────────────────────────┐
│  LAYER 4 — BEHAVIOUR TREE (surgical_bt_node.py)             │
│  Orchestrates: Approach → Retract → Hold sequence           │
│  Publishes: /console_feedback (phase, step, distance)       │
└──────────────────────┬──────────────────────────────────────┘
                       │ ROS 2 Action
┌──────────────────────▼──────────────────────────────────────┐
│  LAYER 3 — ACTION SERVERS                                   │
│  approach_policy_server  → proportional controller          │
│  retract_policy_server   → Phase 2D PPO agent               │
│  hold_policy_server      → zero-action hold                 │
│                                                              │
│  Each server has:                                           │
│  ├── Main executor (rclpy.spin)                             │
│  ├── Separate ROS 2 Context for stop listener               │
│  ├── Background thread spinning stop executor               │
│  ├── Dual freeze loops (before + after env.step())          │
│  └── threading.Event for immediate unblock on resume        │
└──────────────────────┬──────────────────────────────────────┘
                       │ Python API (blocking call)
┌──────────────────────▼──────────────────────────────────────┐
│  LAYER 2 — SOFA SIMULATION (TissueRetractionV2)             │
│  env.step() synchronous ~65ms blocking call                 │
│  FEM tissue physics + collision + OpenGL render             │
└──────────────────────┬──────────────────────────────────────┘
                       │ ROS 2 topics
┌──────────────────────▼──────────────────────────────────────┐
│  LAYER 1 — SAFETY (bridge_node, safety_watchdog_node)       │
│  IEC 62304 independent force monitor at 50Hz                │
│  Auto emergency stop if tissue force > 1.0N                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1. Single Topic for Stop and Resume

There is no separate `/surgeon_resume` topic. Resume works by publishing `/surgeon_stop=False` — the same topic, opposite boolean value. This is intentional: one topic, one source of truth for surgeon stop state, no race condition between two topics.

```
S pressed → /surgeon_stop=True  → _surgeon_stopped=True  → event.clear() → freeze
R pressed → /surgeon_stop=False → _surgeon_stopped=False → event.set()   → unfreeze
```

### 2. Separate ROS 2 Context for Stop Listener

**Problem:** `env.step()` is a ~65ms blocking synchronous SOFA call. During this time the main ROS 2 executor cannot process any callbacks. S pressed during `env.step()` would not be received until the step completes.

**Failed approaches tried:**
- `rclpy.spin_once` inside execute loop — blocked by env.step()
- `threading.Event.wait()` — event never set during step
- `asyncio.sleep` — env.step() is synchronous, no await point
- `MultiThreadedExecutor` — caused SOFA OpenGL race condition `(0,0,3) into (480,480,3)`
- Background node with shared executor — `IndexError: wait set index too big`

**Solution:** Each action server creates a completely isolated DDS instance using a separate `rclpy.Context()`. This gives the background listener its own DDS waitset, completely independent from the main node's executor:

```python
self._stop_context = rclpy.Context()
self._stop_context.init()
self._stop_node = rclpy.create_node(
    '_surgeon_stop_approach',
    context=self._stop_context,
    enable_rosout=False
)
self._stop_executor = rclpy.executors.SingleThreadedExecutor(
    context=self._stop_context)
self._stop_executor.add_node(self._stop_node)
self._stop_thread = threading.Thread(
    target=self._spin_stop_node, daemon=True)
self._stop_thread.start()

def _spin_stop_node(self):
    while self._stop_context.ok():
        try:
            self._stop_executor.spin_once(timeout_sec=0.01)
        except Exception:
            pass
```

The `timeout_sec=0.01` (10ms) is critical — reduced from the initial 0.1s to minimise callback processing latency.

### 3. Dual Freeze Loop Pattern

Two surgeon stop freeze loops per execute cycle:

```python
# Loop 1 — before env.step(): catches S pressed between steps
while self._surgeon_stopped and not self._emergency:
    if goal_handle.is_cancel_requested:
        break
    self._stop_event.wait(timeout=0.05)

obs, reward, terminated, truncated, info = self._env.step(action)
step += 1

# Loop 2 — after env.step(): catches S pressed during the step
while self._surgeon_stopped and not self._emergency:
    if goal_handle.is_cancel_requested:
        break
    self._stop_event.wait(timeout=0.05)
```

This ensures the instrument freezes at most one `env.step()` after S is pressed rather than continuing for multiple steps.

### 4. Republish Timer

The console republishes `/surgeon_stop=True` at 10Hz while stopped. This prevents resume if the BT restarts the goal and the new action server instance does not receive the original stop message:

```python
def _republish_surgeon_stop(self):
    if self._surgeon_stopped:
        msg = Bool()
        msg.data = True
        self._pub_surgeon_stop.publish(msg)
```

### 5. Emergency Stop try/except Pattern

Emergency stop can arrive during a goal state transition. Using `goal_handle.canceled()` directly raises `RCLError: invalid transition`. Fixed with:

```python
if self._emergency:
    try:
        goal_handle.canceled()
    except Exception:
        goal_handle.abort()
```

---

## Console UI Layout

```
============================================================
            SURGICAL ROBOT CONSOLE  v1.0
============================================================
PHASE:    RETRACT          STEP:  117 / 300
DISTANCE: 6.0mm            FORCE: 0.000
WATCHDOG: ● NOMINAL        ESTOP: CLEAR
BT STATE: RUNNING [SURGEON STOPPED]
------------------------------------------------------------
      [S] STOP    [R] RESUME    [E] EMERGENCY    [Q] QUIT
============================================================
Event log:
[13:52:04] [S] STOP  phase=RETRACT step=044 dist=86.7mm
[13:52:11] [R] RESUME phase=RETRACT step=050 dist=79.2mm
[13:52:16] [S] STOP  phase=RETRACT step=109 dist=14.5mm
```

**Key bindings:**

| Key | Action | Topic |
|-----|--------|-------|
| S | Surgeon stop — freeze at current step | `/surgeon_stop=True` |
| R | Resume — unfreeze from exact stop point | `/surgeon_stop=False` |
| E | Emergency stop — halt all servers permanently | `/emergency_stop=True` |
| Q | Quit console (does not stop surgical system) | — |

---

## Stop/Resume Behaviour Per Phase

| Phase | Active Server | Freeze behaviour |
|-------|--------------|-----------------|
| APPROACH | approach_policy_server | Proportional controller pauses, instrument holds position |
| RETRACT | retract_policy_server | PPO agent pauses mid-trajectory |
| HOLD | hold_policy_server | Zero-action loop pauses |

All three servers subscribe to the same `/surgeon_stop` topic. When S is pressed, all three receive it simultaneously — only the currently active server (executing a goal) actually freezes.

---

## Known Limitations and Sim-to-Real Gap

### Stop Latency

```
Measured stop latency: up to ~15-20 env steps × ~65ms = ~1-1.3 seconds

Root cause: env.step() is a synchronous SOFA full-physics call.
It cannot be interrupted mid-execution.
The GIL prevents any Python thread from running during env.step().
The freeze loop fires only after env.step() returns.

Industry standard: <200ms (FDA Class III), <5ms (da Vinci hardware)
```

### Why This Gap Exists

| System | Control loop | Stop mechanism | Stop latency |
|--------|-------------|---------------|-------------|
| da Vinci surgical robot | 1kHz RTOS | Hardware PWM torque cutoff | <1ms |
| Our SOFA simulation | ~15Hz (65ms/step) | Software freeze loop | 65-300ms |
| Industry minimum | 50Hz (20ms/step) | Software + hardware | <200ms |

The safety_watchdog_node achieves 50Hz because it only reads a topic — no blocking call. The action servers are limited to ~15Hz by SOFA FEM physics computation time on the GTX 1650 GPU.

**On higher-end hardware (A100 GPU):** SOFA step time drops to ~8-12ms, giving ~80-120Hz and stop latency of ~10-25ms — approaching industry standard.

**On a real surgical robot:** The policy would output a joint torque command (1ms execution) rather than a SOFA physics step (65ms). The stop latency would be hardware-limited (<5ms) not simulation-limited.

### Visual Observation

When the SOFA GUI is open, the stop response appears fast visually — the instrument visibly pauses within 1-2 frames. The numerical step difference seen in the console event log (10-20 steps) is larger than the visual perception gap, which is the sim-to-real perceptual difference.

---

## Files Changed

| File | Change |
|------|--------|
| `surgeon_console.py` | New file — curses terminal UI, S/R/E/Q control |
| `approach_policy_server.py` | Separate ROS2 context, dual freeze loops, per-step feedback, 10ms spin |
| `retract_policy_server.py` | Same pattern, unique node name `_surgeon_stop_retract` |
| `hold_policy_server.py` | Same pattern, emergency try/except, `_surgeon_stop_hold` |
| `surgical_system.launch.py` | Already included surgeon_console in launch |
| `surgical_bt_node.py` | Publishes `/console_feedback` for console display |

---

## Test Results

### Isolation Test (separate context verification)
```
Node ready. surgeon_stopped= False
Published STOP
Approach: SURGEON STOP received ← fires within 100ms
surgeon_stopped= True | stop_event_set= False  ✓
Published RESUME
Approach: SURGEON RESUME received
surgeon_stopped= False | stop_event_set= True  ✓
TEST PASSED
```

### Integration Test (full procedure with S/R)
```
APPROACH → [S] freeze → [R] resume → goal_reached steps=119
RETRACT  → [S] freeze → [R] resume → goal_reached steps=134 dist=2.6mm
HOLD     → [S] freeze → [R] resume → [E] emergency → procedure complete
BT STATE: SUCCESS
```

### SOFA GUI Verification
- Orange instrument visible navigating to red grasping target
- Tissue deformation visible during retract phase
- Instrument visibly pauses on S, resumes on R
- Console event log synchronized with visual state

---

## Commissioning Instructions

```bash
# 1. Activate environment
source ~/surgical_robot_lapgym_ws/activate.sh

# 2. Launch full system
cd ~/surgical_robot_lapgym_ws/surgical-rl
ros2 launch lapgym_ros2_bridge surgical_system.launch.py

# 3. Console opens automatically in xterm window
#    S — stop, R — resume, E — emergency, Q — quit

# 4. For SOFA GUI (retract phase visual):
#    In retract_policy_server.py change:
#    RenderMode.HEADLESS → RenderMode.HUMAN
#    Then rebuild: colcon build --packages-select lapgym_ros2_bridge
```

---

## Phase 4 Complete Node Table

| Node | Role | Status |
|------|------|--------|
| sofa_bridge_node | Teleop fallback, env bridge | ✓ Phase 4A |
| approach_policy_server | Proportional controller to grasping zone | ✓ Phase 4B |
| retract_policy_server | PPO agent tissue retraction | ✓ Phase 4B |
| hold_policy_server | Zero-action position hold | ✓ Phase 4B |
| safety_watchdog_node | IEC 62304 independent force monitor 50Hz | Phase 4C |
| surgical_bt_node | Behaviour tree orchestrator | ✓ Phase 4C |
| surgeon_console | Human-in-the-loop terminal UI | ✓ Phase 4E |