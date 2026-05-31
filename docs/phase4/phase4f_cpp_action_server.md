# Phase 4F — C++ Action Server

**Author:** Subhash Arockiadoss  
**Status:** Complete  
**Tag:** v4.6-phase4f-cpp  
**Date:** May 2026

---

## Overview

Phase 4F ports the approach policy action server from Python (rclpy)
to C++ (rclcpp), demonstrating production-pattern ROS 2 C++ development.

The Python version (Phase 4B) and C++ version run identical proportional
controller logic with identical safety patterns. The middleware contract
(Retract.action interface) is unchanged — demonstrating language-agnostic
ROS 2 architecture.

---

## Why C++ for This Layer

Python's Global Interpreter Lock (GIL) forces single-threaded execution
of Python bytecode. During env.step() (~65ms SOFA physics call), the GIL
is held and no Python callback can fire — including surgeon stop.

PyTorch escapes this by releasing the GIL before launching CUDA kernels
on GPU. SOFA's SofaPython3 binding does not release the GIL — it runs
on CPU and holds the lock for the full physics step duration.

C++ has no GIL. Threads run truly in parallel on separate CPU cores.
std::atomic<bool> provides safe shared state between parallel threads
without the performance cost of a mutex.
Python GIL problem:          C++ solution:
env.step() holds GIL 65ms    sofaStep() on Thread A
→ stop callback blocked      → stop callback on Thread B
→ delay up to 65ms           → fires immediately (~10ms)
→ instrument keeps moving    → std::atomic flag set
→ execute() sees it next check

---

## Architecture
C++ approach_policy_server_cpp     Python sofa_step_service
(rclcpp action server)             (rclpy service server)
│                                  │
│── SofaStep request ─────────────>│
│   action=[ax, ay, az]            │
│   reset=False                    │  env.step(action)
│                                  │  ← SOFA FEM physics
│<── SofaStep response ────────────│
│    obs[7], dist, tool_xyz        │
│                                  │
Proportional controller C++        SOFA bindings stay Python
std::atomic<bool> stop flags       PPO stays Python (PyTorch)
MultiThreadedExecutor              Perception stays Python
ELF 64-bit binary

---

## Key Engineering Decisions

### 1. std::atomic\<bool\> for Stop Flags

```cpp
std::atomic<bool> surgeon_stopped_{false};
std::atomic<bool> emergency_{false};
```

Two threads access these simultaneously:
- execute() loop reads them every iteration
- stop_callback_group_ thread writes them on message arrival

Plain bool with two truly parallel threads is undefined behaviour
in C++. The compiler may cache the value in a CPU register —
one thread writes true, other thread still reads cached false.

std::atomic guarantees:
- Write on one thread immediately visible to all other threads
- No torn reads (reading half-written value)
- Memory ordering enforced across CPU cores

IEC 62304 relevance: Class C software — stop flag corruption
would mean emergency stop silently fails. Atomic is the correct
implementation of SR-010 through SR-014 in C++.

### 2. Three Callback Groups + MultiThreadedExecutor

```cpp
client_callback_group_   // /sofa_step service responses
stop_callback_group_     // /surgeon_stop, /emergency_stop  
default                  // action server goals, cancels
```

sofaStep() calls future.wait_for() which blocks execute()
thread ~65ms per physics step. Without separate callback groups,
SingleThreadedExecutor cannot process service responses or stop
callbacks during this period.

MultiThreadedExecutor assigns each callback group to its own
thread — true parallel execution on separate CPU cores.

Python comparison:

| Concern | Python solution | C++ solution |
|---------|----------------|-------------|
| Stop during blocking call | Separate rclpy.Context | Separate CallbackGroup |
| Thread safety | threading.Event | std::atomic<bool> |
| Parallel execution | Independent DDS instance | MultiThreadedExecutor |

### 3. future.wait_for() not spin_until_future_complete()

```cpp
// WRONG — crash: adds node to second executor
rclcpp::spin_until_future_complete(get_node_base_interface(), future);

// CORRECT — blocks thread only, executor unaffected
auto status = future.wait_for(std::chrono::seconds(5));
```

MultiThreadedExecutor already owns the node. 
spin_until_future_complete tries to add it to another executor — crash.
future.wait_for() is pure C++ stdlib — no executor involvement.

### 4. Proportional Controller Values

```cpp
GRASPING_TARGET = {-0.0485583f, 0.0085f, 0.0356076f}  // metres
APPROACH_THRESHOLD = 0.025f   // 25mm handoff to PPO retract agent
APPROACH_GAIN = 2.0f          // action = clip(direction × 2.0, -3, 3)
```

GRASPING_TARGET from Phase 4A world-frame coordinate analysis.
Threshold chosen so PPO retract agent starts within its training
distribution. Gain produces consistent 3mm/step reduction.

Known limitation: target is constexpr — hardcoded at compile time.
Real surgical robot: target from pre-operative imaging + registration,
delivered as ROS 2 topic subscription at runtime.

---

## Verified Results
Test: C++ action server + Python SOFA service
Goal: max_steps=200.0, render=false
step  5  | Dist: 67.6mm
step 10  | Dist: 64.6mm
step 15  | Dist: 61.6mm
step 20  | Dist: 58.6mm
...
step 75  | Dist: 25.6mm
step 76  | Dist: 24.4mm ← goal_reached
Result:
success: true
steps_taken: 76
final_distance: 0.0244m
termination: goal_reached
status: SUCCEEDED

---

## Gap This Phase Fills

| Gap | Before | After |
|-----|--------|-------|
| C++ in project | Python only | rclcpp action server |
| Build system | setuptools | ament_cmake + CMakeLists.txt |
| Thread safety | threading.Event | std::atomic<bool> |
| Concurrency | separate rclpy.Context | MultiThreadedExecutor |
| Executable | Python script | ELF 64-bit binary |

---

## Remaining Gaps

| Gap | Path to fill |
|-----|-------------|
| Qt GUI | 2-week project — surgical workstation widget |
| Real-time control (EtherCAT, QNX) | Industry experience |
| Real hardware | First industry role |
| C++ unit tests (Google Test) | Phase 5 addition |

---

## Files Added

| File | Description |
|------|-------------|
| `ros2_packages/lapgym_ros2_bridge_cpp/src/approach_policy_server.cpp` | C++ rclcpp action server |
| `ros2_packages/lapgym_ros2_bridge_cpp/CMakeLists.txt` | ament_cmake build |
| `ros2_packages/lapgym_ros2_bridge_cpp/package.xml` | Package manifest |
| `ros2_packages/lapgym_ros2_bridge/lapgym_ros2_bridge/sofa_step_service.py` | Python SOFA bridge |
| `ros2_packages/lapgym_interfaces/srv/SofaStep.srv` | Custom service definition |