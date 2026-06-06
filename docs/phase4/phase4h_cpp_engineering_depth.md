# Phase 4H — C++ Engineering Depth

**Tag:** `v4.10-phase4h-cpp-depth`
**Status:** Complete ✓
**Files changed:**
- `ros2_packages/lapgym_ros2_bridge_cpp/src/approach_policy_server.cpp`
- `ros2_packages/lapgym_ros2_bridge_cpp/CMakeLists.txt`
- `ros2_packages/lapgym_ros2_bridge_cpp/test/test_proportional_controller.cpp` ← new

---

## Why Phase 4H Exists

After Phase 4F/4G the C++ code was rated 6.5/10 by a surgical robotics
hiring assessment. The gap analysis identified four specific weaknesses:

```
Gap 1: Thin wrapper — no non-trivial C++ algorithm
Gap 2: No RAII — no class owning its own state
Gap 3: No unit tests — no Google Test coverage
Gap 4: No real-time awareness — no timing measurement
```

Phase 4H addresses all four gaps in a single coherent phase.
Each addition answers a different question a hiring engineer asks
when reviewing C++ code:

```
"Do you speak the vocabulary?"     → Eigen ✓
"Do you structure code properly?"  → ProportionalController class ✓
"Do you verify your code?"         → Google Tests ✓
"Do you understand timing?"        → Step timing logs ✓
```

---

## What Was Built — Four Dimensions

### Dimension 1 — Eigen Refactor (Vocabulary)

**What changed:**

```cpp
// Before Phase 4H — scalar arithmetic, 9 lines, error-prone
static constexpr std::array<float, 3> GRASPING_TARGET = {
    -0.0485583f, 0.0085f, 0.0356076f
};
float ex   = GRASPING_TARGET[0] - tool_x;
float ey   = GRASPING_TARGET[1] - tool_y;
float ez   = GRASPING_TARGET[2] - tool_z;
float dist = std::sqrt(ex*ex + ey*ey + ez*ez);
float norm = dist + 1e-8f;
float ax   = std::max(-3.0f, std::min(3.0f, (ex/norm) * APPROACH_GAIN));
float ay   = std::max(-3.0f, std::min(3.0f, (ey/norm) * APPROACH_GAIN));
float az   = std::max(-3.0f, std::min(3.0f, (ez/norm) * APPROACH_GAIN));

// After Phase 4H — Eigen vector operations, 4 lines, self-documenting
const Eigen::Vector3f GRASPING_TARGET(-0.0485583f, 0.0085f, 0.0356076f);
Eigen::Vector3f tool(state->tool_world_x,
                     state->tool_world_y,
                     state->tool_world_z);
Eigen::Vector3f error = GRASPING_TARGET - tool;
float           dist  = error.norm();
Eigen::Vector3f action = (error.normalized() * APPROACH_GAIN)
                          .cwiseMax(-3.0f).cwiseMin(3.0f);
```

**Why Eigen:**
Every surgical robotics C++ codebase uses Eigen. It is in the JD at
CMR Surgical, Medtronic, and ARTORG. More importantly:

- `error.norm()` is self-documenting — immediately readable as vector magnitude
- `error.normalized()` is self-documenting — immediately readable as unit vector
- Three separate floats allow component mixup bugs — `Eigen::Vector3f` does not
- The type system enforces that X, Y, Z belong together as a 3D position

**What each Eigen method does:**
```
error.norm()          → sqrt(x²+y²+z²)     vector magnitude
error.normalized()    → error / error.norm() unit vector (direction only)
.cwiseMax(-3.0f)      → max(-3, each element) clips lower bound
.cwiseMin( 3.0f)      → min(+3, each element) clips upper bound
```

---

### Dimension 2 — ProportionalController Class (RAII + Structure)

**What changed:**

A dedicated class was added before `ApproachPolicyServerCpp`:

```cpp
class ProportionalController
{
public:
    ProportionalController(float gain, float max_action)
    : gain_(gain), max_action_(max_action)
    {}

    Eigen::Vector3f compute(const Eigen::Vector3f& error)
    {
        return (error.normalized() * gain_)
                .cwiseMax(-max_action_)
                .cwiseMin( max_action_);
    }

    void reset() {}

private:
    float gain_;
    float max_action_;
};
```

The class is initialised in the `ApproachPolicyServerCpp` constructor list:
```cpp
: Node("approach_policy_server_cpp"),
  surgeon_stopped_(false),
  emergency_(false),
  controller_(APPROACH_GAIN, 3.0f)   // ← owned as member state
```

And declared as a member variable:
```cpp
ProportionalController controller_;
```

Used in the execute loop as one line:
```cpp
Eigen::Vector3f action = controller_.compute(error);
```

**Why this class matters — RAII explained:**

RAII stands for Resource Acquisition Is Initialisation. It is the most
fundamental C++ design pattern. It means: a resource (memory, state,
a file handle, a thread) is acquired when an object is constructed and
released when it is destroyed — automatically.

Before the class: `APPROACH_GAIN` and `3.0f` were hardcoded constants
scattered inline in the execute loop. They were not owned by anything.

After the class: `gain_` and `max_action_` are owned by
`ProportionalController`. When `ApproachPolicyServerCpp` is destroyed,
`controller_` is automatically destroyed too — taking its state with it.
No manual cleanup. No possibility of using a stale value.

The destructor pattern in `ApproachPolicyServerCpp` already demonstrates
RAII for threads:
```cpp
~ApproachPolicyServerCpp() {
    stop_executor_->cancel();  // release resource
    stop_thread_.join();       // release resource
}
```
The `ProportionalController` class extends this pattern to algorithm state.

**Why KI=0, KD=0 (pure proportional only):**

A full PID controller adds Integral (accumulated past error) and Derivative
(rate of change) terms. In SOFA approach navigation these add no measurable
benefit — the proportional controller already achieves 100% goal rate.
The `reset()` method exists so a PID subclass could override it cleanly
if needed in future. This is correct engineering judgment, not a limitation.

---

### Dimension 3 — Step Timing Logs (Real-Time Awareness)

**What changed:**

The `sofaStep()` call in the execute loop is now wrapped with timing:

```cpp
// -- Step timing + SOFA call -----------------------------------------
// Measures actual sofaStep() latency — demonstrates real-time awareness.
// RCLCPP_DEBUG only shows with --log-level debug (no normal output spam)
auto step_start  = std::chrono::steady_clock::now();
auto result_step = sofaStep({action.x(), action.y(), action.z()});
auto step_ms     = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - step_start).count();
RCLCPP_DEBUG(get_logger(),
    "Step %d sofaStep took %ldms", step, step_ms);
step++;
```

**Why this matters:**

Every surgical robot runs a deterministic real-time control loop.
Real-time C++ patterns require knowing how long each operation takes.
`SCHED_FIFO` and `pthread_setschedparam` (true real-time scheduling)
are not added here because they are meaningless without hardware —
SOFA physics takes ~65ms per step which is not a real-time constraint.

But measuring and logging the latency demonstrates you know the pattern
exists and that you are measuring it. The `RCLCPP_DEBUG` log level means
it does not appear in normal output — only when explicitly enabled with
`--log-level debug`. This is correct production behaviour.

**To see timing logs in action:**
```bash
ros2 run lapgym_ros2_bridge approach_policy_server_cpp \
  --ros-args --log-level debug
```

**What the numbers mean:**
SOFA `env.step()` takes ~65ms on a GTX 1650. This is the fundamental
latency floor — not a GIL problem, not a ROS2 problem. It is the time
physics simulation takes. On real hardware (EtherCAT servo at 1kHz)
this would be ~1ms. The ~65× gap is the documented sim-to-real
latency difference.

---

### Dimension 4 — Google Tests (Industry Verification)

**What changed:**

New file: `test/test_proportional_controller.cpp`
New CMakeLists.txt block:
```cmake
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_proportional_controller
    test/test_proportional_controller.cpp
  )
  target_link_libraries(test_proportional_controller
    Eigen3::Eigen
  )
endif()
```

**Five tests, all passing:**

```
[  PASSED  ] ProportionalController.DirectionIsCorrect
[  PASSED  ] ProportionalController.ClipsToMaxBound
[  PASSED  ] ProportionalController.ClipsToNegativeBound
[  PASSED  ] ProportionalController.UnitErrorGivesGainMagnitude
[  PASSED  ] ProportionalController.DiagonalErrorCorrectDirection

100% tests passed, 0 tests failed out of 1
Total Test time (real) = 0.10 sec
```

**What each test verifies:**

| Test | What it proves |
|------|---------------|
| DirectionIsCorrect | +X error → +X action. Controller points toward target |
| ClipsToMaxBound | Large error with gain=5 → action clipped to 3.0 |
| ClipsToNegativeBound | Large -X error with gain=5 → action clipped to -3.0 |
| UnitErrorGivesGainMagnitude | Unit error × gain=2.0 → action=2.0 (unclipped) |
| DiagonalErrorCorrectDirection | 45° XY error → equal X and Y action components |

**Important: tests caught a real bug in the test expectations.**

The first run failed on `ClipsToMaxBound` and `ClipsToNegativeBound`.
The error message:
```
action.x() evaluates to 2
3.0f evaluates to 3
```

Root cause: with `gain=2.0` and unit error, the action is `2.0` —
which never reaches `max_action=3.0`, so clipping never fires.
The fix was to use `gain=5.0f` in those two tests so the unclipped
action would be `5.0`, which clips to `3.0` correctly.

This is the value of unit tests — they caught an incorrect assumption
about the algorithm before it could cause confusion in production.

**Why no ROS2 dependency in tests:**

The test file copies `ProportionalController` locally and tests pure
C++ logic. No nodes, no services, no SOFA required. This means:
- Tests run in 0.10s with zero infrastructure
- Tests can run in CI without a full ROS2 environment
- The algorithm logic is verified independently of the ROS2 plumbing

This is the correct pattern for IEC 62304 Class C unit verification —
test the safety-relevant algorithm in isolation.

**Run tests:**
```bash
source /opt/ros/humble/setup.bash
cd ~/surgical_robot_lapgym_ws/ros2_ws
colcon build --packages-select lapgym_ros2_bridge_cpp
ctest --test-dir build/lapgym_ros2_bridge_cpp --output-on-failure
```

---

## Complete Phase 4 C++ Evolution

```
Phase 4B  Python action servers    — baseline, GIL problem
Phase 4F  C++ approach server      — rclcpp, atomic, callback groups
Phase 4G  C++ hold + retract       — hybrid architecture complete
Phase 4H  C++ engineering depth    — Eigen + RAII + tests + timing
```

The progression tells a coherent story: identified the GIL problem,
solved it with C++, then deepened the C++ quality with correct
vocabulary, structure, verification, and timing awareness.
This is the engineering maturity that surgical robotics companies hire for.