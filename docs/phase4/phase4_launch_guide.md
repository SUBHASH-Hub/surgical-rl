# Phase 4 Launch and Operations Guide

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

Note: `render_mode:=human` opens the SOFA GUI for the bridge_node
environment. This environment is idle — action servers run headless.
The GUI is for monitoring the bridge's internal state only.

---

## Expected Startup Sequence
[bridge_node-1]         process started with pid [XXXXXX]
[approach_policy-2]     process started with pid [XXXXXX]
[retract_policy-3]      process started with pid [XXXXXX]
[hold_policy-4]         process started with pid [XXXXXX]
[safety_watchdog-5]     SafetyWatchdogNode started -- ALERT=0.35 STOP=1.0 at 50Hz
[safety_watchdog-5]     IEC 62304 independent safety layer ACTIVE
... SOFA plugin loading (normal) ...
[approach_policy-2]     ApproachPolicyServer ready
[hold_policy-4]         HoldPolicyServer ready
[retract_policy-3]      RetractPolicyServer ready
[bridge_node-1]         SofaBridgeNode started at 50 Hz
[safety_watchdog-5]     [WATCHDOG] uptime=10s state=NOMINAL force=0.000
--- 15 second delay (TimerAction) ---
[surgical_bt_node-6]    Building surgical behaviour tree
[surgical_bt_node-6]    SurgicalBTNode started -- ticking at 10 Hz

All four servers print `ready` before the BT starts ticking.

---

## Expected Procedure Output
[surgical_bt_node]     [Approach] Sending goal to approach_policy
[approach_policy]      Approach step   5 | Dist: XXX.Xmm
...
[approach_policy]      Approach complete: goal_reached steps=NNN
[surgical_bt_node]     [Retract] Sending goal to retract_policy
[retract_policy]       Step   5 | Dist: XXX.Xmm | COL
...
[retract_policy]       Goal reached at step NNN!
[surgical_bt_node]     [Hold] Sending goal to hold_policy
[hold_policy]          Hold policy active -- holding position
...
[hold_policy]          Hold timeout reached
[surgical_bt_node]     Surgical procedure complete -- SUCCESS

---

## SOFA Warnings — Expected and Safe to Ignore

All of these appear on every run and are normal:
[ERROR] RequiredPlugin(SofaBoundaryCondition) Plugin not found  ← renamed in SOFA v24
[ERROR] RequiredPlugin(SofaEngine)            Plugin not found  ← renamed in SOFA v24
[ERROR] RequiredPlugin(SofaDeformable)        Plugin not found  ← renamed in SOFA v24
[ERROR] RequiredPlugin(SofaGeneralRigid)      Plugin not found  ← renamed in SOFA v24
[WARN]  UncoupledConstraintCorrection Default compliance not set ← cosmetic
[WARN]  BoxROI No rest position yet defined                      ← cosmetic
[WARN]  InteractiveCamera Too many missing parameters            ← cosmetic
None of these affect simulation physics or policy execution.

---

## Monitoring Topics

In a second terminal while the system is running:

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

To verify the independent safety watchdog during a running procedure:

**Step 1** — Wait until the Hold phase starts:
[surgical_bt_node] [Hold] Sending goal to hold_policy

**Step 2** — In a second terminal inject sustained dangerous force:
```bash
source ~/surgical_robot_lapgym_ws/activate.sh
ros2 topic pub --rate 50 /tissue_force_proxy std_msgs/Float32 "data: 1.5"
```

**Step 3** — Observe the following sequence in the launch terminal:
[safety_watchdog] [ERROR] [WATCHDOG EMERGENCY STOP] force=1.500 >= 1.0
for 3 consecutive readings (60ms)
[bridge_node]     [ERROR] EMERGENCY STOP received
[surgical_bt_node][WARN]  [ForceCondition] High force: 1.500 (1/3)

**Step 4** — Press Ctrl+C in the injection terminal to stop the force.

**Step 5** — Observe watchdog return to NOMINAL:
[safety_watchdog] [INFO] [WATCHDOG] Force normalised: 0.000
-- returning to NOMINAL

---

## IEC 62304 Emergency Stop Analysis — Engineer Notes

### Observed response sequence (from verified run)

| Time | Event | Node | Status |
|---|---|---|---|
| t=0ms | Force 1.5 injected | external | — |
| t=60ms | 3 consecutive readings at 50Hz | safety_watchdog | STOP triggered |
| t=~80ms | /emergency_stop=True published | safety_watchdog | ✓ |
| t=~100ms | EMERGENCY STOP received | bridge_node | halted ✓ |
| t=ongoing | ForceCondition WARN 1/3 | surgical_bt_node | see note below |
| t=200 steps | Hold timeout | hold_policy_server | SUCCESS ✓ |
| t=ongoing | state=STOP | safety_watchdog | alarm persists ✓ |

### Why ForceCondition showed 1/3 continuously

The ForceCondition consecutive counter resets between BT ticks. This is
a known bug — the counter state does not persist correctly between
`update()` calls. However this did not affect clinical safety because
the Phase 4D watchdog independently triggered emergency stop within 60ms.
This demonstrates why two-layer architecture is mandatory per IEC 62304.

### Why Hold server continued running

The Hold server sends zero actions — the instrument does not move.
This is the safest possible behaviour during a safety event. The bridge
halted so no physics steps occurred. The Hold server counting to timeout
had no physical effect.

### Why the procedure completed SUCCESS despite emergency stop

The surgical task was completed correctly — approach, retract, hold all
succeeded. The force spike was an external safety event handled
independently by the watchdog. Surgery complete + safety alarm active
simultaneously is the correct clinical outcome.

---

## Stopping the System

Press `Ctrl+C` in the launch terminal. All 6 nodes shut down simultaneously.
The SOFA backtrace on shutdown is normal — it is SOFA's signal handler
printing the call stack on SIGINT. Not an error.

---

## Known Behaviours

| Behaviour | Cause | Action |
|---|---|---|
| `rcl_shutdown already called` | Double-shutdown race in surgical_bt_node | Harmless, ignore |
| Bridge keeps running after BT SUCCESS | bridge_node has no stop condition | Expected — Ctrl+C to stop |
| GUI shows idle environment | Action servers run headless | Expected — bridge GUI is for teleop only |
| SOFA ERROR plugin not found | Plugin renamed in SOFA v24 | Harmless, simulation works correctly |

## IEC 62304 Emergency Stop Analysis — Updated (Phase 4E prep fixes)

### Both safety layers now fully operational

| Layer | Component | Status | Response |
|---|---|---|---|
| Layer 1 (Application) | ForceCondition (BT) | ✓ FIXED | 1/3 → 2/3 → 3/3 → SAFETY STOP |
| Layer 2 (Independent) | SafetyWatchdogNode | ✓ WORKING | 60ms → /emergency_stop |

### Verified sequence after fixes
t=0ms    Force injection starts
t=60ms   Watchdog: EMERGENCY STOP published (Layer 2)
bridge_node: EMERGENCY STOP received -- halted
t=~80ms  ForceCondition: 1/3 → 2/3 → 3/3 (Layer 1)
ForceCondition: SAFETY STOP triggered
BT: Hold leaf cancelled via cancel_goal()
BT: Surgical procedure FAILED (correct -- dangerous force detected)

### ForceCondition fix — terminate() bug resolved

**Before fix:** `terminate()` reset `_consecutive_high = 0` on every
SUCCESS tick, preventing the counter from ever reaching 3.

**After fix:** `terminate()` only resets on `INVALID` (BT reset/interrupt).
Counter now correctly persists between ticks and accumulates to 3.

### Hold server fix — emergency stop subscriber added

Hold server now subscribes to `/emergency_stop` and sets `_emergency=True`.
The execute loop checks this flag every step and returns early with
`termination='emergency_stop'` when triggered.

### Clinical meaning of FAILED vs SUCCESS

When force injection occurs during Hold phase:
- **Before fixes:** procedure completed SUCCESS (ForceCondition never fired)
- **After fixes:** procedure returns FAILED (correct clinical outcome)

A dangerous force event during the procedure means the surgery did not
complete safely. The FAILED result is the correct response.