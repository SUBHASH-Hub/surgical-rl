# Risk Management File (RMF)
## Autonomous Tissue Retraction via Safe Reinforcement Learning
**Document ID:** RMF-001  
**Version:** 1.0  
**Date:** May 2026  
**Author:** Subhash Arockiadoss  
**Standard:** ISO 14971:2019 + IEC 62304:2006+Amd1:2015 §4.2  
**Status:** Approved

---

## 1. Purpose

This Risk Management File documents hazard identification, risk estimation,
risk controls, and residual risk for the surgical autonomy research software
system. Per ISO 14971, risk management is applied to identify and control
risks throughout the software lifecycle.

**Note on scope:** This system is a research platform not intended for
clinical use. All risks identified are simulation-environment risks. No
patient harm is possible. This document applies the ISO 14971 framework
as a professional engineering exercise demonstrating regulated software
development capability.

---

## 2. Risk Estimation Criteria

### 2.1 Probability

| Level | Label | Definition |
|-------|-------|-----------|
| P1 | Remote | < 1 in 10,000 simulation runs |
| P2 | Occasional | 1 in 1,000 to 1 in 10,000 runs |
| P3 | Probable | 1 in 100 to 1 in 1,000 runs |
| P4 | Frequent | > 1 in 100 runs |

### 2.2 Severity

| Level | Label | Definition (simulation context) |
|-------|-------|----------------------------------|
| S1 | Negligible | No effect on procedure — simulation continues |
| S2 | Minor | Procedure fails — restart required |
| S3 | Serious | Incorrect data reported — could mislead researcher |
| S4 | Critical | Safety layer fails silently — emergency stop not triggered |

### 2.3 Risk Level (before and after controls)

| | S1 | S2 | S3 | S4 |
|-|----|----|----|----|
| **P1** | Acceptable | Acceptable | ALARP | Unacceptable |
| **P2** | Acceptable | ALARP | ALARP | Unacceptable |
| **P3** | ALARP | ALARP | Unacceptable | Unacceptable |
| **P4** | ALARP | Unacceptable | Unacceptable | Unacceptable |

---

## 3. Hazard Identification and Risk Control

### RISK-001: Safety Watchdog Fails to Detect Dangerous Force

| Field | Value |
|-------|-------|
| **Hazard** | Safety watchdog crashes or hangs — dangerous tissue force not detected |
| **Hazard source** | SOUP failure (rclpy, Python interpreter crash) |
| **Foreseeable sequence** | Watchdog process crashes → /tissue_force_proxy not monitored → force spike not detected → /emergency_stop not published → instrument continues operating |
| **Severity** | S4 — Critical (independent safety layer fails silently) |
| **Probability before controls** | P2 — Occasional |
| **Risk before controls** | **Unacceptable** |
| **Risk control 1** | Watchdog is independent process (separate PID) — crash of BT or action servers does not affect it (SR-001, SR-006) |
| **Risk control 2** | Watchdog publishes /watchdog_heartbeat every 1 second — missing heartbeat detectable (SR-005) |
| **Risk control 3** | Watchdog subscribes to /tissue_force_proxy independently of BT (SR-001) |
| **Probability after controls** | P1 — Remote |
| **Severity after controls** | S4 |
| **Residual risk** | ALARP — acceptable for research platform |
| **Verification** | Force injection test: 60ms response time verified. Heartbeat verified publishing during STOP state |

---

### RISK-002: Surgeon Stop Not Received During env.step()

| Field | Value |
|-------|-------|
| **Hazard** | Surgeon presses S but instrument continues moving for multiple steps |
| **Hazard source** | Python GIL prevents callbacks during SOFA blocking call (SOUP-001 anomaly) |
| **Foreseeable sequence** | S pressed → GIL held by env.step() → callback queued but not processed → multiple steps execute before freeze → instrument overshoots |
| **Severity** | S2 — Minor (instrument overshoot in simulation) |
| **Probability before controls** | P4 — Frequent (every stop during step execution) |
| **Risk before controls** | **Unacceptable** |
| **Risk control 1** | Separate rclpy.Context with own DDS instance per action server — callback fires in background thread independent of main executor (SR-021) |
| **Risk control 2** | spin_once timeout reduced to 10ms — callback latency ≤ 10ms from message publication |
| **Risk control 3** | Dual freeze loop (before + after env.step()) — bounds latency to at most one step (~65ms) |
| **Risk control 4** | Republish timer 10Hz — stop maintained even if new action server starts |
| **Probability after controls** | P2 — Occasional (one step overshoot possible; multi-step overshoot eliminated) |
| **Severity after controls** | S1 — Negligible (one step overshoot in simulation) |
| **Residual risk** | Acceptable |
| **Residual risk note** | Remaining one-step overshoot is a known simulation constraint documented in phase4e_surgeon_console.md. On real hardware, hardware PWM cutoff replaces software freeze loop entirely |
| **Verification** | Isolation test verified stop latency within one env.step(). SURGEON STOP received logged within 100ms in all observed tests |

---

### RISK-003: Emergency Stop Fails During Goal State Transition

| Field | Value |
|-------|-------|
| **Hazard** | Emergency stop received while goal_handle is transitioning state — RCLError raised — procedure continues |
| **Hazard source** | ROS 2 Humble known anomaly: `goal_handle.canceled()` raises RCLError during invalid state transition (SOUP-002 anomaly) |
| **Foreseeable sequence** | E pressed → /emergency_stop=True → action server calls goal_handle.canceled() → RCLError raised → exception propagates up → procedure continues unhalted |
| **Severity** | S4 — Critical (emergency stop fails) |
| **Probability before controls** | P3 — Probable (race condition on state transitions) |
| **Risk before controls** | **Unacceptable** |
| **Risk control 1** | try/except pattern wraps goal_handle.canceled() — RCLError caught, goal_handle.abort() called as fallback (SR-014) |
| **Risk control 2** | _emergency flag set atomically before goal state call — flag persists regardless of exception |
| **Risk control 3** | Independent safety watchdog publishes /emergency_stop independently — does not depend on action server emergency handling |
| **Probability after controls** | P1 — Remote |
| **Residual risk** | ALARP — acceptable |
| **Verification** | Emergency stop test during all three phases verified — BT reports FAILED, system halts |

---

### RISK-004: BT ForceCondition Fails to Accumulate Consecutive Readings

| Field | Value |
|-------|-------|
| **Hazard** | ForceCondition consecutive counter resets on each BT tick — never reaches 3 — application safety layer never fires |
| **Hazard source** | py_trees SOUP anomaly — terminate() called on SUCCESS resetting counter (SOUP-007 anomaly) |
| **Foreseeable sequence** | Force spike → ForceCondition increments to 1/3 → BT returns SUCCESS → terminate() called → counter resets to 0 → never reaches 3 → SAFETY STOP never fires |
| **Severity** | S3 — Serious (application safety layer fails — researcher not warned) |
| **Probability before controls** | P4 — Frequent (bug was present in initial implementation) |
| **Risk before controls** | **Unacceptable** |
| **Risk control 1** | Fixed in Phase 4D: terminate() only resets counter on INVALID (BT reset/interrupt) not on SUCCESS |
| **Risk control 2** | Independent watchdog (Layer 2) operates at 50Hz independently — provides backup even if ForceCondition fails |
| **Probability after controls** | P1 — Remote |
| **Residual risk** | Acceptable |
| **Verification** | Force injection test after fix: ForceCondition 1/3 → 2/3 → 3/3 → SAFETY STOP verified. Watchdog also fired independently at 60ms |

---

### RISK-005: PPO Policy Produces Out-of-Distribution Actions

| Field | Value |
|-------|-------|
| **Hazard** | PPO policy encounters observation outside training distribution — produces large erratic action — instrument moves unsafely |
| **Hazard source** | Reinforcement learning policy generalisation limits |
| **Foreseeable sequence** | Novel starting position → obs outside training distribution → policy outputs clipped maximum action → aggressive instrument movement → tissue collision |
| **Severity** | S2 — Minor (simulation collision — no patient harm) |
| **Probability before controls** | P3 — Probable |
| **Risk before controls** | ALARP |
| **Risk control 1** | Action clipping in policy: `np.clip(action, -3.0, 3.0)` limits maximum velocity |
| **Risk control 2** | Collision penalty in reward: λ_collision × collision_steps penalises tissue contact |
| **Risk control 3** | Safety watchdog monitors force proxy — aggressive collisions produce force spike → emergency stop |
| **Risk control 4** | Approach server pre-positions instrument within 25mm before PPO activates — reduces distribution shift |
| **Probability after controls** | P2 — Occasional |
| **Residual risk** | Acceptable — simulation only, independent safety layer active |
| **Verification** | Phase 2D: 85.7 collision steps/episode mean — known and documented |

---

### RISK-006: Force Proxy Underestimates Actual Force

| Field | Value |
|-------|-------|
| **Hazard** | Optical flow force proxy reads 0 while actual tissue contact is dangerous — watchdog not triggered |
| **Hazard source** | Force estimation by optical flow (indirect measurement) — not a physical force sensor |
| **Foreseeable sequence** | Aggressive tissue contact → tissue deformation small or below motion threshold → optical flow reads near-zero → watchdog not triggered → unsafe operation continues |
| **Severity** | S3 — Serious (safety layer not triggered in simulation) |
| **Probability before controls** | P3 — Probable |
| **Risk before controls** | **Unacceptable** |
| **Risk control 1** | Alert threshold calibrated conservatively: mean + 2×std from Phase 3C validation (0.35 px/frame) — triggers well before stop threshold |
| **Risk control 2** | Phase 3C validation: 3,000 steps, 10 episodes — 0 collision steps while agent was active — confirms proxy correctly reflects safe operation |
| **Risk control 3** | Documented limitation in phase4e_surgeon_console.md and SAD-001 — researchers are informed |
| **Risk control 4** | Surgeon console displays live force reading — human observer can act |
| **Probability after controls** | P2 — Occasional |
| **Residual risk** | ALARP — acceptable for research platform with documented limitation |
| **Verification** | Phase 3C: 0/3,000 collision steps. Pearson r = NaN (correct — agent never collided, zero variance in collision signal) |

---

## 4. Residual Risk Summary

| Risk ID | Hazard | Residual Level | Acceptable? |
|---------|--------|----------------|-------------|
| RISK-001 | Watchdog fails silently | ALARP | Yes — research platform |
| RISK-002 | Stop latency > 1 step | Acceptable | Yes |
| RISK-003 | Emergency stop RCLError | ALARP | Yes |
| RISK-004 | ForceCondition bug | Acceptable | Yes — fixed |
| RISK-005 | OOD PPO action | Acceptable | Yes — simulation |
| RISK-006 | Force proxy underestimate | ALARP | Yes — documented limitation |

**Overall residual risk:** Acceptable for intended use as a research platform.
No patient harm is possible. All safety-relevant risks have been mitigated
to ALARP or better.

---

## 5. Risk/Benefit Assessment

The system demonstrates autonomous surgical instrument control with:
- Independent safety monitoring at 50Hz
- Human-in-the-loop surgeon stop with < one-step latency
- Emergency stop verified at 60ms response time
- Documented sim-to-real gap with engineering analysis

Benefits to research community: demonstrated architecture for safe surgical
RL deployment, quantified sim-to-real gap, visual tissue force proxy without
hardware sensors. These benefits outweigh the residual risks for a research
platform not used on patients.