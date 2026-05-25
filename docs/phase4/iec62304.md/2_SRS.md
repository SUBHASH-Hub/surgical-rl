# Software Requirements Specification (SRS)
## Autonomous Tissue Retraction via Safe Reinforcement Learning
**Document ID:** SRS-001  
**Version:** 1.0  
**Date:** May 2026  
**Author:** Subhash Arockiadoss  
**Standard:** IEC 62304:2006+Amd1:2015 §5.2  
**Status:** Approved

---

## 1. Purpose

This document specifies all functional and safety requirements for the
surgical autonomy research software system. Requirements are numbered
for traceability to design, implementation, and test evidence.

---

## 2. System Context

The system performs autonomous tissue retraction of gallbladder tissue in
SOFA FEM simulation. It is intended as a research platform demonstrating
safe RL-based surgical autonomy with human-in-the-loop oversight.

**Users:** Research engineers, robotics engineers evaluating surgical autonomy architectures.

**Operating environment:** Ubuntu 22.04, ROS 2 Humble, SOFA v25.12, GTX 1650.

---

## 3. Functional Requirements

### 3.1 Simulation Interface

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR-001 | The system shall connect to SOFA TissueRetractionV2 environment and reset it on startup | Must | Phase 4A |
| FR-002 | The system shall publish instrument position on `/joint_states` at ≥ 50Hz | Must | Phase 4A |
| FR-003 | The system shall publish tissue force proxy on `/tissue_force_proxy` at ≥ 50Hz | Must | Phase 4A, 3C |
| FR-004 | The system shall publish camera image on `/camera/image_raw` at ≥ 50Hz | Should | Phase 4A |
| FR-005 | The system shall accept instrument delta commands on `/joint_target` for teleoperation | Must | Phase 4A |
| FR-006 | The system shall support RenderMode.HEADLESS for automated runs | Must | Phase 4A |
| FR-007 | The system shall support RenderMode.HUMAN for visual verification | Should | Phase 4E |

### 3.2 Autonomous Approach

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR-010 | The system shall navigate the instrument to within 25mm of the grasping target using a proportional controller | Must | Phase 4B |
| FR-011 | The approach server shall complete navigation within 400 steps | Must | Phase 4B |
| FR-012 | The approach server shall publish step feedback every step | Must | Phase 4E |
| FR-013 | The grasping target shall be at world position [-0.0486, 0.0085, 0.0356] metres | Must | Phase 4B |

### 3.3 Autonomous Retraction

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR-020 | The system shall retract tissue to the end position using the Phase 2D PPO policy | Must | Phase 4B |
| FR-021 | The retract server shall achieve final distance ≤ 3mm from end position | Must | Phase 2D |
| FR-022 | The retract server shall complete within 300 steps | Must | Phase 4B |
| FR-023 | The retract server shall publish step feedback every step | Must | Phase 4E |
| FR-024 | The retract server shall load checkpoint from `logs/checkpoints/phase2_ppo_tissue_retraction_20260409_211946/ppo_tissue_final` | Must | Phase 4B |

### 3.4 Hold Phase

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR-030 | The system shall hold instrument at current position by publishing zero-delta actions | Must | Phase 4B |
| FR-031 | The hold server shall maintain position for up to 500 steps | Must | Phase 4B |
| FR-032 | The hold server shall report success on preemption (hold is always successful) | Must | Phase 4B |

### 3.5 Task Orchestration

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR-040 | The system shall execute Approach → Retract → Hold in sequence | Must | Phase 4C |
| FR-041 | The BT shall tick at 10Hz | Must | Phase 4C |
| FR-042 | The BT shall report SUCCESS when all three phases complete | Must | Phase 4C |
| FR-043 | The BT shall report FAILURE if any phase returns failure | Must | Phase 4C |
| FR-044 | The BT shall publish phase/step/distance on `/console_feedback` every BT tick | Must | Phase 4E |

### 3.6 Human Interface

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR-050 | The surgeon console shall display: phase, step, distance, force, watchdog status, ESTOP status | Must | Phase 4E |
| FR-051 | The surgeon console shall refresh at ≥ 10Hz | Must | Phase 4E |
| FR-052 | The surgeon console shall provide S key for surgeon stop | Must | Phase 4E |
| FR-053 | The surgeon console shall provide R key for surgeon resume | Must | Phase 4E |
| FR-054 | The surgeon console shall provide E key for emergency stop | Must | Phase 4E |
| FR-055 | The surgeon console shall provide Q key for clean exit | Must | Phase 4E |
| FR-056 | The surgeon console shall log all S/R/E events with timestamp, phase, step, distance | Must | Phase 4E |

---

## 4. Safety Requirements

### 4.1 Force Monitoring

| ID | Requirement | Class | Source |
|----|-------------|-------|--------|
| SR-001 | The safety watchdog shall monitor `/tissue_force_proxy` at ≥ 50Hz independent of all other nodes | **C** | IEC 62304, Phase 4D |
| SR-002 | The safety watchdog shall enter ALERT state when force ≥ 0.35 px/frame | **C** | Phase 3C calibration |
| SR-003 | The safety watchdog shall publish `/emergency_stop=True` when force ≥ 1.0 px/frame for 3 consecutive readings (60ms) | **C** | Phase 4D |
| SR-004 | The 3-consecutive-reading requirement shall prevent false positives from transient noise spikes | **B** | Phase 4D design |
| SR-005 | The safety watchdog shall publish `/watchdog_heartbeat` every 1 second to prove liveness | **B** | Phase 4D |
| SR-006 | The safety watchdog process shall survive crash of the BT node | **C** | IEC 62304 independence |

### 4.2 Emergency Stop

| ID | Requirement | Class | Source |
|----|-------------|-------|--------|
| SR-010 | All action servers shall subscribe to `/emergency_stop` independently | **C** | Phase 4E |
| SR-011 | All action servers shall halt execution within one env.step() cycle (~65ms) of receiving `/emergency_stop=True` | **C** | Phase 4E |
| SR-012 | The sofa_bridge_node shall halt simulation on `/emergency_stop=True` | **C** | Phase 4D |
| SR-013 | The emergency stop state shall persist until system restart — no automatic reset | **C** | Phase 4D, clinical safety |
| SR-014 | Emergency stop shall work correctly when received during goal state transition (try/except pattern) | **C** | Phase 4E fix |

### 4.3 Surgeon Stop

| ID | Requirement | Class | Source |
|----|-------------|-------|--------|
| SR-020 | The surgeon stop shall freeze instrument motion within one env.step() cycle of S key press | **B** | Phase 4E |
| SR-021 | Surgeon stop shall use a separate rclpy.Context to receive callbacks during env.step() | **B** | Phase 4E design |
| SR-022 | The system shall resume from the exact step where stop was applied — no reset | **B** | Phase 4E |
| SR-023 | Surgeon stop shall be republished at 10Hz while active to prevent race conditions | **B** | Phase 4E |
| SR-024 | Resume (R) shall be blocked when emergency stop is active | **B** | Phase 4E |

### 4.4 Preemption

| ID | Requirement | Class | Source |
|----|-------------|-------|--------|
| SR-030 | All action servers shall check `is_cancel_requested` as the first operation every step | **C** | Phase 4B |
| SR-031 | Maximum preemption latency shall be bounded by one env.step() cycle | **C** | Phase 4B |
| SR-032 | All action servers shall accept CancelResponse.ACCEPT unconditionally | **C** | Phase 4B |

---

## 5. Performance Requirements

| ID | Requirement | Value | Source |
|----|-------------|-------|--------|
| PR-001 | Safety watchdog check rate | ≥ 50Hz | IEC 62304 |
| PR-002 | Force proxy publication rate | ≥ 50Hz | Phase 4A |
| PR-003 | BT tick rate | 10Hz | Phase 4C |
| PR-004 | Surgeon console refresh rate | ≥ 10Hz | Phase 4E |
| PR-005 | Emergency stop response | ≤ 100ms from force threshold to halt | Phase 4D |
| PR-006 | Approach completion | ≤ 400 steps | Phase 4B |
| PR-007 | Retract completion | ≤ 300 steps | Phase 4B |

---

## 6. Interface Requirements

| ID | Requirement | Source |
|----|-------------|--------|
| IR-001 | Custom action interface defined in `lapgym_interfaces/action/Retract.action` | Phase 4B |
| IR-002 | All inter-node communication via ROS 2 topics, actions, or services | Phase 4 arch |
| IR-003 | Single launch command shall start entire system | Phase 4C |
| IR-004 | System shall source activate.sh before launch | Phase 4A |

---

## 7. Requirements Not Met (Documented Limitations)

| ID | Requirement | Status | Reason |
|----|-------------|--------|--------|
| NM-001 | Stop latency < 5ms (da Vinci standard) | Not met | SOFA env.step() ~65ms blocking — hardware limitation |
| NM-002 | Control loop at 1kHz | Not met | SOFA FEM physics ~15Hz on GTX 1650 |
| NM-003 | Real F/T sensor integration | Not met | Simulation only — optical flow proxy used |
| NM-004 | Clinical deployment readiness | Not in scope | Research platform only |