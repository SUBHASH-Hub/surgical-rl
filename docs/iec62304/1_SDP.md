# Software Development Plan (SDP)
## Autonomous Tissue Retraction via Safe Reinforcement Learning
**Document ID:** SDP-001  
**Version:** 1.0  
**Date:** May 2026  
**Author:** Subhash Arockiadoss  
**Standard:** IEC 62304:2006+Amd1:2015  
**Status:** Approved

---

## 1. Purpose and Scope

This Software Development Plan (SDP) describes the development lifecycle,
methods, tools, and standards applied to the surgical autonomy software
system built as part of the project:

**"Autonomous Tissue Retraction via Safe Reinforcement Learning"**

The system autonomously controls a laparoscopic instrument within SOFA FEM
physics simulation to perform tissue retraction for gallbladder surgery
training and research.

**Intended use:**
Research software for autonomous tissue retraction training and demonstration
in SOFA laparoscopic simulation. Not intended for clinical use on human
patients.

**Out of scope:**
Clinical deployment, use on human patients, CE marking, FDA 510(k) submission.

---

## 2. Safety Classification

Per IEC 62304 §4.3 software safety classification:

| Software Item | Classification | Justification |
|---------------|---------------|---------------|
| safety_watchdog_node | **Class C** | Independent safety monitor — failure could result in undetected dangerous tissue force; injury possible if this item fails silently |
| surgical_bt_node | **Class C** | Orchestrates autonomous surgical instrument motion — incorrect sequencing could cause uncontrolled movement |
| retract_policy_server | **Class C** | Controls autonomous instrument motion via PPO policy — direct actuator control |
| approach_policy_server | **Class C** | Controls autonomous instrument approach — direct actuator control |
| hold_policy_server | **Class B** | Zero-action hold — failure results in no motion, not dangerous motion |
| surgeon_console | **Class B** | Human interface — failure degrades oversight but independent watchdog continues |
| sofa_bridge_node | **Class B** | Physics bridge — failure stops simulation but not a clinical risk |

**System-level classification: Class C**
Any Class C item elevates the system to Class C per IEC 62304 §4.3(c).

---

## 3. Development Lifecycle Model

This project follows an **iterative phase-gate model** with retrospective
documentation. Each phase was completed and verified before the next began.

```
Phase 1 → Phase 2 → Phase 3 → Phase 4A → 4B → 4C → 4D → 4E
  ↑           ↑         ↑          ↑
Gate review: results verified, git tag applied, docs written
```

Retrospective documentation: IEC 62304 compliance documentation was
produced retrospectively against the completed system. This is an accepted
practice for research prototypes being brought into regulated frameworks.
All design decisions are documented as-built.

---

## 4. Development Activities

### 4.1 Requirements Analysis
- Clinical requirements identified from laparoscopic cholecystectomy literature
- System requirements derived and numbered in SRS-001
- Safety requirements identified via ISO 14971 risk analysis (RMF-001)
- Requirements reviewed against Phase 4 architecture before implementation

### 4.2 Architecture Design
- System decomposed into 7 independent ROS 2 nodes
- Safety architecture designed with two independent layers per IEC 62304
- Architecture documented in SAD-001
- Key design decisions documented with rationale in phase docs

### 4.3 Detailed Design
- Each node designed with single responsibility principle
- Interfaces defined via ROS 2 action/topic/service contracts
- Custom action interface defined in `lapgym_interfaces/action/Retract.action`
- Surgeon stop pattern designed with separate rclpy.Context per server

### 4.4 Implementation
- Primary language: Python 3.10
- ROS 2 Humble as middleware framework
- Version control: Git with tagged releases per phase
- Repository: github.com/SUBHASH-Hub/surgical-rl

### 4.5 Verification and Validation
- Unit-level: isolation tests per node (documented in phase docs)
- Integration: full system launch with verified procedure completion
- Safety: force injection test verifying 60ms emergency stop response
- Results documented in TRACEABILITY-001

---

## 5. Development Tools

| Tool | Version | Purpose | Qualified? |
|------|---------|---------|-----------|
| Git | 2.x | Version control, configuration management | Yes — industry standard |
| Python | 3.10.12 | Primary development language | Yes — pinned version |
| ROS 2 Humble | 2022.x | Middleware framework | Yes — LTS release |
| SOFA | v25.12.00 | FEM physics simulation | Yes — pinned version |
| Ubuntu | 22.04 LTS | Development/execution platform | Yes — LTS release |
| Weights & Biases | 0.25.1 | Training metrics logging | No — research use only |
| VS Code | Latest | IDE | No — development only, not in product |

---

## 6. Configuration Management

### 6.1 Version Control
All source code is managed in Git at github.com/SUBHASH-Hub/surgical-rl.

**Branch strategy:**
- `main` — stable, tagged releases only
- Feature development committed directly to main for research prototype

**Tagging convention:**
| Tag | Component | Status |
|-----|-----------|--------|
| v1.0-phase1-complete | Simulation baseline | Released |
| v2.4-phase2-complete | PPO training | Released |
| v3.4-phase3d-complete | Perception pipeline | Released |
| v4.0-phase4a-complete | ROS2 bridge | Released |
| v4.1-phase4b-complete | Action servers | Released |
| v4.2-phase4c-complete | Safety watchdog | Released |
| v4.4-phase4d-complete | Behaviour tree | Released |
| v4.5-phase4e-complete | Surgeon console | Released |
| v4.7-phase4e-iec62304 | IEC 62304 artifacts | This release |

### 6.2 Dependency Management
All Python dependencies pinned in `requirements.txt`.
ROS 2 packages pinned to Humble LTS.
SOFA version pinned to v25.12.00.
SOUP analysis documented in SOUP-001.

---

## 7. Problem Resolution

Defects identified during development were root-caused and documented in
phase result files before proceeding to the next phase:

| Phase | Problem | Root Cause | Resolution |
|-------|---------|-----------|-----------|
| Phase 2A | Agent observation-blind | goal_xyz missing from obs | Added to 7D observation |
| Phase 2C | Reward collapse at λ=0.8 | Curriculum too aggressive | Reduced λ_max to 0.5 |
| Phase 4E | SURGEON STOP not firing during env.step() | Python GIL blocks callbacks | Separate rclpy.Context per server |
| Phase 4E | Emergency stop RCLError | Invalid goal state transition | try/except with abort() fallback |
| Phase 4C | ForceCondition counter reset bug | terminate() called on SUCCESS | Fixed: reset only on INVALID |
| Phase 3C | Optical flow always zero | Python object aliasing — same array pointer | Added .copy() on frame store |

All problems resolved before git tag applied. No open defects at v4.5-phase4e-complete.

---

## 8. Document Control

| Document | ID | Version | Status |
|---------|-----|---------|--------|
| Software Development Plan | SDP-001 | 1.0 | Approved |
| Software Requirements Specification | SRS-001 | 1.0 | Approved |
| Software Architecture Document | SAD-001 | 1.0 | Approved |
| SOUP Analysis | SOUP-001 | 1.0 | Approved |
| Risk Management File | RMF-001 | 1.0 | Approved |
| Requirements Traceability Matrix | TRACEABILITY-001 | 1.0 | Approved |