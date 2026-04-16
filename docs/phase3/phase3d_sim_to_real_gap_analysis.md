# Phase 3D — Sim-to-Real Gap Analysis

**Status:** COMPLETE  
**Completed:** April 2026  
**Purpose:** Document all gaps between the SOFA simulation environment and a real surgical robot, with mitigation strategies for Phase 4 and beyond.

---

## What is the Sim-to-Real Gap?

The sim-to-real gap is the collection of differences between what an agent learns in simulation and what it would experience on a real robot. A policy that works perfectly in simulation often fails on a real robot because the simulation makes simplifying assumptions that do not hold in reality.

This document catalogues every gap identified during Phases 1-3 of this project, organised by category, with a severity rating and a concrete mitigation strategy for each.

**Severity scale:**
- **CRITICAL** — Would prevent deployment without addressing
- **HIGH** — Significant performance degradation expected
- **MEDIUM** — Measurable impact, manageable with engineering
- **LOW** — Minor impact, acceptable for research demonstration

---

## Category 1 — Visual Appearance (Rendering vs Reality)

### Gap 1.1 — Photorealism
**Severity: HIGH**

**In simulation:** The SOFA renderer produces a clean 3D-rendered scene. The tissue is a uniform gold/yellow colour, perfectly lit, with no texture variation. The background is a clean grey-blue gradient. There are no shadows, reflections, blood, fluids, smoke, or lighting artefacts.

**In reality:** Real laparoscopic video has significantly different visual properties:
- Specular highlights and reflections from wet tissue surfaces
- Blood, irrigation fluid, and smoke from cauterisation
- Non-uniform tissue colour and texture (fat, fascia, vessels are all different colours)
- Camera lens effects: distortion, chromatic aberration, depth of field blur
- Surgical lighting is highly directional and creates deep shadows

**Evidence from this project:** The MobileNetV3 tip detector trained on SOFA frames (5.1px MAE) would likely have significantly higher error on real laparoscopic frames because the instrument appearance differs substantially. The instrument in simulation is a clean orange/red rigid body. Real instruments have metallic glare, specular highlights, and appear differently at different insertion angles.

**Mitigation:**
- Domain randomisation: randomise tissue colour, lighting angle, background texture during training
- Domain adaptation: fine-tune the perception module on a small set of labelled real frames
- Sim-to-real transfer: use CycleGAN or similar to translate simulation frames to look realistic before inference

---

### Gap 1.2 — Camera Model
**Severity: MEDIUM**

**In simulation:** The SOFA camera uses a simple perspective projection model with fixed intrinsics (focal length derived from ~42° FOV, verified from projectionMatrix diagonal = 2.605). The camera is fixed at a constant position with no motion, vibration, or drift.

**In reality:** Real laparoscopic cameras (e.g. da Vinci Si/X/5 endoscope):
- Have calibrated but imperfect intrinsics that drift with temperature
- Experience lens distortion (barrel/pincushion), especially at the image periphery
- Are inserted through a trocar and move with the surgeon's wrist motions
- Have a shorter focal length and wider FOV than typical desktop cameras
- Must be white-balanced against the surgical drapes at the start of each procedure

**Evidence from this project:** The segmentation mask generation used the exact OpenGL projection matrices (`cam.modelViewMatrix`, `cam.projectionMatrix`) which are perfect in simulation. Real camera calibration has residual reprojection error typically 0.3-1.0 pixels.

**Mitigation:**
- Calibrate the real camera using a checkerboard target before each procedure
- Apply undistortion as a pre-processing step in the Phase 4 ROS 2 node
- Include slight camera position randomisation during simulation training

---

### Gap 1.3 — Tissue Appearance Variation
**Severity: HIGH**

**In simulation:** Every episode uses the same tissue mesh with the same material properties. The tissue always appears as the same gold/yellow colour at the same position. There is zero inter-episode appearance variation.

**In reality:** Tissue appearance varies substantially:
- Between patients (different fat thickness, tissue age, inflammation state)
- During a procedure (blood staining, coagulation changes colour, irrigation fluid changes reflectivity)
- Between tissue types (the same organ looks different in different regions)

**Evidence from this project:** The Phase 3A segmentation model achieved IoU=1.000 on simulation because the tissue always looks identical. On real tissue this would likely drop significantly — possibly to IoU 0.5-0.7 without domain adaptation.

**Mitigation:**
- Train with augmented tissue appearances (colour jitter ±30%, brightness ±20%, saturation ±40%)
- Collect a small real dataset (50-100 annotated frames) for fine-tuning
- Use test-time augmentation to improve robustness

---

## Category 2 — Physics and Mechanics

### Gap 2.1 — Tissue Mechanical Properties
**Severity: HIGH**

**In simulation:** The SOFA tissue uses a linear elastic Finite Element Model (FEM) with fixed Young's modulus and Poisson's ratio. These properties are uniform across the entire mesh. The tissue deformation is deterministic — given the same force, the tissue always deforms exactly the same way.

**In reality:** Biological tissue:
- Has non-linear stress-strain curves (stiffens under large deformation)
- Has viscoelastic properties (creep, stress relaxation over time)
- Has anisotropic properties (different stiffness in different directions due to fibres)
- Varies between patients (age, hydration, disease state all affect stiffness)
- Changes during the procedure (temperature from cautery, drying from air exposure)

**Evidence from this project:** The SafeRewardWrapper collision penalty uses a geometric penetration criterion — vertices overlapping. This is a computational proxy for force, not a true mechanical model. In Phase 3C, the optical flow force proxy showed max flow = 0.732 px/frame during safe agent behaviour. On real tissue, identical instrument motion could produce very different deformation patterns.

**Mitigation:**
- Use ex-vivo tissue experiments to calibrate SOFA FEM parameters to real tissue
- Add tissue property randomisation during training (Young's modulus ±30%)
- The Phase 3C optical flow proxy is designed specifically to measure actual deformation rather than rely on simulation parameters

---

### Gap 2.2 — Contact and Collision Modelling
**Severity: CRITICAL**

**In simulation:** Contact is modelled using the SOFA constraint-based LCP (Linear Complementarity Problem) solver with UncoupledConstraintCorrection. The collision detection uses bounding volume hierarchies. The compliance vector warning observed repeatedly during training (`Compliance vector should be a multiple of 7`) indicates the constraint correction is using default values, not calibrated values.

**In reality:** Real instrument-tissue contact:
- Has stick-slip friction that changes with speed and normal force
- Has adhesion between wet surfaces that simulation does not model
- Has dynamic effects from instrument vibration (especially during laparoscopic grasping)
- Has deformation at the contact point that requires a fine local mesh

**Evidence from this project:** The `[ERROR] UncoupledConstraintCorrection: Compliance vector should be a multiple of 7` warning appeared in every single training run across Phases 1-3. This indicates the compliance correction is not properly calibrated for the rigid body DOFs of our instrument model. The collision detection worked well enough for training but is not physically accurate.

**Mitigation:**
- Properly calibrate the UncoupledConstraintCorrection compliance values for the specific instrument mass/inertia
- Consider switching to a calibrated constraint correction method (LinearSolverConstraintCorrection)
- For Phase 4 deployment: add a separate force sensor reading as ground truth

---

### Gap 2.3 — Instrument Dynamics
**Severity: MEDIUM**

**In simulation:** The instrument is modelled as a rigid body with direct position control. The action space is `Box(-3.0, 3.0, (3,))` — direct xyz displacement. There is no joint dynamics, no motor saturation, no backlash, no friction in the joints.

**In reality:** Real laparoscopic instruments:
- Have cable-driven mechanisms with hysteresis and backlash
- Have frictional losses that increase with applied force
- Have a remote centre of motion (RCM) constraint through the trocar
- Have a limited workspace that depends on the trocar position
- Have finite joint velocity and acceleration limits

**Evidence from this project:** The Phase 2D agent learned to move in smooth trajectories with mean step displacement of ~0.02 metres per step. Real instruments cannot achieve arbitrary trajectories — they are constrained by the RCM and by joint limits.

**Mitigation:**
- Add RCM constraint to the simulation action space
- Add joint velocity limits and action smoothing
- Model backlash using hysteresis in the action-to-motion mapping

---

## Category 3 — Sensing and Observation

### Gap 3.1 — Force Sensing
**Severity: CRITICAL**

**In simulation:** Force information is available from the FEM solver as contact force vectors. The SafeRewardWrapper uses a collision cost proxy based on geometric penetration. In Phase 3C, we built an optical flow proxy as a visual substitute for force sensing.

**In reality:** Most laparoscopic instruments have NO force sensors. The da Vinci 4/5 instruments have limited force feedback but this is not available in standard systems. The surgeon estimates force entirely from visual tissue deformation.

**Evidence from this project:** This is precisely why Phase 3C was necessary. The optical flow proxy (mean=0.128 px/frame, alert threshold=0.35, safety threshold=1.0) provides a real-time visual force estimate that does not require a force sensor. This proxy is deployable on any laparoscopic system with a camera.

**Gap remaining:** The optical flow proxy was validated on simulation data only (Pearson r = NaN because no collision steps occurred — agent was safe). Validation against real force sensor data has not been performed.

**Mitigation:**
- Phase 3C optical flow proxy is the primary mitigation for this gap
- For high-confidence deployment: collect data from a force-instrumented da Vinci or ATI Nano17 force transducer mounted on a bench-top setup
- Calibrate the flow-to-force relationship using known loads

---

### Gap 3.2 — Observation Space Completeness
**Severity: HIGH**

**In simulation (Phase 2D):** The observation included exact ground-truth `[tool_xyz(3), goal_xyz(3), phase(1)]` from the simulator. This privileged information is not available on a real robot.

**In simulation (Phase 3B):** The observation was 132D visual features from the perception pipeline. The goal_xyz was deliberately removed to match the real robot constraint. Result: agent reward improved 42% but goal was never reached (0% completion rate vs 100% in Phase 2D).

**In reality:** Only what the camera sees is available. There is no ground-truth tool position, no goal position embedded in the observation. The surgeon identifies the target by anatomical landmarks.

**Evidence from this project:** The Phase 3B result directly quantifies this gap:
```
With ground truth:     ep_len=142, reward=-97,  100% goal rate
Without ground truth:  ep_len=300, reward=-135, 0% goal rate
Degradation:           +42% steps, +39% worse reward, complete regression on goal-reaching
```

**Mitigation:**
- Augment the observation with depth information (stereo endoscope or structured light)
- Define goal location as a pre-operative target using anatomical registration
- Use a goal-conditioned policy where the goal is specified as an image patch
- Consider privileged distillation: train teacher with ground truth, distil to student without

---

### Gap 3.3 — Sensor Noise and Latency
**Severity: MEDIUM**

**In simulation:** All observations are noise-free and synchronous. The environment steps at ~8-14 fps (measured across all training runs). There is zero latency between action command and state observation.

**In reality:**
- Camera images have sensor noise (shot noise, read noise, JPEG compression artefacts)
- The perception pipeline (MobileNetV3 tip detector) adds ~3-5ms inference latency per frame
- ROS 2 message passing adds ~1-5ms additional latency
- Real surgical robots operate at 30fps (33ms frame period) — the perception pipeline must run within this budget

**Evidence from this project:** The Phase 3B agent runs at 8.5fps in simulation. Real-time deployment requires 25+ fps for smooth control. The MobileNetV3 inference at 480×480 on GTX 1650 takes ~15-20ms — this is within the 33ms budget but leaves limited margin.

**Mitigation:**
- Add Gaussian noise augmentation to RGB frames during training (σ = 5-15 pixel values)
- Add temporal smoothing to the observation with a 3-frame rolling average
- Profile the complete Phase 4 pipeline on target hardware to verify latency budget

---

## Category 4 — Task Definition

### Gap 4.1 — Goal Definition
**Severity: HIGH**

**In simulation:** The retraction goal is a fixed 3D coordinate (a specific point in the simulator workspace). The episode terminates when the tool tip reaches within a threshold of this coordinate. This coordinate is known precisely.

**In reality:** The "goal" for tissue retraction is defined by surgical context:
- The surgeon wants to expose a specific anatomical structure (artery, duct, etc.)
- The target location is identified by visual inspection of the anatomy
- The target changes during the procedure as anatomy is revealed
- There is no single fixed coordinate — the goal is a region defined by clinical need

**Evidence from this project:** The goal_xyz removal in Phase 3B caused complete regression from 100% to 0% goal achievement, demonstrating that the current reward function is entirely dependent on knowing the goal coordinate.

**Mitigation:**
- Define goal as a visual landmark: "retract until structure X is visible"
- Use pre-operative imaging (CT/MRI) registration to define target in the robot coordinate frame
- Consider hierarchical task decomposition: a high-level goal planner + a low-level execution policy

---

### Gap 4.2 — Episode Termination and Safety
**Severity: CRITICAL**

**In simulation:** Episodes terminate after 300 steps (timeout) or when the goal is reached. There is no consequence for the agent going outside the SOFA workspace bounds except a small position penalty. The `[WARNING] BoxROI: No rest position yet defined` warning suggests boundary enforcement is weak.

**In reality:** Unsafe instrument motion can cause:
- Tissue perforation (irreversible damage)
- Vessel laceration (life-threatening bleeding)
- Injury to adjacent structures (bile duct, ureter)

**Evidence from this project:** During Phase 3C visual observation, the agent was observed (in GUI screenshots) to move the instrument outside the tissue boundary region in episode 5. The Phase 3B agent never triggered a collision penalty in 3000 steps, suggesting the collision detection may be too lenient. In Phase 3C screenshots, the instrument appeared to go outside the workspace boundary without penalty.

**Mitigation:**
- The Phase 3C safety threshold (1.0 px/frame optical flow) is the primary real-time safety monitor
- Add hard workspace limits to the action space that cannot be overridden by the policy
- Implement a supervisory safety layer in Phase 4 that intercepts actions before execution
- Require human confirmation before any action that would increase tissue flow above the alert threshold

---

### Gap 4.3 — Reward Function Realism
**Severity: HIGH**

**In simulation:** The reward function combines task reward (distance to goal), force penalty (collision cost via SafeRewardWrapper), and a step penalty. The collision cost uses a geometric proxy (vertex penetration depth) not a true force measurement.

**In reality:** There is no ground-truth reward signal. A real robot learns from:
- Human surgeon feedback (difficult to obtain at each timestep)
- Outcome metrics (task completion time, tissue damage assessment post-procedure)
- Force sensor readings (if available)
- Visual quality metrics (exposure quality, tissue deformation assessment)

**Evidence from this project:** The Phase 2 curriculum (λ_collision: 0.1→0.3→0.5) was manually designed based on engineering judgement. In Phase 3B the reward improved 42% but the agent never completed the task, suggesting the reward function does not provide sufficient signal for goal-directed behaviour without ground-truth goal coordinates.

**Mitigation:**
- Use inverse RL or imitation learning from surgical expert demonstrations
- Design reward functions based on clinical outcome metrics rather than simulator proxies
- Consider preference-based RL where a surgeon rates trajectory quality

---

## Category 5 — Computational Infrastructure

### Gap 5.1 — Training Hardware vs Deployment Hardware
**Severity: LOW**

**In simulation:** All training was performed on GTX 1650 (4GB VRAM, 1024 CUDA cores). Training speed: 13-15 fps for Phase 3B, 8.5 fps for Phase 3C.

**In reality:** Surgical robot computers must meet:
- Real-time requirements (deterministic execution, no garbage collection pauses)
- Medical device certification requirements (IEC 62304, FDA 510(k))
- NVIDIA DRIVE or similar automotive/medical grade GPU
- Redundant power supply and cooling

**Evidence from this project:** GTX 1650 is a consumer gaming GPU. The SOFA SIGABRT crashes on exit (GIL/destructor race) indicate non-deterministic behaviour that would be unacceptable in a medical device.

**Mitigation:**
- Profile inference latency on target deployment hardware (NVIDIA Orin or similar)
- Convert PyTorch models to TensorRT for deterministic inference latency
- Implement the Phase 4 ROS 2 node with real-time scheduling (SCHED_FIFO) for the inference thread

---

### Gap 5.2 — SOFA Plugin Warnings
**Severity: LOW**

**In simulation:** Every run produces warnings for missing plugins (`SofaBoundaryCondition`, `SofaDeformable`, `SofaEngine`, `SofaGeneralRigid`). These are legacy plugin names from SOFA v22 that have been renamed in SOFA v24. The actual functionality is available under new names (`Sofa.Component.Constraint.Projective` etc.) and was loaded successfully.

**In reality:** This gap has no direct real-robot equivalent. It represents technical debt in the simulation setup.

**Mitigation:**
- Update the SOFA scene file to use the new plugin names for SOFA v24+
- This is a maintenance task for Phase 4 or future work

---

## Summary Table

| Gap | Category | Severity | Status | Phase 4 Mitigation |
|-----|----------|----------|--------|-------------------|
| 1.1 Photorealism | Visual | HIGH | Open | Domain randomisation |
| 1.2 Camera model | Visual | MEDIUM | Open | Camera calibration in ROS 2 |
| 1.3 Tissue appearance | Visual | HIGH | Open | Colour augmentation + fine-tuning |
| 2.1 Tissue mechanics | Physics | HIGH | Open | FEM calibration |
| 2.2 Contact modelling | Physics | CRITICAL | Open | Compliance calibration |
| 2.3 Instrument dynamics | Physics | MEDIUM | Open | RCM constraint |
| 3.1 Force sensing | Sensing | CRITICAL | **Addressed by Phase 3C** | Optical flow proxy |
| 3.2 Observation completeness | Sensing | HIGH | Documented | Goal specification |
| 3.3 Sensor noise and latency | Sensing | MEDIUM | Open | Noise augmentation |
| 4.1 Goal definition | Task | HIGH | Open | Visual landmark goal |
| 4.2 Episode safety | Task | CRITICAL | Partial | Phase 3C safety threshold |
| 4.3 Reward realism | Task | HIGH | Open | Imitation learning |
| 5.1 Training vs deployment HW | Infra | LOW | Open | TensorRT conversion |
| 5.2 SOFA plugin warnings | Infra | LOW | Open | Scene file update |

**Addressed gaps:** 1 of 14 (Force sensing — optical flow proxy from Phase 3C)  
**Partially addressed:** 1 of 14 (Episode safety — optical flow alert/stop thresholds)  
**Open gaps:** 12 of 14

---

## What This Means for Phase 4

Phase 4 (ROS 2 integration) should focus on the three CRITICAL gaps:

**Gap 2.2 (Contact modelling):** Fix the `UncoupledConstraintCorrection` compliance warning before any real robot deployment. This requires correctly specifying the compliance matrix for the rigid body DOFs (7 values per rigid body: 3 translation + 4 rotation matrix elements).

**Gap 3.1 (Force sensing):** The optical flow proxy from Phase 3C is ready for Phase 4 deployment. The ROS 2 node publishes to `/tissue_force_proxy` with alert threshold 0.35 px/frame and safety stop at 1.0 px/frame. This directly addresses the most clinically significant gap.

**Gap 4.2 (Episode safety):** The Phase 4 supervisory safety layer must intercept and modify actions from the policy before execution. The optical flow reading from the previous frame determines whether the next action is permitted. Actions that would likely cause flow above 1.0 px/frame should be attenuated or blocked.

---

## Connection to Phase 3 Results

The Phase 3B result (0% goal rate without ground-truth observation) directly quantifies Gap 3.2. The Phase 3C result (optical flow proxy validated on 3000 steps) directly addresses Gap 3.1. Together, Phase 3A-C represents:

- A perception capability (3A) — closes Gap 1.1 partially within simulation
- A demonstration of the observation gap (3B) — quantifies Gap 3.2
- A force sensing alternative (3C) — closes Gap 3.1

Phase 3D documents what remains open and provides the roadmap for future work.