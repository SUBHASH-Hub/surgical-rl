# SOFA Scene Graph Analysis — Tissue Retraction Environment

## Scene Structure
Root node → scene node → Tissue (soft), EndEffector (rigid), visual nodes

## Tissue (Soft Body — FEM)
- Mesh: tissue_volume.msh (FEM volume), tissue_surface.obj (visual)
- Mass: 0.123264 kg (calibrated to full human gallbladder)
- FEM formulation: Corotated linear elasticity
- Young's modulus: 27,040 Pa (calibrated from porcine tissue experiments)
- Poisson ratio: 0.4287 (near-incompressible, appropriate for biological tissue)
- Constraint correction: PRECOMPUTED (compliance matrix cached in .comp file)
- Fixed boundary: top-right edge fixed in bounding box (simulates liver attachment)
- Free boundary: fundus (bottom) — what the instrument grasps and retracts

## EndEffector (Rigid Body)
- Type: Single-jaw laparoscopic grasper
- Grasping distance: 9mm capture radius
- Remote Centre of Motion (RCM): enforced by SOFA constraint
  (simulates trocar port — instrument rotates around fixed abdominal wall point)
- Starting position: randomised within starting_box each episode
  (domain randomisation for RL robustness and sim-to-real transfer)

## Physics Solver
- Constraint solver: LCP (Linear Complementarity Problem)
  maxIterations=1000, tolerance=0.001
- Animation loop: FREEMOTION
  (unconstrained motion first, then constraint correction)

## Phase 2 Implications
- young_modulus can be varied ±50% for tissue stiffness curriculum
- total_mass can be varied for gallbladder fill state randomisation
- Changing material params invalidates .comp cache — 2min recompute required
- Force data accessible via tissue.deformable_object for safety reward in Phase 2
- RCM constraint is already implemented — no additional work needed