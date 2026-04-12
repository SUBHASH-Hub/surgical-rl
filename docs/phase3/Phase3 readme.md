# Phase 3 — Surgical Perception Pipeline

**Status:** In Progress — Month 5–6  
**Clinical Question:** Can a perception module extract surgical state from endoscopic video, replacing ground-truth simulator coordinates the way a real surgical robot must operate?  
**W&B Project:** surgical-rl-phase3 *(to be linked when Phase 3B training begins)*

---

## Overview

Phase 3 builds the visual perception layer that bridges the gap between simulation and clinical reality. Phase 2 proved a PPO agent can learn safe tissue retraction — but it used ground-truth XYZ coordinates from the simulator. No real surgical robot has access to ground truth. Phase 3 replaces that with camera-based perception.

This phase has four steps, each producing a standalone deliverable:

| Step | Title | Status | Deliverable |
|------|-------|--------|-------------|
| [3A](./phase3a_results.md) | Standalone Surgical Perception Module | In Progress (3/5 sub-steps done) | Tip detector + segmentation model + visualisations |
| [3B](./phase3b_results.md) | Multimodal Observation Integration | Pending | PPO retrained on visual observations |
| [3C](./phase3c_results.md) | Force Proxy from Visual Motion | Pending | Optical flow tissue deformation proxy |
| [3D](./phase3d_gap_analysis.md) | Sim-to-Real Gap Analysis | Pending | Written technical analysis (1 page min) |

---

## Architecture: what changes in Phase 3

**Phase 2 observation pipeline (ground-truth):**
```
SOFA simulator → 7D state vector [tool_xyz, goal_xyz, phase] → PPO agent
```

**Phase 3 observation pipeline (visual):**
```
SOFA camera → 480×480 RGB frame → MobileNetV3 → estimated_xyz (3D)
                                               ↓
                                  [visual_features (128D), estimated_xyz (3D), phase (1D)]
                                               ↓
                                          PPO agent (132D input)
```

**Phase 3C addition (force proxy):**
```
Frame(t) + Frame(t-1) → Optical flow → Tissue deformation magnitude → /tissue_force_proxy
```

---

## Why these specific technology choices

| Component | Technology | Industry justification |
|-----------|-----------|----------------------|
| Tip detection backbone | MobileNetV3-Small | Moon Surgical ScoPilot uses this architecture class for real-time instrument tracking |
| Segmentation | Lightweight UNet (<5M params) | Runs real-time on NVIDIA IGX embedded hardware (Holoscan deployment target) |
| Force proxy | OpenCV optical flow (Farneback) | Medtronic Touch Surgery and NVIDIA Holoscan use visual motion for surgical state inference |
| Feature fusion | 128D visual + 3D XYZ + 1D phase = 132D | Multimodal obs matching Phase 3B PPO input spec |

---

## Phase 3 Deliverables Checklist

- [ ] Trained instrument tip detection model — 5.1px MAE on 480×480 ✓ *(3A-iii done)*
- [ ] Trained tissue segmentation model — IoU metric *(3A-iv pending)*
- [ ] Visualised predictions on 10 test frames — PNG overlays *(3A-v pending)*
- [ ] PPO agent retrained on multimodal observation *(3B pending)*
- [ ] Performance comparison table: ground-truth state vs visual observation *(3B pending)*
- [ ] Optical flow tissue force proxy with validation analysis *(3C pending)*
- [ ] Sim-to-real gap analysis — written, one page minimum *(3D pending)*

---

## Git tags

| Tag | Description |
|-----|-------------|
| `v3.0-phase3a-tip-detector` | MobileNetV3 tip detector trained — 5.1px MAE |
| `v3.1-phase3a-complete` | *(pending)* All 5 Phase 3A sub-steps complete |
| `v3.2-phase3b-complete` | *(pending)* PPO retrained on visual observations |
| `v3.3-phase3c-complete` | *(pending)* Optical flow force proxy validated |
| `v3.4-phase3-complete` | *(pending)* Full Phase 3 complete |

---

## Related files

```
scripts/
  test_camera_capture.py       ← 3A-i: verified camera → (480,480,3) uint8
  collect_rgb_frames.py        ← 3A-ii: 7284 frames, 50 episodes, 145.7 steps mean
  train_tip_detector.py        ← 3A-iii: MobileNetV3 regression training
  train_segmentation.py        ← 3A-iv: UNet tissue boundary (pending)
  visualise_predictions.py     ← 3A-v: overlay visualisation (pending)

models/
  tip_detector/
    mobilenetv3_tip_best.pth   ← best val checkpoint (val_loss=0.00621)
    mobilenetv3_tip_final.pth  ← final epoch checkpoint
    training_log.csv           ← per-epoch loss for plotting
    eval_metrics.json          ← test set metrics

data/
  rgb_frames/                  ← 7284 PNGs + labels.csv (480×480×3, uint8)

docs/
  assets/
    camera_test_frame.png      ← sample frame confirming camera render
```