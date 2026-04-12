# Phase 3A — Standalone Surgical Perception Module

**Status:** ✓ COMPLETE — all 5 sub-steps done  
**Clinical Question:** Can instrument tip position and tissue boundaries be extracted from a single endoscopic RGB frame with sufficient accuracy to replace ground-truth simulator coordinates?  
**Completed:** April 2026

---

## Sub-step Overview

| Sub-step | Description | Status | Key result |
|----------|-------------|--------|------------|
| 3A-i | Camera verification | ✓ Done | (480, 480, 3) uint8 confirmed via Strategy A |
| 3A-ii | RGB frame collection | ✓ Done | 7284 frames, 50 episodes, labels.csv |
| 3A-iii | Instrument tip detector | ✓ Done | 5.1px MAE, val_loss=0.00621 |
| 3A-iv | Tissue boundary segmentation | ✓ Done | IoU=1.000 (sim), UNet 1.73M params |
| 3A-v | Prediction visualisation | ✓ Done | 10 demo frames, mean err=5.5px |

---

## Standalone Module Performance Summary

Both models validated as independent components before RL integration (Phase 3B).

| Module | Metric | Value | Threshold | Status |
|--------|--------|-------|-----------|--------|
| Instrument tip detector | Pixel localisation error | 5.1 px on 480×480 | < 10px | ✓ Met |
| Instrument tip detector | Mean XYZ MAE | 0.0214 normalised | — | ✓ |
| Instrument tip detector | Parameters | 1,001,251 | — | ✓ |
| Tissue segmentation | IoU (sim test set) | 1.000 | > 0.5 | ✓ Met (sim) |
| Tissue segmentation | Dice score | 1.000 | — | ✓ |
| Tissue segmentation | Parameters | 1,729,217 | < 5M | ✓ Met |

**Note on segmentation IoU=1.000:** This reflects a known simulation property — the tissue mesh occupies a geometrically consistent region relative to the fixed camera across all episodes. The model learned the tissue boundary location, not tissue appearance features. See 3A-iv for full analysis and sim-to-real implications.

---

## 3A-i — Camera Verification

**Script:** `scripts/test_camera_capture.py`  
**Date completed:** April 2026

### What was verified

Confirmed that the SOFA endoscopic camera renders accessible RGB frames from Python before writing the full data collection pipeline.

**Camera access path confirmed:**
```python
frame = env._env.render()   # Strategy A — works directly
# Returns: numpy array shape (480, 480, 3), dtype uint8, values 0–255
```

**Render mode used:** `HUMAN` (pyglet window — EGL headless available for server deployment)

**Why this matters:** The `TissueRetractionV2` wrapper does not expose `render()` directly. The underlying `SofaEnv` object (`env._env`) owns the OpenGL context and pyglet buffer. Calling `render()` on the wrapper returns None; calling it on `env._env` returns the numpy array.

### Camera model

The SOFA `InteractiveCamera` is a pinhole camera model positioned at a fixed point in 3D space looking down at the Calot's triangle scene — mimicking a laparoscope tip. OpenGL rasterises the FEM scene from this viewpoint onto a 480×480 GPU framebuffer. `render()` copies the framebuffer from GPU VRAM to CPU RAM as a uint8 array.

**SIGABRT on exit:** A known SOFA/SofaPython3 GIL race condition fires on interpreter shutdown. It does not affect data — all buffers are flushed before exit. Resolved by using `os._exit(0)` in collection scripts.

---

## 3A-ii — RGB Frame Collection

**Script:** `scripts/collect_rgb_frames.py`  
**Date completed:** April 2026

### Collection parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| Episodes | 50 | Sufficient diversity across random start positions |
| Agent | Phase 2D PPO checkpoint | Best trained policy, deterministic=True |
| Render mode | HUMAN (headless available) | GTX 1650 with display present — identical pixel data |
| Capture timing | Before `env.step()` | Frame and label describe same simulator state — no off-by-one |

### Dataset statistics

| Metric | Value |
|--------|-------|
| Total frames | 7284 |
| Episodes | 50 |
| Mean steps per episode | 145.7 |
| Min steps (best episode) | 94 |
| Max steps (hardest episode) | 207 |
| Goal rate | 100% (all 50 episodes ended `done`, not `truncated`) |
| Collection speed | 12.4 fps avg |
| Elapsed time | 585.7s (~9.8 minutes) |
| Image resolution | 480 × 480 × 3 pixels |
| Image dtype | uint8 |

### Label distribution

| Axis | Min | Max | Notes |
|------|-----|-----|-------|
| tool_x | -0.965 | +0.981 | Near-full workspace coverage |
| tool_y | -0.933 | +0.944 | Near-full workspace coverage |
| tool_z | -0.942 | +0.937 | Near-full workspace coverage |
| phase | 0.0 (4446 frames) | 1.0 (2838 frames) | 61% GRASPING / 39% RETRACTING |

All XYZ values normalised to [-1, 1] by `TissueRetractionV2`. Full workspace coverage confirms the Phase 2D agent explored diverse trajectories.

### Dataset split (by episode — not random shuffle)

| Split | Episodes | Frames | Reason |
|-------|----------|--------|--------|
| Train | 0–39 | 5839 | Consecutive frames are near-duplicates (~1mm motion per step). Random shuffle causes data leakage. |
| Val | 40–44 | 758 | |
| Test | 45–49 | 687 | Never seen during training. |

---

## 3A-iii — Instrument Tip Detector

**Script:** `scripts/train_tip_detector.py`  
**Date completed:** April 2026  
**Checkpoint:** `models/tip_detector/mobilenetv3_tip_best.pth`

### Architecture

**Backbone:** MobileNetV3-Small, pretrained on ImageNet-1K  
**Why MobileNetV3:** Depthwise separable convolutions give 9× fewer multiply-accumulate operations than standard convolutions. Moon Surgical ScoPilot uses this architecture class for real-time instrument tracking. Deployable on NVIDIA IGX (Holoscan target).

```
Input:  (480, 480, 3) uint8
  ↓ Resize + ImageNet normalise → (224, 224, 3) float32
  ↓ MobileNetV3-Small backbone (frozen Phase 1, unfrozen Phase 2)
  → feature vector (576,) float32
  ↓ FC 576→128, ReLU, Dropout(0.3)
  ↓ FC 128→3, Tanh
Output: [tool_x, tool_y, tool_z] in [-1, 1]
```

**Total parameters:** 1,001,251

### Training — two-phase transfer learning

| Phase | Epochs | Backbone | LR | Purpose |
|-------|--------|----------|----|---------|
| 1 — warmup | 1–10 | Frozen | 1e-3 | Head learns from fixed ImageNet features |
| 2 — fine-tune | 11–30 | Unfrozen | 1e-4 | Full model adapts to surgical scene |

**Why two phases:** If the backbone is unfrozen immediately with a randomly initialised head, large gradients destroy the ImageNet features. Freezing first allows the head to stabilise. This is the standard transfer learning protocol used in production surgical AI systems.

### Training results

| Epoch | Phase | Train MSE | Val MSE | Notes |
|-------|-------|-----------|---------|-------|
| 1 | Frozen | 0.04325 | 0.05222 | Random head, high initial loss |
| 8 | Frozen | 0.01483 | **0.02315** | Best Phase 1 checkpoint |
| 11 | Fine-tune | 0.01112 | 0.01955 | Immediate improvement on unfreeze |
| 28 | Fine-tune | 0.00445 | **0.00621** | Best overall — 88% loss reduction |
| 30 | Fine-tune | 0.00425 | 0.00666 | Final epoch |

**Total training time:** ~14 minutes on GTX 1650

### Test set evaluation (episodes 45–49, 687 frames)

| Metric | Value | Clinical context |
|--------|-------|-----------------|
| **Pixel localisation error** | **5.1 px on 480×480** | Below 10px surgical AI threshold ✓ |
| Mean XYZ MAE | 0.0214 normalised | 1.07% of workspace width |
| X-axis MAE | 0.0156 | Horizontal — easiest axis |
| Y-axis MAE | 0.0250 | Vertical — highest error |
| Z-axis MAE | 0.0235 | Depth — close to Y, better than expected |

**Notable finding:** Z-depth was expected hardest (monocular depth ambiguity) but Y had highest error. SOFA provides consistent depth cues (fixed geometry, static lighting) allowing reliable Z inference. In real endoscopic video, Z would likely become hardest due to specular highlights and occlusion — documented as sim-to-real gap in Phase 3D.

---

## 3A-iv — Tissue Boundary Segmentation

**Script:** `scripts/train_segmentation.py`  
**Date completed:** April 2026  
**Checkpoint:** `models/segmentation/unet_seg_best.pth`

### Label generation — SOFA mesh projection

Tissue segmentation masks were generated by projecting tissue visual mesh vertices (448 vertices, `scene/tissue/visual/OglModel`) through the OpenGL camera matrices onto the 2D image plane using `scripts/generate_seg_masks.py`.

```
Projection pipeline (verified: vertex→pixel in [0,480]):
  vertex_h = [x, y, z, 1.0]
  clip     = projectionMatrix @ modelViewMatrix @ vertex_h
  ndc_x,y  = clip[0]/clip[3], clip[1]/clip[3]
  u = (ndc_x+1)/2 * 480    v = (1-ndc_y)/2 * 480
  Convex hull of projected points → filled binary mask (480×480)
```

**Coverage:** 92% frame match (6701/7284 frames). The 8% gap is caused by SOFA C++ physics non-determinism across separate process runs — seeding the Python RNG does not control SOFA's internal constraint solver threading. Documented as a simulation infrastructure limitation.

**Why mesh projection over colour thresholding:** The correct engineering approach for simulation-based CV — labels come from the same 3D geometry that rendered the image. This is the method used by NVIDIA Isaac Sim and Blender synthetic data pipelines for sim-to-real transfer. HSV colour thresholding was prototyped and rejected because it exploits a simulation artifact (flat uniform tissue colour) that does not exist in real endoscopic video.

### Architecture — UNet with MobileNetV3-Small encoder

```
Input:  (480, 480, 3) → resize to (256, 256, 3)
  ↓ MobileNetV3-Small encoder (pretrained ImageNet)
  s0:(B,16,128,128)  s1:(B,24,64,64)  s2:(B,48,32,32)  s3:(B,576,8,8)
  ↓ UNet decoder with skip connections (3 UpBlocks)
  ↓ ConvTranspose2d → (B,1,256,256) logits → Sigmoid threshold 0.5
Output: binary tissue mask
```

**Total parameters:** 1,729,217 — within <5M roadmap target.

**Why skip connections:** Deep encoder features encode WHAT (tissue vs background) but lose WHERE (exact pixel location). Skip connections re-inject spatial detail from earlier encoder layers — the decoder recovers precise tissue boundary edges.

**Why same MobileNetV3 encoder as tip detector:** Consistent feature space between both perception models — foundation for the 132D multimodal observation in Phase 3B.

### Loss function — BCE + Dice combined

BCE penalises per-pixel classification error. Dice penalises global mask shape error and prevents all-background prediction when tissue is only 21% of pixels. Equal 0.5/0.5 weighting is standard practice.

### Training results

| Epoch | Train loss | Val loss | Val IoU | Notes |
|-------|-----------|---------|---------|-------|
| 1 | 0.4088 | 0.3509 | 0.9080 | Starting from ImageNet weights |
| 6 | 0.1698 | 0.1545 | **1.0000** | IoU saturates |
| 9 | 0.0956 | 0.0863 | 1.0000 | Best checkpoint saved |
| 30 | 0.0138 | 0.0136 | 1.0000 | Loss still decreasing, IoU saturated |

**Total training time:** ~21 minutes on GTX 1650

### Test set evaluation

| Metric | Value |
|--------|-------|
| Test IoU | 1.0000 |
| Test Dice | 1.0000 |

### Analysis — why IoU=1.000 and what it means

**Root cause:** The tissue mesh is a fixed box with a fixed camera. Tissue coverage is 21.1% of image area, constant across all frames. The UNet learned the tissue boundary location (a fixed geometric region) rather than tissue visual appearance features. By epoch 6 it had learned the consistent mask shape.

**What this demonstrates:** UNet architecture correct, BCE+Dice loss functional, MobileNetV3 encoder+decoder pipeline working, training infrastructure complete.

**What this does NOT demonstrate:** Generalisation to new tissue positions, robustness to lighting variation, blood, smoke, or deployability to real endoscopic video.

**Sim-to-real gap:** In real laparoscopic video, tissue position varies with patient anatomy and instrument interaction. Colour and appearance vary with blood, bile, and OR lighting. Production systems require manually annotated real endoscopic frames or domain-randomised synthetic data. Documented in full in Phase 3D.

---

## 3A-v — Prediction Visualisation

**Script:** `scripts/visualise_predictions.py`  
**Date completed:** April 2026  
**Output:** `docs/assets/predictions/`

### What was generated

10 test frames (episodes 45–49, 2 per episode) with three-panel overlay:

```
Panel 1: Original RGB frame
Panel 2: Tip detection  — GREEN crosshair = predicted tip position
                          RED dot         = ground truth tip position
Panel 3: Segmentation   — CYAN fill       = predicted tissue mask
                          WHITE contour   = tissue boundary edge
```

### Per-frame pixel errors

| Frame | Ep | Step | Pixel error |
|-------|----|------|-------------|
| 01 | 45 | 025 | 11.2px |
| 02 | 45 | 075 | 4.1px |
| 03 | 46 | 033 | 5.8px |
| 04 | 46 | 099 | **3.4px** ← best |
| 05 | 47 | 025 | 4.6px |
| 06 | 47 | 075 | **3.2px** ← best |
| 07 | 48 | 047 | 3.7px |
| 08 | 48 | 143 | 5.0px |
| 09 | 49 | 040 | 8.1px |
| 10 | 49 | 122 | 5.9px |
| **Mean** | | | **5.5px** |

**Mean 5.5px is consistent with 5.1px test MAE from training ✓**

Frame 01 (11.2px worst): early in episode, instrument approaching steep angle, Y-axis error dominates — consistent with training result that Y is the hardest axis.

### Visual confirmation (frame_04_pred.png)

Visual inspection of frame 04 confirmed:
- Green crosshair and red dot co-located at instrument-tissue contact point ✓
- Cyan overlay covers the full yellow-green tissue region accurately ✓
- White contour traces tissue boundary edge cleanly ✓
- Instrument shaft correctly excluded from tissue mask ✓
- Background (grey board, blue scene) correctly excluded ✓

Grid image `prediction_grid.png` available for GitHub README.

---

## Phase 3A Engineering Notes

### Tools chosen and why

| Tool | Used for | Why over alternatives |
|------|----------|-----------------------|
| PIL (Pillow) | PNG save, overlay drawing | RGB natively — OpenCV defaults to BGR causing silent channel swap |
| PyTorch + torchvision | Model training | Industry standard; SB3 also uses PyTorch — consistent ecosystem |
| MobileNetV3-Small | Backbone (both models) | 9× fewer ops than ResNet; Moon ScoPilot architecture; NVIDIA IGX deployable |
| Episode-based split | Train/val/test | Prevents data leakage from near-duplicate consecutive frames |
| MSELoss | Tip regression | Standard for 3D coordinate regression |
| BCE + Dice | Segmentation | Handles 21% class imbalance; Dice prevents all-background prediction |
| SOFA mesh projection | Mask label generation | Labels from 3D geometry — correct sim-based CV pipeline |

### OpenCV in Phase 3A

OpenCV was used in `visualise_predictions.py` for contour extraction only (`cv2.erode`). Its primary role enters in Phase 3C for optical flow (Farneback algorithm) to compute tissue deformation as a visual force proxy.

### Segmentation label generation — three approaches evaluated

| Approach | Match rate | Engineering correctness | Decision |
|----------|-----------|------------------------|----------|
| Mesh projection, basic replay | 88–91% | ✓ Correct | Too many mismatches |
| Mesh projection, CSV-driven replay | 92% | ✓ Correct | **Used for training** |
| HSV colour thresholding | 100% | ✗ Simulation artifact | Rejected — not transferable to real video |

CSV-driven mesh projection was chosen as the correct engineering method. The 8% gap is a documented SOFA C++ non-determinism limitation, not a method failure.