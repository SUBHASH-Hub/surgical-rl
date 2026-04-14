"""
Phase 3B — Perception Pipeline
================================
Converts a single 480×480 RGB frame into the 132D multimodal observation
used by the Phase 3B PPO agent.

Output observation layout:
  [visual_features (128D), estimated_xyz (3D), phase_flag (1D)] = 132D total

How it works:
  1. Load trained MobileNetV3-Small checkpoint from Phase 3A
  2. At each env step, receive RGB frame from SOFA camera
  3. Run frame through backbone → 576D feature vector
  4. Run through FC layer → 128D visual features  (rich visual context)
  5. Run through final head → 3D estimated XYZ    (instrument position)
  6. Concatenate with phase flag → 132D observation

Why 128D features AND 3D xyz:
  The 3D xyz alone (like Phase 2D) gives position but no visual context.
  The 128D features give the PPO agent rich information about what the
  camera sees — tissue state, instrument appearance, scene context.
  This is the multimodal observation the roadmap specifies.
  In Phase 4, this same feature vector feeds the ROS 2 perception topic.

Why use the intermediate 128D layer not just xyz:
  The final 3D output has already compressed all visual information into
  3 numbers. The 128D intermediate layer retains richer spatial and
  appearance features that help the policy make better decisions.
  This is the standard approach in visuomotor policy learning
  (RT-2, R3M, and surgical AI papers all use intermediate features).

Author: Subhash Arockiadoss 
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torchvision import transforms, models


# ─────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────

OBS_DIM        = 132   # 128 visual features + 3 xyz + 1 phase
FEATURE_DIM    = 128   # intermediate feature vector dimension
XYZ_DIM        = 3     # estimated tool x, y, z
TIP_CKPT       = Path("models/tip_detector/mobilenetv3_tip_best.pth")
INPUT_SIZE     = 224   # MobileNetV3 input resolution

IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────
# MODEL COMPONENTS (must match train_tip_detector.py exactly)
# ─────────────────────────────────────────────────────────────────────────

class _TipDetectorFull(nn.Module):
    """
    Full MobileNetV3-Small tip detector — matches train_tip_detector.py.
    Used only for loading the checkpoint. PerceptionPipeline then splits
    this into backbone + feature_head + xyz_head.
    """
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.backbone = nn.Sequential(base.features, base.avgpool)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 3),
            nn.Tanh()
        )

    def forward(self, x):
        return self.head(self.backbone(x))


# ─────────────────────────────────────────────────────────────────────────
# PERCEPTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────

class PerceptionPipeline:
    """
    Converts a raw 480×480 RGB frame into a 132D multimodal observation.

    Splits the trained tip detector into three components:
      backbone:      MobileNetV3-Small features → (576,)
      feature_head:  FC 576→128 + ReLU          → (128,) visual features
      xyz_head:      FC 128→3   + Tanh           → (3,)   estimated XYZ

    The Dropout layer from training is excluded at inference time
    (model.eval() disables dropout automatically).

    Usage:
        pipeline = PerceptionPipeline()
        obs_132d = pipeline.get_observation(rgb_frame_480x480, phase_flag)
    """

    def __init__(self, checkpoint: Path = TIP_CKPT,
                 device: str = None):
        """
        Args:
            checkpoint: path to trained MobileNetV3 .pth file
            device:     'cuda', 'cpu', or None (auto-detect)
        """
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._build_pipeline(checkpoint)
        self._build_transform()

    def _build_pipeline(self, checkpoint: Path):
        """
        Load checkpoint and split into backbone, feature_head, xyz_head.

        The trained checkpoint contains the full model state dict with keys:
          backbone.0.* (features), backbone.1.* (avgpool)
          head.0 (Flatten — no params)
          head.1 (Linear 576→128)
          head.2 (ReLU — no params)
          head.3 (Dropout — no params at inference)
          head.4 (Linear 128→3)
          head.5 (Tanh — no params)
        """
        # Load full model to get state dict
        full_model = _TipDetectorFull()
        state_dict = torch.load(checkpoint, map_location=self.device)
        full_model.load_state_dict(state_dict)
        full_model.eval()

        # Split into three independent components
        # backbone: features + avgpool → (batch, 576, 1, 1)
        self.backbone = full_model.backbone.to(self.device)

        # feature_head: Flatten + Linear(576→128) + ReLU
        # Skip Dropout (head[3]) — not needed at inference, eval() disables it
        self.feature_head = nn.Sequential(
            full_model.head[0],   # Flatten
            full_model.head[1],   # Linear 576→128
            full_model.head[2],   # ReLU
        ).to(self.device)

        # xyz_head: Linear(128→3) + Tanh
        self.xyz_head = nn.Sequential(
            full_model.head[4],   # Linear 128→3
            full_model.head[5],   # Tanh
        ).to(self.device)

        # Set all to eval mode — disables dropout, uses running batch norm stats
        self.backbone.eval()
        self.feature_head.eval()
        self.xyz_head.eval()

    def _build_transform(self):
        """Image preprocessing: resize to 224×224 + ImageNet normalisation."""
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    @torch.no_grad()
    def get_observation(self, rgb_frame: np.ndarray,
                        phase_flag: float) -> np.ndarray:
        """
        Convert one RGB frame + phase flag → 132D observation vector.

        Args:
            rgb_frame:  numpy array (480, 480, 3) uint8 — from env._env.render()
            phase_flag: float, 0.0=GRASPING or 1.0=RETRACTING — from state obs

        Returns:
            numpy array (132,) float32
            Layout: [visual_features(128), estimated_xyz(3), phase(1)]

        The @torch.no_grad() decorator disables gradient tracking for
        all operations in this method — essential for inference speed.
        At 12fps environment speed, each call must complete in <80ms.
        On GTX 1650 this runs in ~5-8ms — well within budget.
        """
        # Preprocess frame: (480,480,3) uint8 → (1,3,224,224) float32 tensor
        img_tensor = self.transform(rgb_frame).unsqueeze(0).to(self.device)

        # Forward pass through split pipeline
        backbone_out  = self.backbone(img_tensor)    # (1, 576, 1, 1)
        features_128  = self.feature_head(backbone_out)  # (1, 128)
        estimated_xyz = self.xyz_head(features_128)  # (1, 3)

        # Convert to numpy and build 132D observation
        feat_np = features_128.squeeze(0).cpu().numpy()    # (128,)
        xyz_np  = estimated_xyz.squeeze(0).cpu().numpy()   # (3,)
        phase_np = np.array([phase_flag], dtype=np.float32)  # (1,)

        # Concatenate: [features(128) | xyz(3) | phase(1)] = 132D
        obs_132d = np.concatenate([feat_np, xyz_np, phase_np]).astype(np.float32)
        return obs_132d

    def get_feature_vector(self, rgb_frame: np.ndarray) -> np.ndarray:
        """
        Return only the 128D feature vector (for analysis / debugging).
        Does not include xyz or phase.
        """
        img_tensor   = self.transform(rgb_frame).unsqueeze(0).to(self.device)
        backbone_out = self.backbone(img_tensor)
        features_128 = self.feature_head(backbone_out)
        return features_128.squeeze(0).cpu().numpy()

    def get_estimated_xyz(self, rgb_frame: np.ndarray) -> np.ndarray:
        """
        Return only the 3D estimated XYZ (for analysis / debugging).
        Same as running the full tip detector in inference mode.
        """
        img_tensor    = self.transform(rgb_frame).unsqueeze(0).to(self.device)
        backbone_out  = self.backbone(img_tensor)
        features_128  = self.feature_head(backbone_out)
        estimated_xyz = self.xyz_head(features_128)
        return estimated_xyz.squeeze(0).cpu().numpy()