"""
models.py
=========
Model definitions:
    MLP          - Flatten keypoint sequence, dense layers. Baseline.
    TemporalCNN  - 1D convolutions over time on keypoint sequence. Baseline.
    PoseConv3D   - 3D CNN on stacked heatmap volumes. Main method.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ------------------------------------------------------------------
# Heatmap conversion
# ------------------------------------------------------------------
def keypoints_to_heatmaps(keypoints, h=56, w=56, sigma=3.0, threshold=0.3):
    """
    Convert keypoint array to stacked 2D Gaussian heatmaps.

    Args:
        keypoints: np.ndarray [T, J, 3] — x, y, confidence (normalized 0-1)
        h, w:      heatmap spatial size
        sigma:     Gaussian standard deviation in pixels
        threshold: confidence below this → zero heatmap

    Returns:
        np.ndarray [T, J, H, W] float32
    """
    T, J, _ = keypoints.shape
    heatmaps = np.zeros((T, J, h, w), dtype=np.float32)

    xs = (keypoints[:, :, 0] * w).astype(np.float32)  # [T, J]
    ys = (keypoints[:, :, 1] * h).astype(np.float32)  # [T, J]
    cs = keypoints[:, :, 2]                             # [T, J]

    # Precompute grid
    grid_x = np.arange(w, dtype=np.float32)[None, None, None, :]  # [1,1,1,W]
    grid_y = np.arange(h, dtype=np.float32)[None, None, :, None]  # [1,1,H,1]

    for t in range(T):
        for j in range(J):
            if cs[t, j] < threshold:
                continue
            dx = grid_x[0, 0] - xs[t, j]   # [1, W]
            dy = grid_y[0, :, 0] - ys[t, j] # [H, 1]
            heatmaps[t, j] = cs[t, j] * np.exp(
                -(dx**2 + dy**2) / (2 * sigma**2)
            )

    return heatmaps


# ------------------------------------------------------------------
# MLP
# ------------------------------------------------------------------
class MLP(nn.Module):
    """
    Flatten-then-MLP baseline.
    Input:  [B, T, D]
    Output: [B, n_classes]
    """
    def __init__(self, n_frames, input_dim, n_classes,
                 hidden_dims=(512, 256), dropout=0.3):
        super().__init__()
        in_features = n_frames * input_dim
        layers = []
        prev_dim = in_features
        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        B = x.shape[0]
        return self.net(x.view(B, -1))


# ------------------------------------------------------------------
# Temporal CNN
# ------------------------------------------------------------------
class TemporalCNN(nn.Module):
    """
    1D Temporal CNN baseline.
    Input:  [B, T, D]
    Output: [B, n_classes]
    """
    def __init__(self, input_dim, n_classes, channels=(64, 128, 256),
                 kernel_size=3, dropout=0.3):
        super().__init__()
        conv_layers = []
        in_ch = input_dim
        for out_ch in channels:
            conv_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                          padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(in_ch, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)       # [B, D, T]
        x = self.conv(x)              # [B, C, T]
        x = self.pool(x).squeeze(-1) # [B, C]
        return self.classifier(x)


# ------------------------------------------------------------------
# PoseConv3D
# ------------------------------------------------------------------
class PoseConv3D(nn.Module):
    """
    3D CNN on stacked Gaussian heatmap volumes.
    Follows the PoseConv3D framework (Duan et al., 2021).

    Input:  [B, T, J, H, W]  heatmap volume
    Output: [B, n_classes]
    """
    def __init__(self, n_joints=17, n_classes=19, dropout=0.3):
        super().__init__()

        # Reshape input: treat joints as channels → [B, J, T, H, W]
        # Then apply 3D convolutions over (T, H, W)
        self.conv1 = nn.Sequential(
            nn.Conv3d(n_joints, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),  # downsample spatial
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),  # downsample temporal+spatial
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
        )

        self.pool = nn.AdaptiveAvgPool3d(1)  # [B, 128, 1, 1, 1]

        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        # x: [B, T, J, H, W]
        x = x.permute(0, 2, 1, 3, 4)  # [B, J, T, H, W]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x).squeeze(-1).squeeze(-1).squeeze(-1)  # [B, 128]
        return self.classifier(x)


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------
def build_model(model_name, n_frames, input_dim, n_classes, n_joints=17):
    if model_name == 'MLP':
        return MLP(n_frames=n_frames, input_dim=input_dim,
                   n_classes=n_classes)
    elif model_name == 'TemporalCNN':
        return TemporalCNN(input_dim=input_dim, n_classes=n_classes)
    elif model_name == 'PoseConv3D':
        return PoseConv3D(n_joints=n_joints, n_classes=n_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def input_dim_for(input_type, n_joints=17):
    if input_type == 'xy':
        return n_joints * 2
    elif input_type == 'xyconf':
        return n_joints * 3
    elif input_type == 'heatmap':
        return None  # heatmaps handled separately
    else:
        raise ValueError(f"Unknown input_type: {input_type}")
