"""
models.py
=========
Model definitions:
    MLP            - Flatten keypoint sequence, dense layers. Baseline.
    TemporalCNN    - 1D convolutions over time. Baseline.
    PoseConv3D     - Simple 3D CNN on joint heatmap volumes.
    PoseConv3DLarge- Larger 3D CNN on joint heatmap volumes.
    PoseSlowOnly   - ResNet-inflated SlowOnly backbone (PoseConv3D paper).
"""

import torch
import torch.nn as nn
import numpy as np

LIMBS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
N_LIMBS = len(LIMBS)


def keypoints_to_heatmaps(keypoints, h=56, w=56, sigma=3.0, threshold=0.3):
    T, J, _ = keypoints.shape
    heatmaps = np.zeros((T, J, h, w), dtype=np.float32)
    xs = (keypoints[:, :, 0] * w).astype(np.float32)
    ys = (keypoints[:, :, 1] * h).astype(np.float32)
    cs = keypoints[:, :, 2]
    grid_x = np.arange(w, dtype=np.float32)
    grid_y = np.arange(h, dtype=np.float32)
    for t in range(T):
        for j in range(J):
            if cs[t, j] < threshold:
                continue
            dx = grid_x[None, :] - xs[t, j]
            dy = grid_y[:, None] - ys[t, j]
            heatmaps[t, j] = cs[t, j] * np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
    return heatmaps


def keypoints_to_limb_heatmaps(keypoints, h=56, w=56, sigma=3.0, threshold=0.3):
    T, J, _ = keypoints.shape
    heatmaps = np.zeros((T, N_LIMBS, h, w), dtype=np.float32)
    xs = (keypoints[:, :, 0] * w).astype(np.float32)
    ys = (keypoints[:, :, 1] * h).astype(np.float32)
    cs = keypoints[:, :, 2]
    grid_x = np.arange(w, dtype=np.float32)
    grid_y = np.arange(h, dtype=np.float32)
    GX, GY = np.meshgrid(grid_x, grid_y)
    for t in range(T):
        for k, (a, b) in enumerate(LIMBS):
            ca, cb = cs[t, a], cs[t, b]
            if ca < threshold or cb < threshold:
                continue
            conf = min(ca, cb)
            x1, y1 = xs[t, a], ys[t, a]
            x2, y2 = xs[t, b], ys[t, b]
            dx, dy = x2 - x1, y2 - y1
            seg_len_sq = dx**2 + dy**2
            if seg_len_sq < 1e-6:
                dist_sq = (GX - x1)**2 + (GY - y1)**2
            else:
                t_proj = np.clip(((GX - x1) * dx + (GY - y1) * dy) / seg_len_sq, 0, 1)
                dist_sq = (GX - (x1 + t_proj * dx))**2 + (GY - (y1 + t_proj * dy))**2
            heatmaps[t, k] = conf * np.exp(-dist_sq / (2 * sigma**2))
    return heatmaps


def keypoints_to_joint_limb_heatmaps(keypoints, h=56, w=56, sigma=3.0, threshold=0.3):
    return np.concatenate([
        keypoints_to_heatmaps(keypoints, h, w, sigma, threshold),
        keypoints_to_limb_heatmaps(keypoints, h, w, sigma, threshold)
    ], axis=1)


class MLP(nn.Module):
    def __init__(self, n_frames, input_dim, n_classes, hidden_dims=(512, 256), dropout=0.3):
        super().__init__()
        in_features = n_frames * input_dim
        layers = []
        prev_dim = in_features
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(x.shape[0], -1))


class TemporalCNN(nn.Module):
    def __init__(self, input_dim, n_classes, channels=(64, 128, 256), kernel_size=3, dropout=0.3):
        super().__init__()
        conv_layers = []
        in_ch = input_dim
        for out_ch in channels:
            conv_layers += [nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size//2),
                           nn.BatchNorm1d(out_ch), nn.ReLU(), nn.Dropout(dropout)]
            in_ch = out_ch
        self.conv = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Linear(in_ch, 128), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(128, n_classes))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.classifier(self.pool(self.conv(x)).squeeze(-1))


class PoseConv3D(nn.Module):
    def __init__(self, in_channels=17, n_classes=19, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, 32, (3,3,3), padding=(1,1,1)),
                                   nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d((1,2,2)))
        self.conv2 = nn.Sequential(nn.Conv3d(32, 64, (3,3,3), padding=(1,1,1)),
                                   nn.BatchNorm3d(64), nn.ReLU(), nn.MaxPool3d((2,2,2)))
        self.conv3 = nn.Sequential(nn.Conv3d(64, 128, (3,3,3), padding=(1,1,1)),
                                   nn.BatchNorm3d(128), nn.ReLU(), nn.MaxPool3d((2,2,2)))
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(nn.Linear(128, 256), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(256, n_classes))

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        return self.classifier(self.pool(self.conv3(self.conv2(self.conv1(x)))).flatten(1))


class PoseConv3DLarge(nn.Module):
    def __init__(self, in_channels=17, n_classes=19, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, 64, (3,3,3), padding=(1,1,1)),
                                   nn.BatchNorm3d(64), nn.ReLU(), nn.MaxPool3d((1,2,2)))
        self.conv2 = nn.Sequential(nn.Conv3d(64, 128, (3,3,3), padding=(1,1,1)),
                                   nn.BatchNorm3d(128), nn.ReLU(), nn.MaxPool3d((2,2,2)))
        self.conv3 = nn.Sequential(nn.Conv3d(128, 256, (3,3,3), padding=(1,1,1)),
                                   nn.BatchNorm3d(256), nn.ReLU(), nn.MaxPool3d((2,2,2)))
        self.conv4 = nn.Sequential(nn.Conv3d(256, 256, (3,3,3), padding=(1,1,1)),
                                   nn.BatchNorm3d(256), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Dropout(dropout),
                                        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout),
                                        nn.Linear(256, n_classes))

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        return self.classifier(self.pool(self.conv4(self.conv3(self.conv2(self.conv1(x))))).flatten(1))


class ResBlock3D(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, temporal=False, stride=1):
        super().__init__()
        t_kernel = 3 if temporal else 1
        t_pad    = 1 if temporal else 0
        self.conv1 = nn.Sequential(nn.Conv3d(in_ch, mid_ch, (1,1,1), bias=False),
                                   nn.BatchNorm3d(mid_ch), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv3d(mid_ch, mid_ch, (t_kernel,3,3),
                                             padding=(t_pad,1,1), stride=(1,stride,stride), bias=False),
                                   nn.BatchNorm3d(mid_ch), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv3d(mid_ch, out_ch, (1,1,1), bias=False),
                                   nn.BatchNorm3d(out_ch))
        self.downsample = None
        if in_ch != out_ch or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, (1,1,1), stride=(1,stride,stride), bias=False),
                nn.BatchNorm3d(out_ch))
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv3(self.conv2(self.conv1(x)))
        if self.downsample:
            identity = self.downsample(x)
        return self.relu(out + identity)


class PoseSlowOnly(nn.Module):
    def __init__(self, in_channels=17, n_classes=19, dropout=0.3, base_ch=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_ch, (1,7,7), padding=(0,3,3), stride=(1,2,2), bias=False),
            nn.BatchNorm3d(base_ch), nn.ReLU())
        self.stage2 = nn.Sequential(
            ResBlock3D(base_ch,    base_ch,    base_ch*4,  False, 1),
            ResBlock3D(base_ch*4,  base_ch,    base_ch*4,  False),
            ResBlock3D(base_ch*4,  base_ch,    base_ch*4,  False))
        self.stage3 = nn.Sequential(
            ResBlock3D(base_ch*4,  base_ch*2,  base_ch*8,  False, 2),
            ResBlock3D(base_ch*8,  base_ch*2,  base_ch*8,  False),
            ResBlock3D(base_ch*8,  base_ch*2,  base_ch*8,  False),
            ResBlock3D(base_ch*8,  base_ch*2,  base_ch*8,  False))
        self.stage4 = nn.Sequential(
            ResBlock3D(base_ch*8,  base_ch*4,  base_ch*16, True,  2),
            ResBlock3D(base_ch*16, base_ch*4,  base_ch*16, True),
            ResBlock3D(base_ch*16, base_ch*4,  base_ch*16, True),
            ResBlock3D(base_ch*16, base_ch*4,  base_ch*16, True),
            ResBlock3D(base_ch*16, base_ch*4,  base_ch*16, True),
            ResBlock3D(base_ch*16, base_ch*4,  base_ch*16, True))
        self.stage5 = nn.Sequential(
            ResBlock3D(base_ch*16, base_ch*8,  base_ch*32, True,  2),
            ResBlock3D(base_ch*32, base_ch*8,  base_ch*32, True),
            ResBlock3D(base_ch*32, base_ch*8,  base_ch*32, True))
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(base_ch*32, n_classes))

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        return self.classifier(self.pool(x).flatten(1))


def build_model(model_name, n_frames, input_dim, n_classes, n_joints=17, in_channels=None):
    if in_channels is None:
        in_channels = n_joints
    if model_name == 'MLP':
        return MLP(n_frames=n_frames, input_dim=input_dim, n_classes=n_classes)
    elif model_name == 'TemporalCNN':
        return TemporalCNN(input_dim=input_dim, n_classes=n_classes)
    elif model_name == 'PoseConv3D':
        return PoseConv3D(in_channels=in_channels, n_classes=n_classes)
    elif model_name == 'PoseConv3DLarge':
        return PoseConv3DLarge(in_channels=in_channels, n_classes=n_classes)
    elif model_name == 'PoseSlowOnly':
        return PoseSlowOnly(in_channels=in_channels, n_classes=n_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def input_dim_for(input_type, n_joints=17):
    if input_type == 'xy':
        return n_joints * 2
    elif input_type == 'xyconf':
        return n_joints * 3
    elif input_type in ('heatmap', 'heatmap_limb', 'heatmap_joint_limb'):
        return None
    else:
        raise ValueError(f"Unknown input_type: {input_type}")


def in_channels_for(input_type, n_joints=17):
    if input_type == 'heatmap':
        return n_joints
    elif input_type == 'heatmap_limb':
        return N_LIMBS
    elif input_type == 'heatmap_joint_limb':
        return n_joints + N_LIMBS
    else:
        return None
