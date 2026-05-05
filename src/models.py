"""
models.py
=========
MLP and TemporalCNN model definitions.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Flatten-then-MLP baseline.
    Input:  [B, T, D]
    Output: [B, n_classes]
    """
    def __init__(self, n_frames, input_dim, n_classes, hidden_dims=(512, 256), dropout=0.3):
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
        x = x.view(B, -1)
        return self.net(x)


class TemporalCNN(nn.Module):
    """
    1D Temporal CNN.
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
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
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


def build_model(model_name, n_frames, input_dim, n_classes):
    if model_name == 'MLP':
        return MLP(n_frames=n_frames, input_dim=input_dim, n_classes=n_classes,
                   hidden_dims=(512, 256), dropout=0.3)
    elif model_name == 'TemporalCNN':
        return TemporalCNN(input_dim=input_dim, n_classes=n_classes,
                           channels=(64, 128, 256), kernel_size=3, dropout=0.3)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def input_dim_for(input_type, n_joints=17):
    if input_type == 'xy':
        return n_joints * 2    # 34
    elif input_type == 'xyconf':
        return n_joints * 3    # 51
    else:
        raise ValueError(f"Unknown input_type: {input_type}")
