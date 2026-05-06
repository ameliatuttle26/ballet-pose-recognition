"""
dataset.py
==========
PyTorch Dataset for ballet pose keypoints.

Input representations:
    'xy'                 -> [T, 34]
    'xyconf'             -> [T, 51]
    'heatmap'            -> [T, 17, H, W]
    'heatmap_limb'       -> [T, 16, H, W]
    'heatmap_joint_limb' -> [T, 33, H, W]
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from models import (keypoints_to_heatmaps, keypoints_to_limb_heatmaps,
                    keypoints_to_joint_limb_heatmaps)


class BalletPoseDataset(Dataset):
    def __init__(self, split_json, keypoints_dir, label_type='fine',
                 input_type='xyconf', n_frames=32, normalize=True,
                 heatmap_size=56, heatmap_sigma=3.0):
        self.keypoints_dir  = Path(keypoints_dir)
        self.label_type     = label_type
        self.input_type     = input_type
        self.n_frames       = n_frames
        self.normalize      = normalize
        self.heatmap_size   = heatmap_size
        self.heatmap_sigma  = heatmap_sigma

        with open(split_json) as f:
            self.clips = json.load(f)

        available = []
        for clip in self.clips:
            if (self.keypoints_dir / f"{clip['clip_id']}.npy").exists():
                available.append(clip)

        n_missing = len(self.clips) - len(available)
        if n_missing > 0:
            print(f"[Dataset] Warning: {n_missing} clips missing keypoints, skipping")
        self.clips = available

        label_key = f'label_{label_type}_idx'
        self.n_classes = len(set(c[label_key] for c in self.clips))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        keypoints = np.load(self.keypoints_dir / f"{clip['clip_id']}.npy")
        keypoints = self._resample(keypoints, self.n_frames)

        if self.normalize:
            keypoints = self._normalize(keypoints)

        if self.input_type == 'xy':
            x = torch.tensor(keypoints[:, :, :2].reshape(self.n_frames, -1),
                             dtype=torch.float32)
        elif self.input_type == 'xyconf':
            x = torch.tensor(keypoints.reshape(self.n_frames, -1),
                             dtype=torch.float32)
        elif self.input_type == 'heatmap':
            x = torch.tensor(keypoints_to_heatmaps(
                keypoints, self.heatmap_size, self.heatmap_size,
                self.heatmap_sigma), dtype=torch.float32)
        elif self.input_type == 'heatmap_limb':
            x = torch.tensor(keypoints_to_limb_heatmaps(
                keypoints, self.heatmap_size, self.heatmap_size,
                self.heatmap_sigma), dtype=torch.float32)
        elif self.input_type == 'heatmap_joint_limb':
            x = torch.tensor(keypoints_to_joint_limb_heatmaps(
                keypoints, self.heatmap_size, self.heatmap_size,
                self.heatmap_sigma), dtype=torch.float32)
        else:
            raise ValueError(f"Unknown input_type: {self.input_type}")

        label_key = f'label_{self.label_type}_idx'
        y = torch.tensor(clip[label_key], dtype=torch.long)
        return x, y

    def _resample(self, keypoints, n_frames):
        T = keypoints.shape[0]
        if T == n_frames:
            return keypoints
        indices = np.linspace(0, T - 1, n_frames)
        lo = np.floor(indices).astype(int)
        hi = np.minimum(lo + 1, T - 1)
        alpha = (indices - lo)[:, None, None]
        return (1 - alpha) * keypoints[lo] + alpha * keypoints[hi]

    def _normalize(self, keypoints):
        conf = keypoints[:, :, 2]
        xy = keypoints[:, :, :2].copy()
        visible = conf > 0.1
        if visible.any():
            visible_xy = xy[visible]
            xy_min = visible_xy.min(axis=0)
            xy_max = visible_xy.max(axis=0)
            rng = np.where(xy_max - xy_min == 0, 1, xy_max - xy_min)
            xy = (xy - xy_min) / rng
        keypoints = keypoints.copy()
        keypoints[:, :, :2] = xy
        return keypoints

    def get_class_weights(self):
        label_key = f'label_{self.label_type}_idx'
        labels = [c[label_key] for c in self.clips]
        counts = np.bincount(labels, minlength=self.n_classes)
        counts = np.maximum(counts, 1)
        weights = 1.0 / counts
        weights = weights / weights.sum() * self.n_classes
        return torch.tensor(weights, dtype=torch.float32)
