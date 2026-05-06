"""
predict_clips.py
================
Run all trained models on specific clips and show predictions.
Usage:
    python src/predict_clips.py --config config.yaml
"""

import json
import yaml
import numpy as np
import torch
from pathlib import Path
from dataset import BalletPoseDataset
from models import build_model, input_dim_for, in_channels_for


DEMO_CLIPS = [
    # Easy - Pirouette
    {"clip_id": "ACM_13_9zPDxyiW",  "label": "PirouetteLeft",        "group": "Easy"},
    {"clip_id": "ACM_13_LJ02kFza",  "label": "PirouetteLeft",        "group": "Easy"},
    # Hard - Balance L/R
    {"clip_id": "FRA_8_BJtnXBE6",   "label": "BalanceDCLeft",        "group": "Hard"},
    {"clip_id": "FRA_8_H3kXWwvQ",   "label": "BalanceDCRight",       "group": "Hard"},
    # Interesting - Cabriole
    {"clip_id": "ACM_9_DNLMTzZX",   "label": "CabrioleDerriereRight","group": "Interesting"},
    {"clip_id": "ACM_13_41UlyXu1",  "label": "CabrioleDerriereRight","group": "Interesting"},
]

EXPERIMENTS = [
    {"exp": "exp1", "name": "MLP (xy)"},
    {"exp": "exp2", "name": "MLP (xyconf)"},
    {"exp": "exp3", "name": "TemporalCNN"},
    {"exp": "exp4", "name": "PoseConv3D (joint only)"},
    {"exp": "exp7", "name": "PoseConv3D (joint+limb)"},
    {"exp": "exp8", "name": "PoseSlowOnly (joint+limb)"},
]


def load_model_for_exp(exp_key, config, n_classes, device):
    exp_config = config['experiments'][exp_key]
    model_name  = exp_config['model']
    input_type  = exp_config['input']
    n_frames    = exp_config.get('n_frames', config['dataset']['n_frames'])
    n_joints    = config['dataset']['n_joints']
    in_dim      = input_dim_for(input_type, n_joints=n_joints)
    in_channels = in_channels_for(input_type, n_joints=n_joints)

    model = build_model(model_name, n_frames, in_dim, n_classes,
                        n_joints=n_joints, in_channels=in_channels).to(device)

    ckpt_path = Path('results/models') / f"{exp_config['name']}_best.pt"
    if not ckpt_path.exists():
        return None, exp_config
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model, exp_config


def get_clip_tensor(clip_id, exp_config, config, clips_data):
    from dataset import BalletPoseDataset
    import numpy as np

    kp_path = Path(config['paths']['keypoints_dir']) / f"{clip_id}.npy"
    if not kp_path.exists():
        return None

    keypoints = np.load(kp_path)
    n_frames      = exp_config.get('n_frames', config['dataset']['n_frames'])
    input_type    = exp_config['input']
    heatmap_size  = config['dataset'].get('heatmap_size', 56)
    heatmap_sigma = config['dataset'].get('heatmap_sigma', 3.0)

    # Resample
    T = keypoints.shape[0]
    if T != n_frames:
        indices = np.linspace(0, T-1, n_frames)
        lo = np.floor(indices).astype(int)
        hi = np.minimum(lo+1, T-1)
        alpha = (indices - lo)[:, None, None]
        keypoints = (1-alpha)*keypoints[lo] + alpha*keypoints[hi]

    # Normalize
    conf = keypoints[:, :, 2]
    xy = keypoints[:, :, :2].copy()
    visible = conf > 0.1
    if visible.any():
        vxy = xy[visible]
        rng = np.where(vxy.max(0)-vxy.min(0)==0, 1, vxy.max(0)-vxy.min(0))
        xy = (xy - vxy.min(0)) / rng
    keypoints = keypoints.copy()
    keypoints[:, :, :2] = xy

    from models import (keypoints_to_heatmaps, keypoints_to_joint_limb_heatmaps)

    if input_type == 'xy':
        x = torch.tensor(keypoints[:,:,:2].reshape(n_frames,-1), dtype=torch.float32)
    elif input_type == 'xyconf':
        x = torch.tensor(keypoints.reshape(n_frames,-1), dtype=torch.float32)
    elif input_type == 'heatmap':
        x = torch.tensor(keypoints_to_heatmaps(keypoints, heatmap_size, heatmap_size, heatmap_sigma), dtype=torch.float32)
    elif input_type == 'heatmap_joint_limb':
        x = torch.tensor(keypoints_to_joint_limb_heatmaps(keypoints, heatmap_size, heatmap_size, heatmap_sigma), dtype=torch.float32)
    else:
        return None

    return x.unsqueeze(0)  # [1, T, ...]


def main():
    config = yaml.safe_load(open('config.yaml'))
    clips_data = json.load(open(config['paths']['clips_index']))
    idx_to_fine = clips_data['idx_to_fine']
    fine_to_idx = clips_data['fine_to_idx']
    n_classes = clips_data['n_fine_classes']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load all models
    models = {}
    for exp_info in EXPERIMENTS:
        model, exp_config = load_model_for_exp(exp_info['exp'], config, n_classes, device)
        if model is None:
            print(f"  [SKIP] {exp_info['name']} — checkpoint not found")
            continue
        models[exp_info['exp']] = (model, exp_config, exp_info['name'])

    print(f"\n{'Clip':<25} {'True Label':<25} {'Group':<12}", end='')
    for exp_info in EXPERIMENTS:
        if exp_info['exp'] in models:
            print(f" {exp_info['name']:<25}", end='')
    print()
    print('-' * 200)

    for demo in DEMO_CLIPS:
        clip_id    = demo['clip_id']
        true_label = demo['label']
        group      = demo['group']

        print(f"{clip_id:<25} {true_label:<25} {group:<12}", end='')

        for exp_info in EXPERIMENTS:
            if exp_info['exp'] not in models:
                print(f" {'N/A':<25}", end='')
                continue

            model, exp_config, name = models[exp_info['exp']]
            x = get_clip_tensor(clip_id, exp_config, config, clips_data)

            if x is None:
                print(f" {'NO KP':<25}", end='')
                continue

            with torch.no_grad():
                logits = model(x.to(device))
                pred_idx = logits.argmax(dim=1).item()
                pred_label = idx_to_fine[str(pred_idx)]
                correct = '✓' if pred_label == true_label else '✗'
                print(f" {correct} {pred_label:<23}", end='')

        print()

    print('\n✓ = correct, ✗ = incorrect')


if __name__ == '__main__':
    main()
