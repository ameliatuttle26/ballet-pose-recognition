"""
find_misclassified.py
=====================
Find one correctly classified and one misclassified clip per model.
Rules:
- Within each model: 3 examples, each from a different video/performance
- Prefer variety in true labels
- For misclassified: find clips where prediction is confidently wrong
"""

import json
import yaml
import numpy as np
import torch
from pathlib import Path
from models import build_model, input_dim_for, in_channels_for
from models import keypoints_to_heatmaps, keypoints_to_joint_limb_heatmaps


EXPERIMENTS = [
    {"exp": "exp1", "name": "MLP_xy"},
    {"exp": "exp2", "name": "MLP_xyconf"},
    {"exp": "exp3", "name": "TemporalCNN"},
    {"exp": "exp4", "name": "PoseConv3D_joint"},
    {"exp": "exp7", "name": "PoseConv3D_limb"},
    {"exp": "exp8", "name": "PoseSlowOnly"},
]


def load_model(exp_key, config, n_classes, device):
    exp_config  = config['experiments'][exp_key]
    model_name  = exp_config['model']
    input_type  = exp_config['input']
    n_frames    = exp_config.get('n_frames', config['dataset']['n_frames'])
    n_joints    = config['dataset']['n_joints']
    in_dim      = input_dim_for(input_type, n_joints=n_joints)
    in_channels = in_channels_for(input_type, n_joints=n_joints)
    model = build_model(model_name, n_frames, in_dim, n_classes,
                        n_joints=n_joints, in_channels=in_channels).to(device)
    ckpt = Path('results/models') / f"{exp_config['name']}_best.pt"
    if not ckpt.exists():
        return None, exp_config
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model, exp_config


def get_tensor(kps, input_type, n_frames, heatmap_size, heatmap_sigma):
    T = kps.shape[0]
    if T != n_frames:
        idx = np.linspace(0, T-1, n_frames)
        lo  = np.floor(idx).astype(int)
        hi  = np.minimum(lo+1, T-1)
        a   = (idx - lo)[:, None, None]
        kps = (1-a)*kps[lo] + a*kps[hi]

    xy   = kps[:, :, :2].copy()
    conf = kps[:, :, 2]
    vis  = conf > 0.1
    if vis.any():
        vxy = xy[vis]
        rng = np.where(vxy.max(0)-vxy.min(0)==0, 1, vxy.max(0)-vxy.min(0))
        xy  = (xy - vxy.min(0)) / rng
    kps = kps.copy()
    kps[:, :, :2] = xy

    if input_type == 'xy':
        return torch.tensor(kps[:,:,:2].reshape(n_frames,-1), dtype=torch.float32)
    elif input_type == 'xyconf':
        return torch.tensor(kps.reshape(n_frames,-1), dtype=torch.float32)
    elif input_type == 'heatmap':
        return torch.tensor(keypoints_to_heatmaps(kps, heatmap_size, heatmap_size, heatmap_sigma), dtype=torch.float32)
    elif input_type == 'heatmap_joint_limb':
        return torch.tensor(keypoints_to_joint_limb_heatmaps(kps, heatmap_size, heatmap_size, heatmap_sigma), dtype=torch.float32)


def main():
    config      = yaml.safe_load(open('config.yaml'))
    clips_data  = json.load(open(config['paths']['clips_index']))
    idx_to_fine = clips_data['idx_to_fine']
    n_classes   = clips_data['n_fine_classes']
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    kp_dir     = Path(config['paths']['keypoints_dir'])
    test_clips = json.load(open('data/processed/splits/test.json'))
    test_clips = [c for c in test_clips if (kp_dir / f"{c['clip_id']}.npy").exists()]

    all_clips_meta = {c['clip_id']: c for c in clips_data['clips']}

    results = {}

    for exp_info in EXPERIMENTS:
        model, exp_config = load_model(exp_info['exp'], config, n_classes, device)
        if model is None:
            print(f"{exp_info['name']}: checkpoint not found")
            continue

        input_type    = exp_config['input']
        n_frames      = exp_config.get('n_frames', config['dataset']['n_frames'])
        heatmap_size  = config['dataset'].get('heatmap_size', 56)
        heatmap_sigma = config['dataset'].get('heatmap_sigma', 3.0)

        correct_clips = []
        wrong_clips   = []

        for clip in test_clips:
            clip_id  = clip['clip_id']
            true_idx = clip['label_fine_idx']
            true_lbl = idx_to_fine[str(true_idx)]
            meta     = all_clips_meta.get(clip_id, {})
            video    = meta.get('video_rel_path', '')
            perf     = video.split('/')[0] if video else ''  # e.g. ACM, FRA

            kps = np.load(kp_dir / f"{clip_id}.npy")
            x   = get_tensor(kps, input_type, n_frames, heatmap_size, heatmap_sigma)

            with torch.no_grad():
                logits   = model(x.unsqueeze(0).to(device))
                pred_idx = logits.argmax(1).item()
                pred_lbl = idx_to_fine[str(pred_idx)]

            entry = {
                'clip_id':   clip_id,
                'true':      true_lbl,
                'predicted': pred_lbl,
                'video':     video,
                'perf':      perf,
                'start':     meta.get('start', 0),
                'end':       meta.get('end', 0),
            }

            if pred_idx == true_idx:
                correct_clips.append(entry)
            else:
                wrong_clips.append(entry)

        # Pick 2 correct clips from different performances and labels
        chosen_correct = []
        used_perfs  = set()
        used_labels = set()
        for c in correct_clips:
            if c['perf'] not in used_perfs and c['true'] not in used_labels:
                chosen_correct.append(c)
                used_perfs.add(c['perf'])
                used_labels.add(c['true'])
            if len(chosen_correct) == 2:
                break

        # Pick 1 wrong clip from a different performance than chosen correct
        chosen_wrong = []
        for c in wrong_clips:
            if c['perf'] not in used_perfs:
                chosen_wrong.append(c)
                used_perfs.add(c['perf'])
            if len(chosen_wrong) == 1:
                break
        # Fallback if no new performance available
        if not chosen_wrong and wrong_clips:
            chosen_wrong.append(wrong_clips[0])

        results[exp_info['name']] = {
            'correct': chosen_correct,
            'wrong':   chosen_wrong
        }

        print(f"=== {exp_info['name']} ===")
        print(f"  Total misclassified: {len(wrong_clips)}/{len(test_clips)}")
        print()
        print("  CORRECT examples:")
        for c in chosen_correct:
            print(f"    clip_id: {c['clip_id']}")
            print(f"    true:    {c['true']}")
            print(f"    video:   {c['video']} t={c['start']:.1f}-{c['end']:.1f}")
            print()
        print("  MISCLASSIFIED example:")
        for c in chosen_wrong:
            print(f"    clip_id:   {c['clip_id']}")
            print(f"    true:      {c['true']}")
            print(f"    predicted: {c['predicted']}")
            print(f"    video:     {c['video']} t={c['start']:.1f}-{c['end']:.1f}")
            print()
        print()

    # Save results for visualization script
    with open('results/logs/misclassified_examples.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved to results/logs/misclassified_examples.json")


if __name__ == '__main__':
    main()
