"""
extract_poses.py
================
Step 3 of the pipeline.

For each clip in a split, extracts frames using ffmpeg and runs MMPose
to get keypoints. Saves one .npy file per clip with shape [T, 17, 3]
(T frames, 17 COCO joints, x/y/confidence).

Run on HPC via SLURM. Supports resuming — skips clips already done.

Usage:
    python src/extract_poses.py --config config.yaml --split train
    python src/extract_poses.py --config config.yaml --split train --limit 10
"""

import os
import json
import argparse
import yaml
import subprocess
import shutil
import numpy as np
from pathlib import Path


CKPT_DIR = '/gpfsnyu/home/aat9362/.cache/torch/hub/checkpoints'
MMPOSE_DIR = '/gpfsnyu/home/aat9362/.conda/envs/mmpose-stable/lib/python3.10/site-packages/mmpose/.mim'

DET_CONFIG  = f'{MMPOSE_DIR}/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py'
DET_CKPT    = f'{CKPT_DIR}/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'
POSE_CONFIG = f'{CKPT_DIR}/rtmpose-m_8xb256-420e_body8-256x192.py'
POSE_CKPT   = f'{CKPT_DIR}/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth'


def load_inferencer(device):
    from mmpose.apis import MMPoseInferencer
    inferencer = MMPoseInferencer(
        pose2d=POSE_CONFIG,
        pose2d_weights=POSE_CKPT,
        det_model=DET_CONFIG,
        det_weights=DET_CKPT,
        det_cat_ids=0,
        device=device,
    )
    return inferencer


def extract_frames(video_path, start, end, fps, n_frames, tmp_dir):
    """Extract n_frames evenly spaced between start and end seconds."""
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    duration = end - start
    times = np.linspace(start, end - duration / n_frames, n_frames)
    frame_paths = []

    for i, t in enumerate(times):
        out_path = tmp_dir / f"frame_{i:04d}.jpg"
        cmd = [
            'ffmpeg', '-ss', str(t), '-i', str(video_path),
            '-frames:v', '1', '-q:v', '2', str(out_path),
            '-y', '-loglevel', 'error'
        ]
        subprocess.run(cmd, check=True)
        frame_paths.append(out_path)

    return frame_paths


def run_pose(inferencer, frame_paths):
    """Run MMPose on a list of frames, return [T, 17, 3] array."""
    all_keypoints = []

    for frame_path in frame_paths:
        result = next(inferencer(str(frame_path), return_datasamples=True))
        predictions = result['predictions']

        if predictions and len(predictions[0].pred_instances.keypoints) > 0:
            kps = predictions[0].pred_instances.keypoints[0]       # [17, 2]
            scores = predictions[0].pred_instances.keypoint_scores[0]  # [17]
            kps_with_conf = np.concatenate(
                [kps, scores[:, None]], axis=1
            ).astype(np.float32)  # [17, 3]
        else:
            kps_with_conf = np.zeros((17, 3), dtype=np.float32)

        all_keypoints.append(kps_with_conf)

    return np.stack(all_keypoints, axis=0)  # [T, 17, 3]


def process_clip(clip, inferencer, config, video_base_dir, keypoints_dir):
    clip_id = clip['clip_id']
    out_path = keypoints_dir / f"{clip_id}.npy"

    if out_path.exists():
        return 'skipped'

    video_path = video_base_dir / clip['video_rel_path']
    if not video_path.exists():
        return 'missing'

    fps     = config['dataset']['fps']
    n_frames = config['dataset']['n_frames']
    tmp_dir = Path(f'/tmp/ballet_frames/{clip_id}')

    try:
        frame_paths = extract_frames(
            video_path, clip['start'], clip['end'], fps, n_frames, tmp_dir
        )
        keypoints = run_pose(inferencer, frame_paths)
        np.save(out_path, keypoints)
        return 'done'

    except Exception as e:
        print(f"  [ERROR] {clip_id}: {e}")
        return 'error'

    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


def main(args):
    config = yaml.safe_load(open(args.config))

    video_base_dir = Path(config['paths']['raw_videos'])
    keypoints_dir  = Path(config['paths']['keypoints_dir'])
    splits_dir     = Path(config['paths']['splits_dir'])
    keypoints_dir.mkdir(parents=True, exist_ok=True)

    split_path = splits_dir / f"{args.split}.json"
    if not split_path.exists():
        print(f"[ERROR] Split not found: {split_path}")
        return

    with open(split_path) as f:
        clips = json.load(f)

    if args.limit:
        clips = clips[:args.limit]

    device = 'cuda:0' if not args.cpu else 'cpu'
    print(f"Processing {len(clips)} clips from '{args.split}' split on {device}")

    print("Loading models...")
    inferencer = load_inferencer(device)
    print("Models loaded.")

    counts = {'done': 0, 'skipped': 0, 'missing': 0, 'error': 0}

    for i, clip in enumerate(clips):
        status = process_clip(clip, inferencer, config, video_base_dir, keypoints_dir)
        counts[status] += 1

        if (i + 1) % 10 == 0 or (i + 1) == len(clips):
            print(f"  [{i+1}/{len(clips)}] done={counts['done']} "
                  f"skipped={counts['skipped']} "
                  f"error={counts['error']} "
                  f"missing={counts['missing']}")

    print(f"\nFinished: {counts}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    main(args)
