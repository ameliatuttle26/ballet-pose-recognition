"""
extract_poses.py
================
Step 3 of the pipeline.

Supports chunk-based parallel processing — split the clips across
multiple SLURM array jobs using --chunk and --total-chunks.

Usage:
    # Single job
    python src/extract_poses.py --config config.yaml --split train

    # Chunk mode (for SLURM array)
    python src/extract_poses.py --config config.yaml --split train \
        --chunk 0 --total-chunks 8
"""

import json
import argparse
import yaml
import subprocess
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm


CKPT_DIR   = '/gpfsnyu/home/aat9362/.cache/torch/hub/checkpoints'
MMPOSE_DIR = '/gpfsnyu/home/aat9362/.conda/envs/mmpose-stable/lib/python3.10/site-packages/mmpose/.mim'

DET_CONFIG  = f'{MMPOSE_DIR}/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py'
DET_CKPT    = f'{CKPT_DIR}/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'
POSE_CONFIG = f'{CKPT_DIR}/rtmpose-m_8xb256-420e_body8-256x192.py'
POSE_CKPT   = f'{CKPT_DIR}/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth'


def load_inferencer(device):
    from mmpose.apis import MMPoseInferencer
    return MMPoseInferencer(
        pose2d=POSE_CONFIG,
        pose2d_weights=POSE_CKPT,
        det_model=DET_CONFIG,
        det_weights=DET_CKPT,
        det_cat_ids=0,
        device=device,
    )


def extract_frames(video_path, start, end, n_frames, tmp_dir):
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    duration = end - start
    times = np.linspace(start, end - duration / n_frames, n_frames)
    frame_paths = []
    for i, t in enumerate(times):
        out_path = tmp_dir / f"frame_{i:04d}.jpg"
        subprocess.run([
            'ffmpeg', '-ss', str(t), '-i', str(video_path),
            '-frames:v', '1', '-q:v', '2', str(out_path),
            '-y', '-loglevel', 'error'
        ], check=True)
        frame_paths.append(out_path)
    return frame_paths


def run_pose(inferencer, frame_paths):
    all_keypoints = []
    for frame_path in frame_paths:
        result = next(inferencer(str(frame_path), return_datasamples=True))
        predictions = result['predictions']
        if predictions and len(predictions[0].pred_instances.keypoints) > 0:
            kps    = predictions[0].pred_instances.keypoints[0]
            scores = predictions[0].pred_instances.keypoint_scores[0]
            kps_with_conf = np.concatenate(
                [kps, scores[:, None]], axis=1
            ).astype(np.float32)
        else:
            kps_with_conf = np.zeros((17, 3), dtype=np.float32)
        all_keypoints.append(kps_with_conf)
    return np.stack(all_keypoints, axis=0)


def process_clip(clip, inferencer, config, video_base_dir, keypoints_dir):
    clip_id  = clip['clip_id']
    out_path = keypoints_dir / f"{clip_id}.npy"

    if out_path.exists():
        return 'skipped'

    video_path = video_base_dir / clip['video_rel_path']
    if not video_path.exists():
        return 'missing'

    n_frames = config['dataset']['n_frames']
    tmp_dir  = Path(f'/tmp/ballet_frames/{clip_id}')

    try:
        frame_paths = extract_frames(
            video_path, clip['start'], clip['end'], n_frames, tmp_dir
        )
        keypoints = run_pose(inferencer, frame_paths)
        np.save(out_path, keypoints)
        return 'done'

    except Exception as e:
        tqdm.write(f"  [ERROR] {clip_id}: {e}")
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

    # Chunk mode — each job handles a slice of the clips
    if args.total_chunks > 1:
        chunk_size = len(clips) // args.total_chunks
        start_idx  = args.chunk * chunk_size
        end_idx    = start_idx + chunk_size if args.chunk < args.total_chunks - 1 else len(clips)
        clips      = clips[start_idx:end_idx]
        print(f"Chunk {args.chunk}/{args.total_chunks}: clips {start_idx}-{end_idx} ({len(clips)} clips)")

    if args.limit:
        clips = clips[:args.limit]

    device = 'cpu' if args.cpu else 'cuda:0'
    print(f"Processing {len(clips)} clips from '{args.split}' split on {device}")

    print("Loading models...")
    inferencer = load_inferencer(device)
    print("Models loaded.\n")

    counts = {'done': 0, 'skipped': 0, 'missing': 0, 'error': 0}

    with tqdm(clips, unit='clip', dynamic_ncols=True) as pbar:
        for clip in pbar:
            status = process_clip(
                clip, inferencer, config, video_base_dir, keypoints_dir
            )
            counts[status] += 1
            pbar.set_postfix(
                done=counts['done'],
                skip=counts['skipped'],
                err=counts['error'],
                miss=counts['missing'],
            )

    print(f"\nFinished: {counts}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--chunk', type=int, default=0)
    parser.add_argument('--total-chunks', type=int, default=1)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    main(args)
