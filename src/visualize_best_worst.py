"""
visualize_best_worst.py
=======================
Generate MMPose skeleton overlay videos for best and worst
predicted clips per model. Run as SLURM job on HPC.
"""

import subprocess
from pathlib import Path

CLIPS = [
    # MLP xy
    {"model": "MLP_xy",     "type": "best",  "label": "SissonneFLeft",    "video": "FRA/FRA0001.mp4", "start": 18.8, "end": 19.8},
    {"model": "MLP_xy",     "type": "worst", "label": "BalanceDCRight",   "video": "ACM/ACM0004.mp4", "start": 26.0, "end": 26.8},
    # MLP xyconf
    {"model": "MLP_xyconf", "type": "best",  "label": "SissonneFLeft",    "video": "FRA/FRA0001.mp4", "start": 18.8, "end": 19.8},
    {"model": "MLP_xyconf", "type": "worst", "label": "BalanceDCRight",   "video": "ACM/ACM0004.mp4", "start": 26.0, "end": 26.8},
    # TemporalCNN
    {"model": "TemporalCNN","type": "best",  "label": "PirouetteLeft",    "video": "ACM/ACM0000.mp4", "start": 20.7, "end": 21.2},
    {"model": "TemporalCNN","type": "worst", "label": "SissonneFRight",   "video": "FRA/FRA0001.mp4", "start": 18.0, "end": 18.8},
    # PoseConv3D joint only
    {"model": "PoseConv3D_joint", "type": "best",  "label": "TourEnLair",      "video": "ACM/ACM0000.mp4", "start": 38.3, "end": 39.2},
    {"model": "PoseConv3D_joint", "type": "worst", "label": "SissonneFRight",  "video": "FRA/FRA0001.mp4", "start": 18.0, "end": 18.8},
    # PoseConv3D joint+limb
    {"model": "PoseConv3D_limb",  "type": "best",  "label": "SissonneFLeft",   "video": "FRA/FRA0001.mp4", "start": 18.8, "end": 19.8},
    {"model": "PoseConv3D_limb",  "type": "worst", "label": "CabrioleDevantLeft","video": "GAM/GAM0008.mp4","start": 10.4, "end": 11.0},
]

VIDEO_BASE = Path("data/raw/videos")
OUT_DIR    = Path("data/best_worst_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT_DIR   = '/gpfsnyu/home/aat9362/.cache/torch/hub/checkpoints'
MMPOSE_DIR = '/gpfsnyu/home/aat9362/.conda/envs/mmpose-stable/lib/python3.10/site-packages/mmpose/.mim'


def run_mmpose_on_clip(video_path, start, end, out_name):
    duration = end - start
    tmp_clip = f'/tmp/{out_name}_clip.mp4'

    # Extract clip
    subprocess.run([
        'ffmpeg', '-ss', str(start), '-i', str(video_path),
        '-t', str(duration), '-c:v', 'mpeg4',
        tmp_clip, '-y', '-loglevel', 'error'
    ], check=True)

    # Run MMPose visualization
    subprocess.run([
        'python', f'{MMPOSE_DIR}/demo/topdown_demo_with_mmdet.py',
        f'{MMPOSE_DIR}/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py',
        f'{CKPT_DIR}/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth',
        f'{CKPT_DIR}/rtmpose-m_8xb256-420e_body8-256x192.py',
        f'{CKPT_DIR}/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth',
        '--input', tmp_clip,
        '--output-root', str(OUT_DIR),
        '--device', 'cuda:0'
    ], capture_output=True)

    # Rename output
    src = OUT_DIR / f'{out_name}_clip.mp4'
    dst = OUT_DIR / f'{out_name}.mp4'
    if src.exists():
        src.rename(dst)
    print(f'  Saved: {dst}')


def main():
    seen = set()
    for clip in CLIPS:
        out_name = f"{clip['model']}_{clip['type']}_{clip['label']}"
        video_path = VIDEO_BASE / clip['video']

        if not video_path.exists():
            print(f'[SKIP] Video not found: {video_path}')
            continue

        # Skip duplicates (same video segment for different models)
        key = f"{clip['video']}_{clip['start']}_{clip['end']}"
        if key in seen:
            # Just copy the already-generated file
            src_name = next(
                f"{c['model']}_{c['type']}_{c['label']}"
                for c in CLIPS
                if f"{c['video']}_{c['start']}_{c['end']}" == key
                and f"{c['model']}_{c['type']}_{c['label']}" != out_name
            )
            src = OUT_DIR / f'{src_name}.mp4'
            dst = OUT_DIR / f'{out_name}.mp4'
            if src.exists():
                import shutil
                shutil.copy(src, dst)
                print(f'  Copied: {dst}')
            continue
        seen.add(key)

        print(f'Processing {out_name}...')
        run_mmpose_on_clip(video_path, clip['start'], clip['end'], out_name)

    print('\nDone!')


if __name__ == '__main__':
    main()
