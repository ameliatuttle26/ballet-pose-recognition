"""
visualize_clips.py
==================
Generate skeleton overlay videos for one clip per class.
Run interactively on HPC GPU node.

Usage:
    python src/visualize_clips.py
"""

import subprocess
import json
import numpy as np
from pathlib import Path


CLIPS = {
    "BalanceDCLeft":        ("PAR/PAR0000.mp4", 18.4, 19.8),
    "BalanceDCRight":       ("PAR/PAR0008.mp4", 1.0,  2.6),
    "CabrioleDerriereLeft": ("SIE/SIE0033.mp4", 5.6,  6.6),
    "CabrioleDerriereRight":("SOL/SOL0038.mp4", 11.7, 12.8),
    "CabrioleDevantLeft":   ("SIE/SIE0030.mp4", 57.1, 58.1),
    "CabrioleDevantRight":  ("LA2/LA20007.mp4", 8.0,  9.1),
    "Courus":               ("BLU/BLU0028.mp4", 46.4, 51.4),
    "EchappeSecond":        ("ESM/ESM0036.mp4", 0.0,  2.1),
    "ExtDerriereOnLeft":    ("GIS/GIS0016.mp4", 29.1, 31.2),
    "ExtDerriereOnRight":   ("LA1/LA10018.mp4", 59.6, 61.7),
    "ExtSecondLeft":        ("LFS/LFS0002.mp4", 14.2, 15.3),
    "ExtSecondRight":       ("OD2/OD20005.mp4", 26.8, 28.5),
    "GrandJeteLeft":        ("GPM/GPM0008.mp4", 47.9, 49.0),
    "GrandJeteRight":       ("OD1/OD10016.mp4", 53.2, 54.5),
    "PirouetteLeft":        ("ESM/ESM0023.mp4", 84.7, 86.3),
    "PirouetteRight":       ("KIT/KIT0013.mp4", 35.5, 37.0),
    "SissonneFLeft":        ("TCM/TCM0019.mp4", 20.3, 21.5),
    "SissonneFRight":       ("TCM/TCM0006.mp4", 18.2, 19.4),
    "TourEnLair":           ("SIE/SIE0004.mp4", 10.3, 12.0),
}

VIDEO_BASE = Path("data/raw/videos")
OUT_DIR    = Path("data/demo_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT_DIR   = '/gpfsnyu/home/aat9362/.cache/torch/hub/checkpoints'
MMPOSE_DIR = '/gpfsnyu/home/aat9362/.conda/envs/mmpose-stable/lib/python3.10/site-packages/mmpose/.mim'


def extract_clip(video_path, start, end, out_path):
    """Cut clip from video using ffmpeg."""
    duration = end - start
    subprocess.run([
        'ffmpeg', '-ss', str(start), '-i', str(video_path),
        '-t', str(duration), '-c:v', 'libx264', '-c:a', 'aac',
        str(out_path), '-y', '-loglevel', 'error'
    ], check=True)


def run_visualization(clip_path, out_path, inferencer):
    """Run MMPose on video clip and save annotated output."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (w, h)
    )

    # COCO skeleton connections
    SKELETON = [
        (0,1),(0,2),(1,3),(2,4),        # head
        (5,6),(5,7),(7,9),(6,8),(8,10), # arms
        (5,11),(6,12),(11,12),          # torso
        (11,13),(13,15),(12,14),(14,16) # legs
    ]

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame temporarily
        tmp_path = f'/tmp/vis_frame_{frame_idx:04d}.jpg'
        cv2.imwrite(tmp_path, frame)

        # Run pose
        result = next(inferencer(tmp_path, return_datasamples=True))
        predictions = result['predictions']

        if predictions and len(predictions[0].pred_instances.keypoints) > 0:
            kps    = predictions[0].pred_instances.keypoints[0].astype(int)
            scores = predictions[0].pred_instances.keypoint_scores[0]

            # Draw skeleton
            for i, j in SKELETON:
                if scores[i] > 0.3 and scores[j] > 0.3:
                    cv2.line(frame, tuple(kps[i]), tuple(kps[j]),
                             (0, 255, 0), 2)

            # Draw joints
            for k, (x, y) in enumerate(kps):
                if scores[k] > 0.3:
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()


def main():
    from mmpose.apis import MMPoseInferencer
    from mmpose.utils import register_all_modules
    register_all_modules()

    print("Loading models...")
    inferencer = MMPoseInferencer(
        pose2d=f'{CKPT_DIR}/rtmpose-m_8xb256-420e_body8-256x192.py',
        pose2d_weights=f'{CKPT_DIR}/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth',
        det_model=f'{MMPOSE_DIR}/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py',
        det_weights=f'{CKPT_DIR}/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth',
        det_cat_ids=0,
        device='cuda:0'
    )
    print("Models loaded.\n")

    for label, (video_rel, start, end) in CLIPS.items():
        print(f"Processing {label}...")
        video_path = VIDEO_BASE / video_rel

        if not video_path.exists():
            print(f"  [SKIP] Video not found: {video_path}")
            continue

        # Extract clip
        clip_path = OUT_DIR / f"{label}_clip.mp4"
        extract_clip(video_path, start, end, clip_path)

        # Run visualization
        out_path = OUT_DIR / f"{label}_pose.mp4"
        run_visualization(clip_path, out_path, inferencer)
        print(f"  Saved: {out_path}")

    print(f"\nDone. Videos saved to {OUT_DIR}")


if __name__ == '__main__':
    main()
