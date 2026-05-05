"""
visualize_local.py
==================
Generate skeleton overlay videos using pre-extracted keypoints.
Runs entirely locally on CPU — no MMPose installation needed.

Usage:
    python src/visualize_local.py

Requires:
    pip install opencv-python numpy
"""

import numpy as np
import cv2
import json
from pathlib import Path


# ------------------------------------------------------------------
# One example clip per class
# ------------------------------------------------------------------
CLIPS = {
    "BalanceDCLeft":         ("PAR/PAR0000.mp4", 18.4, 19.8),
    "BalanceDCRight":        ("PAR/PAR0008.mp4", 1.0,  2.6),
    "CabrioleDerriereLeft":  ("SIE/SIE0033.mp4", 5.6,  6.6),
    "CabrioleDerriereRight": ("SOL/SOL0038.mp4", 11.7, 12.8),
    "CabrioleDevantLeft":    ("SIE/SIE0030.mp4", 57.1, 58.1),
    "CabrioleDevantRight":   ("LA2/LA20007.mp4", 8.0,  9.1),
    "Courus":                ("BLU/BLU0028.mp4", 46.4, 51.4),
    "EchappeSecond":         ("ESM/ESM0036.mp4", 0.0,  2.1),
    "ExtDerriereOnLeft":     ("GIS/GIS0016.mp4", 29.1, 31.2),
    "ExtDerriereOnRight":    ("LA1/LA10018.mp4", 59.6, 61.7),
    "ExtSecondLeft":         ("LFS/LFS0002.mp4", 14.2, 15.3),
    "ExtSecondRight":        ("OD2/OD20005.mp4", 26.8, 28.5),
    "GrandJeteLeft":         ("GPM/GPM0008.mp4", 47.9, 49.0),
    "GrandJeteRight":        ("OD1/OD10016.mp4", 53.2, 54.5),
    "PirouetteLeft":         ("ESM/ESM0023.mp4", 84.7, 86.3),
    "PirouetteRight":        ("KIT/KIT0013.mp4", 35.5, 37.0),
    "SissonneFLeft":         ("TCM/TCM0019.mp4", 20.3, 21.5),
    "SissonneFRight":        ("TCM/TCM0006.mp4", 18.2, 19.4),
    "TourEnLair":            ("SIE/SIE0004.mp4", 10.3, 12.0),
}

# COCO 17-joint skeleton connections
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # arms
    (5, 11), (6, 12), (11, 12),               # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
]

# Joint colors by body part (BGR)
JOINT_COLORS = {
    'head':  (255, 200, 0),   # yellow
    'arms':  (0, 165, 255),   # orange
    'torso': (0, 255, 0),     # green
    'legs':  (255, 0, 128),   # pink
}

JOINT_COLOR_MAP = [
    JOINT_COLORS['head'],   # 0 nose
    JOINT_COLORS['head'],   # 1 left eye
    JOINT_COLORS['head'],   # 2 right eye
    JOINT_COLORS['head'],   # 3 left ear
    JOINT_COLORS['head'],   # 4 right ear
    JOINT_COLORS['arms'],   # 5 left shoulder
    JOINT_COLORS['arms'],   # 6 right shoulder
    JOINT_COLORS['arms'],   # 7 left elbow
    JOINT_COLORS['arms'],   # 8 right elbow
    JOINT_COLORS['arms'],   # 9 left wrist
    JOINT_COLORS['arms'],   # 10 right wrist
    JOINT_COLORS['torso'],  # 11 left hip
    JOINT_COLORS['torso'],  # 12 right hip
    JOINT_COLORS['legs'],   # 13 left knee
    JOINT_COLORS['legs'],   # 14 right knee
    JOINT_COLORS['legs'],   # 15 left ankle
    JOINT_COLORS['legs'],   # 16 right ankle
]

SKELETON_COLOR = (0, 255, 128)  # bright green for bones

VIDEO_BASE  = Path("data/demo_videos")
KP_DIR      = Path("data/processed/keypoints")
OUT_DIR     = Path("data/demo_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_clip_id(label, video_rel, start):
    """Find the keypoint file for this clip."""
    clips_data = json.load(open("data/processed/clips.json"))
    for clip in clips_data['clips']:
        if (clip['label_fine'] == label and
            clip['video_rel_path'] == video_rel and
            abs(clip['start'] - start) < 0.5):
            return clip['clip_id']
    return None


def draw_skeleton_on_frame(frame, keypoints, scores, conf_threshold=0.3):
    """Draw skeleton overlay on a single frame."""
    h, w = frame.shape[:2]

    # Denormalize keypoints from [0,1] to pixel coords
    kps_px = keypoints.copy()
    kps_px[:, 0] *= w
    kps_px[:, 1] *= h
    kps_px = kps_px.astype(int)

    # Draw bones
    for i, j in SKELETON:
        if scores[i] > conf_threshold and scores[j] > conf_threshold:
            pt1 = tuple(kps_px[i])
            pt2 = tuple(kps_px[j])
            cv2.line(frame, pt1, pt2, SKELETON_COLOR, 2, cv2.LINE_AA)

    # Draw joints
    for k in range(len(kps_px)):
        if scores[k] > conf_threshold:
            pt = tuple(kps_px[k])
            color = JOINT_COLOR_MAP[k]
            cv2.circle(frame, pt, 5, color, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 5, (255, 255, 255), 1, cv2.LINE_AA)  # white outline

    return frame


def process_clip(label, video_rel, start, end):
    """Generate a skeleton overlay video for one clip."""
    video_path = VIDEO_BASE / video_rel
    if not video_path.exists():
        print(f"  [SKIP] Video not found: {video_path}")
        return

    # Find keypoint file
    clip_id = find_clip_id(label, video_rel, start)
    if clip_id is None:
        print(f"  [SKIP] No clip_id found for {label}")
        return

    kp_path = KP_DIR / f"{clip_id}.npy"
    if not kp_path.exists():
        print(f"  [SKIP] Keypoints not extracted yet: {kp_path}")
        return

    # Load keypoints [T, 17, 3]
    keypoints = np.load(kp_path)
    T = keypoints.shape[0]

    # Open video and seek to start
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Seek to start frame
    start_frame = int(start * fps)
    end_frame   = int(end * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    out_path = OUT_DIR / f"demo_{label}.mp4"
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (w, h)
    )

    # Map T keypoint frames evenly to video frames
    n_video_frames = end_frame - start_frame
    frame_idx = 0

    while frame_idx < n_video_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Map video frame to keypoint frame
        kp_idx = int(frame_idx / n_video_frames * T)
        kp_idx = min(kp_idx, T - 1)

        kps    = keypoints[kp_idx, :, :2]  # [17, 2]
        scores = keypoints[kp_idx, :, 2]   # [17]

        frame = draw_skeleton_on_frame(frame, kps, scores)

        # Add label text
        cv2.putText(frame, label, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(frame, label, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 1)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"  Saved: {out_path}")


def main():
    print(f"Generating skeleton overlay videos...")
    print(f"Output directory: {OUT_DIR}\n")

    for label, (video_rel, start, end) in CLIPS.items():
        print(f"Processing {label}...")
        process_clip(label, video_rel, start, end)

    print(f"\nDone! Videos saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
