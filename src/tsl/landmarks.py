"""MediaPipe Holistic landmark extraction, shared by offline caching and the live demo (PRE-07).

`extract_frames` is the only place MediaPipe is called. It returns raw per-frame
landmarks, with NaN for anything not detected. Normalisation happens later in
`tsl.features`, so the cache never has to be rebuilt when preprocessing changes (PRE-05).

Frames are passed as given (never flipped); see docs/decisions.md D-08.
"""
import cv2
import mediapipe as mp
import numpy as np

N_HAND, N_POSE = 21, 33


def _to_array(landmarks, n, with_visibility=False):
    cols = 4 if with_visibility else 3
    if landmarks is None:
        return np.full((n, cols), np.nan, dtype=np.float32)
    rows = [[p.x, p.y, p.z, p.visibility] if with_visibility else [p.x, p.y, p.z] for p in landmarks.landmark]
    return np.asarray(rows, dtype=np.float32)


def extract_frames(frames_bgr):
    """Run Holistic on a sequence of BGR frames (one clip / one recording).

    A fresh Holistic instance is used per clip, so tracking state never leaks between clips.
    Returns dict with left_hand (F,21,3), right_hand (F,21,3), pose (F,33,4: x,y,z,visibility),
    all NaN where not detected, and image_size (width, height).
    """
    left, right, pose = [], [], []
    size = None
    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hol:
        for frame in frames_bgr:
            size = (frame.shape[1], frame.shape[0])
            res = hol.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            left.append(_to_array(res.left_hand_landmarks, N_HAND))
            right.append(_to_array(res.right_hand_landmarks, N_HAND))
            pose.append(_to_array(res.pose_landmarks, N_POSE, with_visibility=True))
    if size is None:
        raise ValueError("No frames to process")
    return dict(left_hand=np.stack(left), right_hand=np.stack(right), pose=np.stack(pose),
                image_size=np.asarray(size, dtype=np.int32))


def read_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def extract_video(path):
    return extract_frames(read_video(path))
