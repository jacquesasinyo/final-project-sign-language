"""Push-to-record webcam demo (RT-01..RT-03) and the RT-04 consistency check.

Live and file input share one path: frames -> tsl.landmarks.extract_frames ->
tsl.features.clip_features -> model (PRE-07). Nothing here re-implements preprocessing.

Usage:
    python -m tsl.realtime --run outputs/runs/<run>                      # live demo
    python -m tsl.realtime --run outputs/runs/<run> --video <clip.mp4>   # RT-04 check
    python -m tsl.realtime --check-hands                                  # D-08 orientation check
    python -m tsl.realtime --list-cameras                                 # find the built-in camera index
Add --camera N to choose a camera (macOS may give index 0 to an iPhone via Continuity Camera).

Keys (live): Space = start/stop recording, f = toggle input flip, q = quit.
The preview is always shown mirrored (selfie view); only `flip_input` changes what MediaPipe sees.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from tsl.features import ClipFailed, clip_features
from tsl.landmarks import extract_frames, read_video
from tsl.utils import load_config


def load_run(run_dir):
    import keras
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "config.json").read_text())
    if not (run_dir / "model.keras").exists():
        raise SystemExit(f"No model.keras in {run_dir}")
    return keras.models.load_model(run_dir / "model.keras"), meta["label_names"], meta["config"]["seq_len"]


def predict_frames(frames, model, seq_len):
    """Shared prediction path. Returns (probs, features, raw) or raises ClipFailed / ValueError."""
    raw = extract_frames(frames)
    no_hands = np.isnan(raw["left_hand"]).all() and np.isnan(raw["right_hand"]).all()
    if no_hands:
        raise ValueError("no hand detected")
    feats = clip_features(raw, seq_len)
    probs = model.predict(feats[None], verbose=0)[0]
    return probs, feats, raw


def consistency_check(cfg, model, labels, seq_len, video):
    """RT-04: feed an AUTSL file through the live path and compare with the cached offline features."""
    video = Path(video)
    t0 = time.perf_counter()
    probs, live_feats, _ = predict_frames(read_video(video), model, seq_len)
    dt = time.perf_counter() - t0
    import csv
    with open(cfg["manifest_path"], newline="", encoding="utf-8") as f:
        row = next((r for r in csv.DictReader(f) if r["sample_id"] == video.name.replace("_color.mp4", "")), None)
    if row is None:
        print(f"WARNING: {video.name} is not in the manifest (not one of the selected classes); "
              "the prediction is meaningless. Pick a clip from outputs/manifest.csv.")
    print(f"true label: {row['label'] if row else 'n/a'}")
    print(f"live path: {labels[probs.argmax()]} ({probs.max():.3f}) in {dt:.2f}s")
    cached = Path(cfg["cache_dir"]) / video.parent.name / (video.name.replace("_color.mp4", ".npz"))
    if cached.exists():
        off_feats = clip_features(dict(np.load(cached)), seq_len)
        off_probs = model.predict(off_feats[None], verbose=0)[0]
        print(f"offline cache: {labels[off_probs.argmax()]} ({off_probs.max():.3f}); "
              f"max |feature diff| {np.abs(live_feats - off_feats).max():.2e}; "
              f"same class: {off_probs.argmax() == probs.argmax()}")
    else:
        print(f"(no cached offline features at {cached} to compare)")


def put(img, text, y, color=(255, 255, 255)):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def open_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {index}; try --list-cameras")
    return cap


def list_cameras(max_index=4):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        print(f"camera {i}: " + (f"{frame.shape[1]}x{frame.shape[0]}" if ok else "not available"))
        cap.release()
    print("The built-in MacBook camera is usually 1280x720; an iPhone (Continuity Camera) is often larger.")


def check_hands(cfg):
    """D-08: show which Holistic label (LEFT/RIGHT) each of your hands gets, raw vs flipped input."""
    import mediapipe as mp
    cap = open_camera(cfg["demo"]["camera_index"])
    flip_input = cfg["demo"]["flip_input"]
    with mp.solutions.holistic.Holistic() as hol:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped delivering frames (disconnected?); exiting.")
                break
            inp = cv2.flip(frame, 1) if flip_input else frame
            res = hol.process(cv2.cvtColor(inp, cv2.COLOR_BGR2RGB))
            view = cv2.flip(frame, 1)  # selfie preview
            h, w = view.shape[:2]
            for lm, name, col in [(res.left_hand_landmarks, "LEFT", (255, 0, 0)),
                                  (res.right_hand_landmarks, "RIGHT", (0, 0, 255))]:
                if lm:
                    x = lm.landmark[0].x if flip_input else 1 - lm.landmark[0].x  # into preview coordinates
                    cv2.putText(view, name, (int(x * w), int(lm.landmark[0].y * h)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 3)
            put(view, f"input flipped: {flip_input}  (f = toggle, q = quit)", 30)
            put(view, "Raise ONLY your dominant hand and note its label", 60)
            cv2.imshow("hand check", view)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("f"):
                flip_input = not flip_input
    cap.release()
    cv2.destroyAllWindows()


def live(cfg, model, labels, seq_len, flip_input):
    d = cfg["demo"]
    cap = open_camera(d["camera_index"])
    recording, frames, t_start, message = False, [], 0.0, "Press Space to record a sign"

    def finish():
        dur = time.monotonic() - t_start
        if dur < d["min_seconds"]:
            return f"Too short ({dur:.1f}s < {d['min_seconds']}s), try again"
        t0 = time.perf_counter()
        try:
            probs, _, _ = predict_frames(frames, model, seq_len)
        except (ValueError, ClipFailed) as e:
            return f"No prediction: {e}"
        delay = time.perf_counter() - t0
        i = int(probs.argmax())
        if probs[i] < d["confidence_threshold"]:
            return f"uncertain / no prediction ({probs[i]:.2f})  [{delay:.2f}s]"
        return f"{labels[i]}  ({probs[i]:.2f})  [{delay:.2f}s]"

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera stopped delivering frames (disconnected?); exiting.")
            break
        if recording:
            frames.append(cv2.flip(frame, 1) if flip_input else frame)
            if time.monotonic() - t_start >= d["max_seconds"]:  # RT-02 auto-stop
                recording = False
                message = finish()
        view = cv2.flip(frame, 1)
        if recording:
            cv2.circle(view, (view.shape[1] - 30, 30), 12, (0, 0, 255), -1)
            put(view, f"REC {time.monotonic() - t_start:.1f}s", 30, (0, 0, 255))
        else:
            put(view, message, 30)
        put(view, "Space: record  q: quit", view.shape[0] - 15)
        cv2.imshow("TID demo", view)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord(" "):
            if recording:
                recording = False
                message = "Processing..."
                cv2.imshow("TID demo", view)
                cv2.waitKey(1)
                message = finish()
            else:
                recording, frames, t_start = True, [], time.monotonic()
    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--run", help="run directory with model.keras")
    ap.add_argument("--video", help="RT-04: feed this video file through the live path")
    ap.add_argument("--check-hands", action="store_true", help="D-08 handedness check (webcam)")
    ap.add_argument("--camera", type=int, help="camera index (overrides demo.camera_index in the config)")
    ap.add_argument("--list-cameras", action="store_true", help="probe camera indices 0-3")
    ap.add_argument("--flip-input", action=argparse.BooleanOptionalAction, default=None,
                    help="flip webcam frames before MediaPipe (default: demo.flip_input in the config, D-08)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.list_cameras:
        return list_cameras()
    if args.camera is not None:
        cfg["demo"]["camera_index"] = args.camera
    if args.check_hands:
        return check_hands(cfg)
    if not args.run:
        raise SystemExit("--run is required (a directory under outputs/runs/)")
    model, labels, seq_len = load_run(args.run)
    if args.video:
        return consistency_check(cfg, model, labels, seq_len, args.video)
    flip = cfg["demo"]["flip_input"] if args.flip_input is None else args.flip_input
    print(f"input flip: {flip}")
    live(cfg, model, labels, seq_len, flip)


if __name__ == "__main__":
    main()
