"""Push-to-record webcam demo (RT-01..RT-03) and the RT-04 consistency check.

Live and file input share one path: frames -> tsl.landmarks.extract_frames ->
tsl.features.clip_features -> model (PRE-07). Nothing here re-implements preprocessing.

Usage:
    python -m tsl.realtime --run outputs/runs/<run>                      # live demo
    python -m tsl.realtime --run outputs/runs/<run> --video <clip.mp4>   # RT-04 check
    python -m tsl.realtime --check-hands                                  # D-08 orientation check
    python -m tsl.realtime --list-cameras                                 # find the built-in camera index
    python -m tsl.realtime --run <run> --eval --signer jacques --condition bright --reps 2   # logged evaluation
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
    import tsl.models  # noqa: F401  registers custom layers (M1 AttentionPool) before loading
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


TRIAL_FIELDS = ["trial", "target", "predicted", "correct", "confidence", "top3", "status", "latency_s",
                "duration_s", "n_frames", "timestamp"]


def live_eval(cfg, model, labels, seq_len, flip_input, run_dir, signer, condition, reps, seed):
    """Logged webcam evaluation (spec LIVE-01..05): prompts each sign `reps` times in a seeded random order.

    Every attempt is appended to outputs/live/<session>/trials.csv as soon as it happens, and its raw
    landmarks are saved, so any model can be re-scored later. Too-short recordings are logged but the same
    sign is prompted again; a recording with no hand detected counts as a failed trial.
    """
    import csv
    import random
    from datetime import datetime

    from tsl.train import git_commit

    d = cfg["demo"]
    plan = [w for w in labels for _ in range(reps)]
    random.Random(seed).shuffle(plan)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    session = Path("outputs/live") / f"{stamp}_{signer}_{condition}"
    (session / "landmarks").mkdir(parents=True)
    (session / "meta.json").write_text(json.dumps(dict(
        timestamp=stamp, run=str(run_dir), signer=signer, condition=condition, reps=reps, seed=seed,
        flip_input=flip_input, camera_index=d["camera_index"], min_seconds=d["min_seconds"],
        max_seconds=d["max_seconds"], labels=labels, plan=plan, git_commit=git_commit()), indent=2, ensure_ascii=False))
    log = open(session / "trials.csv", "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(log, fieldnames=TRIAL_FIELDS)
    writer.writeheader()
    print(f"Session: {session}  ({len(plan)} trials)")

    cap = open_camera(d["camera_index"])
    i, attempt, recording, frames, t_start = 0, 0, False, [], 0.0
    feedback = ""

    def finish(t_stop):
        nonlocal i, attempt, feedback
        attempt += 1
        target, dur = plan[i], t_stop - t_start
        row = dict(trial=i + 1, target=target, predicted="", correct=0, confidence="", top3="", status="ok",
                   latency_s="", duration_s=round(dur, 3), n_frames=len(frames),
                   timestamp=datetime.now().isoformat(timespec="seconds"))
        if dur < d["min_seconds"]:
            row["status"] = "too_short"
            feedback = f"Too short ({dur:.1f}s) - sign '{target}' again"
        else:
            try:
                raw = extract_frames(frames)
                if np.isnan(raw["left_hand"]).all() and np.isnan(raw["right_hand"]).all():
                    raise ValueError("no hand detected")
                np.savez_compressed(session / "landmarks" / f"trial{i + 1:03d}_a{attempt}.npz", **raw)
                probs = model.predict(clip_features(raw, seq_len)[None], verbose=0)[0]
                top = np.argsort(probs)[::-1][:3]
                row.update(predicted=labels[top[0]], correct=int(labels[top[0]] == target),
                           confidence=round(float(probs[top[0]]), 4),
                           top3="|".join(f"{labels[k]}:{probs[k]:.3f}" for k in top))
                feedback = f"{'OK' if row['correct'] else 'WRONG'}: predicted {row['predicted']} ({probs[top[0]]:.2f})"
            except (ValueError, ClipFailed) as e:
                row["status"] = "no_hand" if "no hand" in str(e) else "failed"
                feedback = f"No prediction ({e}) - counted as a failed trial"
            row["latency_s"] = round(time.perf_counter() - t_stop_perf, 3)
            i += 1
            attempt = 0
        writer.writerow(row)
        log.flush()

    t_stop_perf = 0.0
    while i < len(plan):
        ok, frame = cap.read()
        if not ok:
            print("Camera stopped delivering frames (disconnected?); session saved so far.")
            break
        if recording:
            frames.append(cv2.flip(frame, 1) if flip_input else frame)
            if time.monotonic() - t_start >= d["max_seconds"]:  # RT-02 auto-stop
                recording, t_stop_perf = False, time.perf_counter()
                finish(time.monotonic())
                continue
        view = cv2.flip(frame, 1)
        put(view, f"Trial {i + 1}/{len(plan)}   SIGN: {plan[i] if i < len(plan) else ''}", 30, (0, 255, 255))
        if recording:
            cv2.circle(view, (view.shape[1] - 30, 30), 12, (0, 0, 255), -1)
            put(view, f"REC {time.monotonic() - t_start:.1f}s", 60, (0, 0, 255))
        else:
            put(view, feedback or "Space: start recording", 60)
        put(view, f"{signer} / {condition}   Space: record   q: stop session", view.shape[0] - 15)
        cv2.imshow("TID live evaluation", view)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord(" "):
            if recording:
                recording, t_stop_perf = False, time.perf_counter()
                put(view, "Processing...", 90)
                cv2.imshow("TID live evaluation", view)
                cv2.waitKey(1)
                finish(time.monotonic())
            else:
                recording, frames, t_start = True, [], time.monotonic()
    cap.release()
    cv2.destroyAllWindows()
    log.close()
    print(f"Saved {i}/{len(plan)} trials to {session}. Report: python scripts/live_report.py {session}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--run", help="run directory with model.keras")
    ap.add_argument("--video", help="RT-04: feed this video file through the live path")
    ap.add_argument("--check-hands", action="store_true", help="D-08 handedness check (webcam)")
    ap.add_argument("--camera", type=int, help="camera index (overrides demo.camera_index in the config)")
    ap.add_argument("--list-cameras", action="store_true", help="probe camera indices 0-3")
    ap.add_argument("--eval", action="store_true", help="logged evaluation session (LIVE): outputs/live/")
    ap.add_argument("--signer", help="--eval: who is signing (e.g. jacques)")
    ap.add_argument("--condition", help="--eval: recording condition (e.g. bright, dim)")
    ap.add_argument("--reps", type=int, default=1, help="--eval: repetitions per sign")
    ap.add_argument("--seed", type=int, default=0, help="--eval: seed of the randomised sign order")
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
    if args.eval:
        if not (args.signer and args.condition):
            raise SystemExit("--eval needs --signer and --condition")
        return live_eval(cfg, model, labels, seq_len, flip, args.run, args.signer, args.condition, args.reps, args.seed)
    live(cfg, model, labels, seq_len, flip)


if __name__ == "__main__":
    main()
