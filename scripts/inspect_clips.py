"""Phase 1 inspection of AUTSL clips with MediaPipe Holistic (PRE-00, spec §17).

Runs Holistic on a seeded sample of train clips from the selected classes and reports:
- extraction time per clip (to estimate the Phase 2 extraction runtime);
- per-hand detection rates;
- leading/trailing idle frames: frames before the first / after the last frame in which
  a wrist is raised above its elbow (pose landmarks 15/16 vs 13/14);
- which hand is dominant per signer (more wrist motion) and on which image side it
  appears, to fix the orientation / left-right convention.
Saves a few annotated frames to outputs/inspect/ for visual checking.

Exploration only: nothing here is used for training. The Phase 2 extractor will be
the single shared implementation (PRE-07).

Usage:
    python scripts/inspect_clips.py [--n 50]
"""
import argparse
import csv
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from tsl.utils import load_config

L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 13, 14, 15, 16


def run_clip(path, holistic):
    """Per-frame arrays: hand presence (F,2), pose wrists/elbows xy (F,4,2) with NaN if missing."""
    cap = cv2.VideoCapture(str(path))
    hands, arms = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        hands.append([res.left_hand_landmarks is not None, res.right_hand_landmarks is not None])
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            arms.append([[lm[i].x, lm[i].y] for i in (L_ELBOW, R_ELBOW, L_WRIST, R_WRIST)])
        else:
            arms.append([[np.nan, np.nan]] * 4)
        last = (frame, res)
    cap.release()
    return np.array(hands), np.array(arms), last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out-dir", default="outputs/inspect")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root, out = Path(cfg["data_root"]), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(cfg["manifest_path"], newline="", encoding="utf-8") as f:
        train = [r for r in csv.DictReader(f) if r["split"] == "train"]
    sample = random.Random(cfg["seed"]).sample(train, args.n)

    per_clip, per_signer = [], defaultdict(lambda: np.zeros(2))
    draw = mp.solutions.drawing_utils
    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hol:
        run_clip(root / sample[0]["video_path"], hol)  # warm-up, not timed
        for k, r in enumerate(sample):
            t0 = time.perf_counter()
            hands, arms, (frame, res) = run_clip(root / r["video_path"], hol)
            dt = time.perf_counter() - t0
            n = len(hands)
            # raised = wrist above elbow (smaller y); idle = frames outside first..last raised frame
            raised = (arms[:, 2, 1] < arms[:, 0, 1]) | (arms[:, 3, 1] < arms[:, 1, 1])
            idx = np.flatnonzero(raised)
            lead, trail = (idx[0], n - 1 - idx[-1]) if len(idx) else (n, 0)  # never raised: all idle
            # dominant hand = subject side whose wrist moves more
            motion = np.nansum(np.abs(np.diff(arms[:, 2:, :], axis=0)), axis=(0, 2))  # [left, right]
            per_signer[r["signer_id"]] += motion
            # image side of the subject's left wrist (x > 0.5 = right half of image)
            lw_x = np.nanmedian(arms[:, 2, 0])
            per_clip.append(dict(sample_id=r["sample_id"], label=r["label"], frames=n, seconds=round(dt, 2),
                                 fps_proc=round(n / dt, 1), left_rate=round(hands[:, 0].mean(), 3),
                                 right_rate=round(hands[:, 1].mean(), 3), lead_idle=int(lead),
                                 trail_idle=int(trail), dominant="right" if motion[1] > motion[0] else "left",
                                 left_wrist_x=round(float(lw_x), 3)))
            if k < 4:
                img = frame.copy()
                draw.draw_landmarks(img, res.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)
                draw.draw_landmarks(img, res.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS,
                                    draw.DrawingSpec(color=(255, 0, 0), thickness=2))  # blue = left
                draw.draw_landmarks(img, res.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS,
                                    draw.DrawingSpec(color=(0, 0, 255), thickness=2))  # red = right
                cv2.imwrite(str(out / f"{r['sample_id']}_last.jpg"), img)
            print(f"{k + 1}/{len(sample)} {r['sample_id']} {n} frames {dt:.1f}s")

    with open(out / "inspect_clips.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_clip[0]))
        w.writeheader()
        w.writerows(per_clip)

    a = {k: np.array([c[k] for c in per_clip], dtype=float)
         for k in ("frames", "seconds", "fps_proc", "left_rate", "right_rate", "lead_idle", "trail_idle",
                   "left_wrist_x")}
    total_frames = sum(int(x) for x in a["frames"])
    lines = [f"clips: {len(per_clip)}, frames: {total_frames}",
             f"processing: {a['seconds'].sum():.1f}s total, {a['seconds'].mean():.2f}s/clip, "
             f"{total_frames / a['seconds'].sum():.1f} frames/s",
             f"hand detection rate: left mean {a['left_rate'].mean():.3f}, right mean {a['right_rate'].mean():.3f}"]
    for k in ("lead_idle", "trail_idle"):
        p = np.percentile(a[k], [0, 25, 50, 75, 100])
        lines.append(f"{k} frames: min {p[0]:.0f}, q1 {p[1]:.0f}, median {p[2]:.0f}, q3 {p[3]:.0f}, max {p[4]:.0f}")
    idle_frac = (a["lead_idle"] + a["trail_idle"]) / a["frames"]
    lines.append(f"idle fraction of clip: median {np.median(idle_frac):.2f}, max {idle_frac.max():.2f}")
    lines.append(f"subject's LEFT wrist image x: median {np.nanmedian(a['left_wrist_x']):.3f} "
                 f"(> 0.5 means it appears on the image right = un-mirrored camera view)")
    lines.append("dominant hand per signer (wrist motion left/right): " + ", ".join(
        f"s{s}: {'R' if m[1] > m[0] else 'L'} ({m[0]:.1f}/{m[1]:.1f})" for s, m in sorted(
            per_signer.items(), key=lambda kv: int(kv[0]))))
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
