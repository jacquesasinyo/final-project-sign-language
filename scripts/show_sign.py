"""Play AUTSL reference clips of a sign, to learn it before testing the live model.

Clips come from different training signers (training split only, never test). Playback loops
in an OpenCV window with the label shown.
Keys: space = pause, n = next clip, s = toggle slow motion, m = toggle mirror view, q = quit.

Usage:
    python scripts/show_sign.py --list                  # words in the vocabulary
    python scripts/show_sign.py selam                   # 3 clips, demo vocabulary
    python scripts/show_sign.py selam --n 5 --mirror    # 5 clips, mirror view (copy like a mirror)
    python scripts/show_sign.py yatak --config configs/default.yaml   # benchmark vocabulary
    python scripts/show_sign.py selam --paths           # just print the clip paths
"""
import argparse
import csv
import os
import random
from pathlib import Path

import cv2

from tsl.utils import load_classes, load_config


def pick_clips(cfg, label, n, seed):
    """Up to n training clips of `label`, each from a different signer."""
    with open(cfg["manifest_path"], newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "train" and r["label"] == label]
    by_signer = {}
    for r in sorted(rows, key=lambda r: r["sample_id"]):
        by_signer.setdefault(r["signer_id"], r)
    signers = sorted(by_signer)
    random.Random(seed).shuffle(signers)
    return [Path(cfg["data_root"]) / by_signer[s]["video_path"] for s in signers[:n]]


def play(paths, label, mirror):
    i, slow, paused = 0, False, False
    while True:
        cap = cv2.VideoCapture(str(paths[i]))
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
        k_frame = 0
        while True:
            fr = frames[k_frame].copy()
            if mirror:
                fr = cv2.flip(fr, 1)
            text = f"{label}  clip {i + 1}/{len(paths)}  {'SLOW ' if slow else ''}{'MIRROR' if mirror else ''}"
            cv2.putText(fr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(fr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(fr, "space pause  n next  s slow  m mirror  q quit", (10, fr.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("TID reference", fr)
            key = cv2.waitKey(100 if slow else 33) & 0xFF
            if key == ord("q"):
                cv2.destroyAllWindows()
                return
            if key == ord("n"):
                break
            if key == ord("s"):
                slow = not slow
            if key == ord("m"):
                mirror = not mirror
            if key == ord(" "):
                paused = not paused
            if not paused:
                k_frame = (k_frame + 1) % len(frames)
                if k_frame == 0:
                    cv2.waitKey(400)  # short pause between loops
        i = (i + 1) % len(paths)


def main():
    os.chdir(Path(__file__).resolve().parents[1])  # config and data paths are relative to the repo root
    ap = argparse.ArgumentParser()
    ap.add_argument("word", nargs="?", help="Turkish gloss, e.g. selam")
    ap.add_argument("--config", default="configs/demo.yaml")
    ap.add_argument("--n", type=int, default=3, help="number of different signers")
    ap.add_argument("--seed", type=int, default=0, help="change to see other signers")
    ap.add_argument("--mirror", action="store_true", help="start in mirror view")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--paths", action="store_true", help="print clip paths instead of playing")
    args = ap.parse_args()
    cfg = load_config(args.config)
    labels = sorted(c["tr"] for c in load_classes(cfg["classes_file"])[0])
    if args.list:
        print("Words:", ", ".join(labels))
        return
    if args.word:
        if args.word not in labels:
            raise SystemExit(f"{args.word!r} is not in this vocabulary. Words: {', '.join(labels)}")
        paths = pick_clips(cfg, args.word, args.n, args.seed)
        if args.paths:
            print("\n".join(map(str, paths)))
        else:
            play(paths, args.word, args.mirror)
        return
    # No word given (e.g. VS Code's Run button): pick words interactively until quit.
    while True:
        print("\n" + "\n".join(f"{i:>2}. {w}" for i, w in enumerate(labels, 1)))
        choice = input("Number or word to play (Enter = quit): ").strip()
        if not choice:
            return
        word = labels[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(labels) else choice
        if word not in labels:
            print(f"{choice!r} is not in the list.")
            continue
        print(f"Playing {word}: q = back to the list, n = next signer, m = mirror, s = slow, space = pause")
        play(pick_clips(cfg, word, args.n, args.seed), word, args.mirror)


if __name__ == "__main__":
    main()
