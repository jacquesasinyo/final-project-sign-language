"""Build the dataset manifest for the selected classes (DS-04, DS-07).

One row per colour clip of the classes in configs/classes.yaml, in every official
split. Video metadata is read with OpenCV; the frame count is obtained by decoding
(grab) every frame, not trusted from the container header. Landmark-related columns
are filled by the Phase 2 extractor.

Fails if a labelled clip is missing or unreadable, or if any two splits share a signer.

Usage:
    python scripts/prepare_dataset.py [--config configs/default.yaml]
"""
import argparse
import csv
from pathlib import Path

import cv2
from tqdm import tqdm

from tsl.data import LABEL_FILES, assert_signers_disjoint, read_labels, signer_of
from tsl.utils import load_classes, load_config

COLUMNS = ["sample_id", "class_id", "label", "signer_id", "split", "video_path",
           "frame_count", "fps", "width", "height", "duration_s",
           "extraction_status", "left_hand_rate", "right_hand_rate"]


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    if n == 0 or fps <= 0:
        raise SystemExit(f"Unreadable video {path} (frames={n}, fps={fps})")
    return n, fps, width, height


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["data_root"])
    classes, _ = load_classes(cfg["classes_file"])
    names = {c["id"]: c["tr"] for c in classes}

    rows = []
    for split, label_file in LABEL_FILES.items():
        if not (root / label_file).exists():
            raise SystemExit(f"Missing label file {root / label_file} (DS-03): stop and report.")
        labels = read_labels(root / label_file)
        samples = sorted(s for s, c in labels.items() if c in names)
        for sid in tqdm(samples, desc=split):
            rel = Path(split) / f"{sid}_color.mp4"
            if not (root / rel).exists():
                raise SystemExit(f"Labelled clip missing: {root / rel}")
            n, fps, w, h = video_info(root / rel)
            rows.append(dict(sample_id=sid, class_id=labels[sid], label=names[labels[sid]],
                             signer_id=signer_of(sid), split=split, video_path=rel.as_posix(),
                             frame_count=n, fps=round(fps, 3), width=w, height=h,
                             duration_s=round(n / fps, 3), extraction_status="pending",
                             left_hand_rate="", right_hand_rate=""))

    signers = {s: {r["signer_id"] for r in rows if r["split"] == s} for s in LABEL_FILES}
    try:
        assert_signers_disjoint(signers)
    except ValueError as e:
        raise SystemExit(f"DS-04 violated: {e}")

    out = Path(cfg["manifest_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out}: {len(rows)} clips; signers "
          + ", ".join(f"{s} {len(v)}" for s, v in signers.items()))


if __name__ == "__main__":
    main()
