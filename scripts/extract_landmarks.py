"""Extract and cache MediaPipe landmarks for every clip in the manifest (PRE-05).

One .npz of raw landmarks per clip at <cache_dir>/<split>/<sample_id>.npz. Resumable:
clips whose cache file exists are skipped. Afterwards the manifest's extraction_status
and per-hand detection rates are filled (DS-07), and failures are logged with a reason (PRE-04).

Usage:
    python scripts/extract_landmarks.py [--workers 4] [--limit N]
"""
import argparse
import csv
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from tsl.features import ClipFailed, normalise
from tsl.utils import load_config

os.environ.setdefault("GLOG_minloglevel", "2")


def work(job):
    src, dst = job
    if Path(dst).exists():
        return dst, "cached", 0.0
    from tsl.landmarks import extract_video  # import per worker process
    t0 = time.perf_counter()
    try:
        raw = extract_video(src)
    except Exception as e:  # unreadable video etc.
        return dst, f"error: {e}", 0.0
    tmp = Path(dst).with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **raw)
    os.replace(tmp, dst)  # atomic: an interrupted run never leaves a half-written cache file
    return dst, "extracted", time.perf_counter() - t0


def status_of(path):
    """(status, left_rate, right_rate) for a cached clip."""
    raw = dict(np.load(path))
    left = float((~np.isnan(raw["left_hand"]).any(axis=(1, 2))).mean())
    right = float((~np.isnan(raw["right_hand"]).any(axis=(1, 2))).mean())
    try:
        normalise(raw)
        status = "ok"
    except ClipFailed as e:
        status = f"failed: {e}"
    return status, round(left, 4), round(right, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="only the first N clips (timing tests)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root, cache = Path(cfg["data_root"]), Path(cfg["cache_dir"])
    manifest = Path(cfg["manifest_path"])
    with open(manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0])

    todo = rows[: args.limit] if args.limit else rows
    jobs = []
    for r in todo:
        dst = cache / r["split"] / f"{r['sample_id']}.npz"
        dst.parent.mkdir(parents=True, exist_ok=True)
        jobs.append((str(root / r["video_path"]), str(dst)))

    t0, done, errors = time.perf_counter(), 0, {}
    with Pool(args.workers) as pool:
        for dst, what, _ in pool.imap_unordered(work, jobs, chunksize=4):
            done += 1
            if what.startswith("error"):
                errors[dst] = what
            if done % 100 == 0 or done == len(jobs):
                el = time.perf_counter() - t0
                print(f"{done}/{len(jobs)} clips, {el / 60:.1f} min elapsed, "
                      f"~{el / done * (len(jobs) - done) / 60:.1f} min left", flush=True)

    for r in rows:
        dst = cache / r["split"] / f"{r['sample_id']}.npz"
        if str(dst) in errors:
            r["extraction_status"], r["left_hand_rate"], r["right_hand_rate"] = errors[str(dst)], "", ""
        elif dst.exists():
            r["extraction_status"], r["left_hand_rate"], r["right_hand_rate"] = status_of(dst)
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    done_rows = [r for r in rows if r["extraction_status"] != "pending"]
    failed = [r for r in done_rows if r["extraction_status"] != "ok"]
    rates = np.array([[float(r["left_hand_rate"] or 0), float(r["right_hand_rate"] or 0)] for r in done_rows])
    print(f"processed {len(done_rows)}/{len(rows)}; failed {len(failed)} "
          f"({100 * len(failed) / max(len(done_rows), 1):.2f}%); mean hand detection rate "
          f"left {rates[:, 0].mean():.3f}, right {rates[:, 1].mean():.3f}")
    for r in failed:
        print(f"  FAILED {r['sample_id']}: {r['extraction_status']}")


if __name__ == "__main__":
    main()
