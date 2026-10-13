"""Report metrics for logged live evaluation sessions (spec LIVE-04).

Combines one or more sessions from `python -m tsl.realtime --eval ...` and writes
<out>.json with accuracy (+ 95% CI), macro-F1, per-class precision/recall/F1, confusion matrix,
most-confused pairs, latency median/p95, and accuracy per signer and per condition.

Usage:
    python scripts/live_report.py outputs/live/<session> [more sessions] [--out outputs/live/report_dev]
"""
import argparse
import csv
import json
from pathlib import Path

from tsl.evaluate import live_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+", help="session folders under outputs/live/")
    ap.add_argument("--out", help="output path without extension (default: inside the first session)")
    args = ap.parse_args()

    trials, labels, runs = [], None, set()
    for s in map(Path, args.sessions):
        meta = json.loads((s / "meta.json").read_text(encoding="utf-8"))
        if labels is not None and meta["labels"] != labels:
            raise SystemExit(f"{s} uses a different vocabulary; report sessions separately.")
        labels = meta["labels"]
        runs.add(meta["run"])
        with open(s / "trials.csv", newline="", encoding="utf-8") as f:
            trials += [dict(r, signer=meta["signer"], condition=meta["condition"]) for r in csv.DictReader(f)]
    if len(runs) > 1:
        print(f"WARNING: sessions used different models: {sorted(runs)}")

    rep = live_summary(trials, labels)
    rep.update(sessions=args.sessions, runs=sorted(runs))
    out = Path(args.out) if args.out else Path(args.sessions[0]) / "report"
    out.with_suffix(".json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))

    lo, hi = rep["accuracy_ci95"]
    print(f"Trials: {rep['n_trials']} (no prediction: {rep['no_prediction']})")
    print(f"Top-1 accuracy: {rep['accuracy']:.3f}  ({rep['n_correct']}/{rep['n_trials']}, 95% CI {lo:.2f}-{hi:.2f})")
    print(f"Macro-F1: {rep['macro_f1']:.3f}")
    if rep["latency_s"]:
        print(f"Latency (stop -> prediction): median {rep['latency_s']['median']:.2f}s, p95 {rep['latency_s']['p95']:.2f}s")
    for key in ("accuracy_by_signer", "accuracy_by_condition"):
        if key in rep:
            print(f"{key}: " + ", ".join(f"{g} {a:.2f}" for g, a in rep[key].items()))
    print("Per class (recall / precision, n):")
    for name, v in sorted(rep["per_class"].items(), key=lambda kv: kv[1]["recall"]):
        print(f"  {name:<10} {v['recall']:.2f} / {v['precision']:.2f}  n={v['support']}")
    if rep["most_confused"]:
        print("Most confused (count, true -> predicted):", rep["most_confused"])
    print(f"Saved {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
