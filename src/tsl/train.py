"""Train one model on one class subset with one seed (MOD-04..MOD-06, EVAL-05).

Early stopping and best-checkpoint selection use validation macro-F1, with validation loss
as the tie-break (MOD-05). Only train + validation are touched; never the test split.

Each run writes outputs/runs/<timestamp>_<model>_<subset>_s<seed>/ with:
config.json, env.json (versions, git commit, hashes), history.csv, val_metrics.json, model.keras.

Usage:
    python -m tsl.train --model m0 --subset smoke --seed 0
"""
import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import keras
import numpy as np

from tsl.data import load_split
from tsl.evaluate import compute_metrics, macro_f1, most_confused
from tsl.features import N_FEATURES
from tsl.models import build_model
from tsl.utils import load_classes, load_config


class BestOnValF1(keras.callbacks.Callback):
    """Track val macro-F1 each epoch, keep the best weights, stop after `patience` epochs without improvement."""

    def __init__(self, X_val, y_val, n_classes, patience):
        super().__init__()
        self.X_val, self.y_val, self.n_classes, self.patience = X_val, y_val, n_classes, patience
        self.best_key, self.best_weights, self.best_epoch, self.wait = None, None, -1, 0

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(self.X_val, verbose=0).argmax(1)
        f1 = macro_f1(self.y_val, pred, self.n_classes)
        logs["val_macro_f1"] = f1
        key = (round(f1, 6), -logs["val_loss"])  # higher F1 first, then lower val loss
        if self.best_key is None or key > self.best_key:
            self.best_key, self.best_weights, self.best_epoch, self.wait = key, self.model.get_weights(), epoch, 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_commit():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
        return commit + ("-dirty" if dirty else "")
    except subprocess.CalledProcessError:
        return "no-commit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default="m0")
    ap.add_argument("--subset", choices=["smoke", "full"], default="full")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tc = cfg["train"]
    classes, smoke = load_classes(cfg["classes_file"])
    class_ids = sorted(smoke if args.subset == "smoke" else [c["id"] for c in classes])
    names = {c["id"]: c["tr"] for c in classes}
    label_names = [names[c] for c in class_ids]

    keras.utils.set_random_seed(args.seed)  # MOD-06: python, numpy and TF seeds
    X_tr, y_tr, _, skip_tr = load_split(cfg, "train", class_ids)
    X_va, y_va, _, skip_va = load_split(cfg, "val", class_ids)
    print(f"train {X_tr.shape}, val {X_va.shape}; skipped (failed extraction): train {len(skip_tr)}, val {len(skip_va)}")

    model = build_model(args.model, cfg["seq_len"], N_FEATURES, len(class_ids))
    model.compile(optimizer=keras.optimizers.Adam(tc["learning_rate"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])  # MOD-04
    best = BestOnValF1(X_va, y_va, len(class_ids), tc["patience"])
    t0 = time.time()
    hist = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=tc["max_epochs"],
                     batch_size=tc["batch_size"], callbacks=[best], verbose=2)
    train_s = time.time() - t0

    pred = model.predict(X_va, verbose=0).argmax(1)
    metrics = compute_metrics(y_va, pred, label_names)
    metrics.update(best_epoch=best.best_epoch + 1, epochs_run=len(hist.history["loss"]),
                   params=int(model.count_params()), train_seconds=round(train_s, 1),
                   most_confused=most_confused(metrics["confusion_matrix"], label_names))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(cfg["runs_dir"]) / f"{stamp}_{args.model}_{args.subset}_s{args.seed}"
    out.mkdir(parents=True)
    model.save(out / "model.keras")
    (out / "config.json").write_text(json.dumps(dict(
        config=cfg, model=args.model, subset=args.subset, seed=args.seed,
        class_ids=class_ids, label_names=label_names, split="official AUTSL train/val",
        skipped=dict(train=skip_tr, val=skip_va)), indent=2, ensure_ascii=False))
    (out / "env.json").write_text(json.dumps(dict(
        timestamp=stamp, git_commit=git_commit(), python=platform.python_version(), machine=platform.platform(),
        packages={p: version(p) for p in ["tensorflow", "keras", "mediapipe", "numpy", "scikit-learn", "opencv-contrib-python"]},
        classes_sha256=sha256(cfg["classes_file"]), manifest_sha256=sha256(cfg["manifest_path"])), indent=2))
    with open(out / "history.csv", "w", newline="") as f:
        w = csv.writer(f)
        keys = list(hist.history)
        w.writerow(["epoch"] + keys)
        for i in range(len(hist.history["loss"])):
            w.writerow([i + 1] + [hist.history[k][i] for k in keys])
    (out / "val_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\n{out}\nval accuracy {metrics['accuracy']:.4f}, val macro-F1 {metrics['macro_f1']:.4f} "
          f"(best epoch {metrics['best_epoch']}/{metrics['epochs_run']}, {metrics['params']} params, {train_s:.0f}s)")


if __name__ == "__main__":
    main()
