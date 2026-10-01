"""AUTSL label files, signer IDs and the signer-disjointness check (DS-03, DS-04)."""
import csv
import re

# split -> label file, relative to the data root (HF mirror layout, docs/decisions.md D-01)
LABEL_FILES = {
    "train": "train_labels.csv",
    "val": "val_labels.csv",
    "test": "test_labels.csv",
}
SIGNLIST = "SignList_ClassId_TR_EN.csv"
SAMPLE_RE = re.compile(r"^signer(\d+)_sample(\d+)$")


def read_labels(path):
    """sample_id -> class_id."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {row[0].strip(): int(row[1]) for row in csv.reader(f) if row}


def signer_of(sample_id):
    m = SAMPLE_RE.match(sample_id)
    if not m:
        raise ValueError(f"Cannot parse signer from sample id {sample_id!r}")
    return int(m.group(1))


def assert_signers_disjoint(signers_by_split):
    """signers_by_split: split -> iterable of signer IDs. Raises if any two splits share a signer."""
    sets = {s: set(v) for s, v in signers_by_split.items()}
    names = sorted(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = sets[a] & sets[b]
            if shared:
                raise ValueError(f"Signer overlap between {a} and {b}: {sorted(shared)}")


def load_split(cfg, split, class_ids):
    """Features for one split, restricted to class_ids (sorted -> label index 0..N-1).

    Returns X (N, T, F), y (N,), sample_ids, skipped. Clips whose extraction failed
    are skipped and returned in `skipped` so they can be reported (PRE-04).
    """
    from pathlib import Path

    import numpy as np

    from tsl.features import ClipFailed, clip_features

    label_of = {cid: i for i, cid in enumerate(sorted(class_ids))}
    cache = Path(cfg["cache_dir"])
    with open(cfg["manifest_path"], newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split and int(r["class_id"]) in label_of]
    X, y, ids, skipped = [], [], [], []
    for r in rows:
        path = cache / split / f"{r['sample_id']}.npz"
        if not path.exists():
            raise SystemExit(f"Missing cached landmarks {path}; run scripts/extract_landmarks.py")
        try:
            X.append(clip_features(dict(np.load(path)), cfg["seq_len"]))
        except ClipFailed as e:
            skipped.append((r["sample_id"], str(e)))
            continue
        y.append(label_of[int(r["class_id"])])
        ids.append(r["sample_id"])
    return np.stack(X), np.asarray(y), ids, skipped
