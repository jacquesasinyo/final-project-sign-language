"""Config and class-list loading."""
from pathlib import Path

import yaml


def load_config(path="configs/default.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_classes(path="configs/classes.yaml"):
    """Return (classes, smoke_ids): classes is a list of {id, tr, en} sorted by id."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{path} not found; run scripts/select_classes.py first (DS-02).")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["classes"], cfg["smoke_test_class_ids"]
