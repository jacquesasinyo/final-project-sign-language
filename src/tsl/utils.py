"""Config and class-list loading."""
from pathlib import Path

import yaml


def load_config(path="configs/default.yaml"):
    """Load a YAML config. A `base:` key names another config whose top-level keys this one overrides."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:
        merged = load_config(cfg.pop("base"))
        merged.update(cfg)
        return merged
    return cfg


def load_classes(path="configs/classes.yaml"):
    """Return (classes, smoke_ids): classes is a list of {id, tr, en} sorted by id.

    smoke_ids is empty for class files without a smoke-test subset (e.g. the demo vocabulary).
    """
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{path} not found; run scripts/select_classes.py first (DS-02).")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["classes"], cfg.get("smoke_test_class_ids", [])
