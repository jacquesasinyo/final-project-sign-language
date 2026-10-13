import csv
from pathlib import Path

import pytest

from tsl.data import assert_signers_disjoint, signer_of
from tsl.utils import load_classes, load_config

MANIFEST = Path(load_config()["manifest_path"])


def test_config_loads():
    cfg = load_config()
    assert cfg["seq_len"] > 0
    assert not Path(cfg["data_root"]).is_absolute()  # DS-06


def test_classes_file():
    classes, smoke = load_classes()
    ids = [c["id"] for c in classes]
    assert len(ids) == 20 and len(set(ids)) == 20
    assert all(0 <= i <= 225 for i in ids)
    assert len(smoke) == 10 and set(smoke) <= set(ids)


def test_signer_of():
    assert signer_of("signer34_sample1") == 34
    with pytest.raises(ValueError):
        signer_of("sample1")


def test_disjoint_check_detects_overlap():
    assert_signers_disjoint({"train": [1, 2], "val": [3], "test": [4]})
    with pytest.raises(ValueError, match="train and val"):
        assert_signers_disjoint({"train": [1, 2], "val": [2], "test": [4]})


@pytest.mark.skipif(not MANIFEST.exists(), reason="manifest missing: run scripts/prepare_dataset.py")
def test_real_manifest_signers_disjoint():  # DS-04
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_split = {}
    for r in rows:
        by_split.setdefault(r["split"], set()).add(r["signer_id"])
    assert set(by_split) == {"train", "val", "test"}
    assert_signers_disjoint(by_split)


def test_demo_config_inherits_and_stays_separate():
    demo = load_config("configs/demo.yaml")
    base = load_config()
    assert demo["seq_len"] == base["seq_len"] and demo["cache_dir"] == base["cache_dir"]  # same pipeline
    for key in ("classes_file", "manifest_path", "runs_dir"):  # never overwrites benchmark files
        assert demo[key] != base[key]


def test_demo_classes_file():
    classes, smoke = load_classes("configs/demo_classes.yaml")
    ids = [c["id"] for c in classes]
    assert len(ids) == len(set(ids)) == 21 and smoke == []
    bench = {c["id"] for c in load_classes()[0]}
    assert {30, 131, 200}.isdisjoint(ids)  # ben, mendil, tuvalet: unreliable live (D-12)
    assert len(bench & set(ids)) == 13
