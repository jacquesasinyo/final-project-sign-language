import numpy as np
import pytest

from tsl.features import N_FEATURES, ClipFailed, clip_features, normalise, resample_indices


def fake_raw(F=10, size=(512, 512), hands=True):
    rng = np.random.default_rng(0)
    pose = rng.uniform(0.3, 0.7, (F, 33, 4)).astype(np.float32)
    pose[:, 11, :2] = [0.6, 0.4]  # subject's left shoulder
    pose[:, 12, :2] = [0.4, 0.4]  # right shoulder -> width 0.2, centre (0.5, 0.4)
    hand = rng.uniform(0.3, 0.7, (F, 21, 3)).astype(np.float32)
    nan_hand = np.full((F, 21, 3), np.nan, np.float32)
    return dict(left_hand=hand if hands else nan_hand, right_hand=nan_hand.copy(), pose=pose,
                image_size=np.array(size))


def test_resample_indices():
    assert resample_indices(5, 3).tolist() == [0, 2, 4]
    assert resample_indices(2, 4).tolist() == [0, 0, 1, 1]  # short clips repeat frames


def test_shapes_and_flags():
    feats = clip_features(fake_raw(), 48)
    assert feats.shape == (48, N_FEATURES) == (48, 100)
    assert np.all(feats[:, -2] == 1) and np.all(feats[:, -1] == 0)  # left present, right missing
    assert np.all(feats[:, 42:84] == 0)                              # missing right hand -> zeros
    assert np.isfinite(feats).all()


def test_shoulder_normalisation():
    feats = normalise(fake_raw())
    pose = feats[:, 84:98].reshape(-1, 7, 2)
    np.testing.assert_allclose(pose[:, 1], [[0.5, 0.0]] * 10, atol=1e-6)   # left shoulder: +half width
    np.testing.assert_allclose(pose[:, 2], [[-0.5, 0.0]] * 10, atol=1e-6)  # right shoulder


def test_aspect_correction():
    # identical pixel geometry in 640x480 and 512x512: shoulders 128 px apart, a hand point
    # 64 px right of and 32 px below the shoulder midpoint -> same normalised (0.5, 0.25) in both
    def raw_px(w, h):
        raw = fake_raw(F=2, size=(w, h))
        cx, cy = 256, 200
        raw["pose"][:, 11, :2] = [(cx + 64) / w, cy / h]
        raw["pose"][:, 12, :2] = [(cx - 64) / w, cy / h]
        raw["left_hand"][:, 0, :2] = [(cx + 64) / w, (cy + 32) / h]
        return raw
    for w, h in [(640, 480), (512, 512)]:
        np.testing.assert_allclose(normalise(raw_px(w, h))[:, 0:2], [[0.5, 0.25]] * 2, atol=1e-5)


def test_missing_shoulders_carry_forward_and_fail():
    raw = fake_raw()
    raw["pose"][:3, 11] = np.nan  # first frames missing -> back-filled
    assert np.isfinite(normalise(raw)).all()
    raw["pose"][:, 11] = np.nan
    with pytest.raises(ClipFailed):
        normalise(raw)


def test_model_output_shape():
    from tsl.models import build_model
    m = build_model("m0", 48, N_FEATURES, 20)
    assert m(np.zeros((2, 48, N_FEATURES), np.float32)).shape == (2, 20)


def test_live_summary():
    from tsl.evaluate import live_summary
    labels = ["a", "b", "c"]
    trials = [dict(target="a", predicted="a", status="ok", latency_s="0.4", signer="x", condition="bright"),
              dict(target="a", predicted="b", status="ok", latency_s="0.6", signer="x", condition="dim"),
              dict(target="b", predicted="b", status="ok", latency_s="0.5", signer="y", condition="bright"),
              dict(target="c", predicted="", status="no_hand", latency_s="0.5", signer="y", condition="dim"),
              dict(target="c", predicted="", status="too_short", latency_s="", signer="y", condition="dim")]
    r = live_summary(trials, labels)
    assert r["n_trials"] == 4 and r["n_correct"] == 2 and r["no_prediction"] == 1  # too_short excluded
    assert r["accuracy"] == 0.5 and r["per_class"]["c"]["recall"] == 0.0
    assert r["accuracy_by_condition"] == {"bright": 1.0, "dim": 0.0}
    assert r["latency_s"]["median"] == 0.5
    lo, hi = r["accuracy_ci95"]
    assert 0 < lo < 0.5 < hi < 1
