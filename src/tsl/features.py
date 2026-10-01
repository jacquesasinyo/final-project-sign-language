"""Raw landmarks -> fixed-length feature sequence (PRE-01..PRE-04), shared offline and live.

Per frame (100 values, docs/decisions.md D-05):
  left hand 21 x (x, y) | right hand 21 x (x, y) | pose 7 x (x, y) | left present | right present
Pose points: nose, shoulders, elbows, wrists.

Coordinates are first made aspect-correct (x scaled by width/height), so 512x512 AUTSL
clips and 640x480 webcam frames share units. Then each frame is centred on the shoulder midpoint
and scaled by the clip's median shoulder width (PRE-02), and the clip is resampled to T frames by
index selection (PRE-03). Missing hands are zeros plus a presence flag of 0 (PRE-04).
"""
import numpy as np

POSE_IDX = [0, 11, 12, 13, 14, 15, 16]  # nose, shoulders, elbows, wrists
L_SHOULDER, R_SHOULDER = 11, 12
N_FEATURES = 21 * 2 * 2 + len(POSE_IDX) * 2 + 2


class ClipFailed(ValueError):
    """The clip cannot be normalised (e.g. shoulders never detected)."""


def _fill_missing(values, valid):
    """Carry forward the last valid row; back-fill rows before the first valid one."""
    out = values.copy()
    idx = np.flatnonzero(valid)
    first = idx[0]
    out[:first] = values[first]
    last = first
    for i in range(first, len(values)):
        if valid[i]:
            last = i
        else:
            out[i] = values[last]
    return out


def resample_indices(n, T):
    return np.round(np.linspace(0, n - 1, T)).astype(int)


def normalise(raw):
    """Raw landmarks dict -> (F, N_FEATURES) normalised features, before resampling."""
    w, h = raw["image_size"]
    aspect = np.array([w / h, 1.0], dtype=np.float32)
    left = raw["left_hand"][..., :2] * aspect
    right = raw["right_hand"][..., :2] * aspect
    pose = raw["pose"][..., :2] * aspect

    shoulders = pose[:, [L_SHOULDER, R_SHOULDER]]
    valid = ~np.isnan(shoulders).any(axis=(1, 2))
    if not valid.any():
        raise ClipFailed("no frame with both shoulders detected")
    centre = _fill_missing(shoulders.mean(axis=1), valid)                       # (F, 2)
    scale = np.median(np.linalg.norm(shoulders[valid, 0] - shoulders[valid, 1], axis=1))
    if not scale > 0:
        raise ClipFailed(f"invalid shoulder width {scale}")

    def norm(points):
        return (points - centre[:, None, :]) / scale

    left_present = ~np.isnan(left).any(axis=(1, 2))
    right_present = ~np.isnan(right).any(axis=(1, 2))
    pose_sel = norm(pose[:, POSE_IDX])
    pose_sel = np.where(np.isnan(pose_sel), 0.0, pose_sel)  # missing pose frame -> zeros
    F = len(left)
    feats = np.concatenate([
        np.where(left_present[:, None, None], norm(left), 0.0).reshape(F, -1),
        np.where(right_present[:, None, None], norm(right), 0.0).reshape(F, -1),
        pose_sel.reshape(F, -1),
        left_present[:, None].astype(np.float32),
        right_present[:, None].astype(np.float32),
    ], axis=1)
    return np.nan_to_num(feats).astype(np.float32)


def clip_features(raw, T):
    """Raw landmarks dict -> (T, N_FEATURES). Raises ClipFailed if normalisation is impossible."""
    feats = normalise(raw)
    return feats[resample_indices(len(feats), T)]
