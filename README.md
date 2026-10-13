# Turkish Sign Language recognition: Engineering Project I

Signer-independent isolated TİD recognition on a fixed 20-class subset of AUTSL,
using MediaPipe hand and upper-body pose landmarks. Spec: `PROJECT_REQUIREMENTS.md`.
Decisions: `docs/decisions.md`.

## Environment (macOS, Python 3.12)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # exact versions: requirements-lock.txt
pip install -e .
```

## Dataset placement

```bash
bash scripts/download_autsl.sh data   # ~13 GB, resumable; verifies SHA-256 and every zip member
```

This creates `data/raw/AUTSL/{train,val,test}/*_color.mp4` and the label CSVs. The location
is set by `data_root` in `configs/default.yaml`.

## Preprocessing (Phase 1)

```bash
python scripts/class_availability.py --data-root data/raw/AUTSL   # docs/class_availability.md
python scripts/select_classes.py      # configs/classes.yaml (run once; refuses to overwrite)
python scripts/prepare_dataset.py     # outputs/manifest.csv; asserts signer-disjoint splits
python scripts/dataset_stats.py       # docs/dataset_stats.md
python scripts/inspect_clips.py       # optional: 50-clip MediaPipe inspection -> outputs/inspect/
pytest -q
```

## Landmarks, training, demo (Phase 2)

```bash
python scripts/extract_landmarks.py --workers 4               # ~65 min on an M1; resumable
python -m tsl.train --model m0 --subset smoke --seed 0         # 10-class smoke test
python -m tsl.train --model m0 --subset full --seed 0          # 20 classes; val metrics only
python -m tsl.realtime --list-cameras
python -m tsl.realtime --run outputs/runs/<run> --camera 1     # Space = record, q = quit
python -m tsl.realtime --run outputs/runs/<run> --video data/raw/AUTSL/<clip>   # RT-04 check
```

The webcam input is flipped before MediaPipe by default (`demo.flip_input`, D-08).

## Demo vocabulary (separate from the benchmark, D-12)

```bash
python scripts/prepare_dataset.py --config configs/demo.yaml             # outputs/manifest_demo.csv
python scripts/extract_landmarks.py --config configs/demo.yaml --workers 4   # only uncached clips
python -m tsl.train --config configs/demo.yaml --model m0 --subset full --seed 0   # -> outputs/runs_demo/
python -m tsl.realtime --run outputs/runs_demo/<run> --camera 1
```

## Logged live evaluation (O4, LIVE-01..05)

```bash
python scripts/show_sign.py                      # watch reference clips first (menu)
python -m tsl.realtime --run outputs/runs_demo/<run> --camera 1 --eval --signer jacques --condition bright --reps 2
python scripts/live_report.py outputs/live/<session> [more sessions]   # accuracy, F1, confusion, latency
```
