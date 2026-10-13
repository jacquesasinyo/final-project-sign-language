# TİD Engineering Project I (EE4297)

Spec: `PROJECT_REQUIREMENTS.md` v0.3 (read the relevant section before substantial work).
Source of truth for commitments: the signed proposal `EE4297_Project_Proposal_Jacques_Asinyo_Guney_Donmez.docx`
and `Gantt_Chart_Jacques_Asinyo_Guney_Donmez.xlsx`.
Goal: a defensible Engineering Project I whose finish line is the five official commitments O1–O5 (spec §2).
Offline 20-class accuracy is near saturation (M0 val macro-F1 0.969), so the priorities are real-time latency,
controlled live verification, robustness and efficiency, not higher offline F1.
Do not expand scope without explicit approval.

## Session start
Read `docs/progress.md` (status, next step) and `docs/decisions.md` if they exist.
Check the current Gantt task in spec §14.

## Commitments (official target / internal target)
- O1: fixed 20 signs, official AUTSL test once after the freeze; mean test macro-F1 ≥ 0.90 (internal 0.93).
- O2: M0 (previous CNN-LSTM) vs M1 (BiGRU + attention); M1 ≤ 1/3 of M0's parameters and within ~2 pp (internal 1 pp).
- O3: streaming extraction; from recording stop to displayed prediction, median ≤ 0.5 s and p95 ≤ 1.0 s on the
  MacBook Air M1, CPU only (internal ≤ 0.3 s).
- O4: 120 controlled webcam trials on the 21-sign demo vocabulary (2 team signers × 2 conditions), frozen demo model; top-1 ≥ 75% (internal 85%);
  offline-to-live gap reported.
- O5: robustness analysis (missing hands, mirroring/handedness, environment) + ≥ 1 mitigation evaluated.
Internal targets never appear as promises in the proposal or the report.

## Scope
In: isolated TİD recognition, fixed AUTSL 20-class subset, MediaPipe hands + upper-body pose,
signer-independent evaluation, M0 vs M1, streaming push-to-record system, controlled webcam evaluation (LIVE),
robustness analysis (ROB).
Stretch (spec §16; only after approval, once O1–O5 are on track): the 226-class benchmark, other architectures and
ensembles, calibration/top-3, ablations, extra signers, segmentation, unknown-sign rejection, TTS, embedded.
Out: robotics, ASL, continuous/sentence recognition, translation, RGB/I3D/RGB-D models, mobile app.
When a different model, combination, parameter or idea could clearly improve results, propose it with evidence
(the student wants to hear it); implement it only after approval.

## Previous TÜBİTAK prototype
Previous work, not Project I work. Old code lives in `isolated sign language/` (gitignored): read-only.
Inspect before reusing; never rewrite working code just to modernize it.
Comparison details: `docs/previous_project.md/` (spec §13).

## Hard rules
- IMPORTANT: never use the test split for tuning or model selection;
  evaluate it once per frozen configuration.
- IMPORTANT: the final webcam session is held out like the test split: recorded after
  the design freeze (≈30 Nov) and evaluated once. Mitigations are chosen on validation
  and webcam-dev data only.
- IMPORTANT: never fabricate or recall metrics; report only numbers read
  from saved files in `outputs/`.
- Never silently change splits or selected classes (`configs/classes.yaml`).
- Offline extraction and the live demo share ONE feature-extraction
  implementation. No duplicated preprocessing.

- Ask before adding dependencies or changing the ML framework; say why.
- Before any job expected to exceed ~30 min, state the command and estimate.
  Send long-job output to log files, not the terminal.
- A model "improves" on another only per spec EVAL-08 (same protocol,
  mean ± std over seeds).
- Mirroring stays off until a per-sign semantic-safety check is documented (spec PRE-06).
- Webcam recordings and personal data never go into Git; video only of the team
  members (consent), landmarks only for anyone else (KVKK).

## Workflow
Substantial changes: read the relevant requirement, inspect existing code,
state assumptions, propose the smallest implementation, get approval,
implement, run tests, report changes and open issues. Small fixes: just do them.
Prefer simple code over abstraction. When adding a module, briefly explain
what it does and why (the student must be able to defend it).

## Experiments
Every run records config, classes, split, seed, model, package versions,
git commit, and metrics (spec EVAL-05).

## Library docs
Use Context7 for current APIs of MediaPipe and the ML framework; state the
pinned version in the query.
Pinned: Python 3.12.8, MediaPipe 0.10.21 (legacy Holistic), TensorFlow 2.19.1 / Keras 3.15.1.

## Commands
- Activate env: `source .venv/bin/activate` (macOS/zsh)
- Data: `bash scripts/download_autsl.sh data`
- Classes (once): `python scripts/select_classes.py`
- Manifest + stats: `python scripts/prepare_dataset.py && python scripts/dataset_stats.py`
- Tests: `pytest -q`
- Extract landmarks (~65 min, resumable): `python scripts/extract_landmarks.py --workers 4`
- Train: `python -m tsl.train --model m0 --subset {smoke,full} --seed 0`
- Demo: `python -m tsl.realtime --run outputs/runs/<run> --camera <idx>` (`--list-cameras`, `--check-hands`)
- RT-04 check: `python -m tsl.realtime --run outputs/runs/<run> --video data/raw/AUTSL/<clip from manifest>`
- Demo vocabulary (D-12): add `--config configs/demo.yaml` to prepare/extract/train; runs go to `outputs/runs_demo/`
- Logged live evaluation: `python -m tsl.realtime --run <run> --eval --signer <name> --condition <c> --reps <k>`, then `python scripts/live_report.py outputs/live/<session>`

## Progress
Technical decisions go in `docs/decisions.md`. At the end of a substantial
session, overwrite `docs/progress.md` (keep under ~40 lines): done, status,
known problems, next step. 