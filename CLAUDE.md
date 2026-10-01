# TİD Engineering Project I

Spec: `PROJECT_REQUIREMENTS.md` (read the relevant section before substantial work).
Goal: a manageable, defensible Engineering Project I, not the most
sophisticated sign-language system. Do not expand scope without explicit approval.

## Session start
Read `docs/progress.md` (status, next step) and `docs/decisions.md` if they exist.

## Scope
In: isolated TİD recognition, AUTSL 20-class subset, MediaPipe hands +
upper-body pose, signer-independent evaluation, M0 baseline vs M1
(BiGRU + attention), push-to-record webcam demo.
Out: robotics, ASL, continuous/sentence recognition, LLM translation,
mobile deployment, full 226-class training, extra model families
(see spec Section 3).

## Previous TÜBİTAK prototype
Previous work, not Project I work. Old code lives in `<path>`: read-only.
Inspect before reusing; never rewrite working code just to modernize it.
Comparison details: `docs/previous_vs_project1.md`.

## Hard rules
- IMPORTANT: never use the test split for tuning or model selection;
  evaluate it once per frozen configuration.
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

## Progress
Technical decisions go in `docs/decisions.md`. At the end of a substantial
session, overwrite `docs/progress.md` (keep under ~40 lines): done, status,
known problems, next step. 