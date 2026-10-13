# Project I Requirements — Turkish Sign Language Recognition

**Status:** v0.3 (2026-10-07)
**Scope:** Marmara University EE4297 Engineering Project I (advisor: Assoc. Prof. Fulya Çallıalp)
**Source of truth:** the signed proposal `EE4297_Project_Proposal_Jacques_Asinyo_Guney_Donmez.docx` and
`Gantt_Chart_Jacques_Asinyo_Guney_Donmez.xlsx` (2026-10-05). This file turns them into implementable requirements.
**Primary goal:** design, implement and systematically evaluate a real-time, signer-independent recognition
system for isolated Turkish Sign Language (TİD) signs. Offline 20-class accuracy is already near saturation
(M0 validation macro-F1 0.969, single seed; `docs/decisions.md` D-10), so the project focuses on whether that
performance survives real-time use: latency, controlled live verification, robustness and computational efficiency.

**Revision note (v0.3):** the five official commitments (§2) follow the signed proposal. 226 classes, calibration,
top-3 display and ablations moved to stretch (§16). M1 is framed as an efficient model, not a "better" one.
RT-05 (informal test) is replaced by the controlled webcam evaluation (LIVE). Streaming extraction (RT-01) and
robustness analysis (ROB) were added.

Final pipeline:

```text
AUTSL isolated TİD videos (fixed 20-sign vocabulary)
        -> shared MediaPipe extractor (hands + upper-body pose), per frame; streaming in the live system
        -> normalisation + temporal resampling
        -> cached landmark sequences
        -> M0: previous TÜBİTAK CNN-LSTM (baseline)   |   M1: lightweight BiGRU + attention
        -> signer-independent AUTSL evaluation (seeds, per-class, per-signer)
        -> real-time push-to-record system
        -> controlled webcam evaluation + robustness analysis
```

---

## 1. Project context

A previous TÜBİTAK 2209-A prototype recognised a small custom TİD vocabulary in real time using MediaPipe-based
features, a CNN-LSTM-style model (as described in its final report) and text-to-speech. The proposal describes it as
"limited to 4 fingerspelled letters and 7 words". Its details are documented in `docs/previous_project.md/`.

Project I is **not a superset** of that prototype. Fingerspelled letters and the exact phrase vocabulary are not
required. The Project I contribution is:
- a standardised multi-signer benchmark subset (AUTSL);
- reproducible preprocessing;
- rigorous signer-independent evaluation;
- an efficient temporal model;
- a streaming real-time system with measured latency;
- a controlled webcam evaluation of the offline-to-live gap.

Phase 0 shall fill the table below from the actual TÜBİTAK report and code. Agents must not guess any value;
write `UNKNOWN` if it cannot be found.

| Item | Previous prototype |
|---|---|
| Vocabulary / number of classes | |
| Number of signers | |
| Clips per class | |
| Split method (random / signer-independent) | |
| Reported metric and the split it was measured on | |
| Framework and versions | |
| Feature definition | |

---

## 2. Project I objective and official commitments

**Research question.** How much of the offline, signer-independent recognition performance on a fixed 20-sign
AUTSL vocabulary survives in a real-time webcam system on a laptop CPU, which failure modes explain the gap, and
which mitigations reduce it?

**Official commitments (the finish line).** These match the proposal's targets and Main Steps 1–4. *Official*
targets are the ones written in the proposal. *Internal* targets are what we aim for, and they never appear in
the proposal or the report as promises.

| ID | Commitment | Official target | Internal target | Proposal step |
|---|---|---|---|---|
| O1 | Signer-independent recognition: fixed 20-sign system, official AUTSL split, multiple seeds, untouched test split evaluated once after the freeze; accuracy, macro-F1, per-class F1, per-signer results, confusion matrices | mean test macro-F1 ≥ 0.90 | ≥ 0.93 | 1, 4 |
| O2 | Baseline vs efficient model under the same protocol: M0 (retrained previous CNN-LSTM) vs M1 (BiGRU + attention); recognition, parameter count, training cost, inference latency | M1 ≤ 1/3 of M0's parameters (M0 = 561,236 → ≤ 187,078) and within ~2 pp of M0's mean test macro-F1 | within 1 pp | 2 |
| O3 | Real-time pipeline: streaming landmark extraction during recording, CPU only | from stopping the recording to the displayed prediction: median ≤ 0.5 s, p95 ≤ 1.0 s | median ≤ 0.3 s | 3 |
| O4 | Controlled live verification of the frozen system on the **demo vocabulary** (21 signs, `configs/demo_classes.yaml`, D-12/D-13): 120 webcam trials, 2 signers, 2 lighting/background conditions; offline-to-live gap quantified; failure cases analysed | top-1 ≥ 75% | ≥ 85% | 4 |
| O5 | Robustness: quantify missing hand landmarks, input mirroring/handedness and environmental variation; implement and evaluate ≥ 1 justified mitigation | analysis + ≥ 1 mitigation evaluated | mirrored-input gap ≤ 3 pp, only if mirroring is shown to be semantically safe (PRE-06) | 3, 4 |

The reference machine for every latency figure is a MacBook Air M1 with 8 GB RAM, running on CPU only, with no
internet. The proposal says "consumer laptop CPU"; this is the concrete instance.

**Principles**
- Project success does **not** depend on M1 beating M0. If M1 is comparable but smaller or faster, that is a
  result. If it is worse, analyse it (EVAL-06).
- Targets are goals, not pass/fail gates (EVAL-10).
- The improvement over TÜBİTAK is shown through vocabulary, multi-signer data, signer-independent testing, full
  metrics, measured latency and the controlled live evaluation, not through a single F1 increase (EVAL-04).
- The AUTSL test split is the evidence of **signer-independent generalisation**. The webcam evaluation is evidence
  of **real-world system behaviour** with two non-validated signers. Never present one as the other.
- Ambition beyond O1–O5 is welcome internally. When a different model, combination of models, parameters or idea
  could clearly improve results, propose it with evidence (the student asked for this). It stays in §16 and is
  implemented only after approval, once O1–O5 are on track.

---

## 3. Scope

### Required
- Turkish Sign Language only; isolated sign recognition only.
- AUTSL as the primary dataset; the official train/validation/test protocol is used unchanged.
- A fixed **20-class** subset (`configs/classes.yaml`) for the official system and every committed experiment.
- A **10-class subset is used only as a smoke test** of the pipeline. It is not an experimental stage.
- RGB video is the only raw visual modality. MediaPipe landmarks are extracted and cached once.
- Both hands and upper-body pose landmarks are used.
- The test split is never used for model selection or tuning.
- The live system and the controlled webcam evaluation (O4) use the separate **demo vocabulary** (21 signs, D-12/D-13).
  It shares the data, pipeline and protocol, but the benchmark results (O1/O2) are always reported on the 20 classes.
- A real-time push-to-record system with streaming extraction (RT-01).
- The controlled webcam evaluation (LIVE) and the robustness analysis (ROB).

### Optional (stretch, §16)
- A 226-class AUTSL scalability benchmark. It is a separate experiment and never replaces the 20-class system.

### Out of scope (this semester)
- Continuous or sentence-level recognition and translation.
- RGB, I3D or RGB-D video models.
- A mobile app.
- Robotics.
- ASL.
- Chasing AUTSL state of the art.
- Studies with Deaf participants or large-scale data collection.

---

## 4. Dataset requirements

- **DS-01** AUTSL is the main dataset (Hugging Face mirror, verified; D-01).
- **DS-02 Class selection.** Exactly 20 classes are chosen once, before any model is trained, and stored in
  `configs/classes.yaml` together with the selection rule (done: seeded draw, seed 2026, D-02). The rule must not
  involve validation or test results. The subset is a documented limitation: results do not generalise to all 226 classes.
- **DS-03 Official split.** Use the official train/validation/test partitions unchanged. If test labels or signer IDs
  are unavailable, STOP and report. Do not silently create a replacement split.
- **DS-04 Signer separation.** `prepare_dataset.py` asserts that the signer sets of train, validation and test are
  pairwise disjoint and fails otherwise. A unit test repeats this check on the real manifest.
- **DS-05** Raw videos, webcam recordings and large feature files are not committed to Git.
- **DS-06** All dataset paths are configurable; no hard-coded absolute paths.
- **DS-07 Manifest.** The manifest contains at least: sample ID, class ID/label, signer ID, split, source video path,
  extraction status, frame count, fps and per-hand detection rate.
- **DS-08 Dataset statistics.** Generated by script (never by hand) in `docs/dataset_stats.md`.

---

## 5. Feature extraction and preprocessing

### PRE-00 Decision log
Data-dependent parameters are recorded in `docs/decisions.md` (D-03 … D-08 decided in Phase 1):

| Parameter | Decision |
|---|---|
| MediaPipe API and version (identical offline and live) | legacy Holistic, 0.10.21 (D-04) |
| Sequence length T | 48 (D-06) |
| Pose landmarks used | nose, shoulders, elbows, wrists (D-05) |
| Coordinates | x, y; 100 features per frame (D-05) |
| Min / max live recording duration | 1.5 s / 3.0 s (D-06) |
| Leading/trailing idle frames | measured; no trimming (D-07) |
| Frame orientation and left/right convention | AUTSL as stored; live input flipped (`demo.flip_input`, D-08) |

### PRE-01 Landmarks
Extract left-hand landmarks, right-hand landmarks and the selected upper-body pose landmarks. Full face landmarks are
not required. The orientation and hand-assignment convention (D-08) is used identically offline and live.

### PRE-02 Normalisation
- **Centre:** shoulder midpoint (pose landmarks 11 and 12), computed per frame. If missing in a frame, carry forward
  the most recent valid frame; if there is none earlier, back-fill from the first valid frame.
- **Scale:** median shoulder width over the valid frames of the whole clip.
- If no frame has valid shoulders, mark the clip extraction-failed and log the reason.
- Hands and pose are normalised with the same centre and scale, after aspect correction (x scaled by width/height).

### PRE-03 Fixed-length sequence
Resample each clip to T frames using index selection `round(linspace(0, N-1, T))`, after normalisation and
missing-value handling. Short clips repeat frames. No zero-padding and no interpolation across missing values.

### PRE-04 Missing landmarks
A missing hand is encoded as zeros plus a per-hand presence flag. A clip is never dropped because of missing frames.
Extraction failures are logged with reasons; the failure rate and mean hand-detection rate are reported.

### PRE-05 Caching
Extracted landmarks are cached to disk so training never re-runs MediaPipe.

### PRE-06 Augmentation
- Training data only, mild and seeded. Allowed: small temporal speed variation, small in-plane rotation, small scale
  variation, small coordinate noise, and simulated missing-hand frames (hand dropout).
- **Horizontal mirroring is disabled by default**, because handedness and direction can carry meaning in TİD.
- Mirroring (as augmentation or as test-time handedness handling) may be enabled only after a documented per-sign
  check, in `docs/decisions.md`, that it is semantically safe for the 20 signs. Evidence can come from the clips of
  AUTSL's own left-dominant signers or from a TİD-competent reviewer.

### PRE-07 Shared implementation
- **One shared feature-extraction implementation** is used offline and live. Preprocessing code must not be duplicated.
- For O3 the extractor gets a per-frame (streaming) interface. Offline extraction calls the same interface frame by
  frame.
- After the refactor, the extractor must reproduce the existing landmark cache (verified with RT-04 on several clips).
  Otherwise the cache is regenerated (about 65 min).

---

## 6. Model requirements

Few, controlled experiments; no architecture zoo.

- **MOD-01 Baseline (M0).** The previous prototype's CNN-LSTM, re-implemented layer for layer on the shared pipeline (D-10).
- **MOD-02 Efficient model (M1).** BiGRU + dropout + temporal attention pooling + softmax classifier, sized to meet
  O2 (≤ 1/3 of M0's parameters). It must train on CPU and run interactively on the reference machine.
- **MOD-03 Optional ablation.** If attention (or bidirectionality) is a single config switch, one extra run without
  it may be added after O1–O5 are met.
- **MOD-04** Cross-entropy loss unless a documented experiment requires otherwise.
- **MOD-05 Training control.** Early stopping and best-checkpoint selection on validation macro-F1 (tie-break:
  validation loss). Patience and maximum epochs are configurable.
- **MOD-06** Set random seeds and record them for each experiment.
- **MOD-07 Tuning budget.**
  - At most 3 configurations per model (learning rate, hidden size, dropout only), run with seed 0 and all logged.
    Then freeze.
  - Robustness mitigations (ROB-02) are part of the design. They are chosen on validation and webcam-development data
    only, never on the test split or the final webcam session.
- **MOD-08 Final runs.** Train the frozen configurations on the train split only (validation for early stopping):

  ```text
  M0 baseline  x seeds 0, 1, 2
  M1 efficient x seeds 0, 1, 2
  ```

  If time forces fewer seeds, document it and claim only an observed difference.
- **MOD-09 Framework.** TensorFlow/Keras, kept from the previous project (D-03). Do not migrate for style.

---

## 7. Evaluation requirements

A specific accuracy number is a goal, not a guaranteed requirement.

- **EVAL-01** M0 and M1 use the same classes, preprocessing and data split.
- **EVAL-02** Report: top-1 accuracy, macro-F1, per-class precision/recall/F1, per-signer accuracy, confusion matrix,
  parameter count, training time and latency (EVAL-07).
- **EVAL-03** Validation data may be used for model selection. The test set is evaluated once per frozen
  configuration, after the design freeze. Until then, all reported numbers are validation numbers and are labelled as such.
- **EVAL-04** Do not report an "improvement" unless the models share the dataset subset and protocol. Historical
  TÜBİTAK results may be described but are not a controlled comparison.
- **EVAL-05** Every experiment saves: configuration, model name, seed, training history, final metrics, timestamp,
  git commit hash, package versions, and hashes of the class list and manifest.
- **EVAL-06** If M1 does not outperform M0, keep and analyse the result. Do not alter splits, classes or metrics to
  manufacture an improvement.
- **EVAL-07 Latency (O3).** On the reference machine, report separately:
  - MediaPipe time per frame;
  - model forward time per window;
  - end-to-end delay from stopping the recording to the displayed prediction, with streaming extraction.

  Use warm-up runs and enough repetitions for a stable median, and report the p95.
- **EVAL-08 Reporting.** Report mean ± std over seeds for top-1 accuracy and macro-F1 for M0 and M1. If M1 is
  consistently better across seeds, state the observed improvement. If results overlap or are inconsistent, state
  "comparable performance". Do not claim statistical significance unless a test was run.
- **EVAL-09 Error analysis.** List the most confused class pairs, per-signer differences, and whether errors coincide
  with missing-hand frames. Discuss limitations: the subset, few test signers, the webcam domain shift, and
  non-validated live signers.
- **EVAL-10 Targets.** Official and internal targets (§2) are written to `docs/targets.md` before test evaluation,
  and may be revised once. They are goals, not pass/fail gates.
- **EVAL-11 Offline-to-live gap (O4).** For the same frozen demo model, report its AUTSL test accuracy (21 demo classes, evaluated once after the freeze) next to webcam accuracy,
  overall and per class.

### Robustness analysis (O5)
- **ROB-01** Quantify the effect of:
  - random missing hand landmarks (per-frame dropout rates);
  - coordinate noise;
  - horizontal mirroring/handedness;
  - lighting/background (the two LIVE conditions);
  - signer differences (per-signer AUTSL results).

  Synthetic perturbations run on validation during development, and once on the test split at the end.
- **ROB-02** Implement at least one justified mitigation (e.g. augmentation with hand dropout; mirroring only if
  PRE-06 allows it; input-quality guards) and measure its effect with the same tests.
- **ROB-03** All robustness tests are scripted and their results saved under `outputs/`. Exploratory numbers that
  were not saved never go into a report.

---

## 8. Real-time system requirements

- **RT-01 Push-to-record with streaming extraction.**
  - Space starts recording, the user performs one sign, and Space stops recording.
  - Frames go through the shared per-frame extractor *during* recording (PRE-07). After stopping, only the remaining
    frames, normalisation (clip-level median shoulder width), resampling to T and the model forward pass run.
  - Display the predicted label and its confidence.
- **RT-02 Recording guards.** Configurable minimum and maximum recording duration (auto-stop at the maximum). If the
  clip is too short or no hand is detected, show a clear message instead of a prediction.
- **RT-03 Uncertainty (optional).** "Uncertain / no prediction" below a confidence threshold chosen on validation.
  Validation errors are mostly high-confidence, so this is not reliable rejection and must not be described as such.
- **RT-04 Consistency check.** Feed AUTSL video files through the live code path. The prediction must match the
  offline pipeline, and any feature differences are explained. Repeat this after the streaming refactor.
- **RT-06 Optional continuous mode.** Stretch only. If implemented, it must resample by timestamp, gate on hand
  presence and smooth over recent predictions.
- **RT-07** Text-to-speech may be preserved from the previous prototype (stretch).
- **RT-08** Do not claim a fixed FPS until measured.

### Controlled webcam evaluation (O4)
RT-05 (informal live test) is superseded by LIVE-01 … LIVE-05.

- **LIVE-01 Protocol.**
  - **120 trials** in total: the 2 signers are the two team members, under 2 lighting/background conditions, using
    the frozen demo model on the reference machine.
  - The allocation (signs × repetitions per signer and condition) is fixed in `docs/decisions.md` before the
    development session. 120 does not divide evenly over 21 signs × 2 signers × 2 conditions, so state the choice
    explicitly.
  - Trials are recorded with the logger: `python -m tsl.realtime --run <run> --eval --signer <name> --condition <c>
    --reps <k>`. It writes `outputs/live/<session>/` (trials.csv, landmarks, meta.json).
    `scripts/live_report.py` computes the LIVE-04 metrics. Hand-written results are informal and never reported as O4.
  - Sign order is randomised. Any sign excluded is listed with the reason, decided before recording.
- **LIVE-02 Sessions.** A development session may guide robustness work (ROB-02). The final session is recorded after
  the design freeze and evaluated once, like the AUTSL test split.
- **LIVE-03 Data.**
  - For every trial, store the landmarks, the intended label and the recording metadata, so every model can be
    re-scored offline.
  - Raw video only of the team members, with their consent.
  - Any other participant: informed consent, landmarks only, handled under KVKK (Law No. 6698).
  - Webcam data is never committed to Git.
- **LIVE-04 Reporting.**
  - Label the results "controlled real-time webcam evaluation: two non-validated signers, reference machine". It is
    not a user study and not evidence of broad signer generalisation.
  - Report top-1 overall, per sign, per signer and per condition, and the offline-to-live gap (EVAL-11).
  - Report failed trials (no hand detected, too short) separately.
  - With 120 trials the 95% interval is about ±8 pp, so say so.
- **LIVE-05 Target.** Top-1 ≥ 75% (official), ≥ 85% (internal).

---

## 9. Software architecture

Keep the repository simple: one package of roughly 8 modules.

```text
sign language/
├── configs/            # classes.yaml, default.yaml
├── src/tsl/            # landmarks, features, data, models, train, evaluate, realtime, utils
├── scripts/            # thin CLI wrappers
├── tests/
├── docs/               # decisions.md, dataset_stats.md, targets.md, previous_project.md/, progress.md
├── outputs/            # gitignored: manifest, runs, robustness and live results
├── PROJECT_REQUIREMENTS.md
└── README.md
```

Datasets, caches, checkpoints, webcam recordings and generated outputs are excluded from Git.

---

## 10. Configuration and reproducibility

- **SW-01** Important settings live in config files: dataset root, classes, sequence length, batch size, learning rate,
  hidden size, dropout, random seed, min/max recording duration, input flip, and confidence threshold (if RT-03 is used).
- **SW-02** Training, evaluation, robustness tests, latency benchmarks and the live system run from documented commands.
- **SW-03** The README contains: environment setup, dataset placement, preprocessing, training, evaluation, and the
  live-system and live-evaluation commands.
- **SW-04** Missing data or model files produce an understandable error.
- **SW-05** Lightweight tests for:
  - config loading, landmark tensor shapes, sequence resampling, label mapping, model output shape;
  - signer disjointness;
  - **streaming-extractor equivalence** (per-frame output equals the cached landmarks for a clip; skipped if no data).

  No large UI test suite.

---

## 11. Environment

Python 3.12.8 in `.venv`; TensorFlow 2.19.1 / Keras 3.15.1; MediaPipe 0.10.21; NumPy 1.26.4 (D-03). Exact versions are
in `requirements-lock.txt`. Do not add a dependency when an existing one or the standard library suffices.

---

## 12. Project I acceptance criteria

Infrastructure (already met unless noted):
1. 20 classes and the selection rule are in `configs/classes.yaml`; dataset statistics are generated by script.
2. Tests pass, including signer disjointness on the real manifest.
3. Landmark extraction and caching run; the failure rate and hand-detection rate are reported.
4. `docs/decisions.md` records every decision; README commands reproduce the pipeline from cached landmarks.

Commitments:
5. **O1** M0 and M1 are trained under the same protocol (3 seeds each), every run stores the EVAL-05 artefacts, and the
   test metrics (EVAL-02) are exported once per frozen configuration, with targets written beforehand (EVAL-10).
6. **O2** The M0 vs M1 statement follows EVAL-08 and includes parameter count, training time and forward latency.
7. **O3** The streaming pipeline is implemented, RT-04 is repeated, and latency is reported per EVAL-07.
8. **O4** The controlled webcam evaluation is done per LIVE-01 … LIVE-05, with the offline-to-live gap (EVAL-11).
9. **O5** The robustness analysis (ROB-01 … ROB-03) is done, with at least one mitigation evaluated.
10. `docs/previous_project.md/` contains the Phase 0 table and the Previous vs Project I comparison, with evidence (§13).
11. The final report, poster and demo (application and video) are prepared.

No stretch item (§16) is required for completion.

---

## 13. Required "Previous Work vs Project I" document

In `docs/previous_project.md/`, provide the Phase 0 table (§1) and an "Aspect | Previous | Project I | Evidence"
table. State explicitly:

- Previous results are not a controlled comparison to Project I results (different vocabulary, signers and protocol).
- Letter recognition is not carried over.
- The controlled comparison is M0 (the previous architecture, retrained) vs M1 under the Project I protocol.
- Report-vs-evidence discrepancies are noted, e.g. the CNN-LSTM was never executed in the saved notebook (D-10).

Previous prototype: small custom vocabulary, limited custom dataset, MediaPipe features, CNN/LSTM-style model,
real-time demo, text-to-speech.

Project I:
- the AUTSL 20-class subset with signer-independent evaluation;
- reproducible preprocessing and feature cache;
- the controlled baseline vs an efficient model;
- systematic metrics;
- streaming real-time operation with measured latency;
- the controlled webcam evaluation and the robustness analysis.

Do not claim a contribution that has not been implemented and evaluated.

---

## 14. Schedule (from the Gantt chart), cut order and risks

Week 1 starts 29 Sep 2026.

| Dates | Gantt task | Work | Commitment |
|---|---|---|---|
| 29 Sep – 3 Oct ✓ | (preliminary work) | Phases 0–2: data, environment, manifest, classes, landmark cache, M0, demo skeleton | — |
| 5 – 11 Oct | Scope, Requirements & Literature Review | this spec, `docs/targets.md`, Phase 0 table, decision D-11 | — |
| 12 – 25 Oct | Lightweight Temporal Model Development | M1; ≤ 3 configurations per model on validation | O2 |
| 26 Oct – 8 Nov | Real-Time Recognition Pipeline Optimization | streaming extractor, RT-04 repeat, latency benchmark | O3 |
| 9 – 29 Nov | Robustness Analysis and Data Augmentation | ROB-01 suite, mitigation(s), webcam development session, LIVE allocation decided | O5 |
| **30 Nov** | — | **design freeze** | — |
| 30 Nov – 6 Dec | Final Model Selection and Offline Evaluation | final seeds, single AUTSL test evaluation | O1, O2 |
| 7 – 13 Dec | Controlled Real-Time Webcam Evaluation | final webcam session (120 trials) | O4 |
| 14 – 20 Dec | Results Analysis and Engineering Verification | gap, failure and robustness analysis; verification against §2 | O1–O5 |
| 21 Dec – 3 Jan | Final Application, Report, Poster and Demo Preparation | report, poster, demo video, previous-work document | — |

The proposal states the schedule covers 10/2026 – 05/2027. Spring (Project II) is outside this spec and is planned
with the advisor after Project I.

Stretch items (§16) only start when the current task is on schedule. Do not rewrite working components unnecessarily,
and keep the repository runnable after each task.

**Cut order if behind:** stretch items → the development webcam session shortened → 3 seeds reduced to 1 (claims
weakened accordingly). The 120 final webcam trials and the single test evaluation are not cut.

| Risk | Fallback |
|---|---|
| Live accuracy below 75% | Report honestly. The failure analysis (O5) is still a result. The offline benchmark remains the main signer-independence evidence |
| Streaming refactor changes the landmarks | RT-04 on several clips; regenerate the cache if needed (~65 min) |
| Mirroring not semantically safe | Keep it off; report the handedness gap without that mitigation |
| Team signers' execution not validated | Compare with ≥ 2 AUTSL reference signers per sign before the final session; a TİD-competent reviewer is a stretch goal |
| M1 more than 2 pp behind M0 | Report per EVAL-06; deploy the model that better meets O3/O4; still make the efficiency statement |
| Hard test signers (14 and 27 were the most misclassified in ChaLearn 2021) | The 0.90 official target leaves margin; report per-signer results |
| Fanless reference machine throttles during long jobs | Run long jobs in Terminal with `caffeinate`; jobs are resumable |

---

## 15. Instructions for AI coding agents

1. Read this file before proposing or implementing major changes.
2. Inspect the existing codebase before creating replacement code; reuse working components when reasonable.
3. The official commitments O1–O5 (§2) are the finish line. Never promote a stretch item (§16) to a commitment, and
   never present internal targets as promises. Do propose better models or ideas when the evidence supports them;
   implement them only after approval.
4. Prefer the simplest implementation that satisfies the requirement.
5. Do not migrate ML frameworks or add dependencies without a stated, concrete need.
6. Never fabricate dataset statistics, training results, accuracy, latency, plots or experiment outputs.
7. Never use the test split or the final webcam session to tune the model. Never silently change class lists or splits.
8. Keep data paths configurable; never commit local paths, recordings or personal data.
9. Make small, reviewable changes and run relevant tests after each meaningful change.
10. Document assumptions that materially affect the experiment instead of guessing.
11. If required data, previous code or labels are unavailable, stop that part and report exactly what is missing.
12. Before any job expected to run over 30 minutes, state the command and estimated time.
13. Log any deviation from PRE-00 decisions in `docs/decisions.md`.
14. Record package versions and the git commit hash in every experiment output.
15. Keep the repository runnable after each task; update the README when commands change.
16. Optimise for project completion and reproducibility, not architectural sophistication.

---

## 16. Stretch goals (not committed)

Start only once O1–O5 are on track, and only with the student's approval:

- 226-class AUTSL scalability benchmark (same split as ChaLearn 2021; about 11–12 h extraction, about 1 h per training run).
- Further architectures and features: ST-GCN, Transformer/1D-CNN, velocity and hand-local features, model ensembles.
- Confidence calibration, the uncertainty threshold (RT-03), a top-3 display.
- Ablations:
  - random vs signer-independent split;
  - TÜBİTAK-style vs Project I features;
  - attention/bidirectionality (MOD-03);
  - pose-only vs hands + pose.
- More live signers; a TİD-competent validator.
- Automatic segmentation, hands-free and continuous mode (RT-06); rejection of unknown signs.
- Text-to-speech (RT-07); few-shot user calibration; embedded deployment (TFLite, Raspberry Pi).

Not worth pursuing this semester: everything listed under §3 "Out of scope".

---

## 17. Assumptions checklist

Status in `docs/decisions.md` D-09:

- [x] AUTSL access and test labels; signer IDs parsed; per-class counts sufficient.
- [x] Overlap with the previous vocabulary (none exact, D-02).
- [x] Clip duration, fps and resolution; idle time.
- [x] MediaPipe + TF install together; extraction speed; laptop and webcam (30 fps).
- [x] Mirroring and handedness convention for the live input (D-08). Semantic safety of mirroring per sign is still
  open (PRE-06).
- [~] Previous code recoverable (notebooks present, environment broken); Phase 0 table still open.
- [ ] Course requirements: report structure, deadlines, whether a numeric improvement is expected (ask the advisor).
- [ ] Availability of a TİD-competent person to validate sign execution.
