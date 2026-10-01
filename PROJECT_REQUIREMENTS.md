# Project I Requirements — Turkish Sign Language Recognition

**Status:** Draft v0.2
**Scope:** Marmara University Engineering Project I
**Primary goal:** Deliver a clear, measurable improvement over the previous TÜBİTAK Turkish Sign Language (TİD) recognition prototype, in data, evaluation rigor and temporal modelling, without expanding into continuous sentence translation or any other large effort this semester.

Final pipeline:

```text
AUTSL isolated TİD videos
        -> shared MediaPipe extractor (hands + upper-body pose)
        -> normalisation + temporal resampling
        -> cached landmark sequences
        -> M0: LSTM/GRU baseline   |   M1: BiGRU + attention
        -> signer-independent evaluation
        -> push-to-record webcam demo
```

---

## 1. Project context

A previous TÜBİTAK 2209-A prototype recognised a small custom TİD vocabulary in real time using MediaPipe-based features, a CNN-LSTM-style model and text-to-speech. Its details are documented in `docs/previous_project.md`

Project I is **not a superset** of that prototype. Fingerspelled letters (J, A, C, K) and the exact phrase vocabulary are not required in Project I. The Project I contribution is a standardised multi-signer TİD benchmark subset (AUTSL), stronger preprocessing and temporal modelling, rigorous signer-independent evaluation, and a measured real-time demo.

Phase 0 shall fill the table below from the actual TÜBİTAK report and code. Agents must not guess any value; write `UNKNOWN` if it cannot be found.

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

## 2. Project I objective
IMPORTANT CORRECTION: D0N'T LIMIT YOURSELF TO THE PERFORMANCE GOAL SET IN THE PROJECT_REQUIREMENT. IF YOU THINK A DIFFERENT MODEL(OR COMBINATION OF MODELS), PARAMETERS OR IDEAS COULD MAKE THE PREDICTIONS BETTER, LET ME KNOW

Build and evaluate a **signer-independent, isolated Turkish Sign Language recognition system with a push-to-record webcam demo**, on a fixed 20-class subset of AUTSL.

Three improvements over the previous work:

1. A larger, standardised, multi-signer dataset instead of a small custom one.
2. Stronger preprocessing and temporal modelling.
3. Proper evaluation on unseen signers: accuracy, macro F1, per-class metrics, confusion matrix, latency.

---

## 3. Scope

### Required

- Turkish Sign Language only.
- Isolated sign recognition only.
- AUTSL as the primary dataset.
- Fixed **20-class** subset for all reported experiments.
- A **10-class subset is used only as a smoke test** of the pipeline. It is not an experimental stage.
- RGB video is the only raw visual modality.
- MediaPipe landmarks are extracted and cached once.
- Both hands and upper-body pose landmarks are used.
- The official AUTSL train/validation/test protocol is used unchanged.
- The test set is never used for model selection or tuning.
- The final system supports a push-to-record webcam demo for the selected vocabulary.



---

## 4. Dataset requirements

- **DS-01** AUTSL is the main dataset.
- **DS-02 Class selection.** Exactly 20 classes are chosen once, before any model is trained, and stored in `configs/classes.yaml` together with the selection rule. The rule must not involve validation or test results. Acceptable rules: (a) seeded random sample (seed recorded), or (b) a curated list justified by criteria such as overlap with the previous prototype vocabulary and sufficient clips per split. The class subset is a documented limitation: results do not generalise to all 226 classes.
- **DS-03 Official split.** Use the official train/validation/test partitions unchanged. If test labels or signer IDs are unavailable, STOP and report. Do not silently create a replacement split.
- **DS-04 Signer separation.** `prepare_dataset.py` shall assert that the signer sets of train, validation and test are pairwise disjoint and fail with an error otherwise. A unit test repeats this check on the real manifest.
- **DS-05** Raw videos and large feature files are not committed to Git.
- **DS-06** All dataset paths are configurable; no hard-coded absolute paths.
- **DS-07 Manifest.** The preprocessing pipeline saves a manifest with at least: sample ID, class ID/label, signer ID, split, source video path, extraction status, frame count, fps, and per-hand detection rate.
- **DS-08 Dataset statistics.** Generate (by script, never by hand) a table of clips per class per split and signers per split, in `docs/dataset_stats.md`.

---

## 5. Feature extraction and preprocessing

### PRE-00 Decision log
Parameters that depend on the real data are **initial defaults, not requirements**. In Phase 1, after inspecting AUTSL clips and the old code, the agent confirms or changes each and records value and reason in `docs/decisions.md` before Phase 2 starts.

| Parameter | Initial default |
|---|---|
| MediaPipe API and version (identical offline and live) | decided in Phase 0 |
| Sequence length T | 48 (check against clip-duration statistics) |
| Pose landmarks used | nose, shoulders, elbows, wrists |
| Coordinates | x, y only (z optional) |
| Min / max live recording duration | from AUTSL clip-duration statistics |
| Leading/trailing idle frames in AUTSL clips | measured and documented; trimming optional, and if used it must be applied identically offline and live |
| Frame orientation and left/right hand convention | decided and documented |

### PRE-01 Landmarks
Extract left-hand landmarks, right-hand landmarks, and the selected upper-body pose landmarks. Full face landmarks are not required. MediaPipe handedness labels assume mirrored input; the orientation and hand-assignment convention from the decision log must be used identically offline and live.

### PRE-02 Normalisation
- **Centre:** shoulder midpoint (pose landmarks 11 and 12), computed per frame. If missing in a frame, carry forward the most recent valid frame; if none earlier, back-fill from the first valid frame.
- **Scale:** median shoulder width over the valid frames of the whole clip.
- If no frame has valid shoulders, mark the clip extraction-failed and log the reason.
- Hands and pose are normalised with the same centre and scale.

### PRE-03 Fixed-length sequence
Resample each clip to T frames (configurable) using index selection `round(linspace(0, N-1, T))`, applied after normalisation and missing-value handling. Short clips therefore repeat frames. No zero-padding and no interpolation across missing values.

### PRE-04 Missing landmarks
A missing hand is encoded as zeros plus a per-hand presence flag. A clip is never dropped because of missing frames. Extraction failures are logged with reasons; the failure rate and mean hand-detection rate are reported.

### PRE-05 Caching
Extracted landmarks are cached to disk so training never re-runs MediaPipe.

### PRE-06 Augmentation
Training data only, mild, and seeded. Allowed: small temporal speed variation, small in-plane rotation, small scale variation, small coordinate noise. Horizontal mirroring is disabled by default because handedness may be meaningful.

### PRE-07 Shared implementation
**One shared feature-extraction implementation** is used for offline extraction and the live demo. Preprocessing code must not be duplicated.

---

## 6. Model requirements

Few, controlled experiments; no architecture zoo.

- **MOD-01 Baseline (M0).** One simple unidirectional LSTM or GRU classifier on the shared pipeline. If the previous CNN-LSTM can be plugged into the shared pipeline in about one day of work, it may be used instead.
- **MOD-02 Improved model (M1).** BiGRU + dropout + temporal attention pooling + softmax classifier. Small enough to train on CPU or a modest GPU and to run interactively on a laptop.
- **MOD-03 Optional ablation.** If attention (or bidirectionality) is a single config switch, one extra run without it may be added after all acceptance criteria are met. Not required.
- **MOD-04** Cross-entropy loss unless a documented experiment requires otherwise.
- **MOD-05 Training control.** Early stopping and best-checkpoint selection on validation macro-F1 (tie-break: validation loss). Patience and maximum epochs are configurable.
- **MOD-06** Set random seeds and record them for each experiment.
- **MOD-07 Tuning budget.** At most 3 configurations per model (learning rate, hidden size, dropout only), run with seed 0 and all logged. Then freeze.
- **MOD-08 Final runs (default protocol).** Train the frozen configurations on the train split only (validation for early stopping):

  ```text
  M0 baseline x seeds 0, 1, 2
  M1 improved x seeds 0, 1, 2
  ```

  If time forces fewer seeds, document it and claim only an observed difference.
- **MOD-09 Framework.** Do not migrate ML frameworks for style. Inspect the previous project; keep its framework if functional; change only for a concrete compatibility reason, stated in the decision log.

---

## 7. Evaluation requirements

A specific accuracy number is a goal, not a guaranteed requirement.

- **EVAL-01** M0 and M1 use the same classes, preprocessing and data split.
- **EVAL-02** Report: top-1 accuracy, macro F1, per-class precision and recall, confusion matrix, parameter count, and latency (EVAL-07).
- **EVAL-03** Validation data may be used for model selection. The test set is evaluated once per frozen configuration, after the design is frozen.
- **EVAL-04** Do not report an "improvement" unless models share the dataset subset and protocol. Historical TÜBİTAK results may be described but are not a controlled comparison.
- **EVAL-05** Every experiment saves: configuration, model name, seed, training history, final metrics, timestamp, git commit hash, package versions, and hashes of the class list and manifest.
- **EVAL-06** If M1 does not outperform M0, keep and analyse the result. Do not alter splits, classes or metrics to manufacture an improvement.
- **EVAL-07 Latency.** On a documented machine, report separately: MediaPipe time per frame, model forward time per window, and end-to-end delay from end of recording to displayed prediction. Use a few warm-up runs, enough repeated runs for a stable median, and also report a high percentile.
- **EVAL-08 Reporting.** Report mean ± std over seeds for top-1 accuracy and macro-F1 for M0 and M1. If M1 is consistently better across seeds, state the observed improvement. If results overlap or are inconsistent, state "comparable performance". Do not claim statistical significance unless a test was run.
- **EVAL-09 Error analysis.** List the most confused class pairs and note whether errors coincide with frequent missing-hand frames. Discuss limitations: curated subset, few test signers, domain shift to webcam.
- **EVAL-10 Target.** A target macro-F1 is set after the smoke test, written to `docs/targets.md` before test evaluation, and may be revised once. It is a goal, not a pass/fail gate.

---

## 8. Real-time demo requirements

- **RT-01 Primary mode: push-to-record.** Space starts recording, the user performs one sign, Space stops recording. The captured clip goes through the shared pipeline (PRE-07): landmark extraction, normalisation (including the clip-level median shoulder width), resampling to T, prediction. Display predicted label and confidence. Frames may be buffered and processed after stopping, through the same function used offline.
- **RT-02 Recording guards.** A configurable minimum and maximum recording duration (auto-stop at the maximum). If the clip is too short or no hand is detected, show a clear message instead of a prediction.
- **RT-03 Uncertainty.** Show "uncertain / no prediction" below a confidence threshold chosen on the validation set. This is not reliable out-of-vocabulary rejection and must not be described as such.
- **RT-04 Consistency check.** Feed at least one AUTSL video file through the live code path. The predicted class should match the offline pipeline, and any feature differences should be inspected and explained. No fixed numeric tolerance is required.
- **RT-05 Informal live test.** Test at least 10 of the 20 classes, 5 repetitions each. Choose the signs the student can reproduce confidently from reference videos, decide the list *before* seeing live results, and report all signs attempted or excluded and why. Report the hit rate in a separate table labelled: single non-validated signer, different camera, not comparable to AUTSL test results. If a TİD-competent person is available, expand the test or have sign execution validated.
- **RT-06 Optional continuous mode.** Only after all acceptance criteria are met. If implemented, it must resample by timestamp, gate on hand presence, and smooth over recent predictions.
- **RT-07** Text-to-speech may be preserved from the previous prototype but is not a required contribution.
- **RT-08** Do not claim a fixed FPS until measured.

---

## 9. Software architecture

Keep the repository simple: one package of roughly 8 modules.

```text
project/
├── configs/            # classes.yaml, default.yaml
├── src/tsl/            # landmarks, features, data, models, train, evaluate, realtime, utils
├── scripts/            # thin CLI wrappers (optional)
├── tests/
├── docs/               # decisions.md, dataset_stats.md, targets.md, previous_vs_project1.md
├── outputs/            # gitignored
├── PROJECT_REQUIREMENTS.md
└── README.md
```

Datasets, caches, checkpoints and generated outputs are excluded from Git.

---

## 10. Configuration and reproducibility

- **SW-01** Important settings live in config files: dataset root, selected classes, sequence length, batch size, learning rate, hidden size, dropout, confidence threshold, random seed, min/max recording duration.
- **SW-02** Training, evaluation and inference run from documented command-line commands.
- **SW-03** README contains: environment setup, dataset placement, preprocessing, training, evaluation, and demo commands.
- **SW-04** Missing data or model files produce an understandable error.
- **SW-05** Lightweight tests for: config loading, landmark tensor shapes, sequence resampling, label mapping, model output shape, signer disjointness. No large UI test suite.

---

## 11. Environment

Recommended starting point: Python 3.11, unless the previous working environment requires another version. Verify MediaPipe and the ML framework install together early (protobuf conflicts are common). Pin the MediaPipe version.

Minimal dependencies: the existing ML framework (TensorFlow/Keras or PyTorch), MediaPipe, OpenCV, NumPy, scikit-learn, matplotlib, PyYAML, tqdm. Do not add a dependency when an existing one or the standard library suffices.

---

## 12. Project I acceptance criteria

Project I is complete when all of the following hold:

1. 20 classes and the selection rule are recorded in `configs/classes.yaml`; dataset statistics are generated by script.
2. Tests pass, including the signer-disjointness check on the real manifest.
3. Landmark extraction and caching run; the failure rate and hand-detection rate are reported.
4. M0 and M1 are trained under the same protocol (default: 3 seeds each); every run stores the EVAL-05 artefacts.
5. Test metrics (accuracy, macro-F1, per-class precision/recall, confusion matrix, parameter count, latency) are exported.
6. The improvement statement follows EVAL-08.
7. The RT-04 consistency check is done and documented.
8. The push-to-record demo works, and the RT-05 table is reported.
9. `docs/decisions.md` records every Phase 1 decision.
10. README commands let a fresh clone reproduce the pipeline from cached landmarks.
11. `docs/previous_project.md` is complete, with evidence for each claim.
12. No robotics or continuous-recognition work is needed.

---

## 13. Required "Previous Work vs Project I" document

Create `docs/previous_project.md` with the Phase 0 table (Section 1) and an "Aspect | Previous | Project I | Evidence" table. State explicitly:

- Previous results are not a controlled comparison to Project I results (different vocabulary, signers and protocol).
- Letter recognition is not carried over.
- The controlled comparison is M0 vs M1 under the Project I protocol.

Previous prototype: small custom vocabulary, limited custom dataset, MediaPipe features, CNN/LSTM-style model, real-time demo, text-to-speech.
Project I: AUTSL 20-class benchmark subset, signer-independent evaluation, reproducible preprocessing and feature cache, controlled baseline, improved temporal model, systematic metrics, measured real-time performance.

Do not claim a contribution that has not been implemented and evaluated.

---

## 14. Development phases, schedule and risks

Adjust weeks to your real calendar.

| Weeks | Phase | Work |
|---|---|---|
| 1-2 | Phase 0 | Audit previous work (hard cap: 1 week). Fill the Section 1 table. If old code will not run, document it as non-reproducible and describe it from the TÜBİTAK report. |
| 1-2 | Phase 1 | AUTSL access, environment, manifest, class selection, dataset statistics, decision log (PRE-00). |
| 2-4 | Phase 2 | Landmark extraction and caching, 10-class smoke test, train M0, demo skeleton with M0. |
| 5-7 | Phase 3 | Implement M1, up to 3 tuning configs per model, freeze design, write target. |
| 8-9 | Phase 4 | Final seeds, one test evaluation, plots, error analysis, latency. |
| 9-10 | Phase 4 | Demo polish, consistency check, informal live test. |
| 10-12 | Phase 5 | README, decisions, previous-vs-Project-I, report, poster, buffer. |

Do not rewrite working previous components unnecessarily. Keep the repository runnable after each phase.

**Cut order if behind:** continuous mode -> optional ablation -> 3 seeds to 1 seed (weaken claims accordingly).

| Risk | Fallback |
|---|---|
| AUTSL access delayed or test labels missing | Stop and ask the advisor; do not invent a split |
| Old code will not run | Document as non-reproducible; describe from the report |
| MediaPipe / ML framework version conflict | Separate environments, or switch frameworks with a stated reason |
| Webcam accuracy poor due to domain shift | Report honestly; the offline benchmark remains the main result |
| M0 and M1 perform equally | Report as comparable (EVAL-06/08); the contribution rests on data, protocol and evaluation |

---

## 15. Instructions for AI coding agents

1. Read this file before proposing or implementing major changes.
2. Inspect the existing codebase before creating replacement code; reuse working components when reasonable.
3. Do not expand Project I beyond the scope in this document.
4. Prefer the simplest implementation that satisfies the requirement.
5. Do not migrate ML frameworks or add dependencies without a stated, concrete need.
6. Never fabricate dataset statistics, training results, accuracy, latency, plots, or experiment outputs.
7. Never use the test set to tune the model. Never silently change class lists or splits.
8. Keep data paths configurable; never commit local paths.
9. Make small, reviewable changes and run relevant tests after each meaningful change.
10. Document assumptions that materially affect the experiment instead of guessing.
11. If required data, previous code, or labels are unavailable, stop that part and report exactly what is missing.
13. Before any job expected to run over 30 minutes, state the command and estimated time.
14. Where preprocessing is ambiguous, use the initial defaults in PRE-00 and log any deviation in `docs/decisions.md`.
15. Record package versions and the git commit hash in every experiment output.
16. Keep the repository runnable after each phase; update the README when commands change.
17. Optimise for project completion and reproducibility, not architectural sophistication.

---

## 16. Stretch goals

Only after all acceptance criteria are complete:

- attention / bidirectionality ablation (MOD-03)
- pose-only vs hands+pose ablation
- optional continuous mode (RT-06)
- preserved text-to-speech (RT-07)

---

## 17. Assumptions to verify before coding

Report the result of each in `docs/decisions.md`.

- [ ] AUTSL access, license and passwords; test labels present in your copy.
- [ ] Signer IDs can be parsed from filenames or metadata; signer counts per split.
- [ ] Per-class clip counts for candidate classes are sufficient in every split.
- [ ] Overlap between AUTSL classes and the previous vocabulary; fingerspelled letters likely absent.
- [ ] Clip duration, fps and resolution distributions; leading/trailing idle time.
- [ ] Whether AUTSL frames are mirrored; presence of left-handed signers.
- [ ] Previous code, data, models, framework versions and evaluation protocol are recoverable.
- [ ] MediaPipe and the ML framework install together on the chosen Python version.
- [ ] Extraction speed on the development machine (time 50 clips first).
- [ ] Laptop specs and achievable webcam fps.
- [ ] Course requirements: deliverables, report structure, engineering constraints/standards sections, deadlines, whether a numeric improvement is expected (ask the advisor).
- [ ] Availability of a TİD-competent person to validate sign execution.
