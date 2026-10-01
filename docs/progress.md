# Progress

## Done
- Phase 1: AUTSL verified download (D-01), 20 classes + 10 smoke (D-02), env (D-03),
  manifest + `docs/dataset_stats.md`, PRE-00 decisions D-04–D-09.
- Phase 2:
  - Landmark cache: 3,241/3,241 clips, 0 failures; hand detection rate L 0.86 / R 0.90.
  - M0 = previous CNN-LSTM (D-10). Seed 0 val macro-F1: smoke 0.969
    (`outputs/runs/20261001-181221_m0_smoke_s0`), 20 classes 0.969
    (`outputs/runs/20261001-181937_m0_full_s0`). Validation only; test untouched.
  - RT-04 consistency check passed (live path = offline, identical features).
  - D-08 resolved: `demo.flip_input: true`, confirmed by the student's webcam test.
  - Informal live try-out with M0 (no flip): 13/20 signs correct with the right hand;
    bayram, olmaz, pencere, turkiye worked only with the left hand (fixed by the flip,
    retest pending); ben, mendil, tuvalet failed with both hands.
- First git commit.

## Status
Phase 2 complete except `docs/targets.md` (EVAL-10; proposed target needs revising
upward, since M0 val is already 0.97).

## Known problems / open
- Retest live with flip; check ben / mendil / tuvalet execution against references.
- M1 headroom is small (M0 val 0.97); the improvement case may rest on protocol,
  size and latency, and on the live demo. Discuss with the advisor.
- Ideas beyond spec (student asked to raise them): hand-local + velocity features,
  augmentation with hand dropout, seed ensembles, ST-GCN. To discuss.
- Open: advisor (course requirements), TİD validator, Phase 0 table.

## Next step
Phase 3: M1 (BiGRU + attention), ≤3 configs per model on val, freeze, `docs/targets.md`.
