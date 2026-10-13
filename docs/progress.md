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
Spec v0.3 (2026-10-07) follows the signed EE4297 proposal: commitments O1–O5 (D-11).
Demo vocabulary (D-12, 21 signs: 13 kept + 8 new) is separate from the 20-class benchmark:
`configs/demo.yaml`, `outputs/manifest_demo.csv`, `outputs/runs_demo/`. Demo M0 trained: `outputs/runs_demo/20261007-161859_m0_full_s0`, val macro-F1 0.967; weakest:
selam 0.87, hoscakal 0.89 (hoscakal→selam).

Informal live check (D-13): demo M0, flip on, 21/21 signs correct on one try each (hoscakal 0.40, yok 0.90 confidence).
O4 now uses the demo vocabulary; logged evaluation via `tsl.realtime --eval` + `scripts/live_report.py`.

## Known problems / open
- LIVE-01: decide how the 120 webcam trials are allocated.
- PRE-06: per-sign semantic-safety check before any mirroring.
- Open: TİD validator, course report requirements, Phase 0 table, `docs/targets.md`.

## Next step
Logged dev session (e.g. `--reps 2`) to get the first real live metrics; decide the LIVE-01 allocation of the 120
trials. Gantt task now (12–25 Oct): M1 on the benchmark.
