# Current execution checkpoint â€” training and final test complete

**The learning budget is finished. R2 is selected, the frozen test ran once, and operational/assistant replay QA is complete with adverse fidelity findings. No further training or test execution is authorized.** The final report and hashed artifact inventory are the handoff; no further execution remains under this plan.

## Authoritative state

- Baseline A: 24 epochs; B: nine. R1 and R2: nine epochs each. R3 control and balanced: eight fixed epochs each; control 8,597 updates/three AMP skips, balanced 8,598/two skips.
- R3 is rejected: pedestrian AP and initial center coverage remain zero on six validation observations. R2 wins the fixed composite by only 0.000829 over A. [Diagnosis](../../artifacts/final-diagnosis/TRAINING_VALIDATION_DIAGNOSIS.md), [R3 assessment](../../artifacts/revision03-assessment/assessment.json).
- Selected checkpoint: `artifacts/runs/revision02-overlap-match/best.pt`; SHA256 `3dfa52ea6a14d3f080a7d8507489c550419425969e8006773a91502706f9aaf0`. [Frozen receipt](../../artifacts/final/selection.json).
- Canonical source aggregate: `02d7eb2366a6c833caca6687622c9614b8ad97b43885b17c0b8e172d49b010f5`. Manifest: `804f711286751ef5ed3bea1f04a9969c8f58d34cfc8ab05490e6b57daa9d9a0c`. Preserve source/configuration/data and recorded identities.
- Selected runtime: original lifting/NMS, eager FP16, refinement enabled, no compilation. Measured cached-image processing 28.79â€“28.94/s; median 34.19â€“34.28 ms, p95 37.35â€“37.69 ms. The 30 FPS selected-pipeline target is unmet. [Latency](../../artifacts/final-measurements/reference/latency.json).
- Faster combined caches measured 44.22â€“44.64/s but remain experimental: four validation pedestrian geometry guards are unavailable. [Runtime decision](../../artifacts/final/runtime-decision.json).
- Test ledger is `complete`, attempt 1, 1,622 scenes. Road IoU 87.91%; sidewalk 61.39%; car BEV AP50 36.05%; pedestrian AP50 0.355%, recall 3/195. Use the [archived test report](../../artifacts/final/test.json) and [ledger](../../artifacts/final/test_ledger.json), never a new test invocation.

## Final replay and QA

Server URL: [http://127.0.0.1:8772](http://127.0.0.1:8772). From the repository root, when that server is not already running:

```powershell
.venv/Scripts/python.exe scripts/serve_demo.py --predictions artifacts/final/test-scenes.jsonl --metadata artifacts/final/replay-metadata.json --data-root data/kitti360 --port 8772
```

The viewer consumes exported predictions and supplied poses. Display default 0.55 was selected on validation; metric cutoff remains 0.01. Recorded model-call timing and CPU software-renderer timing are separate from fresh pipeline latency.

[Automated QA](../../artifacts/final/replay-qa/qa-report.json) passed. Its original pending-visual marker is completed by the subsequent [assistant visual review](../../artifacts/final/replay-qa/visual-review.json), status complete. Root inspected 11 screenshots, video samples at 3/8/13 s and a separate default-threshold first frame. The [recording](../../artifacts/final/replay-qa/selected-replay.webm) covers the first 24 scenes. Stage 12/13 completion means operational verification and inspection, not faithful reconstruction: clustered/overlapping cars, coarse road boundaries and changing arrangements persist. No claim of inspecting every scene or of reliable pedestrian detection.

## Final handoff

[Final report](../../artifacts/final/RESULTS.md) and [hashed inventory](../../artifacts/final/handoff-manifest.json) bind the completed test ledger, selected checkpoint/configuration, runtime decision, reports, plots, source snapshot and assistant visual review. Unavailable guards and unmet performance/fidelity targets are retained. The reproducible inventory builder is `artifacts/final-handoff-planning/build_handoff_manifest.py`; its filled plan is `artifacts/final/handoff-plan.json`.

Do not restart completed training, refreeze selection, change display policy using test, regenerate test predictions, or run another test. No user action is required. Further data/model study would be a separately authorized project; see [DATASET_NEXT_STEP_RESEARCH.md](../../artifacts/final-diagnosis/DATASET_NEXT_STEP_RESEARCH.md).

## Reproduction and history

Current commands and environment instructions are in [README](../../README.md). Reproduce on validation with fresh output paths. Training commands are historical reference only: `--initialize` resets state for a recorded experiment; same-stage `--resume` restores complete state. The existing baseline launcher also supports Aâ†’B cross-stage `--resume` as weights transfer with fresh B state; do not replace it with `--initialize` against baseline configuration lacking experiment metadata.

The prior working continuation is preserved in [CONTINUATION_HISTORY_THROUGH_REVISION3.md](CONTINUATION_HISTORY_THROUGH_REVISION3.md). Its old process ownership and pending status are historical. Architecture v0.1 remains unchanged; [v0.2](../architecture/2026-09-11-implementation-v0.2.md) records implementation, final results and limitations.
