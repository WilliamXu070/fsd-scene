# Execution map

Goal: implement, train, evaluate, optimize and inspect the approved four-drive camera-only scene pipeline. Completion takes priority over a morning deadline. Maximum three substantive revisions after baseline; never present targets as measured achievements.

| Stages | Owner | Prerequisites | Evidence |
|---|---|---|---|
| 1 isolated environment | Parent | none | artifacts/environment/gpu_smoke.json |
| 2 acquisition, 3 geometry | Data worker; parent integrates | environment | manifests, projection audits and coverage |
| 4 evaluator controls | Metrics worker; parent integrates | environment | perfect/perturbed/duplicate/missing controls |
| 5 pipeline, 7 learnability | Parent | data, model, evaluator | gradients, resume/reload and overfit metrics |
| 6 network components | Model worker; parent integrates | geometry contract | module controls and real samples |
| 8 training, 9 revisions | Parent, one GPU job at a time | correctness gates | checkpoints, epoch logs, held-out validation |
| 10 profiling, 11 final test | Parent | selected trained candidate | timing, parity and sealed test report |
| 12 viewer, 13 visual QA | Viewer worker; parent verifies | scene contract | real-prediction replay and browser evidence |

State files under artifacts/stages report measured stage evidence. Missing status means pending. Preserve dated v0.1 architecture. User-approved plan in conversation controls execution; contracts clarify implementation.

## Clarifications

- Native Windows isolated Python3.12/PyTorch CUDA13 verified on sm_120 by actual forward/backward/update; global environment unchanged.
- Radial depth for all cameras preserves native fisheye rays; 64 bin centers1..80m. Earlier camera-axis depth was an open convention.
- Official ZIP range access permits selected members without full archive downloads. Record member CRC, hashes and provenance. Keep50GB disk free.
- Checkpoint selection equally weights road IoU, sidewalk IoU, car BEV AP50 and pedestrian BEV AP50. Missing tasks invalidate selection.
- Smoke subsets are NOT the full four-drive training subset. Do not label short pilot results as completed training.
- Full Stage A completed24epochs on4,300scenes; Stage B is running from bestA after the verified FP16 atan2 backward correction. The failed B attempt is archived, and the actual-engine recovery gate passed60updates without overflow. No final FPS or sealed-test accuracy is established.

## Current verified milestones

The initial-head and late-refinement learnability gates passed on the preserved16-frame training block. Exact evidence is under artifacts/stages/07_learnability_a.json and07_learnability_b.json. This proves tiny-set learning only. Full acquisition and geometry gates now pass; baseline training is running. The complete manifest contains 4,300 training, 197 validation and 1,622 sealed-test scenes.

The real training-fit replay passed operational browser QA and has a saved recording. Visual fidelity remains unmet: extra/overlapping objects and unsupported ground patches are visible. See artifacts/viewer-qa/training-fit-report.json. The pilot module timing is not a final pipeline benchmark.


Full-data evidence: docs/execution/FULL_DATA_GATE.md. Integrated implementation controls:144passed (artifacts/integration/prebaseline-tests.json). Nominal2Hz sampling is realized as every fifth source frame, median0.522774s (~1.913Hz), with actual timestamps throughout. Validation has six pedestrian observations from one identity and none in camera03; do not claim broad surround pedestrian validation.
