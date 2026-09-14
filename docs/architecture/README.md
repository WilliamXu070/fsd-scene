# Architecture records

Preserved preliminary baseline: [v0.1, 2026-09-11](2026-09-11-preliminary-v0.1.md).

KITTI implementation record: [v0.2, 2026-09-11](2026-09-11-implementation-v0.2.md).

Six-camera nuScenes implementation: [v0.4, 2026-09-12](../../experiments/nuscenes/ARCHITECTURE_v0.4.md).

Project history and proposed experiment rationale: [evolution log, 2026-09-13](2026-09-13-evolution-and-experiment-rationale.md). Separates the trained KITTI refinement experiments from the nuScenes run that stopped after Stage A; new recommendations are not implemented changes.

Continuation readiness: [deep review, 2026-09-13](2026-09-13-continuation-readiness-review.md). Includes the reproduced warm-start blocker, checkpoint audit, temporal-control limitations and revised experiment gates. Read before restarting the nuScenes controller.

Checkpoint branching blocker resolved: [implementation and real-data verification](../../experiments/nuscenes/CHECKPOINT_BRANCHING.md). Other readiness findings remain separate tasks.

Authorized Stage A continuation: [bounded experiment campaign](../../experiments/nuscenes/STAGE_A_CAMPAIGN.md). Opt-in quality scoring, yaw supervision, C2 feature injection and appearance augmentation now have real-data initialization/gradient gates. Campaign02 contains the controlled training/evaluation runs; their presence does not mean an architecture has been promoted.

Measured Stage A outcomes: [campaign results, 2026-09-13](2026-09-13-stage-a-campaign-results.md). Epoch 18 remains retained. The lower global restart LR avoids the high-rate regression, but the selected branch failed the full-validation mean object AP improvement guard. No new architecture or Stage B model was adopted.

Read both documents before architecture changes. It records accepted decisions, proposals, unresolved parameters, supervision requirements and unvalidated timing estimates. Later explicit user decisions take precedence.

Preserve dated versions. For each accepted material change, create a new version, link the supporting experiment, summarize the rationale and update this pointer. Do not silently rewrite the original baseline.

Experiment record template: [TEMPLATE.md](../experiments/TEMPLATE.md).
