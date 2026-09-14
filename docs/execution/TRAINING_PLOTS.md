# Training log plots

The plotting command is CPU-only and never loads a checkpoint, dataset, or test output. It snapshots `train.jsonl` and `epochs.jsonl`, then writes six figures and a provenance/coverage report. Existing `training_curves.png` output and `--run` invocation remain supported.

```powershell
.venv/Scripts/python.exe scripts/plot_training.py --run artifacts/runs/baseline-a --evaluation-scope validation
.venv/Scripts/python.exe scripts/plot_training.py --run artifacts/runs/overfit16-b --output-dir artifacts/training-plots/overfit16-b
```

Default output is the run folder; `--output-dir` allows an independent snapshot. The filenames are:

- `training_curves.png`: compatible compact overview of active task losses and recorded selection score.
- `task_losses.png`: individual depth, road, initial-object and refinement loss terms, grouped by task.
- `validation_curves.png`: separate car/pedestrian custom BEV AP50 and recall, road/sidewalk/other-ground IoU, and selection score. Latest ground-truth counts are displayed.
- `validation_losses.png`: every recorded epoch-evaluation loss term, retaining the actual evaluation-scope label.
- `runtime_curves.png`: recorded training-plus-evaluation epoch seconds, peak allocated/reserved VRAM, observed AMP skips by epoch, and scale at logged updates.
- `gradient_curves.png`: individual component L2 gradient norms and total norm before clipping, with the configured clipping threshold. Component panels use their own explicitly labeled ranges.
- `plot_summary.json`: snapshot sizes/SHA-256, event counts, trailing bytes ignored, nonfinite field paths, logged update boundary, and last logged step/total steps per epoch.

## Evidence semantics

Training points are **within-epoch logged means**, not reconstructed exact epoch means. Each running mean covers only processed microbatches before that logging point; after resume it may cover only the resumed part. Finite forward losses associated with an AMP-skipped optimizer attempt can contribute to the running mean. The summary reports each epoch's last logged step versus its total and whether an epoch-evaluation record exists. It never upgrades a partial logged mean into a full-epoch average.

AMP overflow records are separate events without a `losses` field. They are counted and plotted as skips, not indexed as training-loss records. Stage A refinement is explicitly disabled: its logged zero losses receive an explanatory panel and are omitted from the active-task total comparison.

The parser reads one byte snapshot. An unterminated final JSONL record is deferred even if it currently happens to parse; its ignored byte count is recorded. A malformed newline-terminated record raises a source/line error rather than silently disappearing. Missing metric values remain absent; nonfinite values are disclosed and drawn as gaps. Missing historical peak memory is not inferred from sampled live allocation values.

`--evaluation-scope auto` recognizes a run's `learnability.json` and labels it **training-fit / no held-out claim**. Otherwise it states that the split was not recorded in the epoch log. Use explicit `--evaluation-scope validation` only for a known validation run. A file named `validation_losses.png` does not imply that training-fit metrics are held out; the figure title supplies the actual scope.

## Verification

Eight parser/plotting controls cover AMP records, incomplete tails, corrupt complete records, uncommitted valid tails, last-log coverage, nonfinite metrics, sparse-log rendering and text encoding. Their evidence is `artifacts/training-plots/parser-controls.json`.

Real `overfit16-b` logs rendered successfully with five training-loss logs, five AMP-skip events and four training-fit evaluation epochs. They lack historical peak-memory and gradient-group fields; corresponding panels explicitly show unavailable evidence.

The live `baseline-a` capture rendered 540 training-loss logs, five validation epochs and one AMP-skip event. Seven gradient groups and all five epoch peak-memory records were available. Snapshot files are preserved under `artifacts/training-plots/baseline-a/`; the compatible run-folder figures were updated too. These counts describe the snapshot, not the continuously growing logs.

Representative real figures were visually inspected: loss/metric legends, task grouping, scope labels, missing-data panels, gradient small multiples, epoch timing/VRAM and AMP plots were readable. Source hashes, observed coverage and inspection notes are preserved in `artifacts/training-plots/verification.json`. No new convergence or held-out-performance conclusion is inferred from plotting alone.
