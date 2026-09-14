# Data preparation and validity contract

This workflow uses the official KITTI-360 calibration, poses, four camera views,
Velodyne scans, 3D bounding annotations, and labeled accumulated static point clouds.
No training targets are inferred from model predictions. The raw annotation sources
are preserved and `source_manifest.json` records source URLs, archive metadata,
member CRC32 and SHA256, and hashes of processed replacements.

Commands (project environment):

```
.venv/Scripts/python.exe scripts/prepare_data.py --mode all --limit 50
.venv/Scripts/python.exe scripts/prepare_data.py --mode all
```

`--limit 50` is an explicitly labeled smoke subset, with additional true pedestrian
annotation timestamps and one contiguous 16-observation training block where poses
allow it. It is not the full training experiment. Full sampling is every fifth
camera frame, restricted to frames with official optimized poses and released static-semantic windows. `eligibility_full.json` records this mechanical annotation-coverage filter before downloads. All four cameras
and same-frame Velodyne must exist and timestamp span must be below 60 ms.

Only one source archive is open for acquisition at a time. HTTP range requests
retrieve selected ZIP members and validate their CRC; we do **not** claim a hash of
the entire multi-GB archive when only selected members were downloaded. Resized
704x256 RGB PNGs are lossless processed replacements; original decompressed image
hashes and processed image hashes are distinct. An interrupted transfer retries
three times and preparation is resumable. At least 50 GiB free disk is preserved.

Calibration applies each perspective camera's rectification rotation and native Mei
fisheye intrinsics, including the small tangential distortion coefficients present
in official YAML (the older SDK projection script omits them). Pixel-center-aware
resize transforms match image resampling. Depth is **radial range** for every camera
so backprojection follows unit camera rays. Depth bins are nearest centers of 64
linearly spaced values from 1 to 80 m; invalid/occluded/out-of-range cells are -1.
Depth supervision uses the closest same-frame Velodyne return in each stride-eight
image cell. Fisheye pixels outside the invertible calibrated domain are masked.

Coordinates are ego x-forward/y-left/z-up. KITTI-360 IMU x-forward/y-right/z-down
is converted explicitly. XML primitive vertices are transformed into world and ego
coordinates, then enclosed in an upright yaw-oriented box using the annotated local
x direction as heading. The center is geometric, not bottom center. Mild pitch/roll
is absorbed into upright box extents; this is an annotation conversion, not a claim
that these rough primitives are accurate detailed human/car meshes.

Object targets use raw XML kittiIds 13=car,19=pedestrian. Static objects are restricted
to annotation windows; dynamic objects are used only at their exact annotated frame.
Current LiDAR points within the box and visible in at least one camera are required.
Unsupported footprints remain unknown. Negative detection supervision is conservative:
only currently observed, labeled ground cells, plus positive center neighborhoods,
are enabled. Missing or occluded objects elsewhere are not background examples.
Metrics must respect the same validity masks and report coverage alongside scores.

BEV road targets use official static semantic IDs 7=road,8=sidewalk, and 9/10/22 as
other known ground (parking/rail/terrain). Unknown-ground ID6 is deliberately ignored.
Points require confidence >=0.5; actual points are subsampled at 10 cm without
interpolating unlabeled gaps. Current LiDAR visibility screens occluded surfaces;
a 0.5 m BEV cell needs two retained points and >=75% label agreement. Road masks
are consequently sparse observed-surface labels, not dense amodal road completion.

The manifest records split, per-frame boxes, depth/road coverage, exclusions, and
synchronization spans. `cache/<sequence>/<frame>.npz` contains only numeric targets
and pose. `images/<sequence>/image_XX/<frame>.png` contains the current camera image.
`Kitti360Dataset` exposes ordinary current-frame examples; chronological history and
its boundary resets belong to the trainer/model. Inference must not receive targets.

The qualitative projection audit exports up to 50 train/validation mosaics. Test
sequence0004 is reserved for final selection-independent evaluation and receives only
mechanical input/label preparation checks; its imagery is not in qualitative audit.

Official references:
- https://www.cvlibs.net/datasets/kitti-360/download.php
- https://www.cvlibs.net/datasets/kitti-360/documentation.php
- https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/helpers/project.py
- https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/helpers/annotation.py

Data audit at smoke completion: 125 train frames (679 car,58 pedestrian boxes),49 validation frames (66 car,1 pedestrian),36 test frames mechanically prepared. Full data acquisition is separate. Drive0003 has only one pedestrian identity over34 original timestamps; pedestrian generalization evidence is correspondingly limited. A static pedestrian primitive id19024 in drive0000 spans6.37m at frame8475; this source annotation is retained and flagged as coarse, not silently shrunk to a person template. Frame5355..5430 step5 in train0000 forms a complete16-frame temporal smoke block.

Training-only ground-plane calibration: `--mode ground --manifest manifest.smoke.json` reproduces `ground_plane.json`. Measured smoke median z=-0.934311m (5th/95th frame-median percentiles -0.978253/-0.888394m); this uses125trainingframes and2,139,599 actual road points in2-10m annulus. This IMU-origin height differs from typical roof-LiDAR height. `ground_fov_mask` consumes only calibration/camera availability and this fixed estimated plane; it returns potential field of view, not occlusion certainty. It never accesses target masks.

## Full-cache provenance and verified resume

Full preparation records `cache_sha256`, `cache_bytes`, and a per-frame
`preprocessing_fingerprint`. The manifest includes source-code SHA256 values for
data/geometry/acquisition and the preparation entrypoint, relevant package
versions, and verified hashes of calibration, poses, annotation XML, timestamps,
images, LiDAR, and applicable semantic windows. Each stored input is checked
against the official acquisition manifest before preparation or cache reuse.

Caches now use a preprocessing-fingerprint namespace under `cache/v2_*`, leaving
original frozen smoke cache files untouched. Each complete NPZ is atomically
published with a metadata sidecar. Resume requires an identical processing/input
fingerprint, matching NPZ size/SHA256, valid array shapes/ranges, and matching
recorded coverage counts. Missing, corrupt or stale cache entries are rebuilt.
Corrupt source inputs stop preparation; a subsequent acquisition run verifies
stored hashes and reacquires failed members. Only complete requested-drive output
is published as `manifest.full.json`; intermediate status remains
`manifest.preparing.json`. This does not claim completion when frame exclusions
are present: those remain explicit in the completed manifest.

Real-data recovery probe:
` .venv/Scripts/python.exe artifacts/data-audit/cache_provenance_probe.py `
uses four training frames5355-5370 in an isolated artifact root with unchanged
source-file hardlinks. It demonstrates unchanged cache reuse, deliberate cache
corruption recovery, and stale-fingerprint recovery. All regenerated arrays
match their originals. It does not alter the production smoke manifest or source
files. Evidence: `artifacts/data-audit/cache-provenance-probe.json`.

The first measured four-frame preparation took2.35s including source hashing and
loading the semantic window; a verified repeat took0.63s. Full preparation is
provisionally20-60minutes for6163 candidates, with exact elapsed/generation/reuse
counts now emitted every20frames. That is an estimate, not completed full-run
timing. Processing uses the same geometry and labels as the previous cache code.

A separate official-calibration probe corrupts only an isolated owned copy, verifies
that preparation rejects the source hash mismatch, then reacquires the single
member and proves byte-for-byte recovery. Evidence:
`artifacts/data-audit/acquisition-repair-probe.json`. Reproduce with
`.venv/Scripts/python.exe artifacts/data-audit/acquisition_repair_probe.py`.

## Preparation overlap without publishing a partial dataset

`.venv/Scripts/python.exe scripts/prepare_data.py --mode warm-cache --drives 0000`
prepares already downloaded drive0000 while other drives are still acquired.
It calls `prepare_cache(..., publish=False)` and writes only verified fingerprinted
cache payloads/sidecars plus `manifest.warmup.0000.json`. Its status has
`publication: warm_cache_only` and `preparation.complete` stays false until its
requested drive finishes. It never changes `manifest.json`, `manifest.full.json`,
`manifest.smoke.json`, or `manifest.preparing.json`. Warm status is not a claim
that the full four-drive dataset has passed preparation.

The later standard full preparation uses identical per-frame fingerprints and
reuses these caches after verifying their hashes. Overall acquisition-manifest
changes are recorded as provenance, but do not invalidate a cache when its actual
relevant inputs/code are unchanged. An actual four-frame transfer probe confirms
this behavior and verifies that a subsequent published preparation reuses all four
caches. See `artifacts/data-audit/warm-publication-transfer-probe.json`.

The initial real warmup measured301 frames in70.8s (4.25frames/s), with no exclusions.
This provisional sustained rate includes target generation, source hashing and
cache writes; full-drive totals and timing are in the warmup manifest/log on
completion. Background log: `artifacts/data-warmup-0000.log`.

## Accepted pre-baseline car-axis convention correction (2026-09-11)

The official KITTI-360 XML localx axis is not consistently the car's longaxis.
A full training-drive0000 audit found69 observations across5 instances with
width>length; sourceimages show ordinary parkedcars. The stockasset and box-loss
interfaces use length along positivebox-x, so preparation now canonicalizes cars
only: swap length/width when width>length and set yaw=wrap(yaw+pi/2). Occupied
geometry, center and height are preserved. Pedestrians and alreadycanonical cars
are unchanged. Positiveyaw remains a box-axis convention; physicalfront versus
rear direction is not certified by these annotations and may differby180degrees.

All12,557 existing drive0000 objecttargets were recomputed from originalXML/poses:
exactly69 car parameterizations changed, all other targets were bit-identical.
Maximum finiteprecision corner-set change was1.19e-6m; minimum BEV/3D IoU exceeded
0.99999934. The frozen16-frame gate's99cars and16pedestrians are bit-identical, so
its learnability evidence remains applicable.22 geometry/model regressiontests
passed, including corner/IoU equivalence and unchangednormalcar/pedestrian cases.

Original annotations and old caches remain intact. The completed originalwarm
manifest is archived as `manifest.warmup.0000.original-axes.json`; regeneration
uses namespace `v2_8561bc025c13d78d`. This is a data-interface correctness fix before
the baseline, consuming **zero** substantive post-baseline revisions. Coarse
pedestrian/vehicle primitives were flagged, not filtered or silently reshaped.

Evidence: `artifacts/data-audit/car-canonicalization-accepted.json`,
`artifacts/data-audit/box-quality-0000.json`, and
`docs/execution/BOX_SUPERVISION_AUDIT.md`.


## Full four-drive data gate

Full acquisition and preparation passed the independent integrity/geometry gate: 4,300 training, 197 validation and 1,622 sealed test scenes, with 44 explicitly missing-fisheye exclusions in drive0002. Every consumed source and every cache was rehashed; all four images were decoded in every retained scene. The 50 non-test projection audit timestamps plus four crowded/pedestrian details received assistant visual review. See `FULL_DATA_GATE.md` and `artifacts/data-audit/full-gate-report.json` for exact evidence and limitations.

The implemented cadence selects every fifth source frame. The actual median training spacing is 0.5227738618850708s (~1.913Hz), not exactly2Hz; actual timestamps and discontinuity resets are preserved. Validation has only six pedestrian observations from one source annotation identity and no camera03 pedestrian support, so that class/view has no validation coverage.

TRAIN-only render-ground calibration is now -0.930941852176808m, estimated from 4,300 training frames and 74,462,835 labeled road points. The prior smoke calibration is preserved byte-for-byte in `data/kitti360/ground_plane.smoke.json`. Neither this render height nor the visual audits use test ground truth for model selection.
