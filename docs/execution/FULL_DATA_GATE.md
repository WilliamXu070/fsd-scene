# Full four-drive acquisition and geometry gate

Status: **pass_with_documented_dataset_limitations**. This checks data integrity and supervision validity, not model accuracy.

Prepared 6119 of 6163 eligible timestamps; 44 explicitly excluded. Test-drive content receives mechanical checks only.

| Split | Frames | Car observations | Pedestrian observations | Unique car IDs | Unique pedestrian IDs |
|---|---:|---:|---:|---:|---:|
| train | 4300 | 17080 | 742 | 1216 | 79 |
| val | 197 | 244 | 6 | 45 | 1 |
| test | 1622 | 3762 | 195 | 335 | 18 |

## Class/view supervision evidence

Projected-center counts and current-LiDAR-supported surface counts are separate in the JSON. Every retained train/validation pedestrian observation receives the latter check; cars additionally have broad/crowded frame sampling.

- train car: projected centers by camera [8390, 8382, 10466, 10658]; current-LiDAR support [1756, 1748, 1912, 1909]; checked 3212/17080 observations.
- train pedestrian: projected centers by camera [403, 404, 350, 434]; current-LiDAR support [405, 402, 276, 358]; checked 742/742 observations.
- val car: projected centers by camera [120, 120, 144, 118]; current-LiDAR support [9, 9, 5, 8]; checked 14/244 observations.
- val pedestrian: projected centers by camera [3, 3, 6, 0]; current-LiDAR support [3, 3, 5, 0]; checked 6/6 observations.

Class/view gaps are explicitly unresolved coverage limitations:
- val pedestrian camera 3: No current-LiDAR-supported class observation in this view; do not claim that class/view is validated. Scope: all_retained_class_observations.

## Evidence checks

- PASS: complete_four_drive_publication
- PASS: selected_timestamps_accounted
- PASS: stable_preprocessing_source
- PASS: acquisition_manifest_identity
- PASS: ground_calibration_training_only
- PASS: native_camera_ray_projection_roundtrips
- PASS: every_consumed_source_sha256
- PASS: no_exact_four_camera_scene_duplicates_across_splits
- PASS: every_processed_cache_sha256_and_payload
- PASS: all_four_images_decode_at_expected_shape
- PASS: strict_timestamp_order_and_sync
- PASS: car_longaxis_convention
- PASS: target_dimensions_representable_by_current_inference
- PASS: training_and_validation_classes_present
- PASS: depth_supervision_available_in_each_camera
- PASS: audited_boxes_have_current_lidar_camera_support
- PASS: current_lidar_training_support_for_each_class_and_camera

Every consumed stored source was SHA256-checked against acquisition provenance. Every cache was checked against its fingerprint, payload hash and numeric schema. All four images were decoded for every retained sample. Source ZIP verification is per downloaded member; the entire multi-GB archives were not downloaded or hashed.

Exact current-LiDAR/camera support was rechecked on 635 training/validation timestamps. Native pinhole/Mei ray projection round trips were checked numerically. Full visual audit mosaics were generated separately; assistant visual review is recorded below when completed.

## Exclusions and calibration

Exclusions by reason: `{"missing_image_or_lidar": 44}`.
Exclusions by drive: `{"0000": 0, "0002": 44, "0003": 0, "0004": 0}`.

TRAIN-only fixed ground height: -0.930941852m from 4300 frames and 74462835 labeled road points. The per-frame median 5th/95th percentiles are -0.977391/-0.879457m. The prior smoke calibration remains in `data/kitti360/ground_plane.smoke.json`.

## Provenance

- Dataset manifest SHA256: `804f711286751ef5ed3bea1f04a9969c8f58d34cfc8ab05490e6b57daa9d9a0c`
- Acquisition manifest SHA256: `6380aeaa503eb447c0f89005c628f9de9bb5409b9597679215e4ab10374ac1f7`
- Preprocessing fingerprint: `8561bc025c13d78d09b5765c45111e9fa615c9954e43ecc0b4cbfa06ca562ea7`
- Checked source files: 30767
- Checked caches: 6119
- Mechanical verification elapsed: 231.2s
- Free disk: 353.0GiB

Reproduce: `.venv/Scripts/python.exe artifacts/data-audit/verify_full_dataset.py`.
Full machine-readable evidence: `artifacts/data-audit/full-gate-report.json`.

## Assistant visual review

Completed on 50 broadly sampled training/validation timestamps, viewed in four-camera box and depth overview mosaics and BEV ground maps. Four additional crowded/pedestrian timestamps received full-size review: drive0000 frames2085 and8475, drive0002 frame6855, and validation drive0003 frame650. Three full-size radial-depth mosaics and two enlarged pedestrian crops were also inspected. No test imagery was displayed.

The projections broadly align with visible objects in perspective and native fisheye images. No new coordinate reversal, camera interchange, or systematic timing/calibration error was found. Source boxes remain coarse in documented cases, particularly pedestrian19024; partial occlusion and small image size prevent fine shape verification. Sparse ground maps retain holes and unknown regions, and the depth views cover only supported scan surfaces. This passes the qualitative geometry sanity gate, not a model-accuracy or dense-ground reconstruction gate.

The parent assistant independently spot-checked validation0003/frame545, the first ground overview, and crowded0000/frame2085. This is assistant review, not human review.

The exact viewed image paths, hashes, timestamp selection and observations are recorded in `artifacts/data-audit/full-visual-review/assistant-review.json`.

## Sampling cadence

The frozen implementation selects every fifth source frame and retains its actual timestamp. The measured median training interval is 0.5227738618850708s (approximately 1.913 Hz), rather than exactly 2 Hz. Gaps caused by source coverage and exclusions are longer and reset temporal history. This is a documented difference from nominal2Hz sampling; no targets or source data were changed for this clarification.

## Limitations

- KITTI-360 XML instancing primitives can be coarse, especially some pedestrians; no silent filtering.
- Validation drive0003 contains only six pedestrian observations from one annotation identity. Camera03 has zero pedestrian supervision support. A distinct drive does not prove unseen geography.
- Car long-axis geometry is canonical; physical front/back heading is unverified.
- Sparse visible ground supervision covers 6.574% of the training BEV grid on average. Unknown regions are ignored; dense or amodal road truth is unavailable.
- Current-LiDAR support uses sparse LiDAR visibility screening, not independently labeled dense camera visibility; unmatched occluders can remain.
- Projected-center counts describe geometric image coverage. Current-LiDAR audit is exhaustive for retained non-test pedestrians; car coverage is sampled as reported.
- Sampling uses every fifth source frame: median retained training spacing 0.5227738618850708s, approximately 1.912872 Hz, rather than exact 2 Hz. Actual timestamps and discontinuity resets are preserved.
- Assistant visual review completed on 50 broad plus four detailed non-test timestamps. This does not establish model accuracy or human review.
- Test images and numeric labels received mechanical checks only; no test imagery was displayed or used for model selection.
