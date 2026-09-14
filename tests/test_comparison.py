"""Falsification controls for the report-only optimization acceptance gate."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location("compare_runs", Path(__file__).parents[1] / "scripts/compare_runs.py")
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def evaluation():
    return {
        "split": "val", "diagnostic": None, "checkpoint": "fixtures/model.pt",
        "checkpoint_sha256": "a" * 64, "dataset_manifest_sha256": "b" * 64,
        "configuration_sha256": "c" * 64, "source_sha256": "d" * 64,
        "runtime_backend": {"precision": "fp32", "compiled": False, "stage": "b", "refinement_enabled": True},
        "metrics": {
            "metric_protocol": "custom_KITTI360_subset_all_point_AP_not_official_benchmark",
            "geometry_matching": "class-consistent BEV IoU >= 0.5; conditional on matched detections",
            "temporal_matching": "GT identity; class-consistent BEV IoU >= 0.5; world-aligned error residuals",
            "frames": 6,
            "score_protocol": {"metric_confidence_cutoff": .01, "nms_bev_iou": .5, "max_detections": 100},
            "road": {"iou": {"road": .9, "sidewalk": .8, "other_ground": .7}, "valid_pixels": 1000},
            "objects": {"bev_ap50": {"classes": {
                "car": {"ap": .9, "gt_count": 10, "recall": .9},
                "pedestrian": {"ap": .8, "gt_count": 5, "recall": .8}}}},
            "geometry": {name: {"matched_count": count, "center_error_m": .2, "ground_center_error_m": .15,
                                 "dimension_mae_m": .1, "yaw_error_deg": 3.} for name, count in (("car", 9), ("pedestrian", 4))},
            "temporal": {"matched_transition_count": 8, "center_residual_change_m": .05,
                         "dimension_residual_change_m": .02, "yaw_residual_change_deg": .3,
                         "center_residual_rate_mps": .1, "yaw_residual_rate_degps": .6},
        },
    }


def workload(count=4):
    records=[dict(index=i,sequence="synthetic-validation",frame_id=i*5,timestamp=i*.5,
                  sensor_sha256="1"*64,ego_pose_sha256="2"*64) for i in range(count)]
    return dict(schema_version=1,cached_scenes=count,records=records,ordered_sha256=comparison._json_digest(records))


def replay_order(indices):
    from collections import Counter
    return dict(count=len(indices),indices=indices,order_sha256=comparison._json_digest(indices),
                frequencies={str(k):v for k,v in sorted(Counter(indices).items())})


def benchmark(report, latency=50.):
    return {
        **{key: copy.deepcopy(report[key]) for key in ("checkpoint", *comparison.HASH_KEYS, "runtime_backend")},
        "device": "Synthetic RTX 5070 test fixture", "split": "val", "stage": "b", "refinement_enabled": True,
        "batch_scenes": 1, "cameras": 4, "image_size": [256, 704], "scope": "combined replay fixture, not a measurement",
        "score_protocol": {"metric_confidence_cutoff": .1, "nms_bev_iou": .5, "max_detections": 100},
        "sensor_workload": workload(),
        "steady_state": {"required_history_frames": 3, "repeats": [
            {"repeat": i, "warmup_full_history_updates": 50, "timed_full_history_updates": 500,
             "warmup_workload": replay_order([j%4 for j in range(50)]),
             "timed_workload": replay_order([j%4 for j in range(500)]),
             "timings": {"combined_wall_ms": {"samples": 500, "mean_ms": latency + i,
                                               "median_ms": latency + i - 1, "p95_ms": latency + i + 4}}}
            for i in range(3)]},
    }


def nuscenes_pair():
    base,candidate=evaluation(),evaluation()
    for report in (base,candidate):
        report['dataset']='nuscenes'
        report['runtime_backend']['deterministic_inference']=True
        report['metrics']['metric_protocol']='custom_nuScenes_two_class_all_point_IoU_AP_not_official_NDS'
    candidate['runtime_backend']['precision']='fp16'
    first,second=benchmark(base,60),benchmark(candidate,45)
    for timing in (first,second):
        timing['cameras']=6
        timing['runtime_choices']={'deterministic_algorithms':True}
        timing['benchmark_script_sha256']='9'*64
    return base,candidate,first,second


def test_nuscenes_six_camera_speed_and_accuracy_can_be_compared():
    report=comparison.compare_reports(*nuscenes_pair())
    assert report['optimization_eligible'] is True


@pytest.mark.parametrize('mismatch',['camera_count','determinism','benchmark_script','dataset'])
def test_nuscenes_incompatible_runtime_cannot_establish_optimization(mismatch):
    base,candidate,first,second=nuscenes_pair()
    if mismatch=='camera_count':first['cameras']=second['cameras']=4
    if mismatch=='determinism':second['runtime_choices']['deterministic_algorithms']=False
    if mismatch=='benchmark_script':second['benchmark_script_sha256']='8'*64
    if mismatch=='dataset':candidate['dataset']='kitti360'
    report=comparison.compare_reports(base,candidate,first,second)
    assert report['timing']['status']=='not_comparable'
    assert report['optimization_eligible'] is False


def set_path(report, path, value):
    keys = path.split(".")
    for key in keys[:-1]:
        report = report[key]
    report[keys[-1]] = value


def metric_row(report, name):
    return next(row for row in report["accuracy"]["metric_comparisons"] if row["name"] == name)


def test_identical_accuracy_is_pass_but_cannot_establish_optimization_without_timing():
    report = comparison.compare_reports(evaluation(), evaluation())
    assert report["accuracy"]["status"] == "pass"
    assert len(report["accuracy"]["metric_comparisons"]) == 17
    assert report["timing"]["status"] == "not_supplied"
    assert report["optimization_eligible"] is False
    assert report["manual_tradeoff"]["automatically_selected"] is False
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("path", ["road.iou.road", "road.iou.sidewalk",
                                     "objects.bev_ap50.classes.car.ap", "objects.bev_ap50.classes.pedestrian.ap"])
def test_one_percentage_point_boundary_is_inclusive_but_larger_drop_fails(path):
    base, candidate = evaluation(), evaluation()
    old = comparison._get(base["metrics"], path)
    set_path(candidate["metrics"], path, old - .01)
    assert comparison.compare_reports(base, candidate)["accuracy"]["status"] == "pass"
    set_path(candidate["metrics"], path, old - .01001)
    result = comparison.compare_reports(base, candidate)
    assert result["accuracy"]["status"] == "fail"
    row = metric_row(result, path)
    assert row["passed"] is False and row["delta"] < -.01
    assert row["threshold"] == old - .01


@pytest.mark.parametrize("path", [f"geometry.{cls}.{field}" for cls in comparison.CLASSES for field in comparison.GEOMETRY]
                         + [f"temporal.{field}" for field in comparison.TEMPORAL])
def test_every_error_metric_has_five_percent_gate(path):
    base, candidate = evaluation(), evaluation()
    old = comparison._get(base["metrics"], path)
    set_path(candidate["metrics"], path, old * 1.05)
    assert comparison.compare_reports(base, candidate)["accuracy"]["status"] == "pass"
    set_path(candidate["metrics"], path, old * 1.051)
    assert comparison.compare_reports(base, candidate)["accuracy"]["status"] == "fail"


def test_exact_zero_error_baseline_uses_disclosed_absolute_tolerance():
    base, candidate = evaluation(), evaluation()
    base["metrics"]["geometry"]["car"]["center_error_m"] = 0.
    candidate["metrics"]["geometry"]["car"]["center_error_m"] = 1e-9
    assert comparison.compare_reports(base, candidate)["accuracy"]["status"] == "pass"
    candidate["metrics"]["geometry"]["car"]["center_error_m"] = 1.00001e-9
    assert comparison.compare_reports(base, candidate)["accuracy"]["status"] == "fail"


@pytest.mark.parametrize("path,value", [
    ("split", "test"), ("split", "train"), ("diagnostic", "empty_history"),
    ("dataset_manifest_sha256", "e" * 64), ("metrics.frames", 7),
    ("metrics.metric_protocol", "different matching method"), ("metrics.geometry_matching", "other"),
    ("metrics.temporal_matching", "uncompensated motion"),
    ("metrics.score_protocol.metric_confidence_cutoff", .1), ("metrics.score_protocol.nms_bev_iou", .6),
    ("metrics.score_protocol.max_detections", 50), ("metrics.road.valid_pixels", 1001),
    ("metrics.objects.bev_ap50.classes.car.gt_count", 11),
    ("metrics.objects.bev_ap50.classes.pedestrian.gt_count", 6),
])
def test_incomparable_evaluations_cannot_pass(path, value):
    base, candidate = evaluation(), evaluation()
    set_path(candidate, path, value)
    result = comparison.compare_reports(base, candidate)
    assert result["accuracy"]["status"] == "not_comparable"
    assert result["optimization_eligible"] is False


@pytest.mark.parametrize("path,value", [
    ("checkpoint", ""), ("checkpoint_sha256", "not-a-hash"), ("dataset_manifest_sha256", None),
    ("configuration_sha256", None), ("source_sha256", None), ("metrics.frames", 0),
    ("metrics.road.iou.road", None), ("metrics.road.iou.road", float("nan")),
    ("metrics.road.iou.sidewalk", float("inf")), ("metrics.road.iou.sidewalk", 95),
    ("metrics.geometry.pedestrian.dimension_mae_m", -1), ("metrics.geometry.car.center_error_m", None),
    ("metrics.temporal.yaw_residual_change_deg", None), ("metrics.temporal.center_residual_rate_mps", float("nan")),
    ("metrics.objects.bev_ap50.classes.pedestrian.gt_count", 0), ("metrics.road.valid_pixels", 0),
])
def test_missing_invalid_or_nonfinite_values_never_pass_and_serialize_strictly(path, value):
    base, candidate = evaluation(), evaluation()
    set_path(candidate, path, value)
    result = comparison.compare_reports(base, candidate)
    assert result["accuracy"]["status"] == "not_verifiable"
    assert not result["accuracy"]["all_required_evidence_present"]
    json.dumps(result, allow_nan=False)


def test_entire_absent_geometry_or_temporal_is_not_silently_skipped():
    for section in ("geometry", "temporal"):
        candidate = evaluation()
        candidate["metrics"].pop(section)
        result = comparison.compare_reports(evaluation(), candidate)
        assert result["accuracy"]["status"] == "not_verifiable"


@pytest.mark.parametrize("path", ["geometry.car.matched_count", "geometry.pedestrian.matched_count", "temporal.matched_transition_count"])
def test_zero_match_coverage_cannot_establish_numeric_error_even_if_report_says_zero(path):
    candidate = evaluation()
    set_path(candidate["metrics"], path, 0)
    assert comparison.compare_reports(evaluation(), candidate)["accuracy"]["status"] == "not_verifiable"


def test_changed_nonzero_coverage_is_prominent_without_invented_drop_threshold():
    candidate = evaluation()
    candidate["metrics"]["geometry"]["car"]["matched_count"] = 8
    candidate["metrics"]["temporal"]["matched_transition_count"] = 4
    report = comparison.compare_reports(evaluation(), candidate)
    assert report["accuracy"]["status"] == "pass"
    rows = {row["name"]: row for row in report["accuracy"]["coverage"]}
    assert rows["geometry.car.matched_count"]["delta"] == -1
    assert rows["temporal.matched_transition_count"]["relative_change"] == -.5


def test_source_config_or_checkpoint_changes_are_reported_not_forbidden():
    candidate = evaluation()
    for key in ("checkpoint_sha256", "configuration_sha256", "source_sha256"):
        candidate[key] = "e" * 64
    report = comparison.compare_reports(evaluation(), candidate)
    assert report["accuracy"]["status"] == "pass"
    assert report["identity_differences"]["source_sha256"] is True
    assert report["identity_differences"]["dataset_manifest_sha256"] is False


def test_bound_complete_speed_gain_is_eligible_without_requiring_thirty_fps():
    base, candidate = evaluation(), evaluation()
    candidate["runtime_backend"]["precision"] = "fp16"
    report = comparison.compare_reports(base, candidate, benchmark(base, 60), benchmark(candidate, 45))
    assert report["timing"]["status"] == "pass"
    assert report["timing"]["repeatable_speed_gain"] is True
    assert report["optimization_eligible"] is True
    assert report["timing"]["aggregate"]["candidate_meets_30fps_target"] is False


def test_faster_timing_does_not_excuse_failed_accuracy():
    base, candidate = evaluation(), evaluation()
    candidate["metrics"]["road"]["iou"]["road"] = .85
    report = comparison.compare_reports(base, candidate, benchmark(base, 60), benchmark(candidate, 30))
    assert report["timing"]["repeatable_speed_gain"] is True
    assert report["accuracy"]["status"] == "fail"
    assert report["optimization_eligible"] is False
    assert report["manual_tradeoff"]["available"] is True


@pytest.mark.parametrize("path,value", [
    ("checkpoint_sha256", "e" * 64), ("dataset_manifest_sha256", "e" * 64), ("source_sha256", "e" * 64),
    ("runtime_backend.precision", "fp16"), ("runtime_backend.compiled", True),
    ("device", "Different GPU"), ("split", "train"), ("scope", "GPU-only model timing"),
    ("cameras", 1), ("batch_scenes", 4), ("image_size", [128, 352]),
    ("steady_state.required_history_frames", 0), ("score_protocol.metric_confidence_cutoff", .5),
])
def test_timing_cannot_be_bound_to_different_weights_backend_or_workload(path, value):
    base, candidate = evaluation(), evaluation()
    base_timing, candidate_timing = benchmark(base, 60), benchmark(candidate, 40)
    set_path(candidate_timing, path, value)
    report = comparison.compare_reports(base, candidate, base_timing, candidate_timing)
    assert report["timing"]["status"] == "not_comparable"
    assert report["optimization_eligible"] is False


@pytest.mark.parametrize("field,value", [("warmup_full_history_updates", 49), ("timed_full_history_updates", 499)])
def test_short_pilot_is_not_complete_timing_evidence(field, value):
    base, candidate = evaluation(), evaluation()
    a, b = benchmark(base, 60), benchmark(candidate, 40)
    b["steady_state"]["repeats"][1][field] = value
    report = comparison.compare_reports(base, candidate, a, b)
    assert report["timing"]["status"] == "not_verifiable"
    assert report["optimization_eligible"] is False


def test_missing_benchmark_backend_and_nan_timings_are_not_verifiable():
    base, candidate = evaluation(), evaluation()
    a, b = benchmark(base, 60), benchmark(candidate, 40)
    b.pop("runtime_backend")
    b["steady_state"]["repeats"][0]["timings"]["combined_wall_ms"]["mean_ms"] = float("nan")
    report = comparison.compare_reports(base, candidate, a, b)
    assert report["timing"]["status"] == "not_verifiable"
    json.dumps(report, allow_nan=False)


def test_single_timing_report_and_two_repeat_pilot_are_not_verifiable():
    base, candidate = evaluation(), evaluation()
    assert comparison.compare_reports(base, candidate, benchmark(base))["timing"]["status"] == "not_verifiable"
    a, b = benchmark(base, 60), benchmark(candidate, 40)
    a["steady_state"]["repeats"].pop()
    b["steady_state"]["repeats"].pop()
    assert comparison.compare_reports(base, candidate, a, b)["timing"]["status"] == "not_verifiable"


def test_improvement_in_median_repeat_is_not_claimed_as_consistent_if_one_repeat_slower():
    base, candidate = evaluation(), evaluation()
    a, b = benchmark(base, 60), benchmark(candidate, 40)
    b["steady_state"]["repeats"][2]["timings"]["combined_wall_ms"]["mean_ms"] = 70
    report = comparison.compare_reports(base, candidate, a, b)
    assert report["timing"]["status"] == "pass"
    assert report["timing"]["aggregate"]["speedup_ratio"] > 1
    assert report["timing"]["repeatable_speed_gain"] is False
    assert report["optimization_eligible"] is False


def test_cli_preserves_report_hashes_rejects_overwrite_and_never_reads_checkpoint(tmp_path):
    baseline, candidate, output = [tmp_path / name for name in ("baseline.json", "candidate.json", "comparison.json")]
    for path in (baseline, candidate):
        path.write_text(json.dumps(evaluation()), encoding="utf8")
    assert comparison.main(["--baseline", str(baseline), "--candidate", str(candidate), "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert len(result["input_reports"]["baseline"]["sha256"]) == 64
    assert result["identities"]["baseline"]["checkpoint"] == "fixtures/model.pt"
    with pytest.raises(SystemExit):
        comparison.main(["--baseline", str(baseline), "--candidate", str(candidate), "--output", str(output)])


@pytest.mark.parametrize("phase", ["warmup_workload", "timed_workload"])
def test_same_frame_frequencies_in_different_order_are_not_comparable(phase):
    base,candidate=evaluation(),evaluation()
    a,b=benchmark(base,60),benchmark(candidate,40)
    indices=b["steady_state"]["repeats"][0][phase]["indices"][:]
    indices[0],indices[1]=indices[1],indices[0]
    b["steady_state"]["repeats"][0][phase]=replay_order(indices)
    result=comparison.compare_reports(base,candidate,a,b)
    assert result["timing"]["status"]=="not_comparable"
    assert result["optimization_eligible"] is False


def test_different_cached_scene_workloads_cannot_pass_even_with_same_accepted_indices():
    base,candidate=evaluation(),evaluation()
    a,b=benchmark(base,60),benchmark(candidate,40)
    b["sensor_workload"]=workload(8)
    result=comparison.compare_reports(base,candidate,a,b)
    assert result["timing"]["status"]=="not_comparable"
    assert result["optimization_eligible"] is False


def test_changed_sensor_bytes_with_same_frame_ids_cannot_pass():
    base,candidate=evaluation(),evaluation()
    a,b=benchmark(base,60),benchmark(candidate,40)
    b["sensor_workload"]["records"][0]["sensor_sha256"]="3"*64
    b["sensor_workload"]["ordered_sha256"]=comparison._json_digest(b["sensor_workload"]["records"])
    assert comparison.compare_reports(base,candidate,a,b)["timing"]["status"]=="not_comparable"


@pytest.mark.parametrize("mutation", ["missing_workload","missing_accepted_order","wrong_frequency","out_of_range_index","wrong_order_hash"])
def test_missing_or_inconsistent_workload_evidence_is_not_verifiable(mutation):
    base,candidate=evaluation(),evaluation()
    a,b=benchmark(base,60),benchmark(candidate,40)
    order=b["steady_state"]["repeats"][0]["timed_workload"]
    if mutation=="missing_workload":
        # Both missing prevents a known mismatch being classified instead.
        a.pop("sensor_workload");b.pop("sensor_workload")
    elif mutation=="missing_accepted_order":
        for report in (a,b):report["steady_state"]["repeats"][0].pop("timed_workload")
    elif mutation=="wrong_frequency":
        for report in (a,b):report["steady_state"]["repeats"][0]["timed_workload"]["frequencies"]["0"]+=1
    elif mutation=="out_of_range_index":
        for report in (a,b):report["steady_state"]["repeats"][0]["timed_workload"]=replay_order([99]*500)
    elif mutation=="wrong_order_hash":
        for report in (a,b):report["steady_state"]["repeats"][0]["timed_workload"]["order_sha256"]="0"*64
    result=comparison.compare_reports(base,candidate,a,b)
    assert result["timing"]["status"]=="not_verifiable"
    assert result["optimization_eligible"] is False
    json.dumps(result,allow_nan=False)


def test_complete_workload_timing_still_cannot_excuse_missing_pedestrian_geometry():
    base,candidate=evaluation(),evaluation()
    for report in (base,candidate):
        report["metrics"]["geometry"]["pedestrian"]["matched_count"]=0
        for key in comparison.GEOMETRY:report["metrics"]["geometry"]["pedestrian"][key]=None
    result=comparison.compare_reports(base,candidate,benchmark(base,60),benchmark(candidate,40))
    assert result["timing"]["status"]=="pass"
    assert result["accuracy"]["status"]=="not_verifiable"
    assert result["optimization_eligible"] is False
