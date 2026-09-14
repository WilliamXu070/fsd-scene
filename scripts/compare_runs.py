"""Compare immutable validation reports; never load or select model weights.

All AP/IoU inputs are fractions. The only accuracy tolerances are the approved
one percentage point and five percent relative limits. An exactly zero error
baseline uses an explicit 1e-9 absolute tolerance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path


CLASSES = ("car", "pedestrian")
GEOMETRY = ("center_error_m", "ground_center_error_m", "dimension_mae_m", "yaw_error_deg")
TEMPORAL = ("center_residual_change_m", "dimension_residual_change_m", "yaw_residual_change_deg",
            "center_residual_rate_mps", "yaw_residual_rate_degps")
SCORE_KEYS = ("metric_confidence_cutoff", "nms_bev_iou", "max_detections")
BACKEND_KEYS = ("precision", "compiled", "stage", "refinement_enabled")
HASH_KEYS = ("checkpoint_sha256", "dataset_manifest_sha256", "configuration_sha256", "source_sha256")
ZERO_TOLERANCE = 1e-9


def _get(value, path):
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _count(value):
    return _number(value) and value >= 0 and value == int(value)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _json_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def _workload_valid(value):
    if not isinstance(value,dict) or value.get("schema_version")!=1:return False
    rows=value.get("records")
    if not isinstance(rows,list) or not rows or value.get("cached_scenes")!=len(rows):return False
    for i,row in enumerate(rows):
        if not isinstance(row,dict) or row.get("index")!=i or not _text(row.get("sequence")):return False
        if not _count(row.get("frame_id")) or not _number(row.get("timestamp")):return False
        if not all(_digest(row.get(k)) for k in ("sensor_sha256","ego_pose_sha256")):return False
    return _digest(value.get("ordered_sha256")) and value["ordered_sha256"]==_json_digest(rows)


def _order_valid(value,cached_scenes):
    if not isinstance(value,dict) or not _count(cached_scenes) or cached_scenes<1:return False
    indices=value.get("indices")
    if not isinstance(indices,list) or value.get("count")!=len(indices):return False
    if not all(_count(i) and i<cached_scenes for i in indices):return False
    counts={str(k):v for k,v in sorted(Counter(indices).items())}
    return (value.get("frequencies")==counts and _digest(value.get("order_sha256")) and
            value["order_sha256"]==_json_digest(indices))


def _safe(value):
    """Retain invalid input evidence without emitting invalid JSON numbers."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return value


def _row(name, baseline, candidate, status, reason=None, **extra):
    return dict(name=name, baseline=_safe(baseline), candidate=_safe(candidate),
                status=status, passed=status == "pass", reason=reason, **_safe(extra))


def _equal(rows, name, baseline, candidate, validator=lambda x: x is not None):
    valid = validator(baseline) and validator(candidate)
    status = "not_verifiable" if not valid else "pass" if baseline == candidate else "fail"
    rows.append(_row(name, baseline, candidate, status,
                     "Required metadata missing or invalid" if not valid else None))


def _required(rows, name, baseline, candidate, validator):
    valid = validator(baseline) and validator(candidate)
    rows.append(_row(name, baseline, candidate, "pass" if valid else "not_verifiable",
                     None if valid else "Required metadata missing or invalid"))


def _accuracy_check(name, baseline, candidate, kind, coverage_ok=True):
    bounded = kind == "fraction"
    valid = all(_number(x) and x >= 0 and (not bounded or x <= 1) for x in (baseline, candidate))
    if not valid or not coverage_ok:
        return _row(name, baseline, candidate, "not_verifiable",
                    "Missing, nonfinite or invalid metric" if not valid else "No verified matched coverage",
                    delta=None, threshold=None)
    if bounded:
        threshold = baseline - .01
        # One representable step handles decimal boundary arithmetic, not a new tolerance.
        passed = candidate >= math.nextafter(threshold, -math.inf)
        rule = "candidate >= baseline - 0.01 (one percentage point)"
    else:
        threshold = ZERO_TOLERANCE if baseline == 0 else baseline * 1.05
        passed = candidate <= math.nextafter(threshold, math.inf)
        rule = "candidate <= baseline * 1.05; exactly zero baseline allows 1e-9 absolute"
    return _row(name, baseline, candidate, "pass" if passed else "fail", delta=candidate-baseline,
                threshold=threshold, rule=rule)


def _identities(report):
    return {key: report.get(key) for key in ("checkpoint", *HASH_KEYS, "runtime_backend")}


def _coverage(name, baseline, candidate):
    valid = _count(baseline) and _count(candidate)
    return dict(name=name, baseline=_safe(baseline), candidate=_safe(candidate),
                delta=candidate-baseline if valid else None,
                relative_change=(candidate-baseline)/baseline if valid and baseline > 0 else None,
                changed=baseline != candidate,
                interpretation="Conditional error coverage; changes are reported without an additional acceptance threshold.")


def accuracy_metric_rows(a, b):
    """Canonical numerical limits shared by dataset-specific backend workflows."""
    metrics, coverage = [], []
    for cls in ("road", "sidewalk"):
        path = f"road.iou.{cls}"
        metrics.append(_accuracy_check(path, _get(a, path), _get(b, path), "fraction"))
    for cls in CLASSES:
        path = f"objects.bev_ap50.classes.{cls}.ap"
        metrics.append(_accuracy_check(path, _get(a, path), _get(b, path), "fraction"))
        count_path = f"geometry.{cls}.matched_count"
        counts = (_get(a, count_path), _get(b, count_path))
        coverage.append(_coverage(count_path, *counts))
        for field in GEOMETRY:
            path = f"geometry.{cls}.{field}"
            metrics.append(_accuracy_check(path, _get(a, path), _get(b, path), "error",
                                           all(_count(x) and x > 0 for x in counts)))
    counts = (_get(a, "temporal.matched_transition_count"), _get(b, "temporal.matched_transition_count"))
    coverage.append(_coverage("temporal.matched_transition_count", *counts))
    for field in TEMPORAL:
        path = f"temporal.{field}"
        metrics.append(_accuracy_check(path, _get(a, path), _get(b, path), "error",
                                       all(_count(x) and x > 0 for x in counts)))
    return metrics, coverage


def _accuracy(baseline, candidate):
    checks = []
    _equal(checks, "split", baseline.get("split"), candidate.get("split"), _text)
    for label, report in (("baseline", baseline), ("candidate", candidate)):
        split = report.get("split")
        checks.append(_row(f"{label}.ordinary_validation", "val / no diagnostic",
                           {"split": split, "diagnostic": report.get("diagnostic")},
                           "pass" if split == "val" and report.get("diagnostic") is None else "fail"))
    _required(checks, "checkpoint_identifier", baseline.get("checkpoint"), candidate.get("checkpoint"), _text)
    for key in HASH_KEYS:
        _required(checks, key, baseline.get(key), candidate.get(key), _digest)
    _equal(checks, "identical_dataset_manifest_sha256", baseline.get("dataset_manifest_sha256"),
           candidate.get("dataset_manifest_sha256"), _digest)
    a, b = baseline.get("metrics", {}), candidate.get("metrics", {})
    for key in ("metric_protocol", "geometry_matching", "temporal_matching"):
        _equal(checks, key, _get(a, key), _get(b, key), _text)
    _equal(checks, "frames", _get(a, "frames"), _get(b, "frames"), lambda x: _count(x) and x > 0)
    for key in SCORE_KEYS:
        validator = (lambda x: _count(x) and x > 0) if key == "max_detections" else (
            lambda x: _number(x) and 0 <= x <= 1)
        _equal(checks, f"score_protocol.{key}", _get(a, f"score_protocol.{key}"),
               _get(b, f"score_protocol.{key}"), validator)
    for cls in CLASSES:
        path = f"objects.bev_ap50.classes.{cls}.gt_count"
        _equal(checks, path, _get(a, path), _get(b, path), lambda x: _count(x) and x > 0)
    _equal(checks, "road.valid_pixels", _get(a, "road.valid_pixels"), _get(b, "road.valid_pixels"),
           lambda x: _count(x) and x > 0)
    metrics, coverage = accuracy_metric_rows(a, b)
    if any(x["status"] == "fail" for x in checks):
        status = "not_comparable"
    elif any(x["status"] == "not_verifiable" for x in checks):
        status = "not_verifiable"
    elif any(x["status"] == "fail" for x in metrics):
        status = "fail"
    elif any(x["status"] == "not_verifiable" for x in metrics):
        status = "not_verifiable"
    else:
        status = "pass"
    return dict(status=status, comparability_checks=checks, metric_comparisons=metrics, coverage=coverage,
                all_required_evidence_present=all(x["status"] != "not_verifiable" for x in checks+metrics),
                nongating_diagnostics={key: {"baseline": _get(a, f"objects.{key}"),
                                              "candidate": _get(b, f"objects.{key}")}
                                       for key in ("3d_ap25", "3d_ap50")},
                interpretation="Geometry and temporal errors are conditional on successful matches. Inspect coverage changes and per-class recall alongside their values.")


def _benchmarks(base_eval, cand_eval, baseline, candidate):
    if baseline is None and candidate is None:
        return dict(status="not_supplied", repeatable_speed_gain=None,
                    interpretation="Accuracy-only comparison. No latency or optimization claim is supported.")
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        return dict(status="not_verifiable", repeatable_speed_gain=None,
                    interpretation="Both benchmark reports are required for a timing comparison.")
    checks = []
    dataset=base_eval.get("dataset","kitti360")
    _equal(checks,"evaluation_dataset",dataset,cand_eval.get("dataset","kitti360"),lambda x:x in ("kitti360","nuscenes"))
    for label, evaluation, timing in (("baseline", base_eval, baseline), ("candidate", cand_eval, candidate)):
        for key in ("checkpoint_sha256", "dataset_manifest_sha256", "source_sha256"):
            _equal(checks, f"{label}.timing_to_evaluation.{key}", evaluation.get(key), timing.get(key), _digest)
        for key in BACKEND_KEYS:
            validator = (lambda x: isinstance(x, bool)) if key in ("compiled", "refinement_enabled") else _text
            _equal(checks, f"{label}.timing_to_evaluation.runtime_backend.{key}",
                   _get(evaluation, f"runtime_backend.{key}"), _get(timing, f"runtime_backend.{key}"), validator)
        _equal(checks, f"{label}.validation_timing", "val", timing.get("split"), _text)
        if dataset=="nuscenes":
            _equal(checks,f"{label}.timing_to_evaluation.deterministic_inference",
                   _get(evaluation,"runtime_backend.deterministic_inference"),
                   _get(timing,"runtime_choices.deterministic_algorithms"),lambda x:isinstance(x,bool))
    _required(checks, "valid_sensor_workload_identity", baseline.get("sensor_workload"),
              candidate.get("sensor_workload"), _workload_valid)
    _equal(checks, "same_cached_scene_workload", _get(baseline,"sensor_workload.ordered_sha256"),
           _get(candidate,"sensor_workload.ordered_sha256"), _digest)
    _equal(checks, "same_cached_scene_count", _get(baseline,"sensor_workload.cached_scenes"),
           _get(candidate,"sensor_workload.cached_scenes"), lambda x:_count(x) and x>0)
    for key in ("device", "scope", "stage", "refinement_enabled", "batch_scenes", "cameras", "image_size",
                "steady_state.required_history_frames"):
        validator = _text if key in ("device", "scope", "stage") else (
            (lambda x: isinstance(x, bool)) if key == "refinement_enabled" else (
                (lambda x: isinstance(x, list) and len(x) == 2 and all(_count(n) and n > 0 for n in x))
                if key == "image_size" else _count))
        _equal(checks, key, _get(baseline, key), _get(candidate, key), validator)
    _equal(checks, "one_scene_per_batch", baseline.get("batch_scenes"), 1, _count)
    _equal(checks, "required_camera_count", baseline.get("cameras"), 6 if dataset=="nuscenes" else 4, _count)
    if dataset=="nuscenes":
        _equal(checks,"same_benchmark_script",baseline.get("benchmark_script_sha256"),candidate.get("benchmark_script_sha256"),_digest)
    for key in SCORE_KEYS:
        validator = (lambda x: _count(x) and x > 0) if key == "max_detections" else (
            lambda x: _number(x) and 0 <= x <= 1)
        _equal(checks, f"timing_score_protocol.{key}", _get(baseline, f"score_protocol.{key}"),
               _get(candidate, f"score_protocol.{key}"), validator)
    repeats = []
    for label, report in (("baseline", baseline), ("candidate", candidate)):
        rows = _get(report, "steady_state.repeats")
        rows = rows if isinstance(rows, list) else []
        repeats.append(rows)
        checks.append(_row(f"{label}.repeat_count", 3, len(rows), "pass" if len(rows) >= 3 else "not_verifiable",
                           "At least three repetitions are required" if len(rows) < 3 else None))
        for i, row in enumerate(rows):
            for field, minimum in (("warmup_full_history_updates", 50), ("timed_full_history_updates", 500),
                                   ("timings.combined_wall_ms.samples", 500)):
                count = _get(row, field)
                valid = _count(count) and count >= minimum
                checks.append(_row(f"{label}.repeat_{i}.{field}", minimum, count,
                                   "pass" if valid else "not_verifiable", rule=f"At least {minimum}"))
            for phase,count_field in (("warmup_workload","warmup_full_history_updates"),
                                      ("timed_workload","timed_full_history_updates")):
                order=row.get(phase)
                valid=_order_valid(order,_get(report,"sensor_workload.cached_scenes"))
                valid=valid and order["count"]==row.get(count_field)
                if phase=="timed_workload":valid=valid and order["count"]==_get(row,"timings.combined_wall_ms.samples")
                checks.append(_row(f"{label}.repeat_{i}.{phase}.valid_order_and_counts",True,valid,
                                   "pass" if valid else "not_verifiable",
                                   None if valid else "Accepted order/frequencies missing, inconsistent or out of workload range"))
            for field in ("mean_ms", "median_ms", "p95_ms"):
                value = _get(row, f"timings.combined_wall_ms.{field}")
                checks.append(_row(f"{label}.repeat_{i}.combined_wall_ms.{field}", None, value,
                                   "pass" if _number(value) and value > 0 else "not_verifiable"))
    _equal(checks, "paired_repeat_counts", len(repeats[0]), len(repeats[1]), lambda x: _count(x) and x >= 3)
    for i,(a,b) in enumerate(zip(*repeats)):
        for phase in ("warmup_workload","timed_workload"):
            _equal(checks,f"repeat_{i}.{phase}.same_order",_get(a,f"{phase}.order_sha256"),
                   _get(b,f"{phase}.order_sha256"),_digest)
            _equal(checks,f"repeat_{i}.{phase}.same_frequencies",_get(a,f"{phase}.frequencies"),
                   _get(b,f"{phase}.frequencies"),lambda x:isinstance(x,dict) and bool(x))
    status = ("not_comparable" if any(x["status"] == "fail" for x in checks) else
              "not_verifiable" if any(x["status"] == "not_verifiable" for x in checks) else "pass")
    comparisons = []
    for i, (a, b) in enumerate(zip(*repeats)):
        for field in ("mean_ms", "median_ms", "p95_ms"):
            av, bv = (_get(row, f"timings.combined_wall_ms.{field}") for row in (a, b))
            valid = all(_number(x) and x > 0 for x in (av, bv))
            comparisons.append(dict(repeat=i, metric=field, baseline=_safe(av), candidate=_safe(bv),
                                    delta_ms=bv-av if valid else None,
                                    speedup_ratio=av/bv if valid else None,
                                    lower_latency=bv < av if valid else None))
    means = [row for row in comparisons if row["metric"] == "mean_ms"]
    # No minimum speedup was authorized. Report whether every paired repeat improved.
    gain = all(x["lower_latency"] for x in means) if status == "pass" else None
    aggregate = None
    if status == "pass":
        av = statistics.median(x["baseline"] for x in means)
        bv = statistics.median(x["candidate"] for x in means)
        aggregate = dict(baseline_median_of_repeat_mean_ms=av, candidate_median_of_repeat_mean_ms=bv,
                         speedup_ratio=av/bv, baseline_sequential_fps=1000/av, candidate_sequential_fps=1000/bv,
                         candidate_meets_30fps_target=1000/bv >= 30)
    return dict(status=status, comparability_checks=checks, repeat_comparisons=comparisons,
                repeatable_speed_gain=gain, aggregate=aggregate,
                identities={"baseline": _identities(baseline), "candidate": _identities(candidate)},
                interpretation="Uninstrumented combined wall latency is primary. Repeatable gain means lower mean latency in every paired repeat; no minimum gain is invented. Thirty FPS is a reported target, not an acceptance prerequisite. Device-name equality does not prove an unchanged clock, power, or contention state; inspect benchmark conditions.")


def compare_reports(baseline, candidate, baseline_benchmark=None, candidate_benchmark=None):
    """Return a strict-JSON comparison. No files, model weights, or datasets are read."""
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        raise ValueError("Evaluation reports must be JSON objects")
    accuracy = _accuracy(baseline, candidate)
    timing = _benchmarks(baseline, candidate, baseline_benchmark, candidate_benchmark)
    eligible = accuracy["status"] == "pass" and timing["status"] == "pass" and timing["repeatable_speed_gain"] is True
    return _safe(dict(schema_version=1, accuracy=accuracy, timing=timing,
                      identities={"baseline": _identities(baseline), "candidate": _identities(candidate)},
                      identity_differences={key: baseline.get(key) != candidate.get(key) for key in HASH_KEYS},
                      optimization_eligible=eligible,
                      acceptance_policy={"iou_ap_max_absolute_drop": .01, "error_max_relative_worsening": .05,
                                         "exact_zero_error_absolute_tolerance": ZERO_TOLERANCE,
                                         "object_ap_metric": "per-class custom BEV AP at IoU 0.5"},
                      decision="Eligible for manual acceptance" if eligible else "No automatic optimization acceptance",
                      manual_tradeoff={"available": True, "automatically_selected": False,
                                       "instruction": "Keep both configurations/checkpoints. Any speed/accuracy tradeoff outside the gate requires an explicit recorded decision; this report never replaces or deletes weights."}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--baseline-benchmark", type=Path)
    parser.add_argument("--candidate-benchmark", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output exists; choose a fresh comparison path")
    paths = dict(baseline=args.baseline, candidate=args.candidate,
                 baseline_benchmark=args.baseline_benchmark, candidate_benchmark=args.candidate_benchmark)
    inputs, provenance = {}, {}
    for name, path in paths.items():
        if path is None:
            inputs[name] = None
            continue
        raw = path.read_bytes()
        inputs[name] = json.loads(raw)
        provenance[name] = dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest())
    report = compare_reports(**inputs)
    report["input_reports"] = provenance
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(args.output), accuracy=report["accuracy"]["status"],
                          timing=report["timing"]["status"], optimization_eligible=report["optimization_eligible"])))
    return 0 if report["accuracy"]["status"] == "pass" and report["timing"]["status"] in ("pass", "not_supplied") else 2


if __name__ == "__main__":
    raise SystemExit(main())
