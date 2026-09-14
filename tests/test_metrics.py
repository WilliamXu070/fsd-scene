"""Independent numerical controls for scene evaluation; no dataset or GPU required."""
import json
import math
import numpy as np
import pytest
from fsd.metrics import (
    Evaluator, pairwise_iou, pairwise_ious, detection_match, road_confusion,
    depth_metrics, latency_summary, wrap_angle, _clip_polygon, _corners,
)


def box(x=0., y=0., z=1., length=4., width=2., height=2., yaw=0.):
    return [x, y, z, length, width, height, yaw]


def object_metric(result, metric="bev_ap50", cls="car"):
    return result["objects"][metric]["classes"][cls]


def test_identical_and_disjoint_analytic_iou():
    boxes = [box(), box(x=10)]
    bev, vol = pairwise_ious(boxes, boxes)
    np.testing.assert_allclose(bev, np.eye(2))
    np.testing.assert_allclose(vol, np.eye(2))
    bev, vol = pairwise_ious([box()], [box(z=4)])
    assert bev[0, 0] == pytest.approx(1.)
    assert vol[0, 0] == 0.
    assert pairwise_iou([], boxes).shape == (0, 2)


def test_translation_rotation_and_scale_known_intersections():
    # Two 4x2 rectangles shifted 2m: area intersection=4, union=12.
    assert pairwise_iou([box()], [box(x=2)])[0, 0] == pytest.approx(1 / 3)
    # A 90-degree rotation also leaves a 2x2 overlap.
    assert pairwise_iou([box()], [box(yaw=math.pi / 2)])[0, 0] == pytest.approx(1 / 3)
    # Larger box contains the original: BEV=8/32; 3D=16/128.
    bev, vol = pairwise_ious([box()], [box(length=8, width=4, height=4)])
    assert bev[0, 0] == pytest.approx(.25)
    assert vol[0, 0] == pytest.approx(.125)


def test_polygon_fallback_matches_known_area():
    p = _corners(np.array([box(), box(yaw=math.pi / 2)]))
    assert _clip_polygon(p[0], p[1]) == pytest.approx(4.)
    assert _clip_polygon(p[0], p[0] + 100) == 0.


def test_perfect_predictions_score_one_and_empty_class_is_null():
    evaluator = Evaluator()
    evaluator.update([box(), box(x=10)], [.9, .8], [0, 0], [box(), box(x=10)], [0, 0])
    result = evaluator.compute()
    for metric in evaluator.thresholds:
        assert object_metric(result, metric)["ap"] == 1.
        assert object_metric(result, metric)["recall"] == 1.
        assert object_metric(result, metric, "pedestrian")["ap"] is None
        assert result["objects"][metric]["classes_with_gt"] == 1
    assert result["geometry"]["car"]["center_error_m"] == 0.
    assert result["temporal"]["matched_transition_count"] == 0
    json.dumps(result, allow_nan=False)


def test_missing_all_objects_is_zero_not_null():
    evaluator = Evaluator()
    evaluator.update([], [], [], [box()], [0])
    metric = object_metric(evaluator.compute())
    assert metric["ap"] == metric["recall"] == 0
    assert metric["false_negatives"] == 1


def test_duplicate_preceding_second_true_positive_lowers_ap():
    evaluator = Evaluator()
    evaluator.update([box(), box(), box(x=10)], [.99, .98, .5], [0, 0, 0], [box(), box(x=10)], [0, 0])
    metric = object_metric(evaluator.compute())
    assert metric["ap"] == pytest.approx(5 / 6)
    assert metric["precision"] == pytest.approx(2 / 3)
    assert metric["recall"] == 1
    assert metric["false_positives"] == 1


def test_high_confidence_bad_prediction_degrades_ap():
    evaluator = Evaluator()
    evaluator.update([box(x=10), box()], [.9, .8], [0, 0], [box()], [0])
    assert object_metric(evaluator.compute())["ap"] == .5


def test_wrong_class_is_not_a_match():
    evaluator = Evaluator()
    evaluator.update([box()], [.9], [1], [box()], [0])
    result = evaluator.compute()
    assert object_metric(result)["ap"] == 0
    person = object_metric(result, cls="pedestrian")
    assert person["ap"] is None
    assert person["false_positives"] == 1


def test_descending_matching_searches_best_unclaimed_target():
    status, matched = detection_match([[.9, .8], [.9, .7]], [.1, .9])
    assert status.tolist() == [1, 1]
    assert matched.tolist() == [1, 0]


def test_visibility_mask_ignores_unknown_but_preserves_known_positive_and_duplicate():
    evaluator = Evaluator()
    unknown = np.zeros((2, 160, 160), dtype=bool)
    evaluator.update([box(), box(), box(x=30)], [.9, .8, .99], [0, 0, 0], [box()], [0], detection_valid=unknown)
    metric = object_metric(evaluator.compute())
    assert metric["true_positives"] == 1
    assert metric["false_positives"] == 1  # Duplicate of known positive stays supported.
    assert metric["ignored_predictions"] == 1
    assert metric["ap"] == 1  # Low-scoring duplicate follows full recall.


def test_mask_axes_are_y_row_x_column_and_class_specific():
    evaluator = Evaluator()
    valid = np.zeros((2, 160, 160), dtype=bool)
    valid[0, 82, 84] = True  # car at x2,y1
    evaluator.update([box(x=2, y=1), box(x=1, y=2), box(x=2, y=1)], [.9] * 3,
                     [0, 0, 1], [], [], detection_valid=valid)
    result = evaluator.compute()
    assert object_metric(result)["false_positives"] == 1
    assert object_metric(result)["ignored_predictions"] == 1
    assert object_metric(result, cls="pedestrian")["ignored_predictions"] == 1


def test_distance_bins_use_gt_for_matches_and_prediction_for_false_positive():
    evaluator = Evaluator()
    # Center error crosses 20m boundary but must not move the matched GT bin.
    evaluator.update([box(x=20.1), box(x=45)], [.9, .8], [0, 0], [box(x=19.9)], [0])
    bins = object_metric(evaluator.compute())["distance_bins"]
    assert bins["0_20m"]["true_positives"] == 1
    assert bins["0_20m"]["ap"] == 1
    assert bins["20_40m"]["prediction_count"] == 0
    assert bins["40m_plus"]["false_positives"] == 1
    assert bins["40m_plus"]["ap"] is None


def test_yaw_wrap_and_front_back_are_not_confused():
    evaluator = Evaluator()
    evaluator.update([box(yaw=-math.pi + .01)], [.9], [0], [box(yaw=math.pi - .01)], [0])
    assert evaluator.compute()["geometry"]["car"]["yaw_error_deg"] == pytest.approx(.02 * 180 / math.pi)
    assert abs(wrap_angle(math.pi)) == pytest.approx(math.pi)


def test_road_unknown_labels_are_ignored_and_false_class_predictions_penalized():
    confusion = road_confusion([[0, 1, 99], [1, 2, 0]], [[0, 1, -1], [0, 2, -1]])
    np.testing.assert_array_equal(confusion, [[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    evaluator = Evaluator()
    evaluator.update([], [], [], [], [], road_pred=[[0, 1, 2]], road_target=[[0, 0, -1]])
    result = evaluator.compute()["road"]
    assert result["iou"]["road"] == .5
    assert result["iou"]["sidewalk"] == 0
    assert result["iou"]["other_ground"] is None
    assert result["miou"] == .25
    assert result["valid_pixels"] == 2


def test_empty_road_and_depth_report_null():
    evaluator = Evaluator()
    assert evaluator.compute()["road"]["miou"] is None
    assert depth_metrics([2., 5.], [-1., 0.])["rmse_m"] is None


def test_depth_uses_metric_values_and_rejects_invalid_predictions():
    result = depth_metrics([2., 8., 500.], [1., 10., -1.])
    assert result["count"] == 2
    assert result["mae_m"] == 1.5
    assert result["rmse_m"] == pytest.approx(math.sqrt(2.5))
    assert result["abs_rel"] == pytest.approx(.6)
    result = depth_metrics([np.nan, 10.], [1., 10.])
    assert result["invalid_prediction_count"] == 1
    assert result["rmse_m"] is None  # Never hide a model NaN by dropping it.


def test_temporal_real_movement_is_removed_before_measuring_jitter():
    evaluator = Evaluator()
    for timestamp, x, yaw in [(0., 0., 0.), (.5, 1., .3), (1., 2., .6)]:
        evaluator.update([box(x=x + .1, yaw=yaw + .02)], [.9], [0], [box(x=x, yaw=yaw)], [0],
                         frame_metadata={"sequence": "train", "timestamp": timestamp,
                                         "gt_instance_ids": [7], "ego_to_world": np.eye(4)})
    metric = evaluator.compute()["temporal"]
    assert metric["matched_transition_count"] == 2
    assert metric["center_residual_change_m"] == pytest.approx(0, abs=1e-10)
    assert metric["yaw_residual_change_deg"] == pytest.approx(0, abs=1e-10)


def test_temporal_ego_rotation_and_sequence_resets():
    evaluator = Evaluator()
    rotation = np.eye(4)
    rotation[:2, :2] = [[0, -1], [1, 0]]
    evaluator.update([box(x=.2)], [.9], [0], [box()], [0],
                     frame_metadata={"sequence": "a", "timestamp": 0., "gt_instance_ids": [1], "ego_to_world": np.eye(4)})
    evaluator.update([box(y=-.2)], [.9], [0], [box()], [0],
                     frame_metadata={"sequence": "a", "timestamp": .5, "gt_instance_ids": [1], "ego_to_world": rotation})
    evaluator.update([box(y=.5)], [.9], [0], [box()], [0],
                     frame_metadata={"sequence": "b", "timestamp": 1., "gt_instance_ids": [1], "ego_to_world": np.eye(4)})
    metric = evaluator.compute()["temporal"]
    assert metric["matched_transition_count"] == 1
    assert metric["center_residual_change_m"] == pytest.approx(0)


def test_temporal_jitter_is_detected_but_large_gaps_are_not_comparable():
    evaluator = Evaluator()
    for t, x in [(0., 0.), (.5, .4), (3., -.4)]:
        evaluator.update([box(x=x)], [.9], [0], [box()], [0],
                         frame_metadata={"sequence": "a", "timestamp": t, "gt_instance_ids": [1], "ego_to_world": np.eye(4)})
    result = evaluator.compute()["temporal"]
    assert result["matched_transition_count"] == 1
    assert result["center_residual_change_m"] == .4
    assert result["center_residual_rate_mps"] == .8


def test_latency_reports_throughput_from_mean_not_median():
    result = latency_summary([10., 10., 40.])
    assert result["median_ms"] == 10.
    assert result["mean_ms"] == 20.
    assert result["sequential_fps"] == 50.
    with pytest.raises(ValueError):
        latency_summary([0.])


def test_invalid_contracts_fail_loudly():
    with pytest.raises(ValueError):
        pairwise_iou([box(width=-1)], [])
    with pytest.raises(ValueError):
        Evaluator().update([box()], [.8], [3], [], [])
    with pytest.raises(ValueError):
        Evaluator().update([], [], [], [], [], road_pred=[[0]])
    with pytest.raises(ValueError):
        road_confusion([[9]], [[0]])
