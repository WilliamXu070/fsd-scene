"""Public replay transformations must preserve predictions and exclude targets."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("website_package", Path(__file__).parents[1] / "scripts/package_website_replay.py")
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


def test_road_display_matches_existing_viewer_policy():
    record = {"road": [[0, 1, 2], [0, 1, 2]], "road_confidence": [[.45, .449, .9], [.8, .8, .8]], "road_visible": [[True, True, True], [False, True, False]]}
    result, raw = package.display_road(record)
    assert package.decode_runs(result["runs"]) == [0, 255, 2, 255, 1, 255]
    assert np.array_equal(raw, [[0, 255, 2], [255, 1, 255]])


def test_rle_is_lossless_for_full_size_scene():
    values = np.random.default_rng(42).choice([0, 1, 2, 255], (160, 160))
    assert package.decode_runs(package.encode_runs(values)) == values.ravel().tolist()


def test_prediction_allowlist_excludes_target_fields():
    record = {"boxes": [[1, 2, 3, 4, 2, 1.5, .1]], "scores": [.81], "labels": [0], "track_ids": [3],
              "gt_boxes": [[0]*7], "annotations": "private", "optimizer": "private"}
    exported = package.predicted_objects(record)
    assert set(exported) == {"boxes", "scores", "labels", "track_ids"}
    assert exported["boxes"] is record["boxes"]


def test_paths_cannot_escape_dataset_root(tmp_path):
    assert package.safe_child(tmp_path, "images/camera.jpg").is_relative_to(tmp_path)
    with pytest.raises(ValueError):
        package.safe_child(tmp_path, "../elsewhere/secret")


def test_public_json_has_stable_lf_bytes_on_windows(tmp_path):
    target = tmp_path / "frame.json"
    package.write_json(target, {"frame": 1})
    assert target.read_bytes() == b'{"frame":1}\n'
