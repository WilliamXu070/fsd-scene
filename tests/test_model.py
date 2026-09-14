"""Behavioral contracts for geometry decoding, lifting, supervision and memory.

Small CPU fixtures exercise the real ResNet/FPN pipeline; they are diagnostics,
not evidence of dataset accuracy or a replacement for real-data overfitting.
"""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from fsd.geometry import camera_rays
from fsd.losses import compute_losses, match_boxes
from fsd.model import DepthLift, SceneModel, align_history_boxes, decode_proposals


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def small_config():
    return {"model": {"fpn_channels": 16, "context_channels": 8, "bev_channels": 16,
                      "depth_bins": 4, "depth_min": 1., "depth_max": 8., "depth_chunk": 2,
                      "bev_size": 16, "bev_min": -8., "bev_step": 1., "num_proposals": 8,
                      "history_length": 3, "pretrained": False}}


def real_path_fixture():
    torch.manual_seed(7)
    h, w, views = 64, 96, 2
    base = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float32)
    transform = np.tile(np.eye(4, dtype=np.float32), (views, 1, 1))
    transform[:, :3, :3] = base
    transform[:, 2, 3] = 1.6
    transform[1, 1, 3] = -.5
    k = np.array([[48., 0., 47.5], [0., 48., 31.5], [0., 0., 1.]], dtype=np.float32)
    calibration = {"camera_to_ego": transform, "intrinsics": np.tile(k, (views, 1, 1)),
                   "distortion": np.zeros((views, 4), dtype=np.float32),
                   "xi": np.zeros(views, dtype=np.float32), "camera_model": np.zeros(views, dtype=np.int64)}
    rays, valid = camera_rays(calibration, h // 8, w // 8, h, w)
    batch = {key: torch.from_numpy(value)[None] for key, value in calibration.items()}
    batch.update(images=torch.rand(1, views, 3, h, w), camera_valid=torch.ones(1, views, dtype=torch.bool),
                 rays=torch.from_numpy(rays)[None], ray_valid=torch.from_numpy(valid)[None],
                 ego_to_world=torch.eye(4)[None], timestamp=torch.tensor([1.]), sequence=["fixture"],
                 frame_id=torch.tensor([0]), road=torch.zeros(1, 16, 16, dtype=torch.long),
                 depth_target=torch.ones(1, views, h // 8, w // 8, dtype=torch.long),
                 boxes=[torch.tensor([[3., 0., .75, 2., 1., 1.5, .2]])], labels=[torch.tensor([0])],
                 instance_ids=[torch.tensor([17])], detection_valid=torch.ones(1, 2, 16, 16, dtype=torch.bool))
    return batch


def test_center_reads_metric_geometry_with_subcell_offsets():
    logits = torch.full((1, 2, 8, 8), -10.)
    logits[0, 0, 4, 5] = 10
    geometry = torch.zeros(1, 8, 8, 8)
    geometry[0, :, 4, 5] = torch.tensor([math.log(.25 / .75), math.log(.75 / .25), .8,
                                                  math.log(4.4), math.log(1.8), math.log(1.6), 1., 0.])
    result = decode_proposals(logits, geometry, torch.ones(1, 4, 8, 8),
                              {"num_proposals": 1, "bev_min": -4., "bev_step": .5})
    expected = torch.tensor([-1.375, -1.625, .8, 4.4, 1.8, 1.6, math.pi / 2])
    torch.testing.assert_close(result["boxes"][0, 0], expected, atol=1e-5, rtol=1e-5)
    assert result["labels"].item() == 0


def test_lifting_known_ray_places_probability_and_preserves_gradients():
    config = {"bev_size": 4, "bev_min": 0., "bev_step": 1., "depth_bins": 2,
              "depth_min": 1., "depth_max": 3., "depth_chunk": 1}
    batch = {"rays": torch.tensor([[[[[1., 0., 0.]]]]]), "ray_valid": torch.ones(1, 1, 1, 1, dtype=torch.bool),
             "camera_valid": torch.ones(1, 1, dtype=torch.bool), "camera_to_ego": torch.eye(4)[None, None]}
    context = torch.tensor([[[[[2.]]]]], requires_grad=True)
    logits = torch.tensor([[[[[math.log(.25)]], [[math.log(.75)]]]]], requires_grad=True)
    actual = DepthLift(config)(context, logits, batch)
    torch.testing.assert_close(actual[0, 0, 0], torch.tensor([0., .5, 0., 1.5]))
    actual[0, 0, 0, 3].backward()
    assert context.grad.item() > 0
    assert logits.grad[0, 0, 1].item() > 0 and logits.grad[0, 0, 0].item() < 0
    config["depth_chunk"] = 2
    torch.testing.assert_close(DepthLift(config)(context.detach(), logits.detach(), batch), actual.detach())
    batch["camera_valid"][:] = False
    assert DepthLift(config)(context.detach(), logits.detach(), batch).abs().sum().item() == 0


def test_history_pose_transform_rotates_center_and_heading():
    boxes = torch.tensor([[[5., 1., .75, 4., 2., 1.5, 0.]]])
    previous = torch.eye(4)[None]
    current = torch.eye(4)[None]
    current[0, :2, :2] = torch.tensor([[0., -1.], [1., 0.]])
    current[0, 0, 3] = 2.
    aligned = align_history_boxes(boxes, previous, current)
    torch.testing.assert_close(aligned[0, 0, :3], torch.tensor([1., -3., .75]))
    torch.testing.assert_close(aligned[0, 0, 3:6], boxes[0, 0, 3:6])
    assert aligned[0, 0, 6].item() == pytest.approx(-math.pi / 2)


def test_real_cnn_lift_heads_refinement_backward_and_bn_freeze():
    model = SceneModel(small_config()).train()
    batch = real_path_fixture()
    batch["images"].requires_grad_()
    output = model(batch)
    assert output["road_logits"].shape == (1, 3, 16, 16)
    assert output["depth_logits"].shape == (1, 2, 4, 8, 12)
    assert output["refined_boxes"].shape == (1, 8, 7)
    assert output["spatial_valid"].any()
    loss = compute_losses(output, batch, small_config())
    assert all(torch.isfinite(value) for value in loss.values())
    loss["total"].backward()
    for parameter in (model.image_encoder.backbone.conv1.weight, model.image_encoder.fpn.inner_blocks[0][0].weight,
                      model.depth_head[-1].weight, model.bev_encoder[0][0].weight,
                      model.geometry_head[-1].weight, model.refiner.box_delta.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert batch["images"].grad.abs().sum() > 0
    assert all(not module.training for module in model.image_encoder.backbone.modules()
               if isinstance(module, torch.nn.BatchNorm2d))
    assert not output["state"]["features"].requires_grad


def test_no_valid_image_evidence_preserves_boxes_and_missing_history_is_finite():
    model = SceneModel(small_config()).eval()
    batch = real_path_fixture()
    batch["camera_valid"][:] = False
    with torch.no_grad():
        output = model(batch)
        with_history = model(batch, [output["state"]])
    assert not output["spatial_valid"].any()
    torch.testing.assert_close(output["refined_boxes"], output["initial_boxes"], rtol=0, atol=0)
    torch.testing.assert_close(output["refined_logits"], output["initial_logits"], rtol=0, atol=0)
    assert torch.isfinite(with_history["refined_boxes"]).all()


def test_temporal_memory_rejects_other_sequences_future_and_long_gaps():
    model = SceneModel(small_config()).eval()
    batch = real_path_fixture()
    state = {"features": torch.randn(1, 3, 16), "boxes": torch.tensor([[[2., 0., 1., 4., 2., 1.5, 0.]]]).expand(1, 3, 7),
             "ego_to_world": torch.eye(4)[None], "timestamp": torch.tensor([.5]),
             "sequence": ["fixture"], "valid": torch.ones(1, 3, dtype=torch.bool)}
    _, valid = model.refiner.temporal_memory([state], batch, torch.float32)
    assert valid.all()
    for key, value in (("sequence", ["another"]), ("timestamp", torch.tensor([2.])), ("timestamp", torch.tensor([-5.]))):
        changed = dict(state)
        changed[key] = value
        _, valid = model.refiner.temporal_memory([changed], batch, torch.float32)
        assert not valid.any()


def test_unknown_annotations_produce_no_loss_and_targets_never_affect_inference():
    model = SceneModel(small_config()).eval()
    batch = real_path_fixture()
    with torch.no_grad():
        original = model(batch, refine=False)
    batch["depth_target"][:] = -1
    batch["road"][:] = -1
    batch["detection_valid"][:] = False
    batch["boxes"], batch["labels"] = [torch.empty(0, 7)], [torch.empty(0, dtype=torch.long)]
    with torch.no_grad():
        changed = model(batch, refine=False)
    torch.testing.assert_close(original["initial_boxes"], changed["initial_boxes"], rtol=0, atol=0)
    torch.testing.assert_close(original["road_logits"], changed["road_logits"], rtol=0, atol=0)
    loss = compute_losses(changed, batch, small_config())
    assert loss["total"].item() == 0.


def test_matching_is_one_to_one_and_yaw_is_circular():
    targets = torch.tensor([[3., 0., .75, 4., 2., 1.5, math.pi - .01],
                            [8., 1., .75, 4., 2., 1.5, 0.]])
    proposals = targets[[1, 0]].clone()
    proposals[1, 6] = -math.pi + .01
    labels = torch.tensor([0, 0])
    rows, cols = match_boxes(proposals, torch.tensor([[5., -5.], [5., -5.]]), targets, labels)
    assert dict(zip(rows.tolist(), cols.tolist())) == {0: 1, 1: 0}
    circular_error = 1 - torch.cos(proposals[1, 6] - targets[0, 6])
    assert circular_error < .001


def test_invalid_history_is_identical_to_no_history_and_unix_time_is_precise():
    model = SceneModel(small_config()).eval()
    batch = real_path_fixture()
    batch["timestamp"] = torch.tensor([1470000000.5], dtype=torch.float64)
    with torch.no_grad():
        reference = model(batch)
    state = dict(reference["state"])
    state["sequence"] = ["unrelated"]
    state["timestamp"] = torch.tensor([1470000000.0], dtype=torch.float64)
    with torch.no_grad():
        rejected = model(batch, [state])
    torch.testing.assert_close(reference["refined_boxes"], rejected["refined_boxes"], rtol=0, atol=0)
    state["sequence"] = ["fixture"]
    state["valid"] = torch.ones_like(state["valid"])
    _, valid = model.refiner.temporal_memory([state], batch, torch.float32)
    assert valid.all(), "Subsecond Unix timestamps were rounded before subtraction"


def test_refinement_sampling_is_local_and_gradients_reach_history_attention():
    model = SceneModel(small_config()).eval()
    batch = real_path_fixture()
    features = [torch.randn(1, 2, 16, 8 // (2**level), 12 // (2**level)) for level in range(3)]
    boxes = torch.tensor([[[3., 0., .75, 2., 1., 1.5, .2]]], requires_grad=True)
    tokens, valid = model.refiner.sample_images(boxes, features, batch)
    shifted = boxes.detach().clone()
    shifted[..., 0] += .001
    shifted_tokens, shifted_valid = model.refiner.sample_images(shifted, features, batch)
    assert valid.any() and torch.equal(valid, shifted_valid)
    assert (tokens-shifted_tokens).abs().max() < .05
    tokens.square().sum().backward()
    assert torch.isfinite(boxes.grad).all() and boxes.grad.abs().sum() > 0
    proposals = {"boxes": boxes.detach(), "logits": torch.zeros(1, 1, 2),
                 "features": torch.randn(1, 1, 16), "peak_valid": torch.ones(1, 1, dtype=torch.bool)}
    state = {"features": torch.randn(1, 2, 16), "boxes": boxes.detach().expand(1, 2, 7),
             "ego_to_world": torch.eye(4)[None], "timestamp": torch.tensor([.5]),
             "sequence": ["fixture"], "valid": torch.ones(1, 2, dtype=torch.bool)}
    refined, logits, _, _ = model.refiner(proposals, features, batch, [state])
    (refined.square().sum()+logits.square().sum()).backward()
    grad = model.refiner.temporal_attention.in_proj_weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_disabled_losses_are_finite_for_full_size_fp16_outside_autocast():
    # A valid full-grid logit sum overflows FP16; multiplying inf by zero is NaN.
    centers = torch.full((1, 2, 160, 160), -2.19, dtype=torch.float16, requires_grad=True)
    assert torch.isinf(centers.detach().sum())
    outputs = {"center_logits": centers, "geometry_map": torch.zeros(1, 8, 160, 160, dtype=torch.float16),
               "road_logits": torch.zeros(1, 3, 160, 160, dtype=torch.float16)}
    batch = {"boxes": [torch.empty(0, 7)], "labels": [torch.empty(0, dtype=torch.long)],
             "road": torch.full((1, 160, 160), -1, dtype=torch.long),
             "detection_valid": torch.zeros(1, 2, 160, 160, dtype=torch.bool)}
    losses = compute_losses(outputs, batch, {}, refine=False)
    assert all(torch.isfinite(value) and value.item() == 0 for value in losses.values())
    losses["total"].backward()
    assert torch.isfinite(centers.grad).all() and centers.grad.abs().sum() == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("upstream", [0.0, 1.0])
def test_decoded_small_heading_has_finite_parameter_gradients(device, upstream):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    # Actual proposal67/68 in training drive0000 frame35, from best Stage A.
    # FP16 reciprocal(x*x+y*y) overflows although its analytic derivative is finite.
    geometry_parameter=torch.nn.Parameter(torch.zeros((1,8,1,1),dtype=torch.float32,device=device))
    with torch.no_grad():
        geometry_parameter[0,6,0,0]=-0.00150299072265625
        geometry_parameter[0,7,0,0]=0.00341796875
    geometry=geometry_parameter.to(torch.float16)  # AMP-like output, FP32 parameters.
    center=torch.tensor([[[[8.0]],[[-8.0]]]],dtype=torch.float16,device=device)
    features=torch.zeros((1,4,1,1),dtype=torch.float16,device=device)
    result=decode_proposals(center,geometry,features,{"num_proposals":1,"bev_min":-40.,"bev_step":.5})
    assert result["boxes"].dtype==geometry.dtype
    (result["boxes"][...,6].float().sum()*upstream).backward()
    assert torch.isfinite(geometry_parameter.grad).all()
    if upstream==0:
        assert torch.count_nonzero(geometry_parameter.grad)==0
    else:
        y=geometry[0,6,0,0].detach().double()
        x=(geometry[0,7,0,0]+1e-7).detach().double()
        expected=torch.stack((x/(x*x+y*y),-y/(x*x+y*y))).float()
        # The FP32 derivative is cast back through the FP16 activation path.
        torch.testing.assert_close(geometry_parameter.grad[0,6:8,0,0],expected,rtol=1e-3,atol=1e-3)
