"""CPU-only workload identity controls; no camera data or CUDA execution."""
import copy
import importlib.util
from pathlib import Path

import pytest
import torch

SPEC=importlib.util.spec_from_file_location('benchmark',Path(__file__).parents[1]/'scripts/benchmark.py')
benchmark=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def sample(index=0):
    row={key:torch.ones(1,1) for key in benchmark.INFERENCE_KEYS}
    row.update(sequence=['fixture-training'],frame_id=torch.tensor([index*5]),
               timestamp=torch.tensor([index*.5],dtype=torch.float64),ego_to_world=torch.eye(4)[None],
               images=torch.ones(1,4,3,2,2))
    return row


def test_identity_binds_actual_sensor_bytes_and_order_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda,'_lazy_init',lambda:(_ for _ in ()).throw(AssertionError('CUDA forbidden')))
    original=[sample(0),sample(1)]
    expected=benchmark.sensor_workload(original)
    assert expected==benchmark.sensor_workload(copy.deepcopy(original))
    changed=copy.deepcopy(original);changed[0]['images'][0,0,0,0,0]=0
    assert expected['ordered_sha256']!=benchmark.sensor_workload(changed)['ordered_sha256']
    assert expected['ordered_sha256']!=benchmark.sensor_workload(original[::-1])['ordered_sha256']
    assert expected['ordered_sha256']!=benchmark.sensor_workload(original[:1])['ordered_sha256']


def test_pose_calibration_and_timestamp_are_bound_but_ground_truth_is_excluded():
    original=sample()
    expected=benchmark.sensor_workload([original])
    for key in ('ego_to_world','intrinsics','timestamp'):
        changed=copy.deepcopy(original);changed[key].reshape(-1)[0]+=1
        assert benchmark.sensor_workload([changed])['ordered_sha256']!=expected['ordered_sha256']
    changed=copy.deepcopy(original);changed.update(boxes=torch.randn(2,7),road=torch.ones(3,3),depth_target='must be ignored')
    assert benchmark.sensor_workload([changed])==expected


def test_accepted_order_preserves_frequencies_but_distinguishes_order():
    a=benchmark.replay_order_summary([3,4,3,4]);b=benchmark.replay_order_summary([4,3,3,4])
    assert a['count']==4 and a['frequencies']=={'3':2,'4':2}
    assert a['frequencies']==b['frequencies'] and a['order_sha256']!=b['order_sha256']
    assert benchmark.replay_order_summary([])['count']==0


def test_missing_sensor_key_cannot_produce_workload_identity():
    row=sample();row.pop('images')
    with pytest.raises(KeyError):benchmark.sensor_workload([row])


@pytest.mark.parametrize('requested,expected',[(None,True),(True,True),(False,False)])
def test_timing_determinism_matches_evaluation_and_is_bound(monkeypatch,requested,expected):
    calls=[]
    monkeypatch.setattr(torch,'use_deterministic_algorithms',lambda enabled:calls.append(enabled))
    monkeypatch.setattr(torch.cuda,'_lazy_init',lambda:(_ for _ in ()).throw(AssertionError('CUDA forbidden')))
    config={'runtime':{}}
    if requested is not None:config['runtime']['deterministic_inference']=requested
    assert benchmark.configure_inference_determinism(config) is expected
    assert config['runtime']['deterministic_inference'] is expected
    assert calls==[expected]
