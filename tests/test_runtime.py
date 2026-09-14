"""Stage state and real RNG restoration controls."""
import random
import numpy as np
import torch
from fsd.runtime import atomic_json, history_for, rng_state, restore_rng, seed_all
from fsd.engine import selection_score


def test_rng_restore():
    seed_all(42)
    state=rng_state()
    expected=(random.random(),np.random.rand(),torch.rand(4))
    restore_rng(state)
    assert random.random()==expected[0]
    assert np.random.rand()==expected[1]
    assert torch.equal(torch.rand(4),expected[2])


def test_history_never_crosses_sequence_or_future():
    history=[dict(sequence=['a'],timestamp=torch.tensor([1.],dtype=torch.float64))]
    assert history_for(history,dict(sequence=['b'],timestamp=torch.tensor([1.5])))==[]
    assert history_for(history,dict(sequence=['a'],timestamp=torch.tensor([.5])))==[]
    assert history_for(history,dict(sequence=['a'],timestamp=torch.tensor([3.])))==[]
    assert history_for(history,dict(sequence=['a'],timestamp=torch.tensor([1.5]))) is history


def test_selection_requires_all_tasks():
    assert selection_score({}) is None
    metrics=dict(road=dict(iou=dict(road=.8,sidewalk=.6)),objects=dict(bev_ap50=dict(classes=dict(car=dict(ap=.4),pedestrian=dict(ap=.2)))))
    assert abs(selection_score(metrics)-.5)<1e-8


def test_atomic_status(tmp_path):
    import json
    path=tmp_path/'stage.json'
    atomic_json(path,dict(status='pending'))
    atomic_json(path,dict(status='passed',evidence='file'))
    assert json.loads(path.read_text())['evidence']=='file'
    assert not path.with_suffix('.json.tmp').exists()
