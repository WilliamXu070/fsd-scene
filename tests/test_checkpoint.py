"""Checkpoint completeness checks; real GPU test is scripts/verify_resume.py."""
import copy, importlib.util
from pathlib import Path
import torch
import pytest
from fsd.runtime import history_for,save_checkpoint,seed_all

spec=importlib.util.spec_from_file_location("verify_resume",Path(__file__).resolve().parents[1]/"scripts/verify_resume.py")
resume=importlib.util.module_from_spec(spec);spec.loader.exec_module(resume)

@pytest.fixture(autouse=True)
def cpu_only_rng(monkeypatch):
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)

def components():
    model=torch.nn.Linear(4,2)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=.7)
    scaler=torch.amp.GradScaler("cpu",enabled=False)
    return model,optimizer,scheduler,scaler

def update(model,optimizer,scheduler,scaler):
    x=torch.randn(5,4);target=torch.randn(5,2)
    optimizer.zero_grad(set_to_none=True)
    loss=(model(x)-target).square().mean()
    scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();scheduler.step()
    return loss.detach()

def test_fresh_components_restore_next_update_and_rng_exactly(tmp_path):
    seed_all(53)
    model,opt,schedule,scaler=components()
    update(model,opt,schedule,scaler)
    payload=resume.snapshot(model,opt,schedule,scaler,[],{},[0,1],1)
    save_checkpoint(tmp_path/"state.pt",**payload)
    expected_loss=update(model,opt,schedule,scaler)
    expected=resume.snapshot(model,opt,schedule,scaler,[],{},[0,1],2)
    restored=torch.load(tmp_path/"state.pt",map_location="cpu",weights_only=False)
    model2,opt2,schedule2,scaler2=components()
    _,checks=resume.restore_training_state(restored,model2,opt2,schedule2,scaler2,"cpu")
    assert all(checks.values())
    observed_loss=update(model2,opt2,schedule2,scaler2)
    assert torch.equal(expected_loss,observed_loss)
    observed=resume.snapshot(model2,opt2,schedule2,scaler2,[],{},[0,1],2)
    for key in ("model","optimizer","scheduler","scaler","rng"):
        assert resume.compare_trees(expected[key],observed[key],rtol=0,atol=0)["passed"]

def test_comparator_detects_optimizer_omission_and_parameter_changes():
    seed_all(4)
    model,opt,schedule,scaler=components();update(model,opt,schedule,scaler)
    fresh=torch.optim.AdamW(model.parameters(),lr=.01)
    assert not resume.compare_trees(opt.state_dict(),fresh.state_dict())["passed"]
    a={"weight":torch.ones(3)};b={"weight":torch.tensor([1.,1.,1.01])}
    assert not resume.compare_trees(a,b)["passed"]
    assert resume.tree_digest(a)!=resume.tree_digest(b)

def test_temporal_state_boundary_checks_preserve_subsecond_epoch_time():
    state={"timestamp":torch.tensor([1470000000.],dtype=torch.float64),"sequence":["drive0"]}
    history=[state]
    batch={"timestamp":torch.tensor([1470000000.5],dtype=torch.float64),"sequence":["drive0"]}
    assert history_for(history,batch)==history
    assert history_for(history,dict(batch,sequence=["drive1"]))==[]
    assert history_for(history,dict(batch,timestamp=state["timestamp"]))==[]
    assert history_for(history,dict(batch,timestamp=state["timestamp"]+10))==[]


def test_noise_bound_rejects_divergence_beyond_repeat_control():
    control={"max_abs":1e-5,"rmse":1e-7}
    assert resume.within_repeat_noise(control,{"max_abs":1.1e-5,"rmse":1.1e-7})["passed"]
    assert not resume.within_repeat_noise(control,{"max_abs":1e-3,"rmse":1e-5})["passed"]
