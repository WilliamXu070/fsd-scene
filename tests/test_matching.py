"""Controlled regression tests for optional rotated-IoU assignment cost; CPU only."""
import math
import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment as scipy_assignment
import fsd.losses as losses


def box(x=0., y=0., z=0., yaw=0.):
    return [x,y,z,4.,2.,1.5,yaw]


def legacy_cost(pred,logits,truth,labels):
    pred,truth=pred.float(),truth.float()
    position=torch.cdist(pred[:,:3],truth[:,:3],p=1)/5.
    size=torch.cdist(pred[:,3:6].clamp_min(1e-3).log(),truth[:,3:6].clamp_min(1e-3).log(),p=1)
    angle=1-torch.cos(pred[:,None,6]-truth[None,:,6])
    cls=-logits.float().sigmoid()[:,labels]
    return (position+.5*size+.2*angle+2*cls).detach().numpy()


def legacy_match(pred,logits,truth,labels,**kwargs):
    if not len(pred) or not len(truth):
        return torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long)
    r,c=scipy_assignment(legacy_cost(pred,logits,truth,labels))
    return torch.from_numpy(r),torch.from_numpy(c)


@pytest.mark.parametrize("explicit",[False,True])
def test_zero_weight_preserves_exact_cost_assignment_and_skips_iou(monkeypatch,explicit):
    pred=torch.tensor([box(1),box(6,yaw=math.pi-.01),box(1),box(-4)],dtype=torch.float16)
    truth=torch.tensor([box(6,yaw=-math.pi+.01),box(1),box(-3)],dtype=torch.float32)
    labels=torch.tensor([0,1,0]);logits=torch.tensor([[2.,1.],[3.,-2.],[2.,1.],[.5,.9]],dtype=torch.float16)
    expected=legacy_cost(pred,logits,truth,labels);captured=[]
    def solver(cost):
        captured.append(cost.copy());return scipy_assignment(cost)
    def forbidden(*args,**kwargs):raise AssertionError("Zero weight must not compute IoU")
    monkeypatch.setattr(losses,"linear_sum_assignment",solver)
    monkeypatch.setattr(losses,"pairwise_iou",forbidden)
    args={"match_bev_iou_weight":0.0} if explicit else {}
    r,c=losses.match_boxes(pred,logits,truth,labels,**args)
    er,ec=scipy_assignment(expected)
    np.testing.assert_array_equal(captured[0],expected)
    np.testing.assert_array_equal(r.numpy(),er);np.testing.assert_array_equal(c.numpy(),ec)
    assert captured[0].dtype==np.float32


def test_added_cost_uses_true_rotated_bev_iou(monkeypatch):
    pred=torch.tensor([box(yaw=math.pi/2),box()]);truth=torch.tensor([box()])
    logits=torch.zeros(2,2);labels=torch.tensor([0]);base=legacy_cost(pred,logits,truth,labels)
    captured=[]
    def solver(cost):captured.append(cost.copy());return scipy_assignment(cost)
    monkeypatch.setattr(losses,"linear_sum_assignment",solver)
    r,c=losses.match_boxes(pred,logits,truth,labels,match_bev_iou_weight=2.)
    # Same-center 4x2 rectangles rotated90deg intersect in2x2: IoU=4/(8+8-4)=1/3.
    np.testing.assert_allclose(captured[0]-base,np.array([[4/3],[0]],np.float32),rtol=1e-6,atol=1e-7)
    assert captured[0].dtype==np.float32 and r.tolist()==[1] and c.tolist()==[0]


def test_bev_penalty_ignores_vertical_separation(monkeypatch):
    pred=torch.tensor([box(z=10)]);truth=torch.tensor([box()]);logits=torch.zeros(1,2);labels=torch.tensor([0])
    expected=legacy_cost(pred,logits,truth,labels);captured=[]
    def solver(cost):captured.append(cost.copy());return scipy_assignment(cost)
    monkeypatch.setattr(losses,"linear_sum_assignment",solver)
    losses.match_boxes(pred,logits,truth,labels,match_bev_iou_weight=2.)
    np.testing.assert_array_equal(captured[0],expected)  # BEV overlap1, despite3D overlap0.


@pytest.mark.parametrize("n,m",[(0,0),(0,2),(2,0)])
@pytest.mark.parametrize("weight",[0.,2.])
def test_empty_sides_return_device_local_long_indices_without_iou(monkeypatch,n,m,weight):
    def forbidden(*args,**kwargs):raise AssertionError("Empty assignment must not call IoU")
    monkeypatch.setattr(losses,"pairwise_iou",forbidden)
    pred=torch.tensor([box()]*n).reshape(n,7);truth=torch.tensor([box()]*m).reshape(m,7)
    r,c=losses.match_boxes(pred,torch.zeros(n,2),truth,torch.zeros(m,dtype=torch.long),weight)
    assert r.dtype==c.dtype==torch.long and r.device==c.device==pred.device and len(r)==len(c)==0


@pytest.mark.parametrize("weight",[-1.,float("nan"),float("inf"),-float("inf"),None,"2","bad",True,1e100])
def test_invalid_weight_rejected_even_for_empty_assignment(weight):
    with pytest.raises(ValueError,match="match_bev_iou_weight"):
        losses.match_boxes(torch.empty(0,7),torch.empty(0,2),torch.empty(0,7),torch.empty(0,dtype=torch.long),weight)


@pytest.mark.parametrize("weight",[0.,2.])
@pytest.mark.parametrize("field,value",[("pred",float("nan")),("truth",float("inf")),("logits",float("nan")),("logits",float("inf")),("logits",-float("inf"))])
def test_nonfinite_input_is_not_sanitized_into_assignment(weight,field,value):
    args={"pred":torch.tensor([box()]),"truth":torch.tensor([box()]),"logits":torch.zeros(1,2)}
    args[field][0,0]=value
    with pytest.raises(FloatingPointError,match="Non-finite proposal assignment input"):
        losses.match_boxes(args["pred"],args["logits"],args["truth"],torch.tensor([0]),weight)


def test_nonfinite_iou_fails_before_hungarian(monkeypatch):
    monkeypatch.setattr(losses,"pairwise_iou",lambda *a,**k:np.array([[np.nan]]))
    monkeypatch.setattr(losses,"linear_sum_assignment",lambda *a:pytest.fail("Invalid IoU reached Hungarian"))
    with pytest.raises(FloatingPointError,match="Non-finite BEV overlap"):
        losses.match_boxes(torch.tensor([box()]),torch.zeros(1,2),torch.tensor([box()]),torch.tensor([0]),2.)


def test_invalid_box_geometry_fails_without_dropping_rows():
    pred=torch.tensor([box()]);pred[0,3]=0
    with pytest.raises(ValueError,match="strictly positive dimensions"):
        losses.match_boxes(pred,torch.zeros(1,2),torch.tensor([box()]),torch.tensor([0]),2.)


@pytest.mark.parametrize("n,m",[(3,2),(2,3)])
def test_rectangular_assignment_remains_one_to_one_and_ungated(n,m):
    pred=torch.tensor([box(20+i*5) for i in range(n)])
    truth=torch.tensor([box(i*5) for i in range(m)])
    # All proposals decode as cars, all GT are pedestrians: still no class gate.
    logits=torch.tensor([[5.,-5.]]*n);labels=torch.ones(m,dtype=torch.long)
    r,c=losses.match_boxes(pred,logits,truth,labels,2.)
    assert len(r)==len(c)==min(n,m) and len(set(r.tolist()))==len(r) and len(set(c.tolist()))==len(c)
    assert torch.linalg.vector_norm(pred[r,:2]-truth[c,:2],dim=-1).min()>5


def loss_fixture():
    outputs=dict(center_logits=torch.zeros(1,2,8,8,requires_grad=True),geometry_map=torch.zeros(1,8,8,8,requires_grad=True),
        road_logits=torch.zeros(1,3,8,8),
        refined_boxes=torch.tensor([[box(3),box(.3,yaw=.15),box(-2)]],requires_grad=True),
        refined_logits=torch.tensor([[[3.,-2.],[.1,-1.],[1.,-.5]]],requires_grad=True))
    batch=dict(boxes=[torch.tensor([box(0)])],labels=[torch.tensor([0])],detection_valid=torch.zeros(1,2,8,8,dtype=torch.bool))
    config={"model":{"bev_min":-4.,"bev_step":1.,"bev_size":8},"loss":{}}
    return outputs,batch,config


def run_loss(matcher,monkeypatch,explicit_zero):
    outputs,batch,config=loss_fixture()
    if explicit_zero:config["loss"]["match_bev_iou_weight"]=0.
    with monkeypatch.context() as m:
        m.setattr(losses,"match_boxes",matcher)
        result=losses.compute_losses(outputs,batch,config,refine=True)
        result["total"].backward()
    return {k:v.detach() for k,v in result.items()},{k:v.grad for k,v in outputs.items() if v.requires_grad}


def test_zero_weight_preserves_full_loss_and_gradient_values(monkeypatch):
    real_match=losses.match_boxes
    old_values,old_grad=run_loss(legacy_match,monkeypatch,False)
    for explicit in (False,True):
        values,grad=run_loss(real_match,monkeypatch,explicit)
        for key in old_values:torch.testing.assert_close(values[key],old_values[key],rtol=0,atol=0)
        for key in old_grad:torch.testing.assert_close(grad[key],old_grad[key],rtol=0,atol=0)


def test_config_weight_reaches_matcher_and_existing_refinement_losses_backpropagate(monkeypatch):
    outputs,batch,config=loss_fixture();config["loss"]["match_bev_iou_weight"]=2.
    actual=losses.match_boxes;seen=[]
    def recording(*args,**kwargs):
        seen.append(kwargs["match_bev_iou_weight"])
        return actual(*args,**kwargs)
    def gradient_probe(cost):
        assert not torch.is_grad_enabled()
        return scipy_assignment(cost)
    monkeypatch.setattr(losses,"linear_sum_assignment",gradient_probe)
    monkeypatch.setattr(losses,"match_boxes",recording)
    result=losses.compute_losses(outputs,batch,config,refine=True)
    result["refined_total"].backward()
    assert seen==[2.]
    for name in ("refined_boxes","refined_logits"):
        grad=outputs[name].grad
        assert torch.isfinite(grad).all() and grad.abs().sum()>0
