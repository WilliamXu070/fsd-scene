"""Positive-only focal normalization controls; actual negative masking is retained."""
import copy
import math
import pytest
import torch
import fsd.losses as losses


@pytest.fixture(autouse=True,scope='module')
def cpu_threads():
    old=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def old_focal(logits,target,valid):
    probabilities=logits.float().sigmoid().clamp(1e-5,1-1e-5)
    positive=target.eq(1)
    known=valid.bool()|positive
    negative=target.lt(1)&known
    positive_loss=-(probabilities.log()*(1-probabilities).square()*positive).sum()
    negative_loss=-((1-probabilities).log()*probabilities.square()*(1-target).pow(4)*negative).sum()
    return (positive_loss+negative_loss)/positive.sum().clamp_min(1)


def loss_and_grad(function,logits,target,valid,**kwargs):
    x=logits.detach().clone().requires_grad_(True)
    result=function(x,target,valid,**kwargs)
    loss=result[0] if isinstance(result,tuple) else result
    gradient=torch.autograd.grad(loss,x)[0]
    return loss.detach(),gradient


def controls(dtype=torch.float32,batch=1,size=12):
    gen=torch.Generator().manual_seed(904)
    logits=torch.randn(batch,2,size,size,generator=gen).to(dtype)*3
    target=torch.zeros_like(logits,dtype=torch.float32)
    target[:,0,0,:10]=1;target[:,1,1,1]=1
    # Gaussian shoulder and unknown area have distinct supervision semantics.
    target[:,1,1,2]=.6
    valid=torch.rand(batch,2,size,size,generator=gen)>.3
    valid[:,1,1,1]=False
    return logits,target,valid


@pytest.mark.parametrize('dtype',[torch.float32,torch.float16])
@pytest.mark.parametrize('explicit',[False,True])
@pytest.mark.parametrize('batch',[1,2])
def test_default_loss_and_gradients_are_bitwise_legacy(dtype,explicit,batch):
    logits,target,valid=controls(dtype,batch)
    expected,expected_grad=loss_and_grad(old_focal,logits,target,valid)
    kwargs={'positive_normalization':'global'} if explicit else {}
    for stats in (False,True):
        got,grad=loss_and_grad(losses.center_focal_loss,logits,target,valid,return_stats=stats,**kwargs)
        assert torch.equal(got,expected) and torch.equal(grad,expected_grad)


@pytest.mark.parametrize('positive_class',[None,0,1])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float16])
def test_no_positive_and_single_class_preserve_exact_loss_and_gradient(positive_class,dtype):
    logits,target,valid=controls(dtype)
    if positive_class is None:target[target==1]=0
    else:target[:,1-positive_class][target[:,1-positive_class]==1]=0
    baseline,baseline_grad=loss_and_grad(old_focal,logits,target,valid)
    actual,actual_grad=loss_and_grad(losses.center_focal_loss,logits,target,valid,positive_normalization='present_class')
    assert torch.equal(actual,baseline) and torch.equal(actual_grad,baseline_grad)


def test_mixed_class_positive_coefficients_and_all_negative_gradients():
    logits,target,valid=controls();logits.zero_();valid.fill_(True)
    global_loss,global_grad=loss_and_grad(old_focal,logits,target,valid)
    balanced_loss,balanced_grad=loss_and_grad(losses.center_focal_loss,logits,target,valid,positive_normalization='present_class')
    positive=target==1;negative=target<1
    # At p=.5, derivative of -(1-p)^2 log(p) w.r.t. logit is:
    derivative=.25*math.log(.5)-.125
    torch.testing.assert_close(balanced_grad[:,0][positive[:,0]],torch.full((10,),derivative/20),rtol=1e-7,atol=1e-9)
    torch.testing.assert_close(balanced_grad[:,1][positive[:,1]],torch.tensor([derivative/2]),rtol=1e-7,atol=1e-9)
    assert torch.equal(global_grad[negative],balanced_grad[negative])
    # Equal per-positive focal values give equal positive averages despite changed coefficients.
    torch.testing.assert_close(global_loss,balanced_loss,rtol=2e-7,atol=0)


def test_unknown_cells_zero_gradient_positive_override_and_soft_negatives():
    logits,target,valid=controls();valid.zero_()
    # One known Gaussian-shoulder negative and one ordinary negative only.
    valid[0,1,1,2]=True;valid[0,0,2,2]=True
    global_loss,global_grad=loss_and_grad(old_focal,logits,target,valid)
    balanced_loss,balanced_grad=loss_and_grad(losses.center_focal_loss,logits,target,valid,positive_normalization='present_class')
    unknown=(~valid)&(target<1)
    assert torch.count_nonzero(global_grad[unknown])==torch.count_nonzero(balanced_grad[unknown])==0
    assert balanced_grad[0,1,1,1]!=0  # Positive overrides false valid mask.
    assert balanced_grad[0,1,1,2]!=0  # Gaussian shoulder is still a soft negative.
    assert torch.equal(global_grad[target<1],balanced_grad[target<1])


def test_detached_statistics_match_independent_cell_sums():
    logits,target,valid=controls();logits.requires_grad_(True)
    value,stats=losses.center_focal_loss(logits,target,valid,'present_class',return_stats=True)
    p=logits.detach().float().sigmoid().clamp(1e-5,1-1e-5)
    for cls in range(2):
        pos=target[:,cls]==1;neg=(target[:,cls]<1)&(valid[:,cls]|pos)
        pcell=p[:,cls]
        expected_pos=-(pcell[pos].log()*(1-pcell[pos]).square()).sum()
        expected_neg=-((1-pcell[neg]).log()*pcell[neg].square()*(1-target[:,cls][neg]).pow(4)).sum()
        torch.testing.assert_close(stats['positive_sum'][cls],expected_pos)
        torch.testing.assert_close(stats['negative_sum'][cls],expected_neg)
        assert stats['positive_count'][cls]==pos.sum() and stats['negative_count'][cls]==neg.sum()
    assert all(not x.requires_grad and x.grad_fn is None and x.shape==(2,) for x in stats.values())
    expected=(stats['positive_sum']/stats['positive_count']).mean()+stats['negative_sum'].sum()/11
    torch.testing.assert_close(value,expected)


@pytest.mark.parametrize('option',['bad','Present_Class','',None,0,True,[],{}])
def test_invalid_normalization_is_rejected(option):
    logits,target,valid=controls()
    with pytest.raises(ValueError,match='center_positive_normalization'):
        losses.center_focal_loss(logits,target,valid,positive_normalization=option)


@pytest.mark.parametrize('mode',['global','present_class'])
def test_full_bev_fp16_outside_autocast_has_finite_loss_gradients_and_stats(mode):
    logits,target,valid=controls(torch.float16,size=160)
    logits.fill_(-2.19);logits.requires_grad_(True)
    value,stats=losses.center_focal_loss(logits,target,valid,mode,return_stats=True)
    value.backward()
    assert value.dtype==torch.float32 and torch.isfinite(value) and torch.isfinite(logits.grad).all()
    assert all(torch.isfinite(x).all() for x in stats.values())
    assert stats['positive_count'].tolist()==[10,1]


def complete_fixture():
    gen=torch.Generator().manual_seed(443)
    outputs=dict(center_logits=torch.randn(1,2,8,8,generator=gen,requires_grad=True),
        geometry_map=torch.randn(1,8,8,8,generator=gen,requires_grad=True),
        road_logits=torch.randn(1,3,8,8,generator=gen,requires_grad=True),
        depth_logits=torch.randn(1,4,3,2,2,generator=gen,requires_grad=True),
        refined_boxes=torch.tensor([[[0.,0.,.5,4.,2.,1.5,0.],[2.,2.,.5,.8,.6,1.7,0.]]],requires_grad=True),
        refined_logits=torch.tensor([[[1.,0.],[0.,1.]]],requires_grad=True))
    batch=dict(boxes=[torch.tensor([[0.,0.,.5,4.,2.,1.5,0.],[0.,0.,.5,4.,2.,1.5,0.],[2.,2.,.5,.8,.6,1.7,0.]])],
        labels=[torch.tensor([0,0,1])],detection_valid=torch.ones(1,2,8,8,dtype=torch.bool),
        road=torch.zeros(1,8,8,dtype=torch.long),depth_target=torch.zeros(1,4,2,2,dtype=torch.long))
    config={'model':{'bev_min':-4.,'bev_step':1.,'bev_size':8},'loss':{'match_bev_iou_weight':2.}}
    return outputs,batch,config


def test_compute_diagnostics_count_unique_centers_and_exclude_them_from_total():
    outputs,batch,config=complete_fixture()
    config['loss']['center_positive_normalization']='present_class'
    result=losses.compute_losses(outputs,batch,config)
    assert result['center_car_positive_count']==1 and result['center_pedestrian_positive_count']==1
    stats={key:value for key,value in result.items() if key.startswith(('center_car_','center_pedestrian_'))}
    assert len(stats)==8 and all(value.ndim==0 and not value.requires_grad for value in stats.values())
    expected=result['depth']+result['road']+result['initial_total']+result['refined_total']
    assert torch.equal(result['total'],expected)
    result['total'].backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in outputs.values())


def test_compute_global_preserves_every_existing_value_and_total_gradient(monkeypatch):
    first,batch,config=complete_fixture();second=copy.deepcopy(first)
    actual=losses.compute_losses(first,batch,config)
    actual['total'].backward()
    def old_adapter(logits,target,valid,positive_normalization='global',return_stats=False):
        return (old_focal(logits,target,valid),{}) if return_stats else old_focal(logits,target,valid)
    monkeypatch.setattr(losses,'center_focal_loss',old_adapter)
    expected=losses.compute_losses(second,batch,config)
    expected['total'].backward()
    assert set(actual)-set(expected)=={f'center_{cls}_{stat}' for cls in ('car','pedestrian')
          for stat in ('positive_sum','negative_sum','positive_count','negative_count')}
    for key in expected:assert torch.equal(actual[key],expected[key]),key
    for key in first:assert torch.equal(first[key].grad,second[key].grad),key


def test_compute_rejects_invalid_configuration():
    outputs,batch,config=complete_fixture();config['loss']['center_positive_normalization']='incorrect'
    with pytest.raises(ValueError,match='center_positive_normalization'):
        losses.compute_losses(outputs,batch,config)
