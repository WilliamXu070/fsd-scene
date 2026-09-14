"""Real-data gradient, optimizer, checkpoint and target-isolation gate."""
import argparse, copy, json, os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import torch
from fsd.runtime import load_config, seed_all, to_device, atomic_json, record_stage
from fsd.engine import make_loader, amp_context
from fsd.model import SceneModel
from fsd.losses import compute_losses


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/baseline.yaml')
    args=parser.parse_args()
    config=load_config(args.config)
    config['train']['workers']=0
    seed_all(42)
    raw=next(iter(make_loader(config,'train')))
    batch=to_device(raw,'cuda')
    model=SceneModel(config).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    model.train()
    before=next(model.image_encoder.parameters()).detach().clone()
    with amp_context(config):
        out=model(batch,refine=True)
        losses=compute_losses(out,batch,config,refine=True)
    losses['total'].backward()
    groups={}
    for name,param in model.named_parameters():
        group=name.split('.')[0]
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(),name
            groups[group]=groups.get(group,0.)+param.grad.float().abs().sum().item()
    assert groups.get('image_encoder',0)>0,'No actual image-encoder gradient'
    assert any(v>0 for k,v in groups.items() if 'refin' in k),'No refinement gradient'
    optimizer.step()
    delta=(next(model.image_encoder.parameters())-before).abs().max().item()
    assert delta>0
    torch.use_deterministic_algorithms(True)
    model.eval()
    with torch.no_grad(),amp_context(config):
        reference=model(batch,refine=True)
        altered=dict(batch)
        for key in ('boxes','labels','road','depth','depth_target','detection_valid','instance_ids'):
            altered.pop(key,None)
        no_targets=model(altered,refine=True)
    for key in ('road_logits','initial_boxes','refined_boxes'):
        assert torch.equal(reference[key],no_targets[key]),f'Target leakage: {key}'
    checkpoint=Path('artifacts/pipeline/checkpoint_smoke.pt')
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict()),checkpoint)
    config2=copy.deepcopy(config)
    config2['model']['pretrained']=False
    reloaded=SceneModel(config2).cuda().eval()
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    reloaded.load_state_dict(state['model'])
    with torch.no_grad(),amp_context(config):
        comparison=reloaded(batch,refine=True)
    parity={key:(reference[key]-comparison[key]).abs().max().item()
            for key in ('road_logits','initial_boxes','refined_boxes')}
    assert max(parity.values())==0,parity
    report=dict(sequence=batch['sequence'],frame=int(batch['frame_id'][0]),
                losses={k:float(v) for k,v in losses.items()},gradient_groups=groups,
                backbone_parameter_delta=delta,reload_max_abs_error=parity,
                target_isolation=True,peak_gpu_mb=torch.cuda.max_memory_allocated()/2**20)
    atomic_json('artifacts/pipeline/smoke.json',report)
    record_stage('05_pipeline','passed',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()

