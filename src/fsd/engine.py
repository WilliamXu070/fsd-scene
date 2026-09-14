"""Chronological training and evaluation with explicit temporal-state boundaries."""
from __future__ import annotations
import copy, json, math, time
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from fsd.runtime import (append_jsonl, atomic_json, capture_provenance, history_for, record_stage, restore_rng,
                         rng_state, save_checkpoint, seed_all, to_device)


def amp_context(config):
    return torch.autocast('cuda', dtype=torch.float16) if config['runtime'].get('mixed_precision', True) else nullcontext()


def make_loader(config, split, indices=None):
    from fsd.data import Kitti360Dataset, collate_fn
    dataset = Kitti360Dataset(config['data']['root'], split, config)
    if indices is not None:
        dataset = Subset(dataset, indices)
    workers = config['train'].get('workers', 0)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers,
                      pin_memory=True, collate_fn=collate_fn, persistent_workers=workers > 0,
                      generator=torch.Generator().manual_seed(config['train'].get('seed',42)))


@torch.inference_mode()
def evaluate(model, loader, config, refine=True, diagnostic=None, output_path=None):
    from fsd.metrics import Evaluator, depth_metrics
    from fsd.inference import SceneTracker, select_detections
    if diagnostic=='initial_only':
        refine=False
    model.eval()
    evaluator, tracker = Evaluator(bev_min=config['model'].get('bev_min',-40.),
                                 bev_step=config['model'].get('bev_step',.5)), SceneTracker()
    from fsd.losses import compute_losses
    from fsd.geometry import ground_fov_mask
    validation_losses, depth_reports, frame_count = {}, [], 0
    history, records = [], []
    device = config['runtime'].get('device', 'cuda')
    for step, raw in enumerate(loader):
        batch = to_device(raw, device)
        history = history_for(history, batch)
        if diagnostic == 'blank':
            batch['images'] = torch.zeros_like(batch['images'])
        elif diagnostic == 'shuffled':
            batch['images'] = batch['images'].roll(1, dims=1)
        used_history = [] if diagnostic == 'empty_history' else history
        if diagnostic == 'mismatched_history' and history:
            used_history = [dict(h, features=h.get('features', h.get('proposal_features')).roll(7, dims=1)) for h in history]
        torch.cuda.synchronize()
        start = time.perf_counter()
        with amp_context(config):
            out = model(batch, used_history, refine=refine)
        torch.cuda.synchronize()
        inference_ms = (time.perf_counter()-start)*1000
        validation = compute_losses(out, batch, config, refine=refine)
        for key,value in validation.items():
            validation_losses[key] = validation_losses.get(key,0.) + float(value)
        frame_count += 1
        if 'depth_target' in raw:
            bins = torch.linspace(config['model']['depth_min'], config['model']['depth_max'],
                                  config['model']['depth_bins'], device='cuda')
            radial = (out['depth_logits'].float().softmax(2)*bins[None,None,:,None,None]).sum(2)
            target = batch['depth_target'].long()
            valid_depth = target >= 0
            true_radial = bins[target.clamp_min(0)]
            depth_report=depth_metrics(radial[valid_depth].cpu(), true_radial[valid_depth].cpu())
            if depth_report['count'] and any(depth_report[k] is None for k in ('mae_m','rmse_m','abs_rel','delta_1')):
                raise FloatingPointError(f'Invalid depth predictions on {batch["sequence"]} frame {int(batch["frame_id"][0])}: {depth_report}')
            depth_reports.append(depth_report)
        det = select_detections(out, threshold=config.get('evaluation', {}).get('metric_threshold', 0.01), refined=refine)
        timestamp = float(batch['timestamp'][0])
        sequence = batch['sequence'][0]
        pose = batch['ego_to_world'][0].cpu().numpy()
        track_ids = tracker.update(det, timestamp, pose, sequence)
        road_probs = out['road_logits'][0].float().softmax(0)
        road = road_probs.argmax(0).cpu().numpy()
        metadata = dict(sequence=sequence, timestamp=timestamp,
                        gt_instance_ids=raw['instance_ids'][0], ego_to_world=pose)
        evaluator.update(det['boxes'], det['scores'], det['labels'], raw['boxes'][0], raw['labels'][0],
                         road_pred=road, road_target=raw['road'][0],
                         detection_valid=raw['detection_valid'][0], frame_metadata=metadata)
        if output_path:
            # Fixed training-derived mount calibration and camera FOV; no target masks.
            ground_z=config.get('render',{}).get('ground_z_m',-.9343111743493324)
            road_visible=ground_fov_mask(batch,ground_z=ground_z,bev_size=config['model']['bev_size'],
                bev_min=config['model']['bev_min'],bev_step=config['model']['bev_step'])
            # No target-derived masks or boxes enter the replay file.
            record = dict(sequence=sequence, timestamp=timestamp, frame_id=int(batch['frame_id'][0]),
                          boxes=det['boxes'].tolist(), scores=det['scores'].tolist(), labels=det['labels'].tolist(),
                          track_ids=track_ids.tolist(), road=road.tolist(),
                          road_visible=road_visible[0].cpu().tolist(),road_ground_z_m=ground_z,
                          road_confidence=road_probs.max(0).values.cpu().numpy().round(3).tolist(),
                          ego_to_world=pose.tolist(), inference_ms=inference_ms)
            if 'image_paths' in raw:
                record['image_paths'] = raw['image_paths'][0]
            append_jsonl(output_path, record)
        history.append(out['state'])
        history = history[-config['model'].get('history_length', 3):]
        if step % 100 == 0:
            print(json.dumps(dict(event='evaluate',step=step,total=len(loader),diagnostic=diagnostic)), flush=True)
    result=evaluator.compute()
    result['validation_losses']={k:v/max(frame_count,1) for k,v in validation_losses.items()}
    count=sum(d['count'] for d in depth_reports)
    result['depth']={'count':count,'target_convention':'quantized LiDAR radial range; bin centers, not raw range'}
    if count:
        for key in ('mae_m','abs_rel','delta_1'):
            result['depth'][key]=sum(d[key]*d['count'] for d in depth_reports if d['count'])/count
        result['depth']['rmse_m']=(sum(d['rmse_m']**2*d['count'] for d in depth_reports if d['count'])/count)**.5
    result['score_protocol']={'metric_confidence_cutoff':config.get('evaluation',{}).get('metric_threshold',.01),'nms_bev_iou':.5,'max_detections':100}
    return result


def selection_score(metrics):
    """Equal weight to road/sidewalk IoU and car/pedestrian BEV AP50."""
    road = metrics.get('road',{}).get('iou',{})
    objects = metrics.get('objects',{}).get('bev_ap50',{}).get('classes',{})
    values = [road.get('road'),road.get('sidewalk'),
              objects.get('car',{}).get('ap'),objects.get('pedestrian',{}).get('ap')]
    if any(v is None or not math.isfinite(v) for v in values):
        return None
    return float(np.mean(values))

def train(config, run_dir, stage='a', resume=None, max_updates=None, indices=None, initialize=None):
    from fsd.model import SceneModel
    from fsd.losses import compute_losses
    run = Path(run_dir)
    if initialize and resume:
        raise ValueError('Choose fresh initialization or resume, never both')
    if initialize:
        if run.exists() and any(entry.suffix != '.log' for entry in run.iterdir()):
            raise ValueError('Fresh initialization requires an uninitialized run directory; resume existing progress')
        experiment = config.get('experiment', {})
        if not experiment.get('id') or Path(experiment.get('initialization', '')).resolve() != Path(initialize).resolve():
            raise ValueError('Fresh initialization must be recorded in the experiment configuration')
    run.mkdir(parents=True, exist_ok=True)
    atomic_json(run/'config.json', config)
    if not (run/'provenance.json').exists():
        capture_provenance(run, config)
    if initialize:
        from fsd.sealing import sha256
        atomic_json(run/'initialization.json', dict(checkpoint=str(initialize),
            checkpoint_sha256=sha256(initialize), mode='weights_only',
            optimizer_scheduler_scaler_rng_history_progress='fresh from experiment configuration'))
    if resume:
        import sys
        from fsd.sealing import sha256
        resume_dir=run/'resumes'/f'{len(list((run/"resumes").glob("*")))+1:03d}'
        capture_provenance(resume_dir,config)
        atomic_json(resume_dir/'invocation.json',dict(checkpoint=str(resume),
                    checkpoint_sha256=sha256(resume),argv=sys.argv))
    seed_all(config['train'].get('seed',42))
    loader = make_loader(config, 'train', indices)
    assert len(loader) > 0, 'No usable real training samples'
    model = SceneModel(config).to(config['runtime'].get('device','cuda'))
    from fsd.training_policy import configure_trainable_scope, base_state_digest
    policy = configure_trainable_scope(model, stage, config['train'].get('trainable_scope', 'all'))
    atomic_json(run/'training_policy.json', policy)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config['train']['lr'],
                                 weight_decay=config['train']['weight_decay'])
    epochs = config['train']['epochs_a' if stage == 'a' else 'epochs_b']
    epoch_limit = int(config['train'].get('epoch_limit', epochs))
    if not 1 <= epoch_limit <= epochs:
        raise ValueError('epoch_limit must be within the configured scheduler horizon')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    from fsd.sealing import sha256
    manifest_path=Path(config['data']['root'])/config['data'].get('manifest','manifest.json')
    manifest_hash=sha256(manifest_path)
    scaler = torch.amp.GradScaler('cuda', enabled=config['runtime'].get('mixed_precision',True),
                                  init_scale=config['runtime'].get('amp_init_scale',1024.))
    overflow_count, consecutive_overflows = 0, 0
    start_epoch, skip_step, updates, best, stale = 0, 0, 0, -math.inf, 0
    history = []
    if resume or initialize:
        checkpoint = torch.load(resume or initialize, map_location='cpu', weights_only=False)
        previous_config=copy.deepcopy(checkpoint['config'])
        for section in (('data','model') if initialize else ('data','model','loss')):
            if previous_config.get(section,{}) != config.get(section,{}):
                raise ValueError(f'Resume changes immutable {section} configuration; create a recorded experiment')
        if checkpoint.get('manifest_sha256',manifest_hash)!=manifest_hash:
            raise ValueError('Dataset manifest contents changed since checkpoint')
        model.load_state_dict(checkpoint['model'])
        if resume and checkpoint.get('stage') == stage:
            if previous_config['train'].get('trainable_scope', 'all') != policy['scope']:
                raise ValueError('Resume cannot change trainable scope; initialize a new stage experiment')
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            scaler.load_state_dict(checkpoint['scaler'])
            restore_rng(checkpoint['rng'])
            start_epoch = checkpoint['epoch']
            skip_step = checkpoint.get('next_step', 0)
            updates = checkpoint['updates']
            best, stale = checkpoint.get('best',-math.inf), checkpoint.get('stale',0)
            history = to_device(checkpoint.get('history',[]), 'cuda')
            overflow_count = checkpoint.get('overflow_count',0)
    frozen_base = base_state_digest(model) if policy['scope'] == 'refiner' else None
    if frozen_base is not None:
        if resume and checkpoint.get('stage') == stage and checkpoint.get('frozen_base_sha256') != frozen_base:
            raise ValueError('Frozen base differs from the resumed experiment checkpoint contract')
        atomic_json(run/'frozen_base.json', {'sha256': frozen_base, 'includes': 'All non-refiner parameters and buffers',
                    'initialization': config.get('experiment', {}).get('initialization', str(initialize or resume)),
                    'resume_checkpoint': str(resume), 'checked_each_epoch': True})
    refine = stage == 'b'
    accumulate = config['train'].get('accumulate',4)
    optimizer.zero_grad(set_to_none=True)
    def checkpoint_at(epoch, next_step, path='latest.pt'):
        save_checkpoint(run/path, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), rng=rng_state(),
                        config=config, stage=stage, epoch=epoch, next_step=next_step, updates=updates,
                        best=best, stale=stale, history=history, overflow_count=overflow_count,manifest_sha256=manifest_hash,
                        frozen_base_sha256=frozen_base)
    completed_epochs = start_epoch
    for epoch in range(start_epoch, epoch_limit):
        model.train()
        if not skip_step:
            history=[]
        torch.cuda.reset_peak_memory_stats()
        epoch_start = time.perf_counter()
        totals, count = {}, 0
        for step, raw in enumerate(loader):
            if step < skip_step:
                continue
            batch = to_device(raw, config['runtime'].get('device','cuda'))
            history=history_for(history,batch)
            with amp_context(config):
                outputs=model(batch,history if refine else None,refine=refine)
                losses=compute_losses(outputs,batch,config,refine=refine)
                # The last accumulation group may have fewer microbatches.
                group_size=min(accumulate,len(loader)-(step//accumulate)*accumulate)
                loss=losses['total']/group_size
            if not torch.isfinite(loss):
                atomic_json(run/'failure.json',dict(epoch=epoch,step=step,reason='nonfinite_loss',
                            frame=int(batch['frame_id'][0]), sequence=batch['sequence']))
                raise FloatingPointError('Nonfinite loss; inspect failure.json and last checkpoint')
            scaler.scale(loss).backward()
            count += 1
            for key,value in losses.items():
                totals[key]=totals.get(key,0.)+float(value.detach())
            if refine:
                history=(history+[outputs['state']])[-config['model'].get('history_length',3):]
            if (step+1)%accumulate==0 or step+1==len(loader):
                scaler.unscale_(optimizer)
                # Check before clipping: inf clipping can turn otherwise finite entries into NaNs.
                parameter_norms={name:p.grad.detach().float().norm() for name,p in model.named_parameters()
                                 if p.grad is not None}
                grad_norm=torch.linalg.vector_norm(torch.stack(list(parameter_norms.values())))
                if not torch.isfinite(grad_norm):
                    previous_scale=scaler.get_scale()
                    if not scaler.is_enabled():
                        raise FloatingPointError('Nonfinite FP32 gradients')
                    scaler.step(optimizer)  # GradScaler skips the update after unscale_ found inf.
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    overflow_count+=1
                    consecutive_overflows+=1
                    event=dict(event='amp_overflow',epoch=epoch+1,step=step+1,
                               scale_before=previous_scale,scale_after=scaler.get_scale(),
                               total_skips=overflow_count,consecutive_skips=consecutive_overflows)
                    append_jsonl(run/'train.jsonl',event)
                    print(json.dumps(event),flush=True)
                    if consecutive_overflows>=8 or scaler.get_scale()<1.:
                        atomic_json(run/'failure.json',event)
                        raise FloatingPointError('Persistent gradient overflow; inspect saved evidence')
                    continue
                consecutive_overflows=0
                torch.nn.utils.clip_grad_norm_(model.parameters(),config['train'].get('grad_clip',5.))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                updates+=1
                if updates%10==0:
                    event=dict(event='train',stage=stage,epoch=epoch+1,step=step+1,total_steps=len(loader),
                               updates=updates,losses={k:v/count for k,v in totals.items()},
                               grad_norm=float(grad_norm), elapsed_s=time.perf_counter()-epoch_start,
                               allocated_mb=torch.cuda.memory_allocated()/2**20,amp_scale=scaler.get_scale(),
                               amp_overflow_skips=overflow_count,
                               gradient_groups={group:float(torch.stack([value for name,value in parameter_norms.items()
                                    if name.split('.')[0]==group]).norm())
                                    for group in sorted({name.split('.')[0] for name in parameter_norms})})
                    append_jsonl(run/'train.jsonl',event)
                    print(json.dumps(event),flush=True)
                if updates%50==0:
                    checkpoint_at(epoch,step+1)
                if max_updates and updates>=max_updates:
                    checkpoint_at(epoch,step+1)
                    if step+1<len(loader):
                        return model  # Preserve epoch, scheduler and history at this exact boundary.
                    break
        skip_step=0
        completed_epochs = epoch + 1
        scheduler.step()
        history=[]
        checkpoint_at(epoch+1,0)  # Preserve learned state even if evaluator/logging fails.
        val_loader = make_loader(config,'train',indices) if indices is not None else make_loader(config,'val')
        metrics=evaluate(model,val_loader,config,refine=refine)
        if frozen_base is not None and base_state_digest(model) != frozen_base:
            atomic_json(run/'failure.json', {'reason': 'frozen_base_changed', 'epoch': epoch+1})
            raise RuntimeError('Frozen scene parameters or buffers changed during the refiner experiment')
        score=selection_score(metrics)
        if score is None:
            atomic_json(run/'metric_schema_failure.json',metrics)
            raise ValueError('Metric selection schema not integrated; do not select checkpoint blindly')
        improved=score>best+1e-4
        best=max(best,score)
        stale=0 if improved else stale+1
        event=dict(event='epoch',epoch=epoch+1,stage=stage,score=score,best=best,
                   epoch_seconds=time.perf_counter()-epoch_start,metrics=metrics,
                   peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
                   peak_reserved_mb=torch.cuda.max_memory_reserved()/2**20)
        try:
            append_jsonl(run/'epochs.jsonl',event)
        except ValueError:
            (run/'invalid_metrics.txt').write_text(repr(event),encoding='utf8')
            raise
        print(json.dumps(event),flush=True)
        checkpoint_at(epoch+1,0)
        if improved:
            checkpoint_at(epoch+1,0,'best.pt')
        if indices is not None:
            classes=metrics['objects']['bev_ap50']['classes']
            gate=dict(road_iou=metrics['road']['iou'].get('road'),
                      car_recall=classes['car'].get('recall'),pedestrian_recall=classes['pedestrian'].get('recall'),
                      updates=updates,limit=2000,stage=stage,scope='training subset learnability only')
            passed=all(gate[key] is not None and gate[key]>=.9
                       for key in ('road_iou','car_recall','pedestrian_recall'))
            gate['passed']=passed
            atomic_json(run/'learnability.json',gate)
            if passed:
                checkpoint_at(epoch+1,0,'learned.pt')
                record_stage(f'07_learnability_{stage}','passed',dict(run=str(run),**gate))
                return model
        if max_updates and updates>=max_updates:
            if indices is not None:
                record_stage(f'07_learnability_{stage}','failed',dict(run=str(run),**gate))
            return model
        if epoch+1>=config['train'].get('min_epochs',8) and stale>=config['train'].get('patience',6):
            break
    evidence_stage = (f'07_learnability_{stage}' if indices is not None else
                      config.get('experiment', {}).get('stage_record', f'08_train_{stage}'))
    record_stage(evidence_stage, 'failed' if indices is not None else 'passed',
                 dict(run=str(run),best=best,updates=updates,completed_epochs=completed_epochs,
                      epoch_limit=epoch_limit,scheduler_horizon=epochs,trainable_scope=policy['scope'],
                      frozen_base_sha256=frozen_base))
    return model


