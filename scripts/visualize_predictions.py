"""Real train/validation component diagnostics; sealed test is never supported.

Example:
 .venv/Scripts/python.exe scripts/visualize_predictions.py --checkpoint artifacts/runs/overfit16-a/learned.pt --split train --indices 0,5,10 --output artifacts/model-visuals/overfit16
Use --config only to select a compatible data manifest/root or fixed render
calibration. --no-ground-truth removes annotation plots and GT-dependent scores.
These slow diagnostic plots are not a latency benchmark or a normal replay feed.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from fsd.data import Kitti360Dataset, collate_fn
from fsd.geometry import box_corners, project_numpy
from fsd.inference import select_detections
from fsd.metrics import Evaluator, depth_metrics
from fsd.model import SceneModel
from fsd.runtime import atomic_json, history_for, load_config, to_device

MODEL_INPUTS = {'images','camera_valid','camera_to_ego','intrinsics','distortion','xi','camera_model',
                'rays','ray_valid','ego_to_world','timestamp','sequence','frame_id'}
CAMERA_NAMES = ['Front left','Front right','Left fisheye','Right fisheye']
BOX_EDGES = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
BOX_COLORS = ['#e3943b','#57b6d5']
ROAD_COLORS = ['#aeb6bd','#73b4d1','#bac3a2']
ROAD_CMAP=ListedColormap(ROAD_COLORS); ROAD_CMAP.set_bad('#eeeeee')
ROAD_NORM=BoundaryNorm([-.5,.5,1.5,2.5],3)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def slug(value):
    return ''.join(c if c.isalnum() or c in '_-' else '_' for c in str(value))


def format_number(value,precision=3):
    return 'unavailable' if value is None else f'{value:.{precision}f}'


def image_array(sample,view):
    return sample['images'][view].permute(1,2,0).numpy().clip(0,1)


def save_figure(figure,path,title,footer):
    figure.suptitle(title,fontsize=11,fontweight='bold')
    figure.get_layout_engine().set(rect=(0,.035,1,.965))
    figure.text(.015,.008,footer,fontsize=8,color='#5b6570')
    figure.savefig(path,dpi=135,facecolor='white',bbox_inches='tight')
    plt.close(figure)


def draw_camera_boxes(ax,boxes,labels,scores,calibration,view,image_size):
    if not len(boxes):return
    corners=box_corners(torch.as_tensor(boxes,dtype=torch.float32)).numpy()
    t=np.linspace(0,1,13)
    edges=np.stack([corners[:,a,None,:]*(1-t[None,:,None])+corners[:,b,None,:]*t[None,:,None] for a,b in BOX_EDGES],axis=1)
    uv,valid,_=project_numpy(edges.reshape(-1,3),calibration,image_size)
    uv=uv[view].reshape(len(boxes),12,13,2);valid=valid[view].reshape(len(boxes),12,13)
    center_uv,center_valid,_=project_numpy(np.asarray(boxes)[:,:3],calibration,image_size)
    for i in range(len(boxes)):
        color=BOX_COLORS[int(labels[i])]
        for edge,ok in zip(uv[i],valid[i]):
            # Sampling along each 3D edge respects fisheye curvature. Masked
            # samples break polylines rather than bridging invalid lens regions.
            ax.plot(np.where(ok,edge[:,0],np.nan),np.where(ok,edge[:,1],np.nan),color=color,linewidth=.8,alpha=.9)
        if center_valid[view,i] and i<20:
            text=('car' if labels[i]==0 else 'person')+(f' {scores[i]:.2f}' if scores is not None else ' GT')
            ax.text(*center_uv[view,i],text,fontsize=5.5,color='white',bbox=dict(facecolor=color,alpha=.8,pad=.8,edgecolor='none'))


def camera_figure(sample,boxes,labels,scores,calibration,path,title,footer):
    fig,axs=plt.subplots(2,2,figsize=(16,6.7),constrained_layout=True)
    for view,ax in enumerate(axs.flat):
        image=image_array(sample,view);ax.imshow(image)
        draw_camera_boxes(ax,boxes,labels,scores,calibration,view,image.shape[:2])
        ax.set_xlim(-.5,image.shape[1]-.5);ax.set_ylim(image.shape[0]-.5,-.5)
        ax.set_title(CAMERA_NAMES[view],fontsize=10);ax.axis('off')
    save_figure(fig,path,title,footer)


def draw_bev_boxes(ax,boxes,labels,style='-',alpha=1):
    for b,cls in zip(boxes,labels):
        c,s=np.cos(b[6]),np.sin(b[6]);local=np.array([[-1,-1],[1,-1],[1,1],[-1,1],[-1,-1]])*b[3:5]/2
        xy=local@np.array([[c,s],[-s,c]])+b[:2]
        color=BOX_COLORS[int(cls)];ax.plot(xy[:,0],xy[:,1],style,color=color,linewidth=.9,alpha=alpha)
        ax.plot([b[0],b[0]+c*b[3]/2],[b[1],b[1]+s*b[3]/2],style,color=color,linewidth=.8,alpha=alpha)


def ground_axes(ax,extent,title):
    ax.set(xlim=extent[:2],ylim=extent[2:],xlabel='Ego x forward (m)',ylabel='Ego y left (m)',title=title)
    ax.set_aspect('equal');ax.scatter([0],[0],s=22,c='black',marker='+',zorder=10)


def panels(sample,outputs,bev_rms,detections,calibration,config,path_prefix,scope,confidence,include_gt,history_count,checkpoint_label):
    generated=[]
    boxes,labels,scores=detections['boxes'],detections['labels'],detections['scores']
    seq,frame_id=sample['sequence'],int(sample['frame_id'])
    context=f'{scope} | {seq} frame {frame_id}\nconfidence >= {confidence:.2f} | {checkpoint_label}'
    footer=f'Predicted proposals only; confidence >= {confidence:.2f}; {len(boxes)} detections after NMS. History frames used: {history_count}. GT is never a model input.'
    path=Path(str(path_prefix)+'_cameras_prediction.png');camera_figure(sample,boxes,labels,scores,calibration,path,context+' | Predicted 3D boxes',footer);generated.append(path.name)
    truth_boxes=sample['boxes'].numpy() if include_gt else None
    truth_labels=sample['labels'].numpy() if include_gt else None
    if include_gt:
        path=Path(str(path_prefix)+'_cameras_GT_DIAGNOSTIC.png');camera_figure(sample,truth_boxes,truth_labels,None,calibration,path,context+' | Ground truth diagnostic (separate)',f'Annotated supported boxes only; {len(truth_boxes)} targets. This panel is not an inference result.');generated.append(path.name)
    road_prob=outputs['road_logits'][0].float().softmax(0).cpu().numpy();road_pred=road_prob.argmax(0)
    heat=outputs['center_logits'][0].float().sigmoid().cpu().numpy()
    minimum=config['model'].get('bev_min',-40.);extent_size=config['model'].get('bev_step',.5)*road_pred.shape[0]
    extent=[minimum,minimum+extent_size,minimum,minimum+extent_size]
    cols=3 if include_gt else 2
    fig,axs=plt.subplots(2,cols,figsize=(cols*4.9,9.5),constrained_layout=True)
    axs=axs.ravel()
    axs[0].imshow(road_pred,origin='lower',extent=extent,cmap=ROAD_CMAP,norm=ROAD_NORM,interpolation='nearest')
    draw_bev_boxes(axs[0],boxes,labels);ground_axes(axs[0],extent,'Predicted road classes + predicted boxes')
    feature=np.asarray(bev_rms);vmax=max(float(np.percentile(feature,99)),1e-6)
    im=axs[1].imshow(feature,origin='lower',extent=extent,cmap='magma',vmin=0,vmax=vmax)
    ground_axes(axs[1],extent,'BEV encoder feature RMS (99th-percentile display cap)');fig.colorbar(im,ax=axs[1],shrink=.65,label='Channel RMS activation')
    last=2
    if include_gt:
        target=sample['road'].numpy();masked=np.ma.masked_where(target<0,target)
        axs[2].imshow(masked,origin='lower',extent=extent,cmap=ROAD_CMAP,norm=ROAD_NORM,interpolation='nearest')
        draw_bev_boxes(axs[2],truth_boxes,truth_labels,style='--');ground_axes(axs[2],extent,'GT diagnostic: supported road cells / boxes');last=3
    for cls in range(2):
        ax=axs[last+cls];im=ax.imshow(heat[cls],origin='lower',extent=extent,cmap='viridis',vmin=0,vmax=1)
        ground_axes(ax,extent,('Car' if cls==0 else 'Person')+' center heatmap');fig.colorbar(im,ax=ax,shrink=.65,label='Center confidence')
        if include_gt:
            selected=truth_boxes[truth_labels==cls];ax.scatter(selected[:,0],selected[:,1],s=35,marker='+',c='#ff824f',linewidths=1,label='GT centers (diagnostic)');ax.legend(fontsize=7,loc='upper right')
    if include_gt:
        ax=axs[-1];known=sample['road'].numpy()>=0
        mismatch=np.ma.masked_where(~known,(road_pred!=sample['road'].numpy()).astype(float))
        cmap=ListedColormap(['#83b394','#db8678']);cmap.set_bad('#eeeeee')
        ax.imshow(mismatch,origin='lower',extent=extent,cmap=cmap,vmin=0,vmax=1);ground_axes(ax,extent,'Road error only on labeled cells (green correct)')
    path=Path(str(path_prefix)+'_bev_components.png');save_figure(fig,path,context+' | BEV components',footer+' Road colors: gray road, blue sidewalk, muted green other ground; light gray GT background unknown.');generated.append(path.name)
    bins=torch.linspace(config['model'].get('depth_min',1.),config['model'].get('depth_max',80.),outputs['depth_logits'].shape[2],device=outputs['depth_logits'].device)
    depth=(outputs['depth_logits'][0].float().softmax(1)*bins[None,:,None,None]).sum(1).cpu().numpy()
    depth_columns=3 if include_gt else 2
    fig,axs=plt.subplots(4,depth_columns,figsize=(depth_columns*5.3,9),constrained_layout=True)
    for view in range(4):
        axs[view,0].imshow(image_array(sample,view));axs[view,0].set_title(CAMERA_NAMES[view],fontsize=9)
        cmap=plt.get_cmap('turbo').copy();cmap.set_bad('#eeeeee')
        im=axs[view,1].imshow(depth[view],cmap=cmap,vmin=float(bins[0]),vmax=float(bins[-1]),aspect='auto');axs[view,1].set_title('Predicted expected radial depth (m)',fontsize=9)
        if include_gt:
            targets=sample['depth_target'][view].numpy();gt_depth=bins.cpu().numpy()[np.maximum(targets,0)]
            mask=(targets>=0)&sample['ray_valid'][view].numpy()
            axs[view,2].imshow(np.ma.masked_where(~mask,gt_depth),cmap=cmap,vmin=float(bins[0]),vmax=float(bins[-1]),aspect='auto');axs[view,2].set_title('Sparse LiDAR bin-center GT diagnostic',fontsize=9)
        for ax in axs[view]:ax.axis('off')
    fig.colorbar(im,ax=axs[:,1:].ravel().tolist(),shrink=.65,label='Radial range (m)')
    path=Path(str(path_prefix)+'_depth_components.png');save_figure(fig,path,context+' | Camera depth components','Fixed common depth scale; sparse blank targets are unknown. Predicted depth is a bin-distribution expectation, not camera-axis z depth.');generated.append(path.name)
    report={'sequence':seq,'frame_id':frame_id,'scope':scope,'confidence':confidence,'detections':len(boxes),'history_frames_used':history_count,
            'bev_feature_rms_mean':float(feature.mean()),'bev_feature_rms_max':float(feature.max()),'panels':generated}
    if include_gt:
        evaluator=Evaluator(bev_min=minimum,bev_step=config['model'].get('bev_step',.5))
        evaluator.update(boxes,scores,labels,truth_boxes,truth_labels,road_pred=road_pred,road_target=sample['road'],detection_valid=sample['detection_valid'])
        report['frame_metrics']=evaluator.compute()
        targets=sample['depth_target'].numpy();gt_depth=bins.cpu().numpy()[np.maximum(targets,0)]
        report['depth']=depth_metrics(depth,gt_depth,(targets>=0)&sample['ray_valid'].numpy())
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--config',help='Compatible configuration; only data and render sections override checkpoint')
    parser.add_argument('--split',choices=['train','val'],default='val')
    parser.add_argument('--indices',help='Comma-separated ordered dataset indices, e.g. 0,5,10')
    parser.add_argument('--max-frames',type=int,default=3)
    parser.add_argument('--confidence',type=float,default=.3)
    parser.add_argument('--output',default='artifacts/model-visuals/current')
    parser.add_argument('--no-ground-truth',action='store_true')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args()
    if args.max_frames<1 or not 0<=args.confidence<=1:parser.error('max-frames must be positive and confidence must be within0..1')
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    config=copy.deepcopy(checkpoint['config'])
    if args.config:
        requested=load_config(args.config)
        expected=copy.deepcopy(config['model']);observed=copy.deepcopy(requested.get('model',expected))
        expected.pop('pretrained',None);observed.pop('pretrained',None)
        if expected!=observed:parser.error('Configuration model differs from checkpoint; only data/render overrides are supported')
        for section in ('data','render'):
            if section in requested:config[section]=copy.deepcopy(requested[section])
    config['model']['pretrained']=False
    dataset=Kitti360Dataset(config['data']['root'],args.split,config)
    chosen=sorted(set(int(x) for x in args.indices.split(','))) if args.indices else list(range(min(args.max_frames,len(dataset))))
    chosen=chosen[:args.max_frames]
    if not chosen or min(chosen)<0 or max(chosen)>=len(dataset):parser.error('indices must name existing selected-split frames')
    for idx in chosen:
        row=dataset.samples[idx];prefix=f'{args.split}_{slug(row["sequence"])}_{row["frame_id"]:010d}'
        if any(root.glob(prefix+'*.png')):parser.error('Diagnostic image output already exists; select a fresh directory')
    device=torch.device(args.device)
    model=SceneModel(config).to(device).eval();model.load_state_dict(checkpoint['model'])
    captured={}
    def capture_bev(module,inputs,output):captured['bev_rms']=output[0].detach().float().square().mean(0).sqrt().cpu().numpy()
    hook=model.bev_encoder.register_forward_hook(capture_bev)
    refine=checkpoint.get('stage')=='b';history=[];reports=[]
    label=f'checkpoint {Path(args.checkpoint).name}; stage {checkpoint.get("stage","unknown")}; updates {checkpoint.get("updates","unverified")}'
    scope='Training-fit diagnostic (not held-out)' if args.split=='train' else 'Validation diagnostic (not sealed test)'
    try:
        # Walk chronologically to preserve the same temporal state as engine evaluation.
        # Only selected frames are drawn; no latency is inferred from this script.
        for idx in (range(max(chosen)+1) if refine else chosen):
            sample=dataset[idx]
            batch=to_device(collate_fn([sample]),device)
            model_batch={key:value for key,value in batch.items() if key in MODEL_INPUTS}
            history=history_for(history,model_batch)
            mixed=args.device=='cuda' and config.get('runtime',{}).get('mixed_precision',True)
            precision_context=torch.autocast('cuda',dtype=torch.float16) if mixed else nullcontext()
            with torch.inference_mode(),precision_context:
                outputs=model(model_batch,history,refine=refine)
            if idx in chosen:
                for key in ('road_logits','depth_logits','center_logits','initial_boxes','refined_boxes'):
                    if not torch.isfinite(outputs[key]).all():raise FloatingPointError(f'Nonfinite {key} at dataset index {idx}')
                detections=select_detections(outputs,threshold=args.confidence,refined=refine)
                prefix=root/f'{args.split}_{slug(sample["sequence"])}_{int(sample["frame_id"]):010d}'
                report=panels(sample,outputs,captured['bev_rms'],detections,dataset.calibration,config,prefix,scope,args.confidence,not args.no_ground_truth,len(history) if refine else 0,label)
                report['dataset_index']=idx;reports.append(report)
                atomic_json(root/(prefix.name+'_diagnostic.json'),report)
                print(json.dumps({'event':'visualized','index':idx,'frame':int(sample['frame_id']),'panels':report['panels']}),flush=True)
            if refine:history=(history+[outputs['state']])[-config['model'].get('history_length',3):]
    finally:hook.remove()
    manifest=Path(config['data']['root'])/config['data'].get('manifest','manifest.json')
    atomic_json(root/'report.json',{'scope':scope,'split':args.split,'checkpoint':str(Path(args.checkpoint).resolve()),
                 'checkpoint_sha256':sha256(args.checkpoint),'manifest':str(manifest),'manifest_sha256':sha256(manifest),
                 'ground_truth_diagnostics':not args.no_ground_truth,'model_input_keys':sorted(MODEL_INPUTS),
                 'device':args.device,'precision':('FP16 CUDA autocast' if args.device=='cuda' and config.get('runtime',{}).get('mixed_precision',True) else 'FP32'),'confidence':args.confidence,'history_policy':('chronological replay from selected split start; history resets by sequence/time gap' if refine else 'stage A: temporal refinement disabled; each selected frame independent'),
                 'latency_claim':'none; plotting and hooks make this unsuitable for timing','frames':reports,'created':time.time()})

if __name__=='__main__':main()
