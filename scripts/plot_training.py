"""Plot finite, recorded training/evaluation evidence from live JSONL logs."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

COLORS=['#23779a','#d77a38','#329775','#ac5790','#6d67a7','#869337','#777777']
LOSS_GROUPS={
    'Depth and road':('depth','road_ce','road_dice','road'),
    'Initial detection':('center','initial_position','initial_dimensions','initial_yaw','initial_total'),
    'Object refinement':('refined_class','refined_position','refined_dimensions','refined_yaw','refined_total'),
    'Task totals':('depth','road','initial_total','refined_total','total'),
}
LOSS_LABELS={'depth':'Depth CE','road_ce':'Road CE','road_dice':'Road Dice','road':'Road total','center':'Center focal',
 'initial_position':'Position','initial_dimensions':'Dimensions','initial_yaw':'Heading','initial_total':'Initial total',
 'refined_class':'Classification','refined_position':'Position','refined_dimensions':'Dimensions',
 'refined_yaw':'Heading','refined_total':'Refined total','total':'Combined loss'}


def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def get(record,path):
    for key in path.split('.'):
        if not isinstance(record,dict):return None
        record=record.get(key)
    return record


def read_jsonl_snapshot(path):
    """Read once; a trailing record without newline is not committed by the logger.

    An invalid newline-terminated record is a real corruption error, not a partial
    writer append. It is deliberately reported instead of silently skipped.
    """
    path=Path(path)
    if not path.is_file():return [],dict(path=str(path),exists=False,rows=0,ignored_tail_bytes=0)
    raw=path.read_bytes();last_newline=raw.rfind(b'\n');complete=raw[:last_newline+1]
    tail=raw[last_newline+1:];rows=[]
    for number,line in enumerate(complete.splitlines(),1):
        if not line.strip():continue
        try:record=json.loads(line)
        except (ValueError,UnicodeError) as error:raise ValueError(f'{path}: malformed complete JSON record at line {number}') from error
        if not isinstance(record,dict):raise ValueError(f'{path}: line {number} is not an event object')
        rows.append(record)
    return rows,dict(path=str(path.resolve()),exists=True,snapshot_bytes=len(raw),
                     sha256=hashlib.sha256(raw).hexdigest(),rows=len(rows),ignored_tail_bytes=len(tail))


def nonfinite_paths(record,prefix=''):
    result=[]
    if isinstance(record,dict):
        for key,value in record.items():result.extend(nonfinite_paths(value,f'{prefix}.{key}' if prefix else key))
    elif isinstance(record,list):
        for i,value in enumerate(record):result.extend(nonfinite_paths(value,f'{prefix}[{i}]'))
    elif isinstance(record,float) and not math.isfinite(record):result.append(prefix)
    return result


def prepare(run,evaluation_scope='auto'):
    run=Path(run)
    train_rows,train_source=read_jsonl_snapshot(run/'train.jsonl')
    epoch_rows,epoch_source=read_jsonl_snapshot(run/'epochs.jsonl')
    train=[r for r in train_rows if r.get('event')=='train' and isinstance(r.get('losses'),dict)]
    overflow=[r for r in train_rows if r.get('event')=='amp_overflow']
    epochs=[r for r in epoch_rows if r.get('event')=='epoch']
    config={}
    if (run/'config.json').exists():config=json.loads((run/'config.json').read_text(encoding='utf-8-sig'))
    if evaluation_scope=='auto':
        evaluation_scope='training-fit' if (run/'learnability.json').exists() else 'unrecorded'
    scope={'training-fit':'Training-fit epoch evaluation - no held-out claim',
           'validation':'Validation epoch evaluation - not final test',
           'unrecorded':'Epoch evaluation - split not recorded in log'}[evaluation_scope]
    last_logs={}
    for row in train:
        epoch=row.get('epoch');stage=row.get('stage')
        if finite(epoch):last_logs[(stage,epoch)]=row
    completed={(r.get('stage'),r.get('epoch')) for r in epochs}
    coverage=[]
    for (stage,epoch),row in sorted(last_logs.items(),key=lambda pair:(str(pair[0][0]),pair[0][1])):
        step,total=row.get('step'),row.get('total_steps')
        coverage.append(dict(stage=stage,epoch=epoch,last_logged_step=step,total_steps=total,
                             epoch_evaluation_recorded=(stage,epoch) in completed,
                             log_at_final_step=finite(step) and finite(total) and step==total))
    invalid=[]
    for source,rows in (('train',train_rows),('epochs',epoch_rows)):
        for i,row in enumerate(rows):
            paths=nonfinite_paths(row)
            if paths:invalid.append(dict(source=source,record_index=i,paths=paths))
    summary=dict(run=str(run.resolve()),created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                 scope=scope,scope_argument=evaluation_scope,sources={'train':train_source,'epochs':epoch_source},
                 train_loss_records=len(train),epoch_records=len(epochs),amp_overflow_events=len(overflow),
                 train_event_counts=dict(Counter(r.get('event','missing') for r in train_rows)),
                 epoch_event_counts=dict(Counter(r.get('event','missing') for r in epoch_rows)),
                 last_logged_step_per_epoch=coverage,nonfinite_record_fields=invalid,
                 last_logged_optimizer_update=max((r['updates'] for r in train if finite(r.get('updates'))),default=None),
                 training_loss_interpretation='Within-epoch logged means over processed microbatches before that log; on resume these can cover only the resumed part. They are not reconstructed full-epoch means. Finite forward losses from skipped AMP attempts may contribute to the running mean.',
                 missing_evidence_policy='Absent metrics remain absent; nonfinite values become visible gaps. An unterminated final JSONL record is ignored, while a malformed complete record fails explicitly.',
                 epoch_time_interpretation='Logged epoch_seconds includes training and epoch evaluation. A resumed partial epoch can omit earlier process time; this is not reconstructed total training time.')
    return train,epochs,overflow,config,summary


def series(ax,rows,xkey,ypath,label,color,style='-',scale=1.):
    points=[(get(row,xkey),get(row,ypath)) for row in rows if finite(get(row,xkey))]
    if not points or not any(finite(y) for _,y in points):return False
    ax.plot([x for x,_ in points],[y/scale if finite(y) else float('nan') for _,y in points],
            label=label,color=color,linestyle=style,linewidth=1.55,marker='o' if xkey=='epoch' else None,
            markersize=3.2 if xkey=='epoch' else 0)
    return True


def style_axis(ax,title,xlabel,ylabel,available=True,fraction=False,symlog=False):
    ax.set(title=title,xlabel=xlabel,ylabel=ylabel)
    ax.grid(True,color='#dfe5e8',linewidth=.65,alpha=.8)
    ax.spines[['top','right']].set_visible(False)
    if xlabel=='Epoch':ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if fraction:ax.set_ylim(-.035,1.04)
    if symlog and available:
        ax.set_yscale('symlog',linthresh=1e-3)
        values=[float(y) for line in ax.lines for y in line.get_ydata() if finite(y)]
        if values and min(values)>=0:
            low,high=min(values),max(values)
            ax.set_ylim(low*.8 if low>0 else 0,high*1.25 if high>0 else 1.)
    if available:
        handles,labels=ax.get_legend_handles_labels()
        if handles:ax.legend(handles,labels,loc='best',framealpha=.94,fontsize=8)
    else:ax.text(.5,.5,'Not present in these logs',ha='center',va='center',transform=ax.transAxes,color='#687881')


def save(fig,path,title,footer):
    fig.suptitle(title,x=.06,y=.98,ha='left',fontsize=15,fontweight='bold',color='#283d49')
    fig.text(.06,.018,footer,ha='left',va='bottom',fontsize=8,color='#596b75')
    fig.tight_layout(rect=(.035,.075,.985,.935),h_pad=2.2,w_pad=2.3)
    fig.savefig(path,dpi=150,facecolor='white')
    plt.close(fig)


def loss_panels(rows,validation=False):
    fig,axes=plt.subplots(2,2,figsize=(13.5,8.6))
    prefix='metrics.validation_losses' if validation else 'losses'
    stages={row.get('stage') for row in rows}
    for ax,(group,keys) in zip(axes.flat,LOSS_GROUPS.items()):
        if group=='Object refinement' and stages=={'a'}:
            ax.set_title(group)
            ax.text(.5,.5,'Stage A: refinement is disabled\nIts recorded loss terms are zero.',ha='center',va='center',transform=ax.transAxes,color='#687881')
            ax.set_axis_off()
            continue
        available=False
        for i,key in enumerate(keys):
            if stages=={'a'} and key=='refined_total':continue
            available=series(ax,rows,'epoch' if validation else 'updates',f'{prefix}.{key}',LOSS_LABELS[key],COLORS[i]) or available
        style_axis(ax,group,'Epoch' if validation else 'Optimizer updates','Recorded loss (symlog)',available,symlog=True)
    return fig


def plot_training(run,output_dir=None,evaluation_scope='auto'):
    run=Path(run);output=Path(output_dir) if output_dir else run;output.mkdir(parents=True,exist_ok=True)
    train,epochs,overflows,config,summary=prepare(run,evaluation_scope)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,
                         'axes.labelcolor':'#344d5a','text.color':'#344d5a','axes.prop_cycle':plt.cycler(color=COLORS)})
    scope=summary['scope'];name=run.name
    footer_train='Within-epoch logged means; not full-epoch averages. Last-log coverage and any partial trailing records are disclosed in plot_summary.json.'
    footer_eval='Custom subset metrics and recorded evaluation losses. Missing evidence stays missing; no final-test data are read.'
    outputs=[]
    def write(fig,filename,title,footer):
        save(fig,output/filename,title,footer);outputs.append(filename)
    # Keep the original filename as the compact overview consumed by existing workflows.
    fig,axes=plt.subplots(1,2,figsize=(13.5,4.8))
    available=False
    for i,key in enumerate(('depth','road','initial_total','refined_total','total')):
        if key=='refined_total' and {row.get('stage') for row in train}=={'a'}:continue
        available=series(axes[0],train,'updates',f'losses.{key}',LOSS_LABELS[key],COLORS[i]) or available
    style_axis(axes[0],'Training: within-epoch logged means','Optimizer updates','Recorded loss (symlog)',available,symlog=True)
    available=series(axes[1],epochs,'epoch','score','Recorded selection score',COLORS[0])
    style_axis(axes[1],scope,'Epoch','Selection score',available,fraction=True)
    write(fig,'training_curves.png',f'{name} | training and evaluation overview',footer_train)
    write(loss_panels(train),'task_losses.png',f'{name} | individual training loss terms',footer_train)
    write(loss_panels(epochs,True),'validation_losses.png',f'{name} | {scope}',footer_eval)
    fig,axes=plt.subplots(2,2,figsize=(13.5,8.6))
    for ax,cls in zip(axes[0],('car','pedestrian')):
        available=False
        for i,key in enumerate(('ap','recall')):
            available=series(ax,epochs,'epoch',f'metrics.objects.bev_ap50.classes.{cls}.{key}',
                             'BEV AP50' if key=='ap' else 'BEV recall @ IoU 0.5',COLORS[i],'-' if i==0 else '--') or available
        counts=[get(row,f'metrics.objects.bev_ap50.classes.{cls}.gt_count') for row in epochs]
        count_text=f'latest GT count: {counts[-1]}' if counts and finite(counts[-1]) else 'GT count unrecorded'
        style_axis(ax,f'{cls.title()} - {count_text}','Epoch','Fraction',available,fraction=True)
    available=False
    for i,key in enumerate(('road','sidewalk','other_ground')):
        available=series(axes[1,0],epochs,'epoch',f'metrics.road.iou.{key}',key.replace('_',' ').title(),COLORS[i]) or available
    style_axis(axes[1,0],'Valid-label ground overlap','Epoch','IoU',available,fraction=True)
    available=series(axes[1,1],epochs,'epoch','score','Selection score',COLORS[0])
    style_axis(axes[1,1],'Recorded model-selection score','Epoch','Score',available,fraction=True)
    write(fig,'validation_curves.png',f'{name} | {scope}',footer_eval)
    fig,axes=plt.subplots(2,2,figsize=(13.5,8.6))
    available=series(axes[0,0],epochs,'epoch','epoch_seconds','Logged epoch elapsed',COLORS[0])
    style_axis(axes[0,0],'Epoch elapsed (training + evaluation)','Epoch','Seconds',available)
    available=False
    for i,key in enumerate(('peak_allocated_mb','peak_reserved_mb')):
        available=series(axes[0,1],epochs,'epoch',key,'Peak allocated' if i==0 else 'Peak reserved',COLORS[i],scale=1024) or available
    style_axis(axes[0,1],'Recorded peak GPU memory','Epoch','GiB',available)
    by_epoch=Counter(row['epoch'] for row in overflows if finite(row.get('epoch')))
    known=sorted({row['epoch'] for row in [*train,*epochs,*overflows] if finite(row.get('epoch'))})
    if known:
        axes[1,0].bar(known,[by_epoch.get(epoch,0) for epoch in known],color=COLORS[1],width=.65,label='Observed overflow events')
        axes[1,0].yaxis.set_major_locator(MaxNLocator(integer=True));axes[1,0].set_ylim(0,max(1,max(by_epoch.values(),default=0))*1.25)
    style_axis(axes[1,0],f'AMP skipped updates - {len(overflows)} events','Epoch','Logged skip count',bool(known))
    available=series(axes[1,1],train,'updates','amp_scale','Scale at training log',COLORS[0])
    style_axis(axes[1,1],'Recorded AMP gradient scale','Optimizer updates','Scale',available)
    if available:axes[1,1].set_yscale('log',base=2)
    write(fig,'runtime_curves.png',f'{name} | runtime and AMP evidence',
          'Only recorded epoch timings and peak memory are plotted. Missing historical VRAM values are not inferred from live allocation logs.')
    groups=sorted({key for row in train for key in (row.get('gradient_groups') or {})})
    if groups:
        columns=4;count=len(groups)+1;rows=(count+columns-1)//columns
        fig,grid=plt.subplots(rows,columns,figsize=(16,4.1*rows),squeeze=False)
        axes=list(grid.flat)
        for i,group in enumerate(groups):
            available=series(axes[i],train,'updates',f'gradient_groups.{group}',group,COLORS[i%len(COLORS)])
            style_axis(axes[i],group,'Optimizer updates','Gradient norm (symlog)',available,symlog=True)
            if axes[i].get_legend():axes[i].get_legend().remove()
        total_axis=axes[len(groups)]
        for unused in axes[len(groups)+1:]:unused.set_axis_off()
    else:
        fig,axes=plt.subplots(1,2,figsize=(13.5,5.6))
        style_axis(axes[0],'Component gradient groups','Optimizer updates','Gradient norm',False)
        total_axis=axes[1]
    available=series(total_axis,train,'updates','grad_norm','Total gradient norm',COLORS[0])
    clip=get(config,'train.grad_clip')
    if available and finite(clip):total_axis.axhline(clip,color=COLORS[1],linestyle='--',linewidth=1.2,label=f'Clip threshold: {clip:g}')
    style_axis(total_axis,'Total norm and clipping threshold','Optimizer updates','Gradient norm (symlog)',available,symlog=True)
    write(fig,'gradient_curves.png',f'{name} | component gradients before clipping',
          'Sampled L2 norms at log updates; these are not epoch means. Panel scales follow their recorded range; read tick values when comparing groups.')
    summary['disabled_refinement_display']='Stage A all-zero refinement is annotated in its panel and omitted from active task-total comparisons'
    summary['outputs']=outputs
    summary['gradient_groups_recorded']=groups
    summary['epoch_peak_memory_records']=sum(finite(r.get('peak_allocated_mb')) or finite(r.get('peak_reserved_mb')) for r in epochs)
    summary['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output/'plot_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf8')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True)
    p.add_argument('--output-dir',help='Default is the run directory; training_curves.png filename remains compatible')
    p.add_argument('--evaluation-scope',choices=('auto','validation','training-fit'),default='auto',
                   help='Auto recognizes learnability.json; otherwise the log does not explicitly identify its evaluation split')
    args=p.parse_args()
    result=plot_training(args.run,args.output_dir,args.evaluation_scope)
    print(json.dumps({key:result[key] for key in ('run','scope','train_loss_records','epoch_records','amp_overflow_events','outputs')}))


if __name__=='__main__':main()
