"""Compact, read-only execution status; optionally persist a progress snapshot."""
import argparse
import json
from pathlib import Path
from fsd.runtime import atomic_json


def last_record(path):
    if not path.exists():
        return None
    with path.open('rb') as stream:
        stream.seek(0,2)
        length=stream.tell()
        stream.seek(max(0,length-128*1024))
        lines=stream.read().decode('utf8',errors='replace').splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def summarize_epoch(epoch):
    if epoch is None:
        return None
    m=epoch.get('metrics',{})
    objects=m.get('objects',{}).get('bev_ap50',{}).get('classes',{})
    return dict(epoch=epoch.get('epoch'),epoch_seconds=epoch.get('epoch_seconds'),
                selection_score=epoch.get('score'),road_iou=m.get('road',{}).get('iou',{}),
                objects={name:{k:row.get(k) for k in ('ap','recall','gt_count')}
                         for name,row in objects.items()},depth=m.get('depth'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--write')
    args=p.parse_args()
    result={'stages':{},'runs':{}}
    for path in sorted(Path('artifacts/stages').glob('*.json')):
        result['stages'][path.stem]=json.loads(path.read_text())['status']
    for run in sorted(Path('artifacts/runs').glob('*')):
        if not run.is_dir():
            continue
        epoch=summarize_epoch(last_record(run/'epochs.jsonl'))
        training=last_record(run/'train.jsonl')
        if epoch is not None or training is not None:
            result['runs'][run.name]={'latest_epoch':epoch,
                'latest_step':{k:training.get(k) for k in ('event','epoch','step','updates','elapsed_s','amp_overflow_skips')}
                    if training else None,'checkpoint_exists':(run/'latest.pt').exists()}
    if args.write:
        atomic_json(args.write,result)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
