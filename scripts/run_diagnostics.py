"""Validation-only input/history controls on one fixed trained checkpoint."""
import argparse
import json
import subprocess
import sys
import torch
from pathlib import Path
from fsd.runtime import atomic_json
from fsd.sealing import sha256, identity


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    root=Path(args.output)
    root.mkdir(parents=True,exist_ok=True)
    checkpoint_config=torch.load(args.checkpoint,map_location='cpu',weights_only=False)['config']
    expected=identity(args.checkpoint,checkpoint_config)
    checkpoint_hash=expected['checkpoint_sha256']
    descriptions={
        'normal':'Current images and normal aligned history',
        'blank':'All RGB values replaced by zero for the whole sequence',
        'shuffled':'Camera image order permuted while calibration order remains fixed',
        'empty_history':'Spatial refinement with no temporal memory',
        'mismatched_history':'Historical features permuted across proposals, positions unchanged',
        'initial_only':'Same checkpoint backbone/initial heads, both refinement attentions bypassed',
    }
    summary={}
    for name,description in descriptions.items():
        output=root/(name+'.json')
        if output.exists():
            report=json.loads(output.read_text())
            if (report.get('checkpoint_sha256')!=checkpoint_hash or report.get('split')!='val'
                or report.get('source_sha256')!=expected['source_sha256']
                or report.get('dataset_manifest_sha256')!=expected['manifest_sha256']
                or report.get('diagnostic')!=(None if name=='normal' else name)):
                raise RuntimeError(f'Existing diagnostic belongs to different weights/data scope: {output}')
        else:
            command=[sys.executable,'scripts/evaluate.py','--checkpoint',args.checkpoint,
                     '--split','val','--output',str(output)]
            if name!='normal':
                command+=['--diagnostic',name]
            with (root/(name+'.log')).open('w',encoding='utf8') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            report=json.loads(output.read_text())
        summary[name]={'description':description,'report':str(output),'metrics':report['metrics']}
        atomic_json(root/'summary.json',{'checkpoint':args.checkpoint,'checkpoint_sha256':checkpoint_hash,
                                      'scope':'Validation-only causal controls; not final test','controls':summary})
        print(json.dumps({'complete':name,'report':str(output)}),flush=True)


if __name__=='__main__':
    main()
