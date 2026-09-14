"""Run/resume the two full-data stages only after explicit prerequisite evidence."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from fsd.runtime import load_config, record_stage


def require_pass(stage):
    path=Path('artifacts/stages')/(stage+'.json')
    report=json.loads(path.read_text())
    if report.get('status')!='passed':
        raise RuntimeError(f'Prerequisite is incomplete: {stage}')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/baseline.yaml')
    p.add_argument('--prefix',default='baseline')
    args=p.parse_args()
    for stage in ('01_environment','02_acquisition','03_geometry_full','04_evaluation',
                  '05_pipeline','06_components','07_learnability_a','07_learnability_b'):
        require_pass(stage)
    config=load_config(args.config)
    if config['data'].get('manifest')!='manifest.full.json':
        raise RuntimeError('Full training requires the frozen full-subset manifest')
    for stage in ('a','b'):
        run=Path('artifacts/runs')/(args.prefix+'-'+stage)
        report=Path(f'artifacts/stages/08_train_{stage}.json')
        if report.exists():
            status=json.loads(report.read_text())
            prior=Path(status.get('evidence',{}).get('run',''))
            if status.get('status')=='passed' and prior.resolve()==run.resolve() and (run/'best.pt').exists():
                print(f'Already complete: {run}',flush=True)
                continue
        command=[sys.executable,'scripts/train.py','--config',args.config,'--stage',stage,'--run',str(run)]
        latest=run/'latest.pt'
        if latest.exists():
            command+=['--resume',str(latest)]
        elif stage=='b':
            command+=['--resume',str(Path('artifacts/runs')/(args.prefix+'-a')/'best.pt')]
        log=Path('artifacts')/(args.prefix+'-'+stage+'.log')
        with log.open('a',encoding='utf8') as stream:
            stream.write('\n'+json.dumps({'event':'launch','command':command})+'\n')
            stream.flush()
            print(json.dumps({'stage':stage,'run':str(run),'log':str(log),'command':command}),flush=True)
            evidence=dict(run=str(run),command=command,log=str(log),
                          explanation='Full-data training in progress; inspect checkpoint and epoch records.')
            record_stage('08_train_'+stage,'running',evidence)
            try:
                subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
            except subprocess.CalledProcessError as error:
                record_stage('08_train_'+stage,'failed',dict(evidence,return_code=error.returncode,
                    explanation='Training process failed; saved logs and resumable checkpoints require diagnosis.'))
                raise
        require_pass('08_train_'+stage)
    print('Both full-data training stages completed; validation selection and final-test freeze remain.',flush=True)


if __name__=='__main__':
    main()
