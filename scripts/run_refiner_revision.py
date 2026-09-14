"""Resume the recorded frozen-base revision after its real recovery gate."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from fsd.runtime import load_config, record_stage
from fsd.sealing import sha256, source_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/revision01-refiner-only.yaml')
    parser.add_argument('--run', default='artifacts/runs/revision01-refiner-only')
    parser.add_argument('--gate', default='artifacts/revision01-gate/report.json')
    args = parser.parse_args()
    config = load_config(args.config)
    gate = json.loads(Path(args.gate).read_text())
    if gate.get('passed') is not True:
        raise RuntimeError('The real frozen-base recovery gate has not passed')
    expected = {'source_sha256': source_hash(), 'configuration_file_sha256': sha256(args.config),
                'initial_checkpoint_sha256': sha256(config['experiment']['initialization'])}
    if any(gate.get(key) != value for key, value in expected.items()):
        raise RuntimeError('Recovery gate does not match current source, configuration and initialization')
    for stage in ('08_train_a', '08_train_b'):
        if json.loads(Path(f'artifacts/stages/{stage}.json').read_text())['status'] != 'passed':
            raise RuntimeError(f'Baseline prerequisite is incomplete: {stage}')
    stage = config['experiment']['stage_record']
    if not stage.startswith('09_') or config['train'].get('trainable_scope') != 'refiner':
        raise ValueError('This launcher requires isolated revision evidence and a frozen base')
    run = Path(args.run)
    status_path = Path('artifacts/stages') / (stage + '.json')
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if (status.get('status') == 'passed' and
                Path(status.get('evidence', {}).get('run', '')).resolve() == run.resolve() and
                (run / 'best.pt').exists()):
            print('Revision already complete; evaluate its preserved checkpoint.', flush=True)
            return
    source = run / 'latest.pt'
    start_option = '--resume'
    if not source.exists():
        source = Path(config['experiment']['initialization'])
        start_option = '--initialize'
    command = [sys.executable, 'scripts/train.py', '--config', args.config,
               '--stage', 'b', '--run', str(run), start_option, str(source)]
    log = Path('artifacts') / (run.name + '.log')
    evidence = {'run': str(run), 'command': command, 'log': str(log),
                'gate': args.gate, 'gate_sha256': sha256(args.gate),
                'resume_checkpoint': str(source), 'resume_sha256': sha256(source)}
    record_stage(stage, 'running', evidence)
    print(json.dumps(evidence), flush=True)
    with log.open('a', encoding='utf8') as stream:
        stream.write('\n' + json.dumps({'event': 'launch', **evidence}) + '\n')
        stream.flush()
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        record_stage(stage, 'failed', {**evidence, 'returncode': result.returncode})
        raise RuntimeError(f'Revision failed; inspect {log} before recovery')
    if json.loads(status_path.read_text())['status'] != 'passed':
        raise RuntimeError('Training exited without completing the revision evidence gate')
    print('REFINER_REVISION_TRAINING_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
