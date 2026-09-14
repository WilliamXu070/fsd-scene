"""Immutable model-selection receipt and single logical final-test execution."""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
from fsd.runtime import atomic_json


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_hash():
    digest = hashlib.sha256()
    paths=sorted([*Path('src/fsd').glob('*.py'),*Path('scripts').glob('*.py')])
    for path in paths:
        digest.update(path.as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def identity(checkpoint, config):
    manifest = Path(config['data']['root']) / config['data'].get('manifest', 'manifest.json')
    return dict(checkpoint_sha256=sha256(checkpoint), manifest_sha256=sha256(manifest),
                config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
                source_sha256=source_hash())


def freeze_selection(checkpoint, config, validation_report, reason, receipt_path):
    report = json.loads(Path(validation_report).read_text())
    if report.get('split') != 'val' or report.get('diagnostic') is not None:
        raise ValueError('Selection requires an ordinary validation report')
    if report.get('checkpoint_sha256') != sha256(checkpoint):
        raise ValueError('Validation report must refer to this exact checkpoint')
    current_identity=identity(checkpoint,config)
    if report.get('dataset_manifest_sha256')!=current_identity['manifest_sha256']:
        raise ValueError('Validation data identity differs from selected data')
    if report.get('configuration_sha256')!=current_identity['config_sha256']:
        raise ValueError('Validation configuration differs from selected configuration')
    if report.get('source_sha256')!=current_identity['source_sha256']:
        raise ValueError('Revalidate after inference/evaluation source changes before freezing')
    if config['data'].get('manifest') != 'manifest.full.json':
        raise ValueError('Final selection requires the immutable full-subset manifest')
    for stage in ('08_train_a','08_train_b'):
        status=json.loads(Path(f'artifacts/stages/{stage}.json').read_text())
        if status['status'] != 'passed':
            raise ValueError(f'Full training stage is incomplete: {stage}')
    receipt = dict(identity=identity(checkpoint,config),checkpoint=str(Path(checkpoint).resolve()),
                   validation_report=str(Path(validation_report).resolve()),
                   validation_sha256=sha256(validation_report),selection_reason=reason,
                   created=time.time(),test_split='test',test_drive='0004')
    path=Path(receipt_path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as stream:
        json.dump(receipt,stream,indent=2,allow_nan=False)
    return receipt


def claim_final_test(checkpoint, config, receipt_path, ledger_path, resume=False):
    receipt=json.loads(Path(receipt_path).read_text())
    current=identity(checkpoint,config)
    if receipt['identity'] != current:
        raise ValueError('Frozen checkpoint/configuration/data/code identity changed')
    ledger=Path(ledger_path)
    state=dict(identity=current,status='running',started=time.time(),attempt=1,
               receipt_sha256=sha256(receipt_path))
    ledger.parent.mkdir(parents=True,exist_ok=True)
    if ledger.exists():
        prior=json.loads(ledger.read_text())
        if not resume or prior['status'] not in ('failed','running') or prior['identity'] != current:
            raise ValueError('Final test already claimed or complete; only same frozen failed/interrupted run may resume')
        state['attempt']=prior['attempt']+1
        state['prior_attempt']=prior
        atomic_json(ledger,state)
    else:
        with ledger.open('x',encoding='utf8') as stream:
            json.dump(state,stream,indent=2)
    return state
