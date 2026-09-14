"""Reproducible state, logging and stage evidence shared by command-line tools."""
from __future__ import annotations
import json, os, random, time
from pathlib import Path
import numpy as np
import torch
import yaml


def load_config(path):
    return yaml.safe_load(Path(path).read_text(encoding='utf-8-sig'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf8')
    os.replace(tmp, path)


def record_stage(name, status, evidence=None):
    atomic_json(Path('artifacts/stages') / f'{name}.json',
                dict(stage=name, status=status, updated=time.time(), evidence=evidence or {}))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [to_device(x, device) for x in batch]
    if isinstance(batch, tuple):
        return tuple(to_device(x, device) for x in batch)
    return batch


def save_checkpoint(path, **state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    os.replace(temp, path)


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf8') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')


def history_for(history, batch):
    if not history:
        return []
    last = history[-1]
    sequence = batch.get('sequence')
    if sequence != last.get('sequence'):
        return []
    now = batch['timestamp'].detach().cpu()
    before = last['timestamp'].detach().cpu()
    gap = now - before
    if (gap <= 0).any() or (gap > 1.01).any():
        return []
    return history

def capture_provenance(run_dir, config):
    """Snapshot only project code/config and manifests, not raw data or credentials."""
    import hashlib, platform, sys, zipfile
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    paths = sorted([p for folder in ('src','scripts','configs') for p in Path(folder).rglob('*')
                    if p.is_file() and p.suffix in ('.py','.yaml','.json') and '__pycache__' not in p.parts])
    paths += [p for p in (Path('requirements.lock.txt'),Path('pyproject.toml')) if p.exists()]
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    manifest = Path(config['data']['root'])/config['data'].get('manifest','manifest.json')
    data_hash = hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None
    with zipfile.ZipFile(run/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path,str(path))
        if manifest.exists():
            archive.write(manifest,'dataset_manifest.json')
    atomic_json(run/'provenance.json',dict(source_sha256=hashes,dataset_manifest_sha256=data_hash,
                dataset_manifest=str(manifest),python=sys.version,platform=platform.platform(),torch=torch.__version__,
                cuda=torch.version.cuda,created=time.time()))
