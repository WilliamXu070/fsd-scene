import json
from pathlib import Path
import pytest
from fsd.sealing import claim_final_test, identity


def setup_receipt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path('data').mkdir()
    Path('data/manifest.full.json').write_text('{"samples": []}')
    checkpoint=Path('best.pt')
    checkpoint.write_bytes(b'fixed trained weights fixture')
    config={'data':{'root':'data','manifest':'manifest.full.json'}}
    receipt=Path('selection.json')
    receipt.write_text(json.dumps({'identity':identity(checkpoint,config)}))
    return checkpoint,config,receipt,Path('test_ledger.json')


def test_only_one_completed_final_run(tmp_path,monkeypatch):
    checkpoint,config,receipt,ledger=setup_receipt(tmp_path,monkeypatch)
    state=claim_final_test(checkpoint,config,receipt,ledger)
    state['status']='complete'
    ledger.write_text(json.dumps(state))
    with pytest.raises(ValueError,match='already claimed'):
        claim_final_test(checkpoint,config,receipt,ledger,resume=True)


def test_resume_same_failed_identity(tmp_path,monkeypatch):
    checkpoint,config,receipt,ledger=setup_receipt(tmp_path,monkeypatch)
    state=claim_final_test(checkpoint,config,receipt,ledger)
    state['status']='failed'
    ledger.write_text(json.dumps(state))
    resumed=claim_final_test(checkpoint,config,receipt,ledger,resume=True)
    assert resumed['attempt']==2
    assert resumed['identity']==state['identity']


@pytest.mark.parametrize('changed', ['checkpoint','manifest','configuration'])
def test_selection_rejects_changes(tmp_path,monkeypatch,changed):
    checkpoint,config,receipt,ledger=setup_receipt(tmp_path,monkeypatch)
    if changed=='checkpoint':
        checkpoint.write_bytes(b'different weights')
    elif changed=='manifest':
        Path('data/manifest.full.json').write_text('{"samples": [1]}')
    else:
        config['changed']=True
    with pytest.raises(ValueError,match='identity changed'):
        claim_final_test(checkpoint,config,receipt,ledger)
    assert not ledger.exists()
