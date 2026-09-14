"""Fresh experiments must not overwrite progress or ambiguously resume it."""
from pathlib import Path
import pytest
from fsd.engine import train


def test_initialize_and_resume_are_mutually_exclusive_before_run_creation(tmp_path):
    run = tmp_path / 'new-run'
    with pytest.raises(ValueError, match='never both'):
        train({}, run, resume='prior.pt', initialize='prior.pt')
    assert not run.exists()


@pytest.mark.parametrize('existing', ['latest.pt', 'config.json', 'initialization.json', 'metrics.jsonl'])
def test_initialization_preserves_existing_progress(tmp_path, existing):
    run = tmp_path / 'experiment'
    run.mkdir()
    evidence = run / existing
    evidence.write_bytes(b'preserve me')
    with pytest.raises(ValueError, match='uninitialized'):
        train({}, run, initialize='prior.pt')
    assert evidence.read_bytes() == b'preserve me'
    assert list(run.iterdir()) == [evidence]


@pytest.mark.parametrize('experiment', [{}, {'id': 'revision'}, {'id': 'revision', 'initialization': 'different.pt'}])
def test_initialization_must_match_the_recorded_experiment(tmp_path, experiment):
    run = tmp_path / 'experiment'
    with pytest.raises(ValueError, match='recorded'):
        train({'experiment': experiment}, run, initialize='prior.pt')
    assert not run.exists()
