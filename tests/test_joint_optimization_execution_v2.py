"""CPU-only regressions for execution management, never a CUDA acceptance."""
from pathlib import Path
import copy
import os
import sys
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
import joint_optimization_execution_v2 as v


def query_fixture(compute='', util=0, second=None):
    calls = []
    def query(argv, **kwargs):
        if 'compute-apps' in argv[1]:
            calls.append(1)
            return second if second is not None and len(calls) > 1 else compute
        return f'0, GPU-zero, 20, 24564, {util}\n1, GPU-one, 18, 24564, 0'
    return query


def usage(root=10, data=100):
    return lambda path: NS(free=(data if path == '/mnt/datadisk' else root) * v.sealed.GIB)


@pytest.mark.parametrize('dataset', [None, 3, 5])
def test_idle_gate_preserves_original_policy(dataset):
    assert v.resource_gate(dataset, usage=usage(), query=query_fixture())['status'] == 'PASS'


@pytest.mark.parametrize('root,data,dataset', [(2, 100, 3), (10, 25, 3), (10, 39, 5)])
def test_insufficient_resources(root, data, dataset):
    assert v.resource_gate(dataset, usage=usage(root, data), query=query_fixture())['status'] == 'BLOCKED'


def test_external_gpu_and_unknown_utilization_block():
    assert v.resource_gate(usage=usage(), query=query_fixture('987654321, GPU-zero, 200'))['status'] == 'BLOCKED'
    assert v.resource_gate(usage=usage(), query=query_fixture(util=85))['status'] == 'BLOCKED'
    assert v.resource_gate(usage=usage(), query=query_fixture(util='N/A'))['status'] == 'BLOCKED'


def test_self_context_excluded_but_external_even_same_gpu_blocks():
    own = f'{os.getpid()}, GPU-zero, 200'
    own_gate = v.resource_gate(usage=usage(), query=query_fixture(own, util=85))
    assert own_gate['status'] == 'PASS'
    assert own_gate['excluded_own_compute_processes'] == [own]
    assert own_gate['raw_nvidia_smi']['gpu'].startswith('0, GPU-zero')
    for other_gpu in ['GPU-zero', 'GPU-one']:
        gate = v.resource_gate(usage=usage(), query=query_fixture(own + f'\n987654321, {other_gpu}, 30', util=85))
        assert gate['status'] == 'BLOCKED' and len(gate['external_compute_processes']) == 1


def test_new_external_arrival_between_snapshots_blocks():
    own = f'{os.getpid()}, GPU-zero, 200'
    gate = v.resource_gate(usage=usage(), query=query_fixture(own, util=85, second=own + '\n987654321, GPU-one, 30'))
    assert gate['status'] == 'BLOCKED'


@pytest.mark.parametrize('compute', ['broken', 'N/A, GPU-zero, 100', '0, GPU-zero, 100'])
def test_malformed_process_information_fails_closed(compute):
    assert v.resource_gate(usage=usage(), query=query_fixture(compute))['status'] == 'BLOCKED'


def test_query_failure_fails_closed():
    def fail(*args, **kwargs):
        raise OSError('query unavailable')
    assert v.resource_gate(usage=usage(), query=fail)['status'] == 'BLOCKED'


def dataset_result():
    run = dict(status='PASS', no_nan_inf=True, batches=2, windows=[{}, {}], checkpoint_written=False,
               peak_allocated_bytes=1024, peak_reserved_bytes=2048, seconds=0.5,
               initial_parameters_sha256='same-init', event_order_sha256='same-order', state={'state': 'same'})
    return dict(status='PASS', cuda=dict(backend='CUDA', name='CPU_TEST_DOUBLE'),
                control=copy.deepcopy(run), shrunk=copy.deepcopy(run), seconds=1,
                peak_allocated_bytes=1024, peak_reserved_bytes=2048)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    regression = tmp_path / 'cpu_regression.json'
    v.sealed.save(regression, {'status': 'PASS', 'note': 'test fixture only'})
    monkeypatch.setattr(v, 'require_regressions', lambda out: regression)
    monkeypatch.setattr(v, 'inputs', lambda out: {'frozen-test-input': 'same'})
    plan = {'datasets': {str(e): dict(artifact_root=str(tmp_path / f'formal_{e}'),
                run_dir=str(tmp_path / f'formal_{e}/run'), commands={'training': [], 'cpu_evaluation': []}) for e in [3, 5]}}
    monkeypatch.setattr(v.sealed, 'check_seal', lambda out: plan)
    monkeypatch.setattr(v, 'resource_gate', lambda *a, **kw: dict(status='PASS', errors=[]))
    calls = []
    def execute(out, dest, e):
        calls.append(e)
        for variant in ['control', 'balanced']:
            scoped = dest / f'e{e}_{variant}'
            scoped.mkdir(exist_ok=False)
            v.sealed.save(scoped / 'result.json', dataset_result()[('control' if variant == 'control' else 'shrunk')])
        result = dataset_result()
        for name, scoped_name in [('control', 'control'), ('shrunk', 'balanced')]:
            path = dest / f'e{e}_{scoped_name}/result.json'
            result[name]['artifact_sha256'] = {str(path.relative_to(dest)): v.sealed.sha(path)}
        return result
    monkeypatch.setattr(v, 'cuda_dataset', execute)
    return tmp_path, calls


def test_historical_blocked_gate_and_partial_outputs_never_overwritten(harness):
    out, calls = harness
    old = out / 'smoke_resource_gate.json'
    old.write_bytes(b'{"status":"BLOCKED","historical":true}')
    (out / 'gpu_smoke/e3_control').mkdir(parents=True)
    old_attempt = out / 'smoke_attempts_v2/old_failed'
    old_attempt.mkdir(parents=True)
    (old_attempt / 'exitcode').write_text('1\n')
    before = old.read_bytes()
    dest = v.create_attempt(out)
    assert v.smoke_worker(out, dest) == 0
    assert calls == [3, 5] and old.read_bytes() == before
    assert (old_attempt / 'exitcode').read_text() == '1\n'
    assert (out / 'gpu_smoke/e3_control').is_dir()
    assert v.verify_proof(out)['attempt'] == str(dest)


def test_successful_repeat_gets_unique_artifacts_and_retains_prior_proof(harness):
    out, calls = harness
    first = v.create_attempt(out)
    assert v.smoke_worker(out, first) == 0
    proof_bytes = (out / 'gpu_smoke.json').read_bytes()
    second = v.create_attempt(out)
    assert first != second and v.smoke_worker(out, second) == 0
    assert calls == [3, 5, 3, 5]
    assert (out / 'gpu_smoke.json').read_bytes() == proof_bytes
    assert v.sealed.read(second / 'publication.json')['action'] == 'retained_previous_valid_PASS'
    assert v.verify_proof(out, second / 'gpu_smoke.json')['attempt'] == str(second)


def test_resource_block_records_attempt_and_never_calls_cuda(harness, monkeypatch):
    out, calls = harness
    monkeypatch.setattr(v, 'resource_gate', lambda *a, **kw: dict(status='BLOCKED', errors=['external GPU']))
    dest = v.create_attempt(out)
    assert v.smoke_worker(out, dest) == 2
    assert not calls and not (out / 'gpu_smoke.json').exists()
    assert v.sealed.read(dest / 'status.json')['status'] == 'BLOCKED'
    assert (dest / 'exitcode').read_text().strip() == '2'


def test_e3_pass_e5_failure_then_successful_new_attempt(harness, monkeypatch):
    out, calls = harness
    real_fake = v.cuda_dataset
    def failed(out, dest, e):
        if e == 5:
            (dest / 'e5_partial').mkdir()
            raise RuntimeError('simulated E5 failure')
        return real_fake(out, dest, e)
    monkeypatch.setattr(v, 'cuda_dataset', failed)
    first = v.create_attempt(out)
    assert v.smoke_worker(out, first) == 1
    assert v.sealed.read(first / 'e3_result.json')['status'] == 'PASS'
    assert v.sealed.read(first / 'e5_failure.json')['status'] == 'FAIL'
    assert not (out / 'gpu_smoke.json').exists()
    monkeypatch.setattr(v, 'cuda_dataset', real_fake)
    second = v.create_attempt(out)
    assert v.smoke_worker(out, second) == 0
    assert (first / 'e5_partial').is_dir() and (first / 'exitcode').read_text().strip() == '1'


def test_external_competition_between_e3_e5_blocks_e5(harness, monkeypatch):
    out, calls = harness
    def gate(e=None):
        return dict(status='BLOCKED' if e == 5 else 'PASS', errors=['external arrival'] if e == 5 else [])
    monkeypatch.setattr(v, 'resource_gate', gate)
    dest = v.create_attempt(out)
    assert v.smoke_worker(out, dest) == 2
    assert calls == [3] and not (out / 'gpu_smoke.json').exists()
    assert v.sealed.read(dest / 'status.json')['datasets']['5']['status'] == 'BLOCKED'


def test_same_attempt_cannot_be_reentered_or_clobbered(harness):
    out, calls = harness
    dest = v.create_attempt(out)
    assert v.smoke_worker(out, dest) == 0
    before = {p.name: p.read_bytes() for p in dest.iterdir() if p.is_file()}
    with pytest.raises(FileExistsError):
        v.smoke_worker(out, dest)
    assert before == {p.name: p.read_bytes() for p in dest.iterdir() if p.is_file()}


def test_invalid_existing_canonical_proof_preserved_and_not_reused(harness):
    out, calls = harness
    (out / 'gpu_smoke.json').write_text('{"status":"PASS","historical":true}')
    dest = v.create_attempt(out)
    assert v.smoke_worker(out, dest) == 1
    assert (out / 'gpu_smoke.json').read_text() == '{"status":"PASS","historical":true}'
    with pytest.raises((KeyError, AssertionError)):
        v.verify_formal(out)


def test_formal_wrapper_accepts_linked_proof_and_original_entry_consumes_it(harness, monkeypatch):
    out, calls = harness
    dest = v.create_attempt(out)
    assert v.smoke_worker(out, dest) == 0
    assert v.verify_formal(out)['status'] == 'PASS'
    # Exercise the real sealed run entry: it must consume the new proof and then
    # stop at its own resource gate before launching any training process.
    monkeypatch.setattr(v.sealed, 'resource_gate', lambda e: dict(status='BLOCKED', errors=['test block']))
    monkeypatch.setattr(v.sealed.subprocess, 'Popen', lambda *a, **kw: pytest.fail('formal training must not launch'))
    with pytest.raises(AssertionError, match='test block'):
        v.sealed.run(out)
    assert v.sealed.read(out / 'FORMAL_STATUS.json')['status'] == 'FAILED'


@pytest.mark.parametrize('collision', ['FORMAL_STATUS.json', 'e3_training.log', 'e5_formal_resource_gate.json', 'formal_3'])
def test_formal_partial_outputs_block_before_any_training(harness, collision):
    out, calls = harness
    dest = v.create_attempt(out)
    assert v.smoke_worker(out, dest) == 0
    (out / collision).touch()
    with pytest.raises(AssertionError, match='fixed paths'):
        v.verify_formal(out)


@pytest.mark.parametrize('target', ['identity.json', 'e3_result.json', 'gpu_smoke.json'])
def test_proof_tampering_rejected(harness, target):
    out, calls = harness
    dest = v.create_attempt(out)
    assert v.smoke_worker(out, dest) == 0
    data = v.sealed.read(dest / target)
    if target == 'identity.json':
        data['entry_sha256'] = 'changed'
    elif target == 'e3_result.json':
        data['status'] = 'FAIL'
    else:
        data['datasets']['5']['status'] = 'FAIL'
    v.sealed.save(dest / target, data, mutable=True)
    with pytest.raises(AssertionError):
        v.verify_formal(out)


def test_lock_rejects_overlapping_execution(harness):
    out, calls = harness
    with v.execution_lock(out):
        dest = v.create_attempt(out)
        assert v.smoke_worker(out, dest) == 1
    assert not calls


def test_cuda_cannot_start_without_current_passing_regressions(tmp_path):
    with pytest.raises(RuntimeError, match='regression attestation'):
        v.require_regressions(tmp_path)


def test_actual_prepared_05_remains_sealed():
    out = ROOT / 'analysis/joint_optimization_20261029'
    plan_path = out / 'prepared_05/plan.json'
    plan_before = plan_path.read_bytes()
    plan = v.sealed.check_seal(out)
    assert 'scripts/joint_optimization_pipeline.py' in plan['source_sha256']
    assert plan_path.read_bytes() == plan_before
    assert v.sealed.sha(ROOT / 'scripts/joint_optimization_pipeline.py') == plan['source_sha256']['scripts/joint_optimization_pipeline.py']
    assert v.sealed.sha(out / 'prepared_05/treatment.yml') == plan['config_sha256']


def test_nohup_launcher_detaches_and_records_unique_pid_and_identity(harness, monkeypatch):
    out, calls = harness
    launched = []
    def spawn(argv, **kwargs):
        launched.append((argv, kwargs))
        return NS(pid=987654321)
    monkeypatch.setattr(v.subprocess, 'Popen', spawn)
    assert v.launch_smoke(out) == 0
    assert v.launch_smoke(out) == 0
    assert len(launched) == 2 and not calls
    attempts = list((out / 'smoke_attempts_v2').iterdir())
    assert len(attempts) == 2
    for argv, kw in launched:
        assert argv[0] == 'nohup' and '--mode' in argv and 'smoke' in argv
        assert kw['start_new_session'] is True and kw['stdin'] == v.subprocess.DEVNULL
        dest = Path(argv[-1])
        assert (dest / 'launcher.pid').read_text().strip() == '987654321'
        assert v.validate_identity(out, dest)['scope'] == v.SCOPE


def test_launcher_failure_preserves_attempt_and_exitcode(harness, monkeypatch):
    out, calls = harness
    def fail(*args, **kwargs):
        raise OSError('simulated launch failure')
    monkeypatch.setattr(v.subprocess, 'Popen', fail)
    assert v.launch_smoke(out) == 1
    dest, = (out / 'smoke_attempts_v2').iterdir()
    assert v.sealed.read(dest / 'launch_failure.json')['status'] == 'FAIL'
    assert (dest / 'exitcode').read_text().strip() == '1'
    assert not calls and not (out / 'gpu_smoke.json').exists()


def test_actual_variant_config_or_result_tampering_blocks_formal(harness):
    out, calls = harness
    dest = v.create_attempt(out)
    assert v.smoke_worker(out, dest) == 0
    (dest / 'e5_balanced/result.json').write_text('{}')
    with pytest.raises(AssertionError, match='artifact changed'):
        v.verify_formal(out)
