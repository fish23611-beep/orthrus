"""u4090 普通 SSH: attempt-isolated execution for the sealed prepared_05 protocol.

Only execution management changes. The sealed pipeline and CUDA execute() stay
untouched. `run` verifies the linked CUDA proof before calling sealed.run().
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import difflib
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
import joint_optimization_pipeline as sealed

ENTRY = Path(__file__).resolve()
TEST = ROOT / 'tests/test_joint_optimization_execution_v2.py'
TESTS = [TEST, ROOT / 'tests/test_joint_optimization_pipeline.py',
         ROOT / 'tests/test_shrunk_node_training.py', ROOT / 'tests/test_node_balanced_training.py']
SCOPE = ('Smoke attempt isolation, durable evidence, self-PID resource filtering, '
         'and proof verification before unchanged sealed formal entry. '
         'No loss/model/evaluation/checkpoint/training-protocol changes.')


def unique_dir(parent, prefix):
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix + '_' + time.strftime('%Y%m%dT%H%M%S') + '_', dir=parent))


def code_hashes():
    paths = {ENTRY, *TESTS, ROOT / 'scripts/joint_optimization_pipeline.py',
             ROOT / 'scripts/smoke_node_balanced_training.py',
             ROOT / 'scripts/audit_node_training_frequency.py'}
    return {str(p.relative_to(ROOT)): sealed.sha(p) for p in sorted(paths)}


def regressions(out):
    """Real CPU pytest execution, with immutable log and current-code identity."""
    dest = unique_dir(out / 'execution_v2_cpu_tests', 'cpu')
    before = code_hashes()
    argv = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *map(str, TESTS)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    with (dest / 'tests.log').open('x') as log:
        result = subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    code = result.returncode if before == code_hashes() else 1
    (dest / 'exitcode').write_text(str(code) + '\n')
    sealed.save(dest / 'result.json', dict(status='PASS' if code == 0 else 'FAIL',
        exitcode=code, created_at=sealed.now(), argv=argv, code_sha256=before,
        log_sha256=sealed.sha(dest / 'tests.log'), cuda_visible_devices=''))
    print(dest, 'PASS' if code == 0 else 'FAIL', flush=True)
    return code


def require_regressions(out):
    for path in sorted((out / 'execution_v2_cpu_tests').glob('*/result.json'), reverse=True):
        r = sealed.read(path)
        if (r['status'] == 'PASS' and r['code_sha256'] == code_hashes()
                and (path.parent / 'exitcode').read_text().strip() == '0'
                and r['log_sha256'] == sealed.sha(path.parent / 'tests.log')):
            return path
    raise RuntimeError('Current execution code has no passing CPU regression attestation')


def inputs(out):
    plan = sealed.read(out / 'prepared_05/plan.json')
    paths = {out / 'prepared_05/plan.json', out / 'prepared_05/treatment.yml',
             out / 'e3_shrunk_weights.json', out / 'e5_shrunk_weights.json',
             ROOT / 'scripts/audit_node_training_frequency.py', *ENTRY.parent.parent.glob('config/**/*.yml')}
    paths.update(ROOT / p for p in plan['source_sha256'])
    paths.update(Path(d['control_training_config']) for d in plan['datasets'].values())
    return {str(p.resolve()): sealed.sha(p) for p in sorted(paths)}


def resource_gate(e=None, *, usage=shutil.disk_usage, query=subprocess.check_output):
    """Preserve sealed global GPU/disk policy; exempt only this live process.

    Two process snapshots bracket the GPU query. Any foreign PID, even on the
    other card or owned by the same user, blocks. High utilization is exempted
    only on a GPU on which this PID owns a CUDA context; raw data is retained.
    """
    raw = {}
    try:
        app_args = ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                    '--format=csv,noheader,nounits']
        raw['processes_before'] = query(app_args, text=True).strip()
        raw['gpu'] = query(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu',
                            '--format=csv,noheader,nounits'], text=True).strip()
        raw['processes_after'] = query(app_args, text=True).strip()
        own, foreign, own_uuids = [], [], set()
        for line in set((raw['processes_before'] + '\n' + raw['processes_after']).splitlines()):
            if not line.strip():
                continue
            row = [v.strip() for v in line.split(',')]
            if len(row) != 3:
                raise ValueError('Malformed compute process row')
            pid = int(row[0])
            if pid <= 0:
                raise ValueError('Invalid compute PID')
            if pid == os.getpid():
                own.append(line)
                own_uuids.add(row[1])
            else:
                foreign.append(line)
        adjusted = []
        for line in raw['gpu'].splitlines():
            row = [v.strip() for v in line.split(',')]
            if len(row) != 5:
                raise ValueError('Malformed GPU row')
            int(row[4])  # Unknown utilization must fail closed, including own GPU.
            if row[1] in own_uuids and not foreign:
                row[4] = '0'
            adjusted.append(', '.join(row))
        def filtered(argv, **kwargs):
            return '\n'.join(sorted(foreign)) if 'compute-apps' in argv[1] else '\n'.join(adjusted)
        gate = sealed.resource_gate(e, usage=usage, query=filtered)
        gate.update(raw_nvidia_smi=raw, own_pid=os.getpid(),
                    excluded_own_compute_processes=sorted(own), external_compute_processes=sorted(foreign))
        return gate
    except Exception as exc:
        # Still collect disk evidence using the original safety floors.
        def failed(*args, **kwargs):
            raise RuntimeError(repr(exc))
        gate = sealed.resource_gate(e, usage=usage, query=failed)
        gate.update(raw_nvidia_smi=raw, own_pid=os.getpid())
        return gate


@contextmanager
def execution_lock(out):
    with (out / 'execution_v2.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another v2 Smoke/formal execution owns the lock')
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def create_attempt(out):
    dest = unique_dir(out / 'smoke_attempts_v2', 'smoke')
    shutil.copyfile(ENTRY, dest / ENTRY.name)
    original = ROOT / 'scripts/joint_optimization_pipeline.py'
    shutil.copyfile(original, dest / original.name)
    (dest / 'execution.diff').write_text(''.join(difflib.unified_diff(
        original.read_text().splitlines(True), ENTRY.read_text().splitlines(True),
        fromfile=str(original), tofile=str(ENTRY))))
    sealed.save(dest / 'identity.json', dict(created_at=sealed.now(), attempt=str(dest.resolve()),
        entry=str(ENTRY), entry_sha256=sealed.sha(ENTRY), code_sha256=code_hashes(),
        input_sha256=inputs(out), scope=SCOPE, python=sys.executable,
        cpu_regression=str(require_regressions(out).resolve()),
        cpu_regression_sha256=sealed.sha(require_regressions(out)),
        execution_diff_sha256=sealed.sha(dest / 'execution.diff')))
    return dest


def validate_identity(out, dest):
    identity = sealed.read(dest / 'identity.json')
    assert identity['attempt'] == str(dest.resolve()), 'Attempt identity mismatch'
    assert identity['entry_sha256'] == sealed.sha(ENTRY), 'Execution entry changed'
    assert identity['code_sha256'] == code_hashes(), 'Execution/test sources changed'
    assert identity['input_sha256'] == inputs(out), 'Frozen/config/weight inputs changed'
    assert sealed.sha(dest / ENTRY.name) == identity['entry_sha256']
    assert sealed.sha(dest / 'execution.diff') == identity['execution_diff_sha256']
    assert sealed.sha(identity['cpu_regression']) == identity['cpu_regression_sha256']
    require_regressions(out)
    return identity


def cuda_dataset(out, dest, e):
    """Same data, fitting, cfg_for and execute calls as sealed.smoke()."""
    import torch
    import data_utils
    import smoke_node_balanced_training as s
    from mstc.training_loss import ShrunkNodeContributionWeights
    from mstc.time_gap import TimeGapStatistics
    from audit_node_training_frequency import source_for, windows

    assert torch.cuda.is_available(), 'Real CUDA is required'
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.cuda.synchronize()
    device = dict(backend='CUDA', name=torch.cuda.get_device_name(),
                  index=torch.cuda.current_device(), torch_version=torch.__version__,
                  cuda_version=torch.version.cuda)
    started = time.monotonic()
    old_log = s.training.log
    s.training.log = lambda *a: None
    try:
        control, cache = source_for(e)
        cfg = s.cfg_for(e, dest)
        paths = sorted((cache / 'train').glob('*.TemporalData.simple'))
        capacity = max(max(int(w.src.max()), int(w.dst.max())) for w in windows(paths)) + 1
        weights = ShrunkNodeContributionWeights.fit(windows(paths), num_nodes=capacity)
        expected = sealed.read(out / f'e{e}_shrunk_weights.json')
        assert weights.summary()['counts_sha256'] == expected['counts_sha256'] and weights.tau == expected['tau']
        stats = TimeGapStatistics.load(control / 'metadata/time_statistics.json')
        prepared = [data_utils._prepare_window(cfg, str(p), cfg.dataset_view.mode) for p in paths[:2]]
        graphs, _, _, full, cap = data_utils._eager_load_all_datasets(cfg, prepared, [], [])
        result = {}
        for variant in ['control', 'balanced']:
            result[variant], parameters = s.execute(e, variant, cfg, graphs, full, cap, stats, weights,
                                                    dest, balanced_mode='node_contribution_shrunk')
            del parameters
        assert result['control']['initial_parameters_sha256'] == result['balanced']['initial_parameters_sha256']
        assert result['control']['event_order_sha256'] == result['balanced']['event_order_sha256']
        assert result['control']['state'] == result['balanced']['state']
        result['shrunk'] = result.pop('balanced')
        for name, scoped_name in [('control', 'control'), ('shrunk', 'balanced')]:
            scoped = dest / f'e{e}_{scoped_name}'
            result[name]['artifact_sha256'] = {
                str(path.relative_to(dest)): sealed.sha(path)
                for path in sorted(scoped.iterdir()) if path.is_file()}
        result.update(status='PASS', cuda=device, seconds=time.monotonic() - started,
                      peak_allocated_bytes=max(result[v]['peak_allocated_bytes'] for v in ['control', 'shrunk']),
                      peak_reserved_bytes=max(result[v]['peak_reserved_bytes'] for v in ['control', 'shrunk']))
        del graphs, full, prepared, weights
        gc.collect()
        return result
    finally:
        s.training.log = old_log


def validate_results(proof):
    assert proof['status'] == 'PASS' and proof['formal_training_started'] is False
    assert set(proof['datasets']) == {'3', '5'}, 'Both datasets required'
    for result in proof['datasets'].values():
        assert result['status'] == 'PASS' and result['cuda']['backend'] == 'CUDA'
        a, b = result['control'], result['shrunk']
        for variant in [a, b]:
            assert variant['status'] == 'PASS' and variant['no_nan_inf'] is True
            assert 0 < variant['batches'] <= 100 and len(variant['windows']) == 2
            assert variant['checkpoint_written'] is False
            assert variant['peak_allocated_bytes'] > 0 and variant['seconds'] > 0
        for key in ['initial_parameters_sha256', 'event_order_sha256', 'state']:
            assert a[key] == b[key], key


def verify_proof(out, path=None):
    sealed.check_seal(out)
    path = path or out / 'gpu_smoke.json'
    proof = sealed.read(path)
    validate_results(proof)
    dest = Path(proof['attempt']).resolve()
    assert dest.parent == (out / 'smoke_attempts_v2').resolve(), 'Proof outside attempt root'
    validate_identity(out, dest)
    assert proof['identity_sha256'] == sealed.sha(dest / 'identity.json')
    assert sealed.sha(path) == sealed.sha(dest / 'gpu_smoke.json'), 'Canonical proof differs from attempt'
    assert sealed.read(dest / 'status.json')['status'] == 'PASS'
    assert (dest / 'exitcode').read_text().strip() == '0'
    for e in [3, 5]:
        assert sealed.read(dest / f'e{e}_result.json') == proof['datasets'][str(e)]
        for variant in ['control', 'shrunk']:
            artifacts = proof['datasets'][str(e)][variant]['artifact_sha256']
            assert artifacts, 'CUDA variant artifacts missing'
            for relative, digest in artifacts.items():
                artifact = (dest / relative).resolve()
                assert artifact.is_relative_to(dest), 'Artifact outside attempt'
                assert sealed.sha(artifact) == digest, 'CUDA config/result/cleanup artifact changed'
        assert sealed.read(dest / f'e{e}_resource_gate.json')['status'] == 'PASS'
    return proof


def publish(out, dest):
    target = out / 'gpu_smoke.json'
    try:
        # Atomic create-if-absent; never replace earlier proof, even a broken one.
        os.link(dest / 'gpu_smoke.json', target)
        return dict(action='created', canonical=str(target), attempt=str(dest))
    except FileExistsError:
        previous = verify_proof(out)
        return dict(action='retained_previous_valid_PASS', canonical=str(target),
                    attempt=str(dest), canonical_attempt=previous['attempt'])


def smoke_worker(out, dest):
    assert dest.resolve().parent == (out / 'smoke_attempts_v2').resolve()
    # Claim before opening any worker outputs: the same attempt is never reused.
    sealed.save(dest / 'worker_claim.json', dict(pid=os.getpid(), started_at=sealed.now()))
    with (dest / 'worker.log').open('x', buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        code, state = 1, dict(status='RUNNING', pid=os.getpid(), started_at=sealed.now(), datasets={})
        (dest / 'pid').write_text(str(os.getpid()) + '\n')
        sealed.save(dest / 'status.json', state)
        try:
            with execution_lock(out):
                sealed.check_seal(out)
                validate_identity(out, dest)
                gate = resource_gate()
                sealed.save(dest / 'smoke_resource_gate.json', gate)
                if gate['status'] != 'PASS':
                    state.update(status='BLOCKED', errors=gate['errors'])
                    code = 2
                else:
                    assert not (out / 'FORMAL_STATUS.json').exists(), 'Formal execution already exists'
                    for e in [3, 5]:
                        gate = resource_gate(e)
                        sealed.save(dest / f'e{e}_resource_gate.json', gate)
                        if gate['status'] != 'PASS':
                            state['datasets'][str(e)] = dict(status='BLOCKED', errors=gate['errors'])
                            state.update(status='BLOCKED', errors=gate['errors'])
                            code = 2
                            break
                        state['datasets'][str(e)] = dict(status='RUNNING')
                        sealed.save(dest / 'status.json', state, mutable=True)
                        try:
                            result = cuda_dataset(out, dest, e)
                        except BaseException as exc:
                            sealed.save(dest / f'e{e}_failure.json', dict(status='FAIL', error=repr(exc)))
                            state['datasets'][str(e)] = dict(status='FAIL', error=repr(exc))
                            raise
                        sealed.save(dest / f'e{e}_result.json', result)
                        state['datasets'][str(e)] = dict(status='PASS')
                        sealed.save(dest / 'status.json', state, mutable=True)
                    else:
                        sealed.check_seal(out)
                        validate_identity(out, dest)
                        proof = dict(status='PASS', datasets={str(e): sealed.read(dest / f'e{e}_result.json') for e in [3, 5]},
                                     attempt=str(dest.resolve()), identity_sha256=sealed.sha(dest / 'identity.json'),
                                     formal_training_started=False, finished_at=sealed.now())
                        validate_results(proof)
                        sealed.save(dest / 'gpu_smoke.json', proof)
                        sealed.save(dest / 'publication.json', publish(out, dest))
                        state['status'], code = 'PASS', 0
        except BaseException as exc:
            state.update(status='FAIL', error=repr(exc))
            traceback.print_exc()
        finally:
            state.update(exitcode=code, finished_at=sealed.now())
            sealed.save(dest / 'status.json', state, mutable=True)
            with (dest / 'exitcode').open('x') as f:
                f.write(str(code) + '\n')
            print(json.dumps(state, ensure_ascii=False), flush=True)
    return code


def launch_smoke(out):
    sealed.check_seal(out)
    require_regressions(out)
    dest = create_attempt(out)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', WANDB_MODE='disabled',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    argv = ['nohup', sys.executable, '-u', str(ENTRY), '--analysis', str(out),
            '--mode', 'smoke', '--attempt', str(dest)]
    sealed.save(dest / 'launch.json', dict(argv=argv, created_at=sealed.now(), no_formal_training=True,
        runtime_environment={key: env.get(key) for key in ['CUDA_VISIBLE_DEVICES', 'CUBLAS_WORKSPACE_CONFIG',
            'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE', 'WANDB_MODE']}))
    try:
        with (dest / 'nohup.log').open('x') as log:
            process = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    except OSError as exc:
        sealed.save(dest / 'launch_failure.json', dict(status='FAIL', launcher_pid=os.getpid(), error=repr(exc)))
        with (dest / 'exitcode').open('x') as f:
            f.write('1\n')
        print(dest, 'launch FAIL:', repr(exc), flush=True)
        return 1
    with (dest / 'launcher.pid').open('x') as f:
        f.write(str(process.pid) + '\n')
    print(dest, 'PID', process.pid, flush=True)
    return 0


def verify_formal(out):
    proof = verify_proof(out)
    plan = sealed.check_seal(out)
    collisions = [out / 'FORMAL_STATUS.json', out / 'formal_pipeline.exitcode', out / 'resource_monitor.jsonl']
    for e in [3, 5]:
        d = plan['datasets'][str(e)]
        collisions += [Path(d['artifact_root']), Path(d['run_dir']), out / f'e{e}_formal_resource_gate.json']
        for stage in d['commands']:
            collisions += [out / f'e{e}_{stage}.{suffix}' for suffix in ['log', 'pid', 'exitcode']]
    existing = [str(p) for p in collisions if p.exists() or p.is_symlink()]
    assert not existing, 'Formal fixed paths already exist; never retry/overwrite: ' + repr(existing)
    gate = resource_gate(5)  # Reserve the larger dataset's disk budget before either starts.
    assert gate['status'] == 'PASS', gate
    return dict(status='PASS', smoke_attempt=proof['attempt'], resource_gate=gate,
                commands=plan['datasets'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--mode', choices=['cpu-tests', 'launch-smoke', 'smoke', 'verify-formal', 'run'], required=True)
    parser.add_argument('--attempt', type=Path)
    args = parser.parse_args()
    out = args.analysis.resolve()
    if args.mode == 'cpu-tests':
        return regressions(out)
    if args.mode == 'launch-smoke':
        return launch_smoke(out)
    if args.mode == 'smoke':
        assert args.attempt is not None
        return smoke_worker(out, args.attempt.resolve())
    if args.mode == 'verify-formal':
        print(json.dumps(verify_formal(out), ensure_ascii=False, indent=2))
        return 0
    with execution_lock(out):
        verify_formal(out)
        return sealed.run(out)


if __name__ == '__main__':
    sys.exit(main())
