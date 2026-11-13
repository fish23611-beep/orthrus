"""u4090 ordinary SSH: physical GPU 1 isolation for the frozen E3/E5 recovery protocol.

No sealed sources are changed. E3 executes test/evaluation only against a fresh
run with read-only references to accepted checkpoints. E5 uses frozen commands.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import copy
import difflib
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import time
from unittest.mock import patch

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import joint_optimization_pipeline as sealed
import joint_optimization_execution_v2 as v2
import audit_joint_recovery_v3 as audit
import joint_gpu_isolation_v4 as gpu_guard
ENTRY=Path(__file__).resolve()
TEST=ROOT/'tests/test_joint_optimization_execution_v4.py'
GIB=sealed.GIB
SOURCE=[ENTRY,Path(gpu_guard.__file__).resolve(),ROOT/'scripts/joint_optimization_execution_v3.py',Path(audit.__file__).resolve(),TEST,ROOT/'scripts/joint_optimization_execution_v2.py',
        ROOT/'scripts/evaluate_joint_optimization.py',ROOT/'scripts/compare_e3_node_balanced_treatment.py',
        ROOT/'scripts/restore_e5_frozen_postprocessing.py',ROOT/'scripts/diagnose_e3_e5_low_recall.py',
        *sorted((ROOT/'src').rglob('*.py'))]


def hashes():
    return {str(p.relative_to(ROOT)):sealed.sha(p) for p in SOURCE}


def regression(out):
    dest=v2.unique_dir(out/'execution_v4_cpu_tests','cpu')
    before=hashes()
    tests=[TEST,ROOT/'tests/test_joint_optimization_execution_v3.py',ROOT/'tests/test_joint_optimization_execution_v2.py',ROOT/'tests/test_joint_optimization_pipeline.py',
        ROOT/'tests/test_checkpoint_policy.py',ROOT/'tests/test_checkpoint_resume.py',
        ROOT/'tests/test_e5_cpu_recovery.py',ROOT/'tests/test_shrunk_node_training.py']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    argv=[sys.executable,'-m','pytest','-q','-p','no:cacheprovider',*map(str,tests)]
    with (dest/'tests.log').open('x') as log:
        result=subprocess.run(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    code=result.returncode if hashes()==before else 1
    sealed.save(dest/'result.json',dict(status='PASS' if code==0 else 'FAIL',exitcode=code,code_sha256=before,
        argv=argv,log_sha256=sealed.sha(dest/'tests.log'),at=sealed.now()))
    (dest/'exitcode').write_text(str(code)+'\n')
    print(dest,'PASS' if code==0 else 'FAIL',flush=True)
    return code


def require_tests(out):
    for path in sorted((out/'execution_v4_cpu_tests').glob('*/result.json'),reverse=True):
        r=sealed.read(path)
        if (r['status']=='PASS' and r['code_sha256']==hashes() and
            (path.parent/'exitcode').read_text().strip()=='0' and r['log_sha256']==sealed.sha(path.parent/'tests.log')):
            return path
    raise RuntimeError('Current v4 code has no passing CPU regressions')


def footprint(roots):
    """Only owned paths; never follow symlinks, deduplicate hardlinks."""
    seen=set();allocated=logical=files=0;per_root={}
    for root in map(Path,roots):
        a=b=n=0
        if root.is_symlink() or not root.exists():
            continue
        paths=[root] if root.is_file() else (Path(base)/name for base,dirs,names in os.walk(root,followlinks=False) for name in names)
        for path in paths:
            try:
                st=path.lstat()
            except FileNotFoundError:
                continue  # Atomic temporary renamed/unlinked during scan; next sample/fd trace covers it.
            if not stat.S_ISREG(st.st_mode) or (st.st_dev,st.st_ino) in seen:
                continue
            seen.add((st.st_dev,st.st_ino));a+=st.st_blocks*512;b+=st.st_size;n+=1
        allocated+=a;logical+=b;files+=n
        per_root[str(root)]=dict(allocated_bytes=a,logical_bytes=b,files=n)
    return dict(owned_allocated_bytes=allocated,owned_logical_bytes=logical,files=files,per_root=per_root),seen


def descendants(pid):
    result=[pid]
    for parent in result:
        try:
            result.extend(int(x) for x in Path(f'/proc/{parent}/task/{parent}/children').read_text().split() if int(x) not in result)
        except (FileNotFoundError,ProcessLookupError):
            pass
    return result


def process_writes(pid,roots,seen):
    """Detect writable regular files outside owned roots, and count open-unlinked temps."""
    roots=[Path(p).resolve() for p in roots]
    io={};external=[];deleted=0;seen=set(seen)
    for child in descendants(pid):
        try:
            io[str(child)]={k:int(value.strip()) for line in Path(f'/proc/{child}/io').read_text().splitlines()
                           for k,value in [line.split(':',1)]}
            for fd in Path(f'/proc/{child}/fd').iterdir():
                try:
                    info=Path(f'/proc/{child}/fdinfo/{fd.name}').read_text()
                    flags=int(next(line.split(':',1)[1].strip() for line in info.splitlines() if line.startswith('flags:')),8)
                    if flags & os.O_ACCMODE == os.O_RDONLY:
                        continue
                    st=fd.stat()
                    if not stat.S_ISREG(st.st_mode):
                        continue
                    target=os.readlink(fd);clean=Path(target.removesuffix(' (deleted)'))
                    own=any(clean==root or clean.is_relative_to(root) for root in roots)
                    shared_memory=target.startswith('/dev/shm/') or target.startswith('/memfd:')
                    if not own and not shared_memory:
                        external.append(dict(pid=child,fd=int(fd.name),path=target,allocated_bytes=st.st_blocks*512))
                    if st.st_nlink==0 and (st.st_dev,st.st_ino) not in seen and (own or shared_memory):
                        deleted+=st.st_blocks*512;seen.add((st.st_dev,st.st_ino))
                except (FileNotFoundError,ProcessLookupError):
                    pass
        except (FileNotFoundError,ProcessLookupError):
            pass
    return dict(process_io=io,unaccounted_writable_files=external,open_unlinked_allocated_bytes=deleted,
        cumulative_process_write_bytes=sum(v.get('write_bytes',0) for v in io.values()))


def disk_gate(sample,budget,*,data_free,root_free):
    metrics=dict(owned_allocated_bytes=sample['owned_allocated_bytes']+sample.get('open_unlinked_allocated_bytes',0),
                 data_free_bytes=data_free,root_free_bytes=root_free,
                 unaccounted_writable_files=sample.get('unaccounted_writable_files',[]))
    thresholds=dict(owned_budget_bytes=budget,data_safety_floor_bytes=20*GIB,root_safety_floor_bytes=2*GIB)
    triggers=[]
    if metrics['owned_allocated_bytes']>budget:triggers.append('owned_allocated_bytes > owned_budget_bytes')
    if data_free<20*GIB:triggers.append('data_free_bytes < data_safety_floor_bytes')
    if root_free<2*GIB:triggers.append('root_free_bytes < root_safety_floor_bytes')
    if metrics['unaccounted_writable_files']:triggers.append('writable regular files outside owned roots')
    return dict(status='PASS' if not triggers else 'BLOCKED',metrics=metrics,thresholds=thresholds,triggers=triggers)


def scoped_environment(data, cpu=False):
    env=os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1',WANDB_MODE='disabled',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    redirects={'TMPDIR':'tmp','TMP':'tmp','TEMP':'tmp','XDG_CACHE_HOME':'cache','MPLCONFIGDIR':'cache/matplotlib',
        'TORCH_HOME':'cache/torch','HF_HOME':'cache/huggingface','NUMBA_CACHE_DIR':'cache/numba',
        'CUDA_CACHE_PATH':'cache/cuda','TORCH_EXTENSIONS_DIR':'cache/torch_extensions','TRITON_CACHE_DIR':'cache/triton',
        'JOBLIB_TEMP_FOLDER':'tmp/joblib','WANDB_DIR':'cache/wandb','WANDB_CACHE_DIR':'cache/wandb_cache',
        'WANDB_DATA_DIR':'cache/wandb_data','PYTHONPYCACHEPREFIX':'cache/pycache'}
    for key,relative in redirects.items():
        path=data/relative;path.mkdir(parents=True,exist_ok=True);env[key]=str(path)
    if cpu:env['CUDA_VISIBLE_DEVICES']=''
    return env


def verify_original(accepted):
    run=Path(accepted['run_dir'])
    assert audit.inventory(run)==accepted['original_inventory'],'Accepted original E3 files changed'


def check_identity(out,attempt):
    sealed.check_seal(out);v2.verify_proof(out)
    identity=sealed.read(attempt/'identity.json')
    if (attempt/'cuda_acceptance_link.json').exists():
        link=sealed.read(attempt/'cuda_acceptance_link.json')
        assert link['identity_sha256']==sealed.sha(attempt/'identity.json')
        assert link['proof_sha256']==sealed.sha(attempt/'cuda_check/proof.json')
    assert identity['code_sha256']==hashes(),'v4 code changed after acceptance'
    assert sealed.sha(out/'prepared_05/plan.json')==identity['plan_sha256']
    assert sealed.sha(identity['checkpoint_acceptance'])==identity['checkpoint_acceptance_sha256']
    assert sealed.sha(identity['cpu_tests'])==identity['cpu_tests_sha256']
    require_tests(out)
    return identity


def create_attempt(out,acceptance,target):
    sealed.check_seal(out);v2.verify_proof(out);tests=require_tests(out)
    accepted=sealed.read(acceptance)
    assert accepted['status']=='PASS';verify_original(accepted)
    attempt=v2.unique_dir(out/'recovery_attempts_v4','recovery')
    data=Path('/mnt/datadisk/yby/mstc/artifacts/joint_shrunk_recovery_v4')/attempt.name
    data.mkdir(parents=True,exist_ok=False)
    for e in [3,5]:(attempt/f'E{e}').mkdir();(data/f'E{e}').mkdir()
    for source in [ENTRY,Path(gpu_guard.__file__).resolve(),Path(audit.__file__).resolve(),TEST]:shutil.copyfile(source,attempt/source.name)
    old=ROOT/'scripts/joint_optimization_execution_v3.py'
    (attempt/'execution_v3_to_v4.diff').write_text(''.join(difflib.unified_diff(old.read_text().splitlines(True),
        ENTRY.read_text().splitlines(True),fromfile=str(old),tofile=str(ENTRY))))
    sealed.save(attempt/'identity.json',dict(created_at=sealed.now(),data_root=str(data),code_sha256=hashes(),
        plan_sha256=sealed.sha(out/'prepared_05/plan.json'),treatment_sha256=sealed.sha(out/'prepared_05/treatment.yml'),
        checkpoint_acceptance=str(acceptance.resolve()),checkpoint_acceptance_sha256=sealed.sha(acceptance),
        cpu_tests=str(tests.resolve()),cpu_tests_sha256=sealed.sha(tests),
        target_gpu=target,entry_sha256=sealed.sha(ENTRY),
        scope='Physical GPU 1 binding, target-only gates/continuous monitoring, CUDA mapping acceptance; native scientific protocol unchanged.',
        difference_sha256=sealed.sha(attempt/'execution_v3_to_v4.diff')))
    return attempt


def setup_e3(out,attempt,accepted):
    identity=sealed.read(attempt/'identity.json')
    artifact=Path(identity['data_root'])/'E3/artifacts'
    run=artifact/'THEIA_E3/runs/orthrus/seed_0'
    run.mkdir(parents=True,exist_ok=False)
    original=Path(accepted['run_dir'])
    (run/'checkpoints').symlink_to(original/'checkpoints',target_is_directory=True)
    # The shared data volume does not support copystat/chmod/utime. Copy
    # immutable contents only; preserve identity by SHA256, not inode metadata.
    (run/'metadata').mkdir()
    for source in (original/'metadata').iterdir():
        assert source.is_file(), 'Unexpected metadata subdirectory'
        target=run/'metadata'/source.name
        shutil.copyfile(source,target)
        assert sealed.sha(source)==sealed.sha(target), 'Metadata copy differs'
    runtime=sealed.read(original/'runtime.json')
    sealed.save(run/'runtime.json',dict(training=runtime['training'],model=runtime['model']))
    shutil.copyfile(original/'config_resolved.yml',run/'config_resolved.yml')
    shutil.copyfile(out/'e3_training.exitcode',attempt/'E3/e3_training.exitcode')
    sealed.save(attempt/'E3/recovery_route.json',dict(original_run=str(original),recovered_run=str(run),
        original_checkpoint_references='native read-only loader; hashes checked before and after each E3 stage',
        original_outputs='Retained unchanged; regenerate all validation outputs to keep native RNG trajectory',
        original_training_exitcode_sha256=sealed.sha(out/'e3_training.exitcode'),
        copied_metadata={p.name:sealed.sha(p) for p in (run/'metadata').iterdir()}))
    # Prove the runtime-path change preserves the checkpoint's effective config.
    audit.frozen_cfg(out,3,artifact)
    return artifact,run


def stage_command(out,e,stage,artifact=None):
    assert not (e==3 and stage=='training'),'E3 retraining is prohibited'
    argv=list(sealed.read(out/'prepared_05/plan.json')['datasets'][str(e)]['commands'][stage])
    if artifact is not None:
        argv[argv.index('--artifact-root')+1]=str(artifact)
    return argv


def record_prelaunch_block(management,e,stage,gate):
    sealed.save(management/f'{stage}_status.json',dict(status='BLOCKED',exitcode=2,pid=None,
        started_at=sealed.now(),finished_at=sealed.now(),gate=gate,process_started=False))
    with (management/f'e{e}_{stage}.exitcode').open('x') as f:f.write('2\n')


def run_stage(out,attempt,e,stage,argv,roots,env,*,gpu):
    management=attempt/f'E{e}'
    budget=(6 if e==3 else 20)*GIB
    target=sealed.read(attempt/'identity.json')['target_gpu']
    env=gpu_guard.environment(env,target,cpu=not gpu)
    gpu_guard.require_environment(env,target,cpu=not gpu)
    if gpu:
        gate=gpu_guard.resource_gate(target,e)
        sealed.save(management/f'{stage}_resource_gate.json',gate)
        if gate['status']!='PASS':record_prelaunch_block(management,e,stage,gate)
        if gate['status']!='PASS':raise gpu_guard.TargetGPUBlocked(repr(gate))
    roots=list(roots)+[management]+[p for p in attempt.iterdir() if p.is_file()]
    snapshot,seen=footprint(roots)
    pre=disk_gate(snapshot,budget,data_free=shutil.disk_usage('/mnt/datadisk').free,root_free=shutil.disk_usage('/').free)
    sealed.save(management/f'{stage}_disk_preflight.json',pre)
    if pre['status']!='PASS':record_prelaunch_block(management,e,stage,pre)
    assert pre['status']=='PASS',pre
    state=dict(status='RUNNING',argv=argv,started_at=sealed.now(),gpu=gpu,owned_roots=list(map(str,roots)),
        target_gpu=target,runtime_environment={k:env.get(k) for k in ['CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','MSTC_TARGET_GPU_UUID']})
    status_path=management/f'{stage}_status.json'
    sealed.save(status_path,state)
    process=None;started=time.monotonic();highwater=0;write_highwater=0;gpu_peak_mib=0;own_gpu_seen=False;baseline_free=shutil.disk_usage('/mnt/datadisk').free
    try:
        with (management/f'{stage}.log').open('x') as log:
            process=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            (management/f'{stage}.pid').write_text(str(process.pid)+'\n');state['pid']=process.pid
            next_scan=0
            while True:
                current=time.monotonic();scan_seconds=0
                if current>=next_scan or process.poll() is not None:
                    tick=time.monotonic();snapshot,seen=footprint(roots);scan_seconds=time.monotonic()-tick;next_scan=current+30
                io=dict(process_io={},unaccounted_writable_files=[],open_unlinked_allocated_bytes=0,
                    cumulative_process_write_bytes=write_highwater)
                if process.poll() is None:
                    try:
                        io=process_writes(process.pid,roots,seen)
                    except PermissionError:
                        # /proc/fd briefly denies access while this exact child
                        # exits. A live process with persistent denial still
                        # fails closed; never silently omit an active writer.
                        time.sleep(.01)
                        if process.poll() is None:
                            io=process_writes(process.pid,roots,seen)
                sample=dict(**snapshot,**io)
                gpu_sample=None
                if gpu:
                    allowed={os.getpid()}|set(descendants(process.pid))
                    gpu_sample=gpu_guard.resource_gate(target,own_pids=allowed)
                    card=gpu_sample.get('target_gpu',{})
                    gpu_peak_mib=max(gpu_peak_mib,card.get('memory_used_mib') or 0)
                    own_gpu_seen=own_gpu_seen or any(p['pid']==process.pid for p in gpu_sample.get('own_target_processes',[]))
                    sample['gpu_monitor']=gpu_sample
                    if gpu_sample['status']!='PASS' and process.poll() is None:
                        sealed.save(management/f'{stage}_gpu_gate_trigger.json',gpu_sample)
                        sealed.terminate_own(process)
                        raise gpu_guard.TargetGPUBlocked(repr(gpu_sample))
                gate=disk_gate(sample,budget,data_free=shutil.disk_usage('/mnt/datadisk').free,root_free=shutil.disk_usage('/').free)
                highwater=max(highwater,gate['metrics']['owned_allocated_bytes'])
                write_highwater=max(write_highwater,io['cumulative_process_write_bytes'])
                sample.update(at=sealed.now(),gate=gate,owned_scan_seconds=scan_seconds,
                    sampled_owned_peak_bytes=highwater,process_write_bytes_highwater=write_highwater,
                    global_data_free_delta_observation_only=baseline_free-gate['metrics']['data_free_bytes'],
                    peak_note='Owned footprint sampled every 30s; process I/O and open-unlinked temps every 15s. Between-sample resident peaks cannot be reconstructed exactly.')
                with (management/f'{stage}_monitor.jsonl').open('a') as monitor:
                    monitor.write(json.dumps(sample,ensure_ascii=False)+'\n')
                sealed.save(status_path,dict(**state,monitor=sample),mutable=True)
                if gate['status']!='PASS':
                    sealed.save(management/f'{stage}_gate_trigger.json',sample)
                    sealed.terminate_own(process)
                    raise RuntimeError('Owned-storage/safety/write-path gate: '+repr(gate))
                if process.poll() is not None:break
                time.sleep(15)
            code=process.returncode
        state.update(status='PASS' if code==0 else 'FAIL',exitcode=code)
        assert code==0,f'{e}/{stage} failed ({code})'
        if gpu:assert own_gpu_seen,'No physical GPU 1 execution was observed'
    except BaseException as exc:
        if process is not None:sealed.terminate_own(process)
        state.update(status='BLOCKED' if isinstance(exc,gpu_guard.TargetGPUBlocked) else 'FAIL',error=repr(exc),exitcode=process.returncode if process is not None else 1)
        raise
    finally:
        state.update(finished_at=sealed.now(),wall_seconds=time.monotonic()-started,
            sampled_owned_peak_bytes=highwater,process_write_bytes_highwater=write_highwater,
            sampled_target_gpu_peak_memory_mib=gpu_peak_mib,physical_gpu_execution_observed=own_gpu_seen)
        sealed.save(status_path,state,mutable=True)
        with (management/f'e{e}_{stage}.exitcode').open('x') as f:f.write(str(state.get('exitcode',1))+'\n')
    return state


def evaluate_adapter(out,attempt,e,run):
    """Adapt file routing only; execute the sealed evaluator's original main()."""
    import evaluate_joint_optimization as evaluator
    sealed.check_seal(out)
    context=attempt/f'E{e}'
    original_plan=sealed.read(out/'prepared_05/plan.json')
    routed=copy.deepcopy(original_plan)
    routed['datasets'][str(e)]['run_dir']=str(run)
    manifest=sealed.read(original_plan['datasets'][str(e)]['manifest'])
    assert sealed.sha(original_plan['datasets'][str(e)]['manifest'])==original_plan['datasets'][str(e)]['manifest_sha256']
    route_path=context/'evaluation_route.json'
    sealed.save(route_path,dict(frozen_plan_sha256=sealed.sha(out/'prepared_05/plan.json'),dataset=e,
        runtime_run_dir=str(run),allowed_override='datasets[e].run_dir only; original plan stays immutable',
        manifest_source=original_plan['datasets'][str(e)]['manifest'],
        split_routing_note='Original helper hardcodes THEIA_E3 glob; resolve frozen dataset manifest directly for E3/E5.'))
    def read(path):
        return routed if Path(path)==context/'prepared_05/plan.json' else sealed.read(path)
    def split_specs(split):
        return [w for w in manifest['window_specs'] if w['split_name']==split]
    with patch.object(evaluator,'read',read),patch.object(evaluator.paired,'split_specs',split_specs):
        evaluator.main(context,e)


def acceptance(out,attempt,e,run):
    from detection.checkpoint_policy import validate_selection_manifest
    from yacs.config import CfgNode
    import yaml
    cfg=CfgNode(yaml.safe_load((run/'config_resolved.yml').read_text()))
    selection=sealed.read(run/'checkpoint_selection.json')
    assert set(selection['per_epoch_validation_objective'])=={f'model_epoch_{i}' for i in range(1,7)}
    selected=validate_selection_manifest(selection,cfg)
    assert selection['val_objective_field']=='score_raw'
    assert {p.name for p in (run/'edge_scores/test').iterdir()}=={selected}
    runtime=sealed.read(run/'runtime.json')
    testing=runtime['testing']
    assert testing['selection_validation_passes']==6 and testing['canonical_validation_passes']==1 and testing['test_passes']==1
    assert runtime['training']['processed_event_count']==sealed.read(out/f'e{e}_shrunk_weights.json')['target_events']*6
    actual=sealed.read(run/'metadata/train_loss_weighting.json');expected=sealed.read(out/f'e{e}_shrunk_weights.json')
    assert all(actual[k]==expected[k] for k in actual)
    result=sealed.read(attempt/f'E{e}/e{e}_evaluation/result.json')
    assert result['status']=='PASS' and result['selected_checkpoint']==selected and result['calibrated_csv_copies']==0
    assert result['checkpoint_sha256']==sealed.sha(run/'checkpoints'/selected/'checkpoint.pt')
    assert not list(run.rglob('*calibrated*.csv'))
    sealed.save(attempt/f'E{e}/acceptance.json',dict(status='PASS',selected_checkpoint=selected,
        result_sha256=sealed.sha(attempt/f'E{e}/e{e}_evaluation/result.json'),training_reexecuted=e!=3,
        protocol='Frozen six-epoch val score_raw selection, best-only test, sealed CPU streaming evaluation'))
    return result


def worker(out,attempt):
    gpu_guard.require_environment(os.environ,sealed.read(attempt/'identity.json')['target_gpu'])
    sealed.save(attempt/'worker_claim.json',dict(pid=os.getpid(),started_at=sealed.now()))
    (attempt/'pid').write_text(str(os.getpid())+'\n')
    state=dict(status='RUNNING',started_at=sealed.now(),pid=os.getpid(),datasets={})
    sealed.save(attempt/'status.json',state)
    code=1
    with (attempt/'worker.log').open('x',buffering=1) as log,redirect_stdout(log),redirect_stderr(log):
        try:
            with v2.execution_lock(out):
                identity=check_identity(out,attempt)
                assert sealed.read(attempt/'cuda_check/proof.json')['status']=='PASS'
                assert (attempt/'cuda_check/exitcode').read_text().strip()=='0'
                proof=sealed.read(attempt/'cuda_check/proof.json')
                assert proof['target']==identity['target_gpu']
                assert (attempt/'cuda_acceptance_link.json').exists()
                accepted=sealed.read(identity['checkpoint_acceptance']);verify_original(accepted)
                expected={r['epoch']:r['files']['checkpoint.pt']['sha256'] for r in accepted['checkpoints']}
                assert {r['epoch']:r['sha256'] for r in proof['child']['checkpoint_deserialization']}==expected
                data=Path(identity['data_root'])
                plan=sealed.read(out/'prepared_05/plan.json')
                artifact3,run3=setup_e3(out,attempt,accepted)
                roots3=[data/'E3',Path(plan['datasets']['3']['artifact_root']),
                    *[p for p in out.glob('e3_*') if p.is_file()],out/'resource_monitor.jsonl']
                env3=scoped_environment(data/'E3')
                import yaml
                saved3=yaml.safe_load(Path(plan['datasets']['3']['control_training_config']).read_text())
                for key in ['host','user','port']:env3['ORTHRUS_DB_'+key.upper()]=str(saved3['database'][key])
                state['datasets']['3']=dict(status='RECOVERING',training='REUSED_ACCEPTED_6_EPOCHS')
                sealed.save(attempt/'status.json',state,mutable=True)
                run_stage(out,attempt,3,'checkpoint_validation_and_test',stage_command(out,3,'checkpoint_validation_and_test',artifact3),
                    roots3,env3,gpu=True)
                verify_original(accepted)
                argv=[sys.executable,'-u',str(ENTRY),'--analysis',str(out),'--mode','evaluate',
                    '--attempt',str(attempt),'--dataset','3','--run-dir',str(run3)]
                run_stage(out,attempt,3,'cpu_evaluation',argv,roots3,
                    scoped_environment(data/'E3',cpu=True),gpu=False)
                result3=acceptance(out,attempt,3,run3);verify_original(accepted)
                state['datasets']['3'].update(status='PASS',run_dir=str(run3),metrics=result3['metrics'])
                sealed.save(attempt/'status.json',state,mutable=True)
                check_identity(out,attempt)
                d=plan['datasets']['5'];artifact5=Path(d['artifact_root']);run5=Path(d['run_dir'])
                assert not artifact5.exists(),'E5 artifacts exist; no overwrite/automatic training retry'
                env5=scoped_environment(data/'E5')
                import yaml
                saved=yaml.safe_load(Path(d['control_training_config']).read_text())
                for key in ['host','user','port']:env5['ORTHRUS_DB_'+key.upper()]=str(saved['database'][key])
                state['datasets']['5']=dict(status='RUNNING')
                sealed.save(attempt/'status.json',state,mutable=True)
                roots5=[data/'E5',artifact5]
                run_stage(out,attempt,5,'training',stage_command(out,5,'training'),roots5,env5,gpu=True)
                check_identity(out,attempt)
                run_stage(out,attempt,5,'checkpoint_validation_and_test',stage_command(out,5,'checkpoint_validation_and_test'),roots5,env5,gpu=True)
                argv=[sys.executable,'-u',str(ENTRY),'--analysis',str(out),'--mode','evaluate',
                    '--attempt',str(attempt),'--dataset','5','--run-dir',str(run5)]
                run_stage(out,attempt,5,'cpu_evaluation',argv,roots5,scoped_environment(data/'E5',cpu=True),gpu=False)
                result5=acceptance(out,attempt,5,run5)
                state['datasets']['5'].update(status='PASS',run_dir=str(run5),metrics=result5['metrics'])
                check_identity(out,attempt);verify_original(accepted)
                state['status']='PASS';code=0
        except BaseException as exc:
            import traceback
            traceback.print_exc()
            code=2 if isinstance(exc,gpu_guard.TargetGPUBlocked) else 1
            state.update(status='BLOCKED' if code==2 else 'FAILED',error=repr(exc))
            for dataset_state in state['datasets'].values():
                if dataset_state['status'] in ['RUNNING','RECOVERING']:
                    dataset_state.update(status=state['status'],error=repr(exc),finished_at=sealed.now())
        finally:
            state.update(exitcode=code,finished_at=sealed.now())
            sealed.save(attempt/'status.json',state,mutable=True)
            (attempt/'exitcode').write_text(str(code)+'\n')
            print(json.dumps(state,ensure_ascii=False),flush=True)
    return code


def launch(out,acceptance_path,target):
    attempt=create_attempt(out,acceptance_path,target)
    identity=sealed.read(attempt/'identity.json')
    data=Path(identity['data_root'])
    env=gpu_guard.environment(scoped_environment(data/'cuda_check'),target)
    probe_argv=[sys.executable,'-u',str(ENTRY),'--analysis',str(out),'--mode','probe-child','--attempt',str(attempt)]
    try:
        gpu_guard.cuda_acceptance(target,attempt/'cuda_check',probe_argv,env)
        sealed.save(attempt/'cuda_acceptance_link.json',dict(identity_sha256=sealed.sha(attempt/'identity.json'),
            proof_sha256=sealed.sha(attempt/'cuda_check/proof.json')))
    except BaseException as exc:
        code=2 if isinstance(exc,gpu_guard.TargetGPUBlocked) else 1
        sealed.save(attempt/'status.json',dict(status='BLOCKED' if code==2 else 'FAILED',exitcode=code,error=repr(exc),formal_started=False))
        (attempt/'exitcode').write_text(str(code)+'\n')
        print(attempt,'CUDA_ACCEPTANCE_NOT_PASS',repr(exc),flush=True)
        return code
    argv=['nohup',sys.executable,'-u',str(ENTRY),'--analysis',str(out),'--mode','worker','--attempt',str(attempt)]
    sealed.save(attempt/'launch.json',dict(argv=argv,at=sealed.now(),e3_retraining=False))
    with (attempt/'nohup.log').open('x') as log:
        process=subprocess.Popen(argv,cwd=ROOT,env=gpu_guard.environment(dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),target),stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (attempt/'launcher.pid').write_text(str(process.pid)+'\n')
    print(attempt,'PID',process.pid,flush=True)
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--mode',choices=['cpu-tests','launch','worker','evaluate','probe-child'],required=True)
    parser.add_argument('--gpu-index',type=int,choices=[1],default=1)
    parser.add_argument('--gpu-uuid',default=gpu_guard.GPU1_UUID)
    parser.add_argument('--attempt',type=Path)
    parser.add_argument('--acceptance',type=Path)
    parser.add_argument('--dataset',type=int,choices=[3,5])
    parser.add_argument('--run-dir',type=Path)
    args=parser.parse_args();out=args.analysis.resolve()
    if args.mode=='cpu-tests':return regression(out)
    if args.mode=='launch':return launch(out,args.acceptance.resolve(),gpu_guard.resolve_target(args.gpu_index,args.gpu_uuid))
    if args.mode=='probe-child':
        accepted=sealed.read(sealed.read(args.attempt/'identity.json')['checkpoint_acceptance'])
        return gpu_guard.probe_child(args.attempt/'cuda_check',Path(accepted['run_dir'])/'checkpoints')
    if args.mode=='worker':return worker(out,args.attempt.resolve())
    check_identity(out,args.attempt.resolve())
    evaluate_adapter(out,args.attempt.resolve(),args.dataset,args.run_dir.resolve())
    return 0


if __name__=='__main__':sys.exit(main())
