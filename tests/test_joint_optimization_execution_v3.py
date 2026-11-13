"""CPU regressions: ownership accounting and path-only recovery compatibility."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace as NS

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import joint_optimization_execution_v3 as v


def gate(sample,**kwargs):
    return v.disk_gate(sample,6*v.GIB,data_free=kwargs.get('data',100*v.GIB),root_free=kwargs.get('root',10*v.GIB))


def test_other_users_disk_growth_does_not_consume_our_budget(tmp_path):
    owned=tmp_path/'ours';other=tmp_path/'other';owned.mkdir();other.mkdir()
    (owned/'result').write_bytes(b'x'*65536)
    before,_=v.footprint([owned])
    (other/'unrelated').write_bytes(b'x'*2**20)
    after,_=v.footprint([owned])
    assert after==before
    assert gate(after,data=80*v.GIB)['status']=='PASS'
    assert gate(after,data=19*v.GIB)['status']=='BLOCKED'


@pytest.mark.parametrize('kind', ['budget','data','root','unowned'])
def test_all_safety_gates_remain_active(kind):
    sample=dict(owned_allocated_bytes=0)
    kwargs={}
    if kind=='budget':sample['owned_allocated_bytes']=6*v.GIB+1
    if kind=='data':kwargs['data']=20*v.GIB-1
    if kind=='root':kwargs['root']=2*v.GIB-1
    if kind=='unowned':sample['unaccounted_writable_files']=[{'path':'unowned'}]
    result=gate(sample,**kwargs)
    assert result['status']=='BLOCKED' and result['triggers']
    assert set(result)=={'status','metrics','thresholds','triggers'}


def test_allocated_vs_logical_hardlinks_and_symlinks(tmp_path):
    owned=tmp_path/'ours';owned.mkdir()
    sparse=owned/'sparse'
    with sparse.open('wb') as f:f.truncate(2**30)
    regular=owned/'regular';regular.write_bytes(b'x'*65536)
    os.link(regular,owned/'same_inode')
    other=tmp_path/'other';other.mkdir();(other/'unowned').write_bytes(b'x'*65536)
    (owned/'pointer').symlink_to(other,target_is_directory=True)
    result,_=v.footprint([owned,regular])
    assert result['files']==2
    assert result['owned_allocated_bytes']==sparse.stat().st_blocks*512+regular.stat().st_blocks*512
    assert result['owned_logical_bytes']==2**30+65536


def test_real_process_unlinked_temporary_is_counted_and_outside_write_detected(tmp_path):
    owned=tmp_path/'ours';owned.mkdir();outside=tmp_path/'outside'
    ready=owned/'ready'
    source='''import os,sys,time
from pathlib import Path
root=Path(sys.argv[1]);outside=Path(sys.argv[2])
f=(root/'deleted').open('w+b');f.write(b'x'*65536);f.flush();os.fsync(f.fileno());os.unlink(f.name)
g=outside.open('w+b');g.write(b'x'*8192);g.flush();os.fsync(g.fileno())
(root/'ready').write_text('ready')
time.sleep(20)
'''
    process=subprocess.Popen([sys.executable,'-c',source,str(owned),str(outside)])
    try:
        for _ in range(100):
            if ready.exists():break
            time.sleep(.02)
        assert ready.exists()
        files,seen=v.footprint([owned])
        sample=v.process_writes(process.pid,[owned],seen)
        assert sample['open_unlinked_allocated_bytes']>=65536
        assert sample['cumulative_process_write_bytes']>=65536
        assert any(r['path']==str(outside) for r in sample['unaccounted_writable_files'])
        assert gate(dict(**files,**sample))['status']=='BLOCKED'
    finally:
        process.terminate();process.wait(timeout=5)


def test_temporary_cache_redirects_inside_owned_root(tmp_path):
    env=v.scoped_environment(tmp_path,cpu=True)
    for key in ['TMPDIR','TMP','TEMP','XDG_CACHE_HOME','MPLCONFIGDIR','TORCH_HOME','HF_HOME','NUMBA_CACHE_DIR','JOBLIB_TEMP_FOLDER','WANDB_DIR','WANDB_CACHE_DIR','WANDB_DATA_DIR','PYTHONPYCACHEPREFIX']:
        assert Path(env[key]).is_relative_to(tmp_path) and Path(env[key]).is_dir()
    assert env['HOME']==os.environ['HOME'] and env['CUDA_VISIBLE_DEVICES']==''
    assert env['PYTHONDONTWRITEBYTECODE']=='1'


def test_e3_training_is_unconditionally_prohibited():
    with pytest.raises(AssertionError,match='retraining'):
        v.stage_command(ROOT/'analysis/joint_optimization_20261029',3,'training')


def test_runtime_artifact_override_keeps_frozen_config_identity(tmp_path):
    out=ROOT/'analysis/joint_optimization_20261029'
    a=v.audit.frozen_cfg(out,3)
    b=v.audit.frozen_cfg(out,3,tmp_path/'fresh')
    from mstc.experiment_utils import stable_config_hash,sanitized_config
    assert stable_config_hash(a)==stable_config_hash(b)
    assert sanitized_config(a)==sanitized_config(b)
    assert a._run_dir!=b._run_dir
    assert not (tmp_path/'fresh').exists()


@pytest.mark.parametrize('dataset',[3,5])
def test_sealed_evaluation_adapter_changes_only_runtime_run_and_manifest_route(tmp_path,monkeypatch,dataset):
    import evaluate_joint_optimization as evaluator
    import copy
    out=tmp_path/'original';out.mkdir();(out/'prepared_05').mkdir()
    manifest=tmp_path/'manifest.json'
    v.sealed.save(manifest,{'window_specs':[{'split_name':'train','num_events':11}, {'split_name':'val','num_events':7}, {'split_name':'test','num_events':13}]})
    plan={'source_sha256':{},'datasets':{str(dataset):{'run_dir':'original','manifest':str(manifest),'manifest_sha256':v.sealed.sha(manifest),'protocol':'unchanged'}}}
    v.sealed.save(out/'prepared_05/plan.json',plan)
    before=(out/'prepared_05/plan.json').read_bytes()
    attempt=tmp_path/'attempt';(attempt/f'E{dataset}').mkdir(parents=True)
    run=tmp_path/'fresh_run'
    monkeypatch.setattr(v.sealed,'check_seal',lambda out:plan)
    called=[]
    original_split=evaluator.paired.split_specs
    def original_main(context,e):
        routed=evaluator.read(context/'prepared_05/plan.json')
        expected=copy.deepcopy(plan);expected['datasets'][str(e)]['run_dir']=str(run)
        assert routed==expected
        assert evaluator.paired.split_specs('val')==[{'split_name':'val','num_events':7}]
        assert evaluator.paired.split_specs('test')==[{'split_name':'test','num_events':13}]
        called.append(e)
    monkeypatch.setattr(evaluator,'main',original_main)
    v.evaluate_adapter(out,attempt,dataset,run)
    assert called==[dataset] and (out/'prepared_05/plan.json').read_bytes()==before
    assert evaluator.paired.split_specs is original_split


def test_setup_reuses_original_checkpoints_and_copies_only_metadata(tmp_path,monkeypatch):
    original=tmp_path/'original';(original/'metadata').mkdir(parents=True);(original/'checkpoints').mkdir()
    (original/'checkpoints/checkpoint.pt').write_bytes(b'protected')
    (original/'metadata/time_statistics.json').write_text('{}')
    (original/'config_resolved.yml').write_text('original')
    v.sealed.save(original/'runtime.json',{'training':{'processed_event_count':42},'model':{'parameter_count':1},'stale_testing':'discard'})
    out=tmp_path/'analysis';out.mkdir();(out/'e3_training.exitcode').write_text('0\n')
    attempt=out/'attempt';(attempt/'E3').mkdir(parents=True)
    v.sealed.save(attempt/'identity.json',{'data_root':str(tmp_path/'data')})
    monkeypatch.setattr(v.audit,'frozen_cfg',lambda *a:None)
    def forbidden_stat(*args,**kwargs):
        raise PermissionError('volume does not support metadata mutations')
    monkeypatch.setattr(v.shutil,'copystat',forbidden_stat)
    accepted={'run_dir':str(original)}
    before=v.audit.inventory(original)
    artifact,run=v.setup_e3(out,attempt,accepted)
    assert (run/'checkpoints').is_symlink()
    assert (run/'metadata/time_statistics.json').read_bytes()==(original/'metadata/time_statistics.json').read_bytes()
    assert v.sealed.read(run/'runtime.json')=={'training':{'processed_event_count':42},'model':{'parameter_count':1}}
    assert v.audit.inventory(original)==before
    with pytest.raises(FileExistsError):v.setup_e3(out,attempt,accepted)
    assert v.audit.inventory(original)==before


def test_real_cpu_stage_writes_owned_files_and_saves_log_pid_exitcode(tmp_path,monkeypatch):
    out=tmp_path/'analysis';out.mkdir();attempt=out/'attempt';(attempt/'E3').mkdir(parents=True)
    data=tmp_path/'data';data.mkdir()
    env=v.scoped_environment(data,cpu=True)
    argv=[sys.executable,'-c','from pathlib import Path; import os; Path(os.environ["TMPDIR"],"own").write_bytes(b"x"*65536); print("real child complete")']
    # Real subprocess with a short polling interval only in this test.
    native_sleep=v.time.sleep
    monkeypatch.setattr(v.time,'sleep',lambda seconds:native_sleep(.02))
    result=v.run_stage(out,attempt,3,'cpu_evaluation',argv,[data],env,gpu=False)
    assert result['status']=='PASS'
    context=attempt/'E3'
    assert (context/'e3_cpu_evaluation.exitcode').read_text().strip()=='0'
    assert (context/'cpu_evaluation.pid').exists()
    assert 'real child complete' in (context/'cpu_evaluation.log').read_text()
    assert result['sampled_owned_peak_bytes']>=65536


def test_real_failed_stage_preserves_artifacts_and_exitcode(tmp_path,monkeypatch):
    out=tmp_path/'analysis';out.mkdir();attempt=out/'attempt';(attempt/'E3').mkdir(parents=True)
    data=tmp_path/'data';data.mkdir()
    env=v.scoped_environment(data,cpu=True)
    native_sleep=v.time.sleep;monkeypatch.setattr(v.time,'sleep',lambda seconds:native_sleep(.02))
    with pytest.raises(AssertionError,match='failed'):
        v.run_stage(out,attempt,3,'cpu_evaluation',[sys.executable,'-c','raise RuntimeError("test failure")'],[data],env,gpu=False)
    assert (attempt/'E3/e3_cpu_evaluation.exitcode').read_text().strip()=='1'
    assert v.sealed.read(attempt/'E3/cpu_evaluation_status.json')['status']=='FAIL'


def test_gpu_block_stops_before_any_child_process(tmp_path,monkeypatch):
    out=tmp_path/'analysis';out.mkdir();attempt=out/'attempt';(attempt/'E3').mkdir(parents=True)
    monkeypatch.setattr(v.v2,'resource_gate',lambda e:{'status':'BLOCKED','errors':['other user']})
    monkeypatch.setattr(v.subprocess,'Popen',lambda *a,**k:pytest.fail('must not launch'))
    with pytest.raises(AssertionError,match='other user'):
        v.run_stage(out,attempt,3,'checkpoint_validation_and_test',[],[],{},gpu=True)
    assert v.sealed.read(attempt/'E3/checkpoint_validation_and_test_resource_gate.json')['status']=='BLOCKED'
    status=v.sealed.read(attempt/'E3/checkpoint_validation_and_test_status.json')
    assert status['status']=='BLOCKED' and status['pid'] is None and not status['process_started']
    assert (attempt/'E3/e3_checkpoint_validation_and_test.exitcode').read_text().strip()=='2'


def test_original_seal_and_v2_gpu_proof_still_valid():
    out=ROOT/'analysis/joint_optimization_20261029'
    assert v.sealed.check_seal(out)['datasets']['3']['config_hash']
    assert v.v2.verify_proof(out)['status']=='PASS'


@pytest.mark.parametrize('failure', ['e3_test','e3_acceptance',None])
def test_worker_stage_order_never_retrains_e3_or_starts_e5_before_e3_acceptance(tmp_path,monkeypatch,failure):
    from contextlib import nullcontext
    out=tmp_path/'analysis';(out/'prepared_05').mkdir(parents=True)
    control=tmp_path/'control.yml';control.write_text('database:\n  host: local\n  user: test\n  port: 5432\n')
    plan={'datasets':{str(e):{'control_training_config':str(control),'artifact_root':str(tmp_path/f'artifact{e}'),'run_dir':str(tmp_path/f'run{e}')} for e in [3,5]}}
    v.sealed.save(out/'prepared_05/plan.json',plan)
    attempt=out/'attempt';(attempt/'E3').mkdir(parents=True);(attempt/'E5').mkdir()
    accepted=tmp_path/'accepted.json';v.sealed.save(accepted,{'status':'PASS'})
    identity={'checkpoint_acceptance':str(accepted),'data_root':str(tmp_path/'data')}
    monkeypatch.setattr(v.v2,'execution_lock',lambda out:nullcontext())
    monkeypatch.setattr(v,'check_identity',lambda *a:identity)
    monkeypatch.setattr(v,'verify_original',lambda *a:None)
    monkeypatch.setattr(v,'setup_e3',lambda *a:(tmp_path/'newart3',tmp_path/'run3'))
    monkeypatch.setattr(v,'stage_command',lambda out,e,stage,artifact=None:[str(e),stage])
    calls=[];accepted3=[]
    def stage(out,attempt,e,name,*args,**kwargs):
        calls.append((e,name))
        assert e!=3 or name!='training'
        if e==5:assert accepted3
        if failure=='e3_test' and e==3:raise RuntimeError('simulated inference failure')
    def accept(out,attempt,e,run):
        if e==3:
            if failure=='e3_acceptance':raise RuntimeError('simulated acceptance failure')
            accepted3.append(True)
        return {'metrics':{'tp':1}}
    monkeypatch.setattr(v,'run_stage',stage)
    monkeypatch.setattr(v,'acceptance',accept)
    code=v.worker(out,attempt)
    assert (attempt/'exitcode').read_text().strip()==str(code)
    if failure:
        assert code==1 and all(e==3 for e,_ in calls)
        assert v.sealed.read(attempt/'status.json')['datasets']['3']['status']=='FAILED'
    else:
        assert code==0 and calls==[(3,'checkpoint_validation_and_test'),(3,'cpu_evaluation'),(5,'training'),(5,'checkpoint_validation_and_test'),(5,'cpu_evaluation')]


def test_disk_block_records_terminal_status_and_code_without_child(tmp_path,monkeypatch):
    out=tmp_path/'analysis';out.mkdir();attempt=out/'attempt';(attempt/'E3').mkdir(parents=True)
    monkeypatch.setattr(v.shutil,'disk_usage',lambda path:NS(free=v.GIB))
    monkeypatch.setattr(v.subprocess,'Popen',lambda *a,**k:pytest.fail('must not launch'))
    with pytest.raises(AssertionError):
        v.run_stage(out,attempt,3,'cpu_evaluation',[],[],{},gpu=False)
    assert v.sealed.read(attempt/'E3/cpu_evaluation_status.json')['status']=='BLOCKED'
    assert (attempt/'E3/e3_cpu_evaluation.exitcode').read_text().strip()=='2'
