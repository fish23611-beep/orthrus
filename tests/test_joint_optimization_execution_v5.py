"""CPU-only sharing-policy regressions; CUDA acceptance is a separate real run."""
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
import joint_optimization_execution_v5 as v
import joint_gpu_sharing_v5 as g

TARGET=dict(physical_index=1,uuid='GPU-one',pci_bus_id='00000000:05:00.0',name='RTX4090',
    cuda_visible_devices='1',cuda_device_order='PCI_BUS_ID',logical_device='cuda:0')


def query_fixture(compute='',util=0,second=None,uuid='GPU-one',pci='00000000:05:00.0',temp='50'):
    calls=[]
    def query(argv,**kw):
        if 'compute-apps' in argv[1]:
            calls.append(1)
            return second if second is not None and len(calls)>1 else compute
        return f'0, GPU-zero, 00000000:01:00.0, RTX4090, 10370, 24564, 100, 425, 65\n1, {uuid}, {pci}, RTX4090, 6000, 24564, {util}, 250, {temp}\n'
    return query


def usage(path):return NS(free=100*v.GIB)


def test_idle_gpu_allows_run():
    gate=g.resource_gate(TARGET,usage=usage,query=query_fixture())
    assert gate['status']=='PASS' and g.can_run(gate) and not gate['warnings']


@pytest.mark.parametrize('dataset',[None,3,5])
def test_external_gpu_process_permitted_with_explicit_warning(dataset,tmp_path):
    gate=g.resource_gate(TARGET,dataset,usage=usage,query=query_fixture('987654, GPU-one, 4548',util=98))
    assert gate['status']=='WARNING' and gate['allowed'] and g.can_run(gate)
    assert not gate['errors'] and gate['target_external_processes'][0]['pid']==987654
    assert gate['target_external_processes'][0]['memory_mib']==4548
    assert gate['target_gpu']['temperature_celsius']==50
    path=tmp_path/'warnings.jsonl';g.record_warnings(path,gate)
    record=json.loads(path.read_text())
    assert record['severity']=='WARNING' and record['allowed'] and record['at']==gate['at']
    assert record['target_gpu']['utilization_percent']==98
    assert record['target_gpu']['power_watts']==250
    assert record['external_processes'][0]['memory_mib']==4548


def test_high_utilization_without_process_listing_also_warns():
    gate=g.resource_gate(TARGET,usage=usage,query=query_fixture(util=99))
    assert gate['status']=='WARNING' and g.can_run(gate)
    assert gate['warnings'][0]['code']=='HIGH_TARGET_UTILIZATION'


def test_self_context_and_new_external_process_both_allow_continue():
    own=f'{os.getpid()}, GPU-one, 2000'
    gate=g.resource_gate(TARGET,usage=usage,query=query_fixture(own,util=100))
    assert gate['status']=='PASS' and g.can_run(gate)
    gate=g.resource_gate(TARGET,usage=usage,query=query_fixture(own,util=100,second=own+'\n987654, GPU-one, 4548'))
    assert gate['status']=='WARNING' and g.can_run(gate)
    assert gate['target_external_processes'][0]['pid']==987654


@pytest.mark.parametrize('case',['wrong_uuid','wrong_pci','own_wrong_card','bad_metrics','query_failure'])
def test_binding_and_inspection_failures_are_still_blocked_despite_competitor(case):
    kwargs={'compute':'987654, GPU-one, 4548','util':98}
    if case=='wrong_uuid':kwargs['uuid']='GPU-wrong'
    if case=='wrong_pci':kwargs['pci']='changed'
    if case=='own_wrong_card':kwargs['compute']+=f'\n{os.getpid()}, GPU-zero, 2000'
    if case=='bad_metrics':kwargs['temp']='N/A'
    query=query_fixture(**kwargs)
    if case=='query_failure':
        def query(*a,**k):raise OSError('device inspection error')
    gate=g.resource_gate(TARGET,usage=usage,query=query)
    assert gate['status']=='BLOCKED' and gate['errors'] and not g.can_run(gate)


@pytest.mark.parametrize('path,free',[('/',2*v.GIB-1),('/mnt/datadisk',20*v.GIB-1)])
def test_disk_floor_not_bypassed_by_warning(path,free):
    def measure(p):return NS(free=free if p==path else 100*v.GIB)
    gate=g.resource_gate(TARGET,usage=measure,query=query_fixture('987654, GPU-one, 4548',util=100))
    assert gate['warnings'] and gate['status']=='BLOCKED' and not g.can_run(gate)


def test_status_alone_cannot_override_errors_or_explicit_denial():
    for gate in [dict(status='WARNING',allowed=True,errors=['binding failure']),
                 dict(status='PASS',allowed=False,errors=[])]:
        assert not g.can_run(gate)


def harness(tmp_path,name='attempt'):
    out=tmp_path/'analysis';out.mkdir(exist_ok=True)
    attempt=out/name;(attempt/'E3').mkdir(parents=True)
    v.sealed.save(attempt/'identity.json',{'target_gpu':TARGET})
    data=tmp_path/(name+'_data');data.mkdir()
    return out,attempt,data


def monitoring_gate(*a,**kw):
    own=sorted(set(kw.get('own_pids',[]))-{os.getpid()})
    props=dict(memory_used_mib=6000,utilization_percent=98,power_watts=250,temperature_celsius=60)
    at=v.sealed.now()
    return dict(status='WARNING',allowed=True,errors=[],warnings=[{'code':'EXTERNAL_TARGET_COMPUTE','at':at}],
        at=at,target=TARGET,target_gpu=props,target_external_processes=[{'pid':987654,'gpu_uuid':'GPU-one','memory_mib':4548}],
        own_target_processes=[{'pid':pid,'gpu_uuid':'GPU-one','memory_mib':1200} for pid in own])


def accelerate(monkeypatch):
    native_sleep=v.time.sleep
    monkeypatch.setattr(v.time,'sleep',lambda _:native_sleep(.02))


def test_late_external_process_logs_warning_without_terminating_real_child(tmp_path,monkeypatch):
    out,attempt,data=harness(tmp_path);calls=[]
    def gate(*a,**kw):
        calls.append(kw.get('own_pids'))
        if len(calls)==1:return dict(status='PASS',allowed=True,errors=[],warnings=[])
        return monitoring_gate(*a,**kw)
    monkeypatch.setattr(g,'resource_gate',gate)
    monkeypatch.setattr(v.sealed,'terminate_own',lambda *a:pytest.fail('sharing must not terminate a running child'))
    accelerate(monkeypatch)
    argv=[sys.executable,'-c','import os,time; from pathlib import Path; assert os.environ["CUDA_VISIBLE_DEVICES"]=="1"; time.sleep(.12); Path(os.environ["TMPDIR"],"completed").write_text("done")']
    result=v.run_stage(out,attempt,3,'checkpoint_validation_and_test',argv,[data],v.scoped_environment(data),gpu=True)
    assert result['status']=='PASS' and result['exitcode']==0 and len(calls)>1
    assert (data/'tmp/completed').read_text()=='done'
    assert (attempt/'E3/e3_checkpoint_validation_and_test.exitcode').read_text().strip()=='0'
    warnings=[json.loads(x) for x in (attempt/'E3/checkpoint_validation_and_test_gpu_warnings.jsonl').read_text().splitlines()]
    assert warnings and all(x['severity']=='WARNING' for x in warnings)
    assert not (attempt/'E3/checkpoint_validation_and_test_gpu_gate_trigger.json').exists()


@pytest.mark.parametrize('error',['raise torch.cuda.OutOfMemoryError("simulated CUDA OOM")','raise RuntimeError("CUDA error: invalid device function")'])
def test_actual_child_cuda_error_is_failure_not_warning_success(tmp_path,monkeypatch,error):
    out,attempt,data=harness(tmp_path);spawned=[];native_spawn=v.subprocess.Popen
    def spawn(*args,**kw):
        p=native_spawn(*args,**kw);spawned.append(p);return p
    monkeypatch.setattr(v.subprocess,'Popen',spawn)
    monkeypatch.setattr(g,'resource_gate',monitoring_gate);accelerate(monkeypatch)
    argv=[sys.executable,'-c','import torch; '+error]
    with pytest.raises(AssertionError,match='failed'):
        v.run_stage(out,attempt,3,'checkpoint_validation_and_test',argv,[data],v.scoped_environment(data),gpu=True)
    assert len(spawned)==1 and spawned[0].returncode==1
    state=v.sealed.read(attempt/'E3/checkpoint_validation_and_test_status.json')
    assert state['status']=='FAIL' and state['exitcode']==1
    assert 'CUDA' in (attempt/'E3/checkpoint_validation_and_test.log').read_text()
    assert (attempt/'E3/e3_checkpoint_validation_and_test.exitcode').read_text().strip()=='1'


@pytest.mark.parametrize('low_path',['/','/mnt/datadisk'])
def test_runtime_disk_floor_still_terminates_own_child_with_sharing_warning(tmp_path,monkeypatch,low_path):
    out,attempt,data=harness(tmp_path);counts={};spawned=[];native_spawn=v.subprocess.Popen
    def measure(path):
        counts[path]=counts.get(path,0)+1
        # root: preflight then monitor; data: preflight, baseline, monitor.
        limit=2 if path=='/' else 3
        return NS(free=v.GIB if path==low_path and counts[path]>=limit else 100*v.GIB)
    def spawn(*a,**kw):
        p=native_spawn(*a,**kw);spawned.append(p);return p
    monkeypatch.setattr(v.shutil,'disk_usage',measure)
    monkeypatch.setattr(g,'resource_gate',monitoring_gate)
    monkeypatch.setattr(v.subprocess,'Popen',spawn)
    with pytest.raises(RuntimeError,match='Owned-storage'):
        v.run_stage(out,attempt,3,'checkpoint_validation_and_test',[sys.executable,'-c','import time; time.sleep(10)'],[data],v.scoped_environment(data),gpu=True)
    assert len(spawned)==1 and spawned[0].poll() is not None
    assert v.sealed.read(attempt/'E3/checkpoint_validation_and_test_gate_trigger.json')['gate']['status']=='BLOCKED'


def test_binding_failure_still_refuses_start(tmp_path,monkeypatch):
    out,attempt,data=harness(tmp_path)
    monkeypatch.setattr(g,'resource_gate',lambda *a,**kw:dict(status='BLOCKED',allowed=False,errors=['wrong UUID'],warnings=[]))
    monkeypatch.setattr(v.subprocess,'Popen',lambda *a,**k:pytest.fail('wrong GPU must not launch'))
    with pytest.raises(g.TargetGPUBlocked,match='UUID'):
        v.run_stage(out,attempt,3,'checkpoint_validation_and_test',[],[data],v.scoped_environment(data),gpu=True)
    assert v.sealed.read(attempt/'E3/checkpoint_validation_and_test_status.json')['status']=='BLOCKED'


def test_old_failed_blocked_and_completed_stage_files_are_not_overwritten(tmp_path,monkeypatch):
    out,attempt,data=harness(tmp_path)
    legacy=out/'legacy';legacy.mkdir()
    (legacy/'FAILED.json').write_text('{"status":"FAILED"}')
    (legacy/'BLOCKED.log').write_text('historical GPU block')
    before={p:p.read_bytes() for p in legacy.iterdir()}
    monkeypatch.setattr(g,'resource_gate',monitoring_gate);accelerate(monkeypatch)
    argv=[sys.executable,'-c','print("completed")']
    v.run_stage(out,attempt,3,'checkpoint_validation_and_test',argv,[data],v.scoped_environment(data),gpu=True)
    old={p:p.read_bytes() for p in (attempt/'E3').iterdir() if p.is_file()}
    with pytest.raises(FileExistsError):
        v.run_stage(out,attempt,3,'checkpoint_validation_and_test',argv,[data],v.scoped_environment(data),gpu=True)
    assert all(p.read_bytes()==value for p,value in before.items())
    assert all(p.read_bytes()==value for p,value in old.items())


def test_cuda_acceptance_parent_accepts_shared_gpu_warning(tmp_path,monkeypatch):
    directory=tmp_path/'probe';env=g.environment(dict(os.environ),TARGET)
    source='''import json,sys,time
from pathlib import Path
p=Path(sys.argv[1]);(p/'child_result.json').write_text(json.dumps({'status':'PASS','device_count':1,'logical_device':'cuda:0','test_double':True}))
(p/'ready').write_text('ready')
for _ in range(300):
    if (p/'release').exists():break
    time.sleep(.01)
else:raise RuntimeError('bounded test probe timeout')
'''
    monkeypatch.setattr(g,'resource_gate',monitoring_gate)
    proof=g.cuda_acceptance(TARGET,directory,[sys.executable,'-c',source,str(directory)],env,timeout=10)
    assert proof['status']=='PASS' and proof['nvml_mapping']['status']=='WARNING'
    assert proof['child']['test_double']  # CPU control-flow test only, never a real CUDA acceptance.
    assert len((directory/'resource_warnings.jsonl').read_text().splitlines())==2


def test_v4_frozen_sources_and_e3_checkpoints_unchanged():
    out=ROOT/'analysis/joint_optimization_20261029'
    assert v.sealed.sha(ROOT/'scripts/joint_optimization_execution_v4.py')=='3ecf275dd04a540fb2b3b40546cd4becb0436f6009e27cc17f9fd28d08512243'
    assert v.sealed.sha(ROOT/'scripts/joint_gpu_isolation_v4.py')=='35fb5160500bfa02a4f3501faf6b184077c50052eb19728649433c846bb00242'
    assert v.sealed.check_seal(out)
    accepted=v.sealed.read(out/'recovery_audits_v3/audit_20261030T194155_fpfp30fl/e3_checkpoint_acceptance.json')
    v.verify_original(accepted)
    with pytest.raises(AssertionError,match='retraining'):v.stage_command(out,3,'training')
