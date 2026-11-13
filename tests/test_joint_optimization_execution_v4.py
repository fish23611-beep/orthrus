"""CPU target-GPU isolation and stage lifecycle regressions; no fake GPU PASS."""
from pathlib import Path
import copy
import os
import subprocess
import sys
import time
from types import SimpleNamespace as NS

import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import joint_optimization_execution_v4 as v
import joint_gpu_isolation_v4 as g

TARGET=dict(physical_index=1,uuid='GPU-one',pci_bus_id='00000000:05:00.0',name='RTX4090',
            cuda_visible_devices='1',cuda_device_order='PCI_BUS_ID',logical_device='cuda:0')


def fixture_query(compute='',util0=100,util1=0,second=None,uuid1='GPU-one',power='15'):
    calls=[]
    def query(argv,**kwargs):
        if 'compute-apps' in argv[1]:
            calls.append(1)
            return second if second is not None and len(calls)>1 else compute
        return f'0, GPU-zero, 00000000:01:00.0, RTX4090, 10370, 24564, {util0}, 425\n1, {uuid1}, 00000000:05:00.0, RTX4090, 15, 24564, {util1}, {power}\n'
    return query


def usage(path):return NS(free=100*v.GIB)


def test_physical_gpu1_uuid_index_and_cuda_renumbering_contract():
    t=g.resolve_target(1,'GPU-one',query=fixture_query())
    assert t==TARGET
    env=g.environment({'CUDA_VISIBLE_DEVICES':'0'},t)
    assert env['CUDA_VISIBLE_DEVICES']=='1' and env['CUDA_DEVICE_ORDER']=='PCI_BUS_ID'
    assert t['logical_device']=='cuda:0'
    g.require_environment(env,t)
    assert g.environment(env,t,cpu=True)['CUDA_VISIBLE_DEVICES']==''


@pytest.mark.parametrize('kwargs',[{'index':0,'uuid':'GPU-one'},{'index':1,'uuid':'wrong'}])
def test_wrong_index_or_uuid_is_rejected(kwargs):
    with pytest.raises(AssertionError):g.resolve_target(query=fixture_query(),**kwargs)


@pytest.mark.parametrize('dataset',[None,3,5])
def test_gpu0_busy_does_not_block_idle_gpu1(dataset):
    gate=g.resource_gate(TARGET,dataset,usage=usage,query=fixture_query('999001, GPU-zero, 10322'))
    assert gate['status']=='PASS' and not gate['target_external_processes']
    assert gate['non_target_processes'][0]['pid']==999001
    assert gate['non_target_gpus'][0]['utilization_percent']==100
    assert gate['target_gpu']['power_watts']==15


def test_external_target_gpu_context_blocks_even_at_zero_utilization():
    gate=g.resource_gate(TARGET,usage=usage,query=fixture_query('999001, GPU-one, 20'))
    assert gate['status']=='BLOCKED' and gate['target_external_processes'][0]['pid']==999001


def test_self_context_and_own_controlled_child_are_exempt_but_new_competitor_is_not():
    own=f'{os.getpid()}, GPU-one, 600'
    gate=g.resource_gate(TARGET,usage=usage,query=fixture_query(own,util1=100))
    assert gate['status']=='PASS' and gate['own_target_processes'][0]['pid']==os.getpid()
    child=900003
    gate=g.resource_gate(TARGET,own_pids={child},usage=usage,query=fixture_query(f'{child}, GPU-one, 1000',util1=100))
    assert gate['status']=='PASS'
    gate=g.resource_gate(TARGET,usage=usage,query=fixture_query(own,util1=100,second=own+'\n999001, GPU-one, 20'))
    assert gate['status']=='BLOCKED'


def test_own_context_on_gpu0_is_a_binding_violation():
    gate=g.resource_gate(TARGET,usage=usage,query=fixture_query(f'{os.getpid()}, GPU-zero, 100'))
    assert gate['status']=='BLOCKED' and gate['own_binding_violations']


@pytest.mark.parametrize('case',['missing_uuid','unknown_util','unknown_power','unattributed_util','bad_process','query_failure'])
def test_target_checks_fail_closed(case):
    kwargs={}
    if case=='missing_uuid':kwargs['uuid1']='changed'
    if case=='unknown_util':kwargs['util1']='N/A'
    if case=='unknown_power':kwargs['power']='N/A'
    if case=='unattributed_util':kwargs['util1']=100
    if case=='bad_process':kwargs['compute']='bad'
    query=fixture_query(**kwargs)
    if case=='query_failure':
        def query(*a,**k):raise OSError('NVML unavailable')
    assert g.resource_gate(TARGET,usage=usage,query=query)['status']=='BLOCKED'


@pytest.mark.parametrize('path,free',[('/',2*v.GIB-1),('/mnt/datadisk',20*v.GIB-1)])
def test_global_disk_safety_floors_remain(path,free):
    def measure(p):return NS(free=free if p==path else 100*v.GIB)
    assert g.resource_gate(TARGET,usage=measure,query=fixture_query())['status']=='BLOCKED'


def harness(tmp_path):
    out=tmp_path/'analysis';out.mkdir();attempt=out/'attempt';(attempt/'E3').mkdir(parents=True)
    v.sealed.save(attempt/'identity.json',{'target_gpu':TARGET})
    data=tmp_path/'data';data.mkdir()
    return out,attempt,data


def test_gpu1_busy_preflight_never_spawns_child_and_records_block(tmp_path,monkeypatch):
    out,attempt,data=harness(tmp_path)
    monkeypatch.setattr(g,'resource_gate',lambda *a,**k:{'status':'BLOCKED','errors':['target busy']})
    monkeypatch.setattr(v.subprocess,'Popen',lambda *a,**k:pytest.fail('must not spawn'))
    with pytest.raises(g.TargetGPUBlocked):
        v.run_stage(out,attempt,3,'checkpoint_validation_and_test',[],[data],v.scoped_environment(data),gpu=True)
    assert v.sealed.read(attempt/'E3/checkpoint_validation_and_test_status.json')['status']=='BLOCKED'
    assert (attempt/'E3/e3_checkpoint_validation_and_test.exitcode').read_text().strip()=='2'


def test_real_child_receives_gpu1_and_late_competition_stops_only_own_child(tmp_path,monkeypatch):
    out,attempt,data=harness(tmp_path);calls=[];spawned=[]
    native_spawn=v.subprocess.Popen
    def spawn(*a,**kw):
        assert kw['env']['CUDA_VISIBLE_DEVICES']=='1'
        assert kw['env']['CUDA_DEVICE_ORDER']=='PCI_BUS_ID'
        p=native_spawn(*a,**kw);spawned.append(p);return p
    def gate(*a,**kw):
        calls.append(kw.get('own_pids'))
        if len(calls)==1:return {'status':'PASS','errors':[]}
        return dict(status='BLOCKED',errors=['new target competitor'],target_gpu={'memory_used_mib':1200},own_target_processes=[])
    monkeypatch.setattr(g,'resource_gate',gate);monkeypatch.setattr(v.subprocess,'Popen',spawn)
    with pytest.raises(g.TargetGPUBlocked,match='competitor'):
        v.run_stage(out,attempt,3,'checkpoint_validation_and_test',[sys.executable,'-c','import time; time.sleep(20)'],[data],v.scoped_environment(data),gpu=True)
    assert len(spawned)==1 and spawned[0].poll() is not None
    assert spawned[0].pid in calls[1]
    assert v.sealed.read(attempt/'E3/checkpoint_validation_and_test_status.json')['status']=='BLOCKED'
    assert (attempt/'E3/e3_checkpoint_validation_and_test.exitcode').read_text().strip()=='-15'
    assert (attempt/'E3/checkpoint_validation_and_test_gpu_gate_trigger.json').exists()


def test_real_cpu_stage_is_cuda_hidden_and_keeps_storage_gate(tmp_path,monkeypatch):
    out,attempt,data=harness(tmp_path)
    native_sleep=v.time.sleep;monkeypatch.setattr(v.time,'sleep',lambda _:native_sleep(.02))
    argv=[sys.executable,'-c','import os; assert os.environ["CUDA_VISIBLE_DEVICES"]==""; print("CPU_ONLY")']
    result=v.run_stage(out,attempt,3,'cpu_evaluation',argv,[data],v.scoped_environment(data),gpu=False)
    assert result['status']=='PASS' and result['runtime_environment']['CUDA_VISIBLE_DEVICES']==''
    assert not result['physical_gpu_execution_observed']
    assert (attempt/'E3/e3_cpu_evaluation.exitcode').read_text().strip()=='0'


def test_e3_retraining_still_prohibited_and_config_hash_unaffected(tmp_path):
    out=ROOT/'analysis/joint_optimization_20261029'
    with pytest.raises(AssertionError,match='retraining'):v.stage_command(out,3,'training')
    from mstc.experiment_utils import stable_config_hash
    a=v.audit.frozen_cfg(out,3);b=v.audit.frozen_cfg(out,3,tmp_path/'new')
    assert stable_config_hash(a)==stable_config_hash(b)
    assert v.sealed.check_seal(out) and v.v2.verify_proof(out)['status']=='PASS'


def test_gpu_cache_paths_are_owned(tmp_path):
    env=v.scoped_environment(tmp_path)
    for key in ['CUDA_CACHE_PATH','TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR']:
        assert Path(env[key]).is_relative_to(tmp_path)


def test_nvidia_probe_attestation_is_required_before_formal_worker(tmp_path,monkeypatch):
    out,attempt,data=harness(tmp_path)
    # Identity/environment guard refuses a direct unbound worker before output mutation.
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    with pytest.raises(AssertionError):v.worker(out,attempt)
    assert not (attempt/'worker_claim.json').exists()
