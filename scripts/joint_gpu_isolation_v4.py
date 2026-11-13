"""Physical GPU 1 identity, target-only safety gate, and bounded CUDA acceptance."""
from __future__ import annotations
import csv
import io
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import joint_optimization_pipeline as sealed

GPU1_UUID='GPU-78802eaf-12bf-2a2a-7c9d-22839c0b1162'
GPU_FIELDS='index,uuid,pci.bus_id,name,memory.used,memory.total,utilization.gpu,power.draw'
APP_ARGS=['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits']
GPU_ARGS=['nvidia-smi','--query-gpu='+GPU_FIELDS,'--format=csv,noheader,nounits']


class TargetGPUBlocked(RuntimeError):
    pass


def number(value):
    value=float(value.strip())
    if not math.isfinite(value):raise ValueError('non-finite GPU metric')
    return value


def gpu_rows(raw):
    rows=[]
    for r in csv.reader(io.StringIO(raw)):
        if not r:continue
        if len(r)!=8:raise ValueError('Malformed GPU row')
        row=dict(index=int(r[0]),uuid=r[1].strip(),pci_bus_id=r[2].strip(),name=r[3].strip(),raw=r)
        for key,value in zip(['memory_used_mib','memory_total_mib','utilization_percent','power_watts'],r[4:]):
            try:row[key]=number(value)
            except ValueError:row[key]=None
        rows.append(row)
    return rows


def processes(raw):
    result={}
    for r in csv.reader(io.StringIO(raw)):
        if not r:continue
        if len(r)!=3 or int(r[0])<=0 or not r[1].strip():raise ValueError('Malformed GPU process row')
        pid,uuid=int(r[0]),r[1].strip()
        try:memory=number(r[2])
        except ValueError:memory=None
        result[(pid,uuid)]=dict(pid=pid,gpu_uuid=uuid,memory_mib=memory)
    return list(result.values())


def resolve_target(index=1,uuid=GPU1_UUID,*,query=subprocess.check_output):
    assert index==1,'This execution is explicitly restricted to physical GPU 1'
    rows=gpu_rows(query(GPU_ARGS,text=True))
    selected=[r for r in rows if r['index']==index and r['uuid']==uuid]
    assert len(selected)==1,'Physical GPU index/UUID binding changed or missing'
    row=selected[0]
    return dict(physical_index=index,uuid=uuid,pci_bus_id=row['pci_bus_id'],name=row['name'],
        cuda_visible_devices=str(index),cuda_device_order='PCI_BUS_ID',logical_device='cuda:0')


def resource_gate(target,e=None,*,own_pids=None,usage=shutil.disk_usage,query=subprocess.check_output):
    own={os.getpid()} if own_pids is None else set(own_pids)|{os.getpid()}
    errors=[];raw={}
    result=dict(at=sealed.now(),target=target,allowed_own_pids=sorted(own))
    try:
        root_free=usage('/').free;data_free=usage('/mnt/datadisk').free
        extra=(6 if e==3 else 20)*sealed.GIB if e else 0
        result.update(root_free_bytes=root_free,data_free_bytes=data_free,
            root_safety_floor_bytes=2*sealed.GIB,data_safety_floor_bytes=20*sealed.GIB,additional_start_budget_bytes=extra)
        if root_free<2*sealed.GIB:errors.append('System free space below 2 GiB')
        if data_free<20*sealed.GIB+extra:errors.append('Data free space below safety floor plus stage start budget')
        raw['processes_before']=query(APP_ARGS,text=True).strip()
        raw['gpu']=query(GPU_ARGS,text=True).strip()
        raw['processes_after']=query(APP_ARGS,text=True).strip()
        rows=gpu_rows(raw['gpu'])
        matches=[r for r in rows if r['uuid']==target['uuid'] and r['index']==target['physical_index']]
        if len(matches)!=1:raise ValueError('Target physical GPU index/UUID mismatch')
        gpu=matches[0]
        if gpu['pci_bus_id']!=target['pci_bus_id']:raise ValueError('Target PCI identity changed')
        if any(gpu[k] is None for k in ['memory_used_mib','memory_total_mib','utilization_percent','power_watts']):
            raise ValueError('Unknown target GPU metric')
        merged={}
        for item in processes(raw['processes_before'])+processes(raw['processes_after']):
            merged[(item['pid'],item['gpu_uuid'])]=item
        apps=list(merged.values())
        target_apps=[p for p in apps if p['gpu_uuid']==target['uuid']]
        external=[p for p in target_apps if p['pid'] not in own]
        own_target=[p for p in target_apps if p['pid'] in own]
        non_target=[p for p in apps if p['gpu_uuid']!=target['uuid']]
        leaked=[p for p in non_target if p['pid'] in own]
        if external:errors.append('External compute process on target GPU')
        if leaked:errors.append('Own CUDA context on non-target GPU: binding violation')
        if not own_target and gpu['utilization_percent']>10:errors.append('Unattributed target utilization above 10%')
        result.update(target_gpu=gpu,target_external_processes=external,own_target_processes=own_target,
            non_target_processes=non_target,non_target_gpus=[r for r in rows if r['uuid']!=target['uuid']],
            own_binding_violations=leaked)
    except Exception as exc:
        errors.append('Target resource inspection failed: '+repr(exc))
    result.update(status='PASS' if not errors else 'BLOCKED',errors=errors,raw_nvidia_smi=raw)
    return result


def environment(env,target,*,cpu=False):
    result=env.copy()
    result['CUDA_DEVICE_ORDER']=target['cuda_device_order']
    result['CUDA_VISIBLE_DEVICES']='' if cpu else target['cuda_visible_devices']
    result['MSTC_TARGET_GPU_UUID']=target['uuid']
    return result


def require_environment(env,target,*,cpu=False):
    assert env['CUDA_VISIBLE_DEVICES']==('' if cpu else '1')
    assert env['CUDA_DEVICE_ORDER']=='PCI_BUS_ID'
    assert env['MSTC_TARGET_GPU_UUID']==target['uuid']


def probe_child(directory,checkpoint_root):
    """No training: tiny CUDA backward + unchanged CUDA-0 checkpoint deserialization."""
    import gc
    import torch
    assert os.environ['CUDA_VISIBLE_DEVICES']=='1' and os.environ['CUDA_DEVICE_ORDER']=='PCI_BUS_ID'
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.set_num_threads(2);torch.cuda.set_device(0)
    started=time.monotonic()
    x=torch.eye(8,device='cuda:0',requires_grad=True)
    loss=(x@x).square().sum();loss.backward();torch.cuda.synchronize()
    assert bool(torch.isfinite(loss)) and bool(torch.isfinite(x.grad).all())
    assert x.device.index==0
    checkpoints=[]
    for epoch in range(1,7):
        path=checkpoint_root/f'model_epoch_{epoch}/checkpoint.pt'
        before=sealed.sha(path)
        cp=torch.load(path,weights_only=False)  # Native saved cuda:0 -> visible logical 0 (physical GPU 1).
        devices={str(t.device) for t in cp['model_state_dict'].values() if t.is_cuda}
        assert cp['epoch']==epoch and devices=={'cuda:0'}
        assert all(bool(torch.isfinite(t).all()) for t in cp['model_state_dict'].values())
        del cp;gc.collect()
        assert sealed.sha(path)==before
        checkpoints.append(dict(epoch=epoch,sha256=before,cuda_devices=sorted(devices),status='PASS'))
    torch.cuda.synchronize()
    result=dict(status='PASS',pid=os.getpid(),logical_device='cuda:0',device_count=1,
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],cuda_device_order=os.environ['CUDA_DEVICE_ORDER'],
        gpu_name=torch.cuda.get_device_name(0),cuda_build=torch.version.cuda,torch_version=torch.__version__,
        finite_forward_backward=True,checkpoint_deserialization=checkpoints,seconds=time.monotonic()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        no_training_epochs=True)
    sealed.save(directory/'child_result.json',result)
    (directory/'ready').write_text('ready\n')
    # Hold the live context for the parent's NVML mapping attestation; bounded.
    for _ in range(300):
        if (directory/'release').exists():return 0
        time.sleep(.05)
    raise RuntimeError('Parent mapping attestation did not release probe within 15 seconds')


def cuda_acceptance(target,directory,argv,env,*,timeout=60):
    directory.mkdir(exist_ok=False)
    require_environment(env,target)
    gate=resource_gate(target)
    sealed.save(directory/'resource_gate.json',gate)
    if gate['status']!='PASS':
        sealed.save(directory/'status.json',dict(status='BLOCKED',exitcode=2,gate=gate))
        (directory/'exitcode').write_text('2\n')
        raise TargetGPUBlocked(repr(gate))
    process=None;code=1
    try:
        with (directory/'probe.log').open('x') as log:
            process=subprocess.Popen(argv,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            (directory/'pid').write_text(str(process.pid)+'\n')
            deadline=time.monotonic()+timeout
            for _ in range(int(timeout*10)):
                if (directory/'ready').exists():break
                if process.poll() is not None:raise RuntimeError('CUDA acceptance child failed: '+(directory/'probe.log').read_text())
                if time.monotonic()>=deadline:raise RuntimeError('CUDA acceptance timed out')
                time.sleep(.1)
            assert (directory/'ready').exists(),'CUDA acceptance did not become ready'
            mapping=resource_gate(target,own_pids={os.getpid(),process.pid})
            assert mapping['status']=='PASS',mapping
            assert any(p['pid']==process.pid for p in mapping['own_target_processes']),'CUDA logical/physical mapping not attested'
            child=sealed.read(directory/'child_result.json')
            assert child['status']=='PASS' and child['device_count']==1 and child['logical_device']=='cuda:0'
            (directory/'release').write_text('released\n')
            code=process.wait(timeout=max(1,deadline-time.monotonic()))
            assert code==0
        proof=dict(status='PASS',at=sealed.now(),target=target,nvml_mapping=mapping,child=child,exitcode=0)
        sealed.save(directory/'proof.json',proof)
        sealed.save(directory/'status.json',dict(status='PASS',exitcode=0))
        return proof
    except BaseException as exc:
        if process is not None:sealed.terminate_own(process)
        sealed.save(directory/'status.json',dict(status='FAIL',exitcode=1,error=repr(exc)))
        code=1
        raise
    finally:
        (directory/'exitcode').write_text(str(code)+'\n')
