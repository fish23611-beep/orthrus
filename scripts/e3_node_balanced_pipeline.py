"""Single frozen E3/seed0 intervention; persistent supervisor with disk guard."""
from __future__ import annotations
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace as NS

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
CONTROL=Path('/mnt/datadisk/yby/mstc/artifacts/u4090_e3/THEIA_E3/runs/orthrus/seed_0')
SHARED=Path('/mnt/datadisk/yby/mstc/artifacts/u4090_e3')
MIN_DATA_FREE=20*1024**3
MIN_ROOT_FREE=2*1024**3


def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def save(path,value,mutable=False):
    path=Path(path)
    if mutable:
        temp=path.with_suffix(path.suffix+'.next')
        with temp.open('w',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
        os.replace(temp,path)
    else:
        with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)


def config_args(config_path,artifact_root,cpu=False):
    return NS(dataset='THEIA_E3',model='orthrus',config=str(config_path),cpu=cpu,from_weights=False,seed=0,
        skip_tracing=True,artifact_root=str(artifact_root),shared_artifact_root=str(SHARED),
        shared_artifact_readonly=True,resume_checkpoint=None,inference_checkpoint=None)


def preflight(out,artifact_root):
    import numpy as np
    import torch
    import yaml
    import config
    from mstc.experiment_utils import sanitized_config,stable_config_hash
    from mstc.time_gap import TimeGapStatistics
    from mstc.training_loss import NodeContributionWeights
    from detection.checkpoint_policy import compute_validation_objective
    from audit_node_training_frequency import windows
    import data_utils,factory

    out.mkdir(parents=True,exist_ok=False)
    assert not artifact_root.exists(),'Treatment output must be new'
    free_data=shutil.disk_usage('/mnt/datadisk').free;free_root=shutil.disk_usage('/').free
    assert free_data>=MIN_DATA_FREE+6*1024**3 and free_root>=MIN_ROOT_FREE+1024**3,'Insufficient reserved disk space'
    original=read(ROOT/'analysis/node_balanced_training_20261028/final_verification.json')
    for p,h in original['代码SHA256'].items():assert sha(ROOT/p)==h,p
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    acceptance=read(ROOT/'analysis/seed0_final_acceptance_20261028/e3_acceptance.json')
    assert head==acceptance['git_head']
    manifest=read(ROOT/'analysis/seed0_final_acceptance_20261028/artifact_manifest.json')
    expected={x['path']:x['sha256'] for x in manifest['inputs']}
    identity={}
    for p in [CONTROL/'checkpoints/model_epoch_4/checkpoint.pt',CONTROL/'node_scores/node_predictions.csv',
              CONTROL/'node_scores/metrics.json',CONTROL/'config_resolved.yml',CONTROL/'final_audit/mstc_full.yml',CONTROL/'metadata/time_statistics.json']:
        identity[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size}
        assert identity[str(p)]['sha256']==expected[str(p)],p
    assert read(CONTROL/'runtime.json')['seed']==0
    saved_config=yaml.safe_load((CONTROL/'config_resolved.yml').read_text())
    for name in ['host','user','port']:os.environ['ORTHRUS_DB_'+name.upper()]=str(saved_config['database'][name])
    paths=config.set_task_paths;config.set_task_paths=lambda c:None
    try:
        control=config.get_yml_cfg(config_args(CONTROL/'final_audit/mstc_full.yml',artifact_root,cpu=True))
        treatment=config.get_yml_cfg(config_args(ROOT/'analysis/node_balanced_training_20261028/paired_plan/treatment.yml',artifact_root,cpu=True))
    finally:config.set_task_paths=paths
    cp=torch.load(CONTROL/'checkpoints/model_epoch_4/checkpoint.pt',map_location='cpu',weights_only=False)
    assert cp['epoch']==4 and cp['config_hash']==stable_config_hash(control),'Historical effective config hash not reproduced'
    assert control.detection.gnn_training.use_seed and control.detection.gnn_training.num_epochs==6
    assert cp['optimizer_state_dict']['param_groups'][0]['lr']==control.detection.gnn_training.lr==1e-5
    assert cp['optimizer_state_dict']['param_groups'][0]['weight_decay']==control.detection.gnn_training.weight_decay==1e-5
    def flat(x,p=''):
        if isinstance(x,dict):return {k:v for a,b in x.items() for k,v in flat(b,p+'.'+a if p else a).items()}
        return {p:x}
    a,b=flat(sanitized_config(control)),flat(sanitized_config(treatment))
    differences={k:dict(control=a[k],treatment=b.get(k)) for k in a if a[k]!=b.get(k)}
    assert set(differences)=={'detection.gnn_training.train_loss_weighting','detection.evaluation.checkpoint_policy'}
    objectives={p.name:compute_validation_objective(p,control)[0] for p in sorted((CONTROL/'edge_scores/val').iterdir())}
    assert min(objectives,key=objectives.get)=='model_epoch_4'
    cache=next(SHARED.glob('edge_featurization/THEIA_E3/embed_edges/*/edge_embeds'))
    frequency=read(ROOT/'analysis/node_balanced_training_20261028/e3_train_frequency.json')
    train_paths=[Path(x['path']) for x in frequency['source_tensor_hashes']]
    for item in frequency['source_tensor_hashes']:
        st=Path(item['path']).stat();assert st.st_size==item['bytes'] and st.st_mtime_ns==item['mtime_ns']
    capacity=max(max(int(w.src.max()),int(w.dst.max())) for w in windows(train_paths))+1
    weights=NodeContributionWeights.fit(windows(train_paths),num_nodes=capacity)
    assert weights.summary()['counts_sha256']==frequency['counts_sha256']
    assert weights.num_events==4063357 and weights.num_nodes==410023
    assert abs(frequency['weight_distribution']['mean']-1)<1e-12
    sample=data_utils._prepare_window(control,str(train_paths[0]),control.dataset_view.mode)
    stats=TimeGapStatistics.load(CONTROL/'metadata/time_statistics.json')
    model=factory.build_model(sample,torch.device('cpu'),control,capacity,stats)
    model.load_state_dict(cp['model_state_dict'],strict=True)
    parameter_count=sum(x.numel() for x in model.parameters())
    del model,sample,cp,weights
    overlay=yaml.safe_load((ROOT/'analysis/node_balanced_training_20261028/paired_plan/treatment.yml').read_text())
    with (out/'treatment.yml').open('x') as f:yaml.safe_dump(overlay,f,allow_unicode=True,sort_keys=False)
    save(out/'effective_control.json',sanitized_config(control));save(out/'effective_treatment.json',sanitized_config(treatment))
    protected={}
    for e in [3,5]:
        root=Path(f'/mnt/datadisk/yby/mstc/artifacts/u4090_e{e}/THEIA_E{e}/runs/orthrus/seed_0')
        for p in root.rglob('*'):
            if p.is_file():s=p.stat();protected[str(p)]={'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
    save(out/'protected_inputs.json',protected)
    save(out/'preflight.json',dict(status='PASS',baseline_comparability='PASS',created_at=now(),git_head=head,
        code_sha256={p:sha(ROOT/p) for p in original['代码SHA256']},control_identity=identity,
        historical_checkpoint_config_hash=stable_config_hash(control),checkpoint_config_hash_exact=True,
        control_validation_objectives=objectives,control_selected_epoch=4,seed=0,model_parameters=parameter_count,
        semantic_intervention='仅train_loss_weighting：event_mean → node_contribution_balanced',differences=differences,
        execution_policy_difference='只改变验证选优的执行顺序：Control历史all_checkpoints，其选中epoch4已由六个原始验证目标独立复现；Treatment只测试选中的一个checkpoint',
        frequency=weights_summary(frequency),database_identity={k:str(saved_config['database'][k]) for k in ['host','user','port']},
        storage=dict(root_free_bytes=free_root,data_free_bytes=free_data,estimated_increment_upper_bytes=6*1024**3,
            minimum_data_free_bytes=MIN_DATA_FREE,minimum_root_free_bytes=MIN_ROOT_FREE,
            calibrated_csv_copies=0,temporary_directory=str(artifact_root/'tmp'),
            plan='六个checkpoint及验证raw、一个最佳test raw；CPU流式聚合，不写大型calibrated/event副本'),
        shared_cache=str(cache),artifact_root=str(artifact_root),run_dir=str(artifact_root/'THEIA_E3/runs/orthrus/seed_0'),
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv'],text=True),
        environment=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda),control_reused=True,
        git_diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff'],cwd=ROOT)).hexdigest()))
    print('PREFLIGHT PASS',out,flush=True)


def weights_summary(f):return {k:f[k] for k in ['target_events','observed_nodes','normalization','counts_sha256','weight_distribution']}


def supervise(out):
    pre=read(out/'preflight.json');artifact_root=Path(pre['artifact_root'])
    assert not artifact_root.exists(),'Refuse to reuse an existing output'
    artifact_root.mkdir(parents=True);(artifact_root/'tmp').mkdir()
    os.environ['TMPDIR']=str(artifact_root/'tmp');os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['WANDB_MODE']='disabled'
    for name,value in pre['database_identity'].items():os.environ['ORTHRUS_DB_'+name.upper()]=value
    for p,h in pre['code_sha256'].items():assert sha(ROOT/p)==h,p
    (out/'supervisor.pid').write_text(str(os.getpid())+'\n')
    started=now();baseline_free=shutil.disk_usage('/mnt/datadisk').free
    status=dict(status='RUNNING',stage='training',started_at=started,pid=os.getpid(),run_dir=pre['run_dir'],baseline_comparability='PASS',stages={})
    save(out/'STATUS.json',status,mutable=True)
    common=[sys.executable,'-u',str(ROOT/'src/experiments/run_experiment.py'),'--dataset','THEIA_E3','--config',str(out/'treatment.yml'),
        '--seed','0','--artifact-root',str(artifact_root),'--shared-artifact-root',str(SHARED),'--shared-artifact-readonly']
    stages=[('training',common+['--stages','train']),('validation_selection_and_best_test',common+['--stages','test']),
        ('cpu_paired_evaluation',[sys.executable,'-u',str(ROOT/'scripts/compare_e3_node_balanced_treatment.py'),'--analysis',str(out)])]
    exitcode=0
    try:
        for name,argv in stages:
            status['stage']=name;status['stages'][name]={'status':'RUNNING','started_at':now(),'argv':argv}
            env=os.environ.copy()
            if name=='cpu_paired_evaluation':env['CUDA_VISIBLE_DEVICES']='';env['OMP_NUM_THREADS']='2';env['OPENBLAS_NUM_THREADS']='2'
            with (out/f'{name}.log').open('x') as log:
                process=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                (out/f'{name}.pid').write_text(str(process.pid)+'\n');status['child_pid']=process.pid
                save(out/'STATUS.json',status,mutable=True)
                while process.poll() is None:
                    data_free=shutil.disk_usage('/mnt/datadisk').free;root_free=shutil.disk_usage('/').free
                    status.update(updated_at=now(),data_free_bytes=data_free,root_free_bytes=root_free,
                                  net_data_disk_used_bytes=baseline_free-data_free)
                    save(out/'STATUS.json',status,mutable=True)
                    if data_free<MIN_DATA_FREE or root_free<MIN_ROOT_FREE:
                        os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=60)
                        raise RuntimeError('DISK_GUARD：安全余量不足，已停止本实验子进程，未删除任何文件')
                    time.sleep(15)
                code=process.returncode
            (out/f'{name}.exitcode').write_text(str(code)+'\n')
            status['stages'][name].update(status='PASS' if code==0 else 'FAIL',finished_at=now(),exitcode=code)
            save(out/'STATUS.json',status,mutable=True)
            if code!=0:raise RuntimeError(f'{name} exited {code}')
            if name=='training':
                run=Path(pre['run_dir']);runtime=read(run/'runtime.json')['training']
                assert runtime['processed_event_count']==6*4063357
                assert sorted(p.name for p in (run/'checkpoints').iterdir())==[f'model_epoch_{i}' for i in range(1,7)]
                save(out/'config_resolved_train_snapshot.json',read(out/'effective_treatment.json'))
        status.update(status='COMPLETED',finished_at=now(),exitcode=0)
    except BaseException as exc:
        exitcode=1;status.update(status='FAILED',finished_at=now(),exitcode=1,error=repr(exc))
        print(repr(exc),flush=True)
    finally:
        save(out/'STATUS.json',status,mutable=True);(out/'pipeline.exitcode').write_text(str(exitcode)+'\n')
    return exitcode


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--artifact-root',type=Path);p.add_argument('--preflight',action='store_true')
    a=p.parse_args();out=a.analysis.resolve()
    if a.preflight:preflight(out,a.artifact_root.resolve())
    else:sys.exit(supervise(out))
