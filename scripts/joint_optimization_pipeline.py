"""u4090普通SSH：冻结准备、严格资源门禁、顺序Smoke/seed0；不重训Control。"""
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
GIB=1024**3
SOURCE_FILES=['src/mstc/training_loss.py','src/config.py','src/detection/orthrus_gnn_training.py',
    'src/model.py','src/mstc/experiment_utils.py','scripts/smoke_node_balanced_training.py',
    'scripts/joint_optimization_pipeline.py','scripts/evaluate_joint_optimization.py']


def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()
def save(path,value,mutable=False):
    path=Path(path)
    target=path.with_suffix(path.suffix+'.next') if mutable else path
    with target.open('w' if mutable else 'x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
    if mutable:os.replace(target,path)
def flat(d,p=''):
    if isinstance(d,dict):return {k:v for key,val in d.items() for k,v in flat(val,p+'.'+key if p else key).items()}
    return {p:d}


def resource_gate(e=None, *, usage=shutil.disk_usage, query=subprocess.check_output):
    root_free=usage('/').free;data_free=usage('/mnt/datadisk').free
    budget=(6 if e==3 else 20)*GIB if e else 0
    errors=[]
    if root_free<3*GIB:errors.append('系统盘不足3 GiB（含2 GiB安全余量）')
    if data_free<20*GIB+budget:errors.append(f'数据盘不足安全20 GiB加本实验预算{budget//GIB} GiB')
    try:
        processes=query(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True).strip()
        gpu=query(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
        if processes:errors.append('GPU存在计算进程，拒绝启动共享GPU任务')
        gpu_rows=[r.split(',') for r in gpu.splitlines() if r.strip()]
        if not gpu_rows:errors.append('未发现GPU')
        if any(int(r[-1].strip())>10 for r in gpu_rows):errors.append('GPU利用率超过10%')
    except Exception as exc:
        processes='';gpu='';errors.append('GPU检查失败：'+repr(exc))
    return dict(status='PASS' if not errors else 'BLOCKED',created_at=now(),errors=errors,
        root_free_bytes=root_free,data_free_bytes=data_free,gpu_compute_processes=processes,gpu=gpu,
        data_safety_floor_bytes=20*GIB,additional_budget_bytes=budget)


def prepare(out):
    import yaml
    import config
    from mstc.experiment_utils import sanitized_config,stable_config_hash
    assert (out/'OPTIMIZATION_DESIGN.md').is_file()
    plan=out/'prepared_05';plan.mkdir(exist_ok=False)
    overlay=yaml.safe_load((ROOT/'analysis/e3_seed0_node_balanced_treatment_20261028/treatment.yml').read_text())
    overlay['detection']['gnn_training']['train_loss_weighting']='node_contribution_shrunk'
    with (plan/'treatment.yml').open('x') as f:yaml.safe_dump(overlay,f,allow_unicode=True,sort_keys=False)
    datasets={}
    for e in [3,5]:
        shared=Path(f'/mnt/datadisk/yby/mstc/artifacts/u4090_e{e}')
        control=shared/f'THEIA_E{e}/runs/orthrus/seed_0'
        training_config=control/('config_resolved_train.yml' if e==5 else 'config_resolved.yml')
        saved=yaml.safe_load(training_config.read_text())
        old_env={key:os.environ.get(key) for key in ['ORTHRUS_DB_HOST','ORTHRUS_DB_USER','ORTHRUS_DB_PORT']}
        for key in ['host','user','port']:os.environ['ORTHRUS_DB_'+key.upper()]=str(saved['database'][key])
        artifact=Path(f'/mnt/datadisk/yby/mstc/artifacts/joint_shrunk_20261029/E{e}/seed_0/treatment')
        assert not artifact.exists(),'禁止覆盖已有正式实验'
        args=NS(dataset=f'THEIA_E{e}',model='orthrus',config=str(plan/'treatment.yml'),seed=0,cpu=False,
            from_weights=False,skip_tracing=True,artifact_root=str(artifact),shared_artifact_root=str(shared),
            shared_artifact_readonly=True,resume_checkpoint=None,inference_checkpoint=None)
        try:cfg=config.get_yml_cfg(args)
        finally:
            for key,value in old_env.items():
                if value is None:os.environ.pop(key,None)
                else:os.environ[key]=value
        effective=sanitized_config(cfg)
        historical=read(ROOT/'analysis/e3_seed0_node_balanced_treatment_20261028/effective_control.json') if e==3 else sanitized_config(__import__('yacs.config',fromlist=['CfgNode']).CfgNode(saved))
        a,b=flat(historical),flat(effective)
        # _paths are runtime-only; event_mean omitted in old configs is the legacy default.
        a={k:v for k,v in a.items() if not any(part.startswith('_') for part in k.split('.'))}
        a.setdefault('detection.gnn_training.train_loss_weighting','event_mean')
        differences={k:dict(control=a.get(k),treatment=b.get(k)) for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
        allowed={'detection.gnn_training.train_loss_weighting','detection.evaluation.checkpoint_policy'}
        assert set(differences)<=allowed, differences
        assert cfg.detection.gnn_training.num_epochs==6 and cfg.detection.gnn_training.encoder.batch_size==512
        assert list(cfg.detection.gnn_training.encoder.context.multiscale.neighbor_budgets)==[8,8,8]
        assert cfg.detection.gnn_training.decoder.time_gap.lambda_time==.3
        assert cfg.node_aggregation.topk==5 and cfg.node_aggregation.include_dst and cfg.node_threshold.quantile==.999
        assert cfg.calibration.method=='hierarchical_relation' and cfg.model_selection.method=='min_val_objective'
        assert cfg.detection.evaluation.checkpoint_policy=='val_select_then_test_best'
        cache=Path(cfg.edge_featurization.embed_edges._edge_embeds_dir)
        manifest=next((cache/'_compact_index_v2').glob('manifest__*.json'));m=read(manifest)
        assert m['completed'] and m['dataset']==f'THEIA_E{e}' and m['view_mode']=='host_network_full'
        assert (cache/'_compact_index_v2'/m['compact_file']).is_file()
        graph_splits=[set(saved['dataset'][k]) for k in ['train_files','val_files','test_files']]
        assert all(not x&y for i,x in enumerate(graph_splits) for y in graph_splits[i+1:])
        split_paths={s:{w['relative_path'] for w in m['window_specs'] if w['split_name']==s} for s in ['train','val','test']}
        assert all(split_paths.values()) and all(not split_paths[x]&split_paths[y] for x,y in [('train','val'),('train','test'),('val','test')])
        for w in m['window_specs']:assert (cache/w['relative_path']).is_file()
        baseline=read(ROOT/f'analysis/seed0_final_acceptance_20261028/e{e}_acceptance.json')
        checkpoint=control/'checkpoints'/baseline['selected_epoch']/'checkpoint.pt'
        import torch
        cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
        control_cfg=cfg.clone();control_cfg.detection.gnn_training.train_loss_weighting='event_mean'
        control_cfg.detection.evaluation.checkpoint_policy=historical['detection']['evaluation']['checkpoint_policy']
        assert cp['config_hash']==stable_config_hash(control_cfg),'冻结Control有效配置无法重现checkpoint身份'
        assert cp['epoch']==int(baseline['selected_epoch'].rsplit('_',1)[1])
        del cp
        save(plan/f'e{e}_effective.json',effective)
        common=[sys.executable,'-u',str(ROOT/'src/experiments/run_experiment.py'),'--dataset',f'THEIA_E{e}',
            '--config',str((plan/'treatment.yml').resolve()),'--seed','0','--artifact-root',str(artifact),
            '--shared-artifact-root',str(shared),'--shared-artifact-readonly']
        datasets[str(e)]=dict(dataset=f'THEIA_E{e}',artifact_root=str(artifact),run_dir=str(artifact/f'THEIA_E{e}/runs/orthrus/seed_0'),
            config_hash=stable_config_hash(cfg),config_differences=differences,baseline_source=baseline['source'],
            control_checkpoint=str(checkpoint),control_checkpoint_sha256=sha(checkpoint),control_config_sha256=sha(control/'config_resolved.yml'),control_training_config=str(training_config),
            control_training_config_sha256=sha(training_config),control_checkpoint_config_hash_exact=True,
            historical_config_note='E3历史None字段用有效配置核验；E5训练snapshot与推理snapshot的DB连接和checkpoint执行策略不同；两者训练checkpoint哈希均复现',
            manifest=str(manifest),manifest_sha256=sha(manifest),shared_cache=str(cache),
            splits={k:saved['dataset'][k] for k in ['train_files','val_files','test_files']},
            status={k:'PENDING' for k in ['gpu_smoke','training','checkpoint_validation','test','cpu_evaluation']},
            commands=dict(training=common+['--stages','train'],checkpoint_validation_and_test=common+['--stages','test'],
                cpu_evaluation=[sys.executable,'-u',str(ROOT/'scripts/evaluate_joint_optimization.py'),'--analysis',str(out),'--dataset',str(e)]))
    save(plan/'plan.json',dict(created_at=now(),git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        seed=0,epochs=6,sequential_order=[3,5],control_retraining=False,datasets=datasets,
        source_sha256={p:sha(ROOT/p) for p in SOURCE_FILES},design_sha256=sha(out/'OPTIMIZATION_DESIGN.md'),
        config_sha256=sha(plan/'treatment.yml')))
    save(plan/'resource_preflight.json',resource_gate())
    print('冻结配置与seed0顺序命令已准备；资源状态',read(plan/'resource_preflight.json')['status'],flush=True)


def check_seal(out):
    plan=read(out/'prepared_05/plan.json')
    for p,h in plan['source_sha256'].items():assert sha(ROOT/p)==h,p
    assert sha(out/'OPTIMIZATION_DESIGN.md')==plan['design_sha256']
    assert sha(out/'prepared_05/treatment.yml')==plan['config_sha256']
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==plan['git_head']
    assert (out/'cpu_tests_03.exitcode').read_text().strip()=='0'
    assert read(out/'train_weight_audit.json')['status']=='PASS'
    assert (out/'pipeline_tests_final.exitcode').read_text().strip()=='0'
    for e in [3,5]:
        d=plan['datasets'][str(e)]
        assert sha(d['control_checkpoint'])==d['control_checkpoint_sha256']
        assert sha(d['manifest'])==d['manifest_sha256']
    return plan


def smoke(out):
    plan=check_seal(out);gate=resource_gate()
    save(out/'smoke_resource_gate.json',gate)
    if gate['status']!='PASS':print(json.dumps(gate,ensure_ascii=False));return 2
    import gc
    import torch
    import data_utils
    import smoke_node_balanced_training as s
    from mstc.training_loss import ShrunkNodeContributionWeights
    from mstc.time_gap import TimeGapStatistics
    from audit_node_training_frequency import source_for,windows
    destination=out/'gpu_smoke';destination.mkdir(exist_ok=False)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    previous=s.training.log;s.training.log=lambda *a:None
    overall={}
    try:
        for e in [3,5]:
            # Check again between datasets; do not queue on a newly occupied GPU.
            assert resource_gate()['status']=='PASS'
            control,cache=source_for(e);cfg=s.cfg_for(e,destination)
            paths=sorted((cache/'train').glob('*.TemporalData.simple'))
            capacity=max(max(int(w.src.max()),int(w.dst.max())) for w in windows(paths))+1
            weights=ShrunkNodeContributionWeights.fit(windows(paths),num_nodes=capacity)
            expected=read(out/f'e{e}_shrunk_weights.json')
            assert weights.summary()['counts_sha256']==expected['counts_sha256'] and weights.tau==expected['tau']
            stats=TimeGapStatistics.load(control/'metadata/time_statistics.json')
            prepared=[data_utils._prepare_window(cfg,str(p),cfg.dataset_view.mode) for p in paths[:2]]
            graphs,_,_,full,cap=data_utils._eager_load_all_datasets(cfg,prepared,[],[])
            results={}
            for variant in ['control','balanced']:
                results[variant],parameters=s.execute(e,variant,cfg,graphs,full,cap,stats,weights,destination,
                    balanced_mode='node_contribution_shrunk')
                del parameters
            assert results['control']['initial_parameters_sha256']==results['balanced']['initial_parameters_sha256']
            assert results['control']['event_order_sha256']==results['balanced']['event_order_sha256']
            assert results['control']['state']==results['balanced']['state']
            results['shrunk']=results.pop('balanced');overall[str(e)]=results
            del graphs,full,prepared,weights;gc.collect()
        save(out/'gpu_smoke.json',dict(status='PASS',datasets=overall,formal_training_started=False))
    finally:s.training.log=previous
    return 0


def terminate_own(process):
    if process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=30)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()


def run(out):
    plan=check_seal(out)
    assert read(out/'gpu_smoke.json')['status']=='PASS','两数据集GPU Smoke尚未PASS'
    status_path=out/'FORMAL_STATUS.json'
    assert not status_path.exists(),'禁止覆盖/自动重试正式实验'
    state=dict(status='RUNNING',started_at=now(),pid=os.getpid(),datasets={})
    save(status_path,state);process=None
    try:
        for e in [3,5]:
            d=plan['datasets'][str(e)];gate=resource_gate(e)
            save(out/f'e{e}_formal_resource_gate.json',gate)
            assert gate['status']=='PASS',gate['errors']
            artifact=Path(d['artifact_root']);assert not artifact.exists()
            saved=__import__('yaml').safe_load(Path(d['control_training_config']).read_text())
            env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',WANDB_MODE='disabled',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            for key in ['host','user','port']:env['ORTHRUS_DB_'+key.upper()]=str(saved['database'][key])
            artifact.mkdir(parents=True);(artifact/'tmp').mkdir();env['TMPDIR']=str(artifact/'tmp')
            baseline_free=shutil.disk_usage('/mnt/datadisk').free;budget=(6 if e==3 else 20)*GIB
            state['datasets'][str(e)]={};stages=state['datasets'][str(e)]
            for stage,argv in d['commands'].items():
                stage_env=env.copy()
                if stage=='cpu_evaluation':stage_env['CUDA_VISIBLE_DEVICES']=''
                stages[stage]=dict(status='RUNNING',started_at=now(),argv=argv)
                with (out/f'e{e}_{stage}.log').open('x') as log:
                    process=subprocess.Popen(argv,cwd=ROOT,env=stage_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    (out/f'e{e}_{stage}.pid').write_text(str(process.pid)+'\n');stages[stage]['pid']=process.pid
                    while process.poll() is None:
                        data_free=shutil.disk_usage('/mnt/datadisk').free;root_free=shutil.disk_usage('/').free
                        sample=dict(at=now(),dataset=e,stage=stage,data_free=data_free,root_free=root_free,
                            data_net_growth=baseline_free-data_free,
                            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True),
                            cpu_memory=subprocess.check_output(['ps','-p',str(process.pid),'-o','pid,etimes,%cpu,%mem,rss'],text=True))
                        with (out/'resource_monitor.jsonl').open('a') as monitor:monitor.write(json.dumps(sample,ensure_ascii=False)+'\n')
                        save(status_path,state,mutable=True)
                        if data_free<20*GIB or root_free<2*GIB or baseline_free-data_free>budget:
                            terminate_own(process);raise RuntimeError('磁盘预算/安全水位门禁中止；历史产物未删除')
                        time.sleep(15)
                    code=process.returncode
                (out/f'e{e}_{stage}.exitcode').write_text(str(code)+'\n')
                stages[stage].update(status='PASS' if code==0 else 'FAIL',exitcode=code,finished_at=now())
                save(status_path,state,mutable=True)
                assert code==0,f'{e}/{stage}失败'
                if stage=='training':
                    run_dir=Path(d['run_dir'])
                    assert {p.name for p in (run_dir/'checkpoints').iterdir()}=={f'model_epoch_{i}' for i in range(1,7)}
                    expected=read(out/f'e{e}_shrunk_weights.json')['target_events']*6
                    assert read(run_dir/'runtime.json')['training']['processed_event_count']==expected
                if stage=='checkpoint_validation_and_test':
                    run_dir=Path(d['run_dir']);selection=read(run_dir/'checkpoint_selection.json')
                    chosen=selection['selected_epoch']
                    assert selection['val_objective_field']=='score_raw'
                    assert chosen==min(selection['per_epoch_validation_objective'],key=selection['per_epoch_validation_objective'].get)
                    assert {p.name for p in (run_dir/'edge_scores/test').iterdir()}=={chosen}
                    stages['checkpoint_validation']=dict(status='PASS',selected_epoch=chosen,selection=selection)
                    stages['test']=dict(status='PASS',selected_epoch=chosen)
            assert not list(Path(d['run_dir']).rglob('*calibrated*.csv'))
        state.update(status='COMPLETED',finished_at=now(),exitcode=0);return 0
    except BaseException as exc:
        if process is not None:terminate_own(process)
        state.update(status='FAILED',error=repr(exc),finished_at=now(),exitcode=1);raise
    finally:
        save(status_path,state,mutable=True);(out/'formal_pipeline.exitcode').write_text(str(state.get('exitcode',1))+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--mode',choices=['prepare','preflight','smoke','run'],required=True);a=p.parse_args();out=a.analysis.resolve()
    if a.mode=='prepare':prepare(out)
    elif a.mode=='preflight':print(json.dumps(resource_gate(),ensure_ascii=False,indent=2))
    elif a.mode=='smoke':sys.exit(smoke(out))
    else:sys.exit(run(out))
