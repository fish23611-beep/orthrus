"""Real prefix train windows, <=100 batches, production forward/backward/Adam.

Runs legacy/default/balanced with identical seed and order, writes no checkpoints.
"""
from pathlib import Path
import gc
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace as NS

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
import config
import factory
import data_utils
from detection import orthrus_gnn_training as training
from detection.orthrus_gnn_testing import _cleanup_checkpoint_cuda
from mstc.experiment_utils import sanitized_config
from mstc.time_gap import TimeGapStatistics
from mstc.training_loss import NodeContributionWeights
from audit_node_training_frequency import source_for,windows,write


def digest_tensors(tensors):
    h=hashlib.sha256()
    for x in tensors:
        a=x.detach().cpu().contiguous().numpy();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def cfg_for(e,out):
    args=NS(dataset=f'THEIA_E{e}',model='orthrus',config=str(ROOT/'config/experiments/mstc_full.yml'),cpu=False,
        from_weights=False,seed=0,skip_tracing=True,artifact_root=str(out/'isolated'),resume_checkpoint=None,inference_checkpoint=None)
    original=config.set_task_paths;config.set_task_paths=lambda cfg:None
    try:cfg=config.get_yml_cfg(args)
    finally:config.set_task_paths=original
    return cfg


def legacy_class():
    namespace={}
    source=subprocess.check_output(['git','show','HEAD:src/model.py'],cwd=ROOT,text=True)
    exec(compile(source,'legacy_model.py','exec'),namespace)
    return namespace['MSTCOrthrus']


def state_digest(m):
    h=m.encoder.neighbor_loader.history_store
    active=torch.nonzero(h._count>0).reshape(-1)
    return dict(time_state_sha256=hashlib.sha256(json.dumps(sorted(m.last_seen_per_node.items())).encode()).hexdigest(),
        history_sha256=digest_tensors([active,h._count[active],h._neighbor_id[active],h._event_id[active],h._timestamp_ns[active],h._direction[active]]),
        time_nodes=len(m.last_seen_per_node),history_nodes=active.numel())


def execute(e,variant,cfg,graphs,full,capacity,stats,weights,out, *, balanced_mode='node_contribution_balanced'):
    cfg=cfg.clone();cfg.detection.gnn_training.train_loss_weighting=balanced_mode if variant=='balanced' else 'event_mean'
    scoped=out/f'e{e}_{variant}';scoped.mkdir(exist_ok=False);cfg._run_dir=str(scoped)
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    baseline_allocated=torch.cuda.memory_allocated();started=time.monotonic()
    m=factory.build_model(graphs[0],torch.device('cuda'),cfg,capacity,stats)
    if variant=='legacy':m.__class__=legacy_class()
    init_hash=digest_tensors(list(m.state_dict().values()));m.reset_state()
    optimizer=factory.optimizer_factory(cfg,parameters=set(m.parameters()))
    rows=[];remaining=100;original_loader=training.batch_loader_factory
    sequence=hashlib.sha256();window_counts=[];pending={}
    def loader(cfg,data,indexer):
        nonlocal remaining
        for batch in original_loader(cfg,data,indexer):
            if remaining<=0:break
            sequence.update(batch.global_event_index.detach().cpu().numpy().tobytes())
            pending['batch']=batch
            remaining-=1
            yield batch
    def after_forward(module,args,outputs):
        batch=pending['batch'];balanced=variant=='balanced'
        if balanced:
            loss=weights.reduce(outputs['event_loss'],batch.src,batch.dst)
            w=weights.event_weights(batch.src,batch.dst)
            wmin,wmax,wmean=float(w.min()),float(w.max()),float(w.mean())
        else:loss=outputs['loss'];wmin=wmax=wmean=1.
        for value in outputs.values():
            if isinstance(value,torch.Tensor) and not bool(torch.isfinite(value).all()):raise ValueError('nonfinite forward')
        rows.append(dict(batch=len(rows),events=int(batch.src.numel()),loss=float(loss.detach()),
            unweighted_loss=float(outputs['loss'].detach()),weight_min=wmin,weight_max=wmax,weight_mean=wmean))
    def before_step(opt,args,kwargs):
        squared=0.
        for p in m.parameters():
            if p.grad is not None:
                if not bool(torch.isfinite(p.grad).all()):raise ValueError('nonfinite gradient')
                squared+=float(p.grad.detach().double().square().sum())
        rows[-1]['gradient_norm']=float(np.sqrt(squared))
        rows[-1]['allocated_bytes_before_step']=torch.cuda.memory_allocated()
    fh=m.register_forward_hook(after_forward);oh=optimizer.register_step_pre_hook(before_step)
    training.batch_loader_factory=loader
    try:
        for i,g in enumerate(graphs):
            if remaining<=0:break
            n_before=len(rows);g.to('cuda')
            objective=training.train(g,full,m,optimizer,cfg,loss_weighting=weights if variant=='balanced' else None)
            g.to('cpu');window_counts.append(dict(window=i,batches=len(rows)-n_before,logged_objective=float(objective)))
            print('SMOKE',e,variant,'window',i,'batches',len(rows),flush=True)
    finally:
        training.batch_loader_factory=original_loader;fh.remove();oh.remove();pending.clear()
    torch.cuda.synchronize()
    assert rows and len(rows)<=100 and len(window_counts)==2
    state=state_digest(m);parameters={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    elapsed=time.monotonic()-started
    result=dict(status='PASS',variant=variant,batches=len(rows),events=sum(r['events'] for r in rows),windows=window_counts,
        seconds=elapsed,initial_parameters_sha256=init_hash,event_order_sha256=sequence.hexdigest(),state=state,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        batch_records=rows,weighted_loss_event_mean=sum(r['loss']*r['events'] for r in rows)/sum(r['events'] for r in rows),
        unweighted_loss_event_mean=sum(r['unweighted_loss']*r['events'] for r in rows)/sum(r['events'] for r in rows),
        gradient_norm_min=min(r['gradient_norm'] for r in rows),gradient_norm_max=max(r['gradient_norm'] for r in rows),
        no_nan_inf=True,checkpoint_written=False,seed=0)
    write(scoped/'config.json',sanitized_config(cfg));write(scoped/'result.json',result)
    _cleanup_checkpoint_cuda(m,torch.device('cuda'))
    m.to('cpu')
    weights._device_inverse.clear()
    del optimizer,m,fh,oh;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
    result['cleanup_before_cublas_workspace_clear_bytes']=torch.cuda.memory_allocated()
    # cuBLAS retains workspaces after the first GEMM independently of model
    # tensors. Clear only after training; this does not alter any step/kernel.
    clear_workspaces=getattr(torch._C,'_cuda_clearCublasWorkspaces',None)
    if clear_workspaces is not None:clear_workspaces()
    torch.cuda.synchronize();torch.cuda.empty_cache()
    result['cleanup_allocated_bytes']=torch.cuda.memory_allocated()
    write(scoped/'cleanup.json',dict(baseline_allocated_bytes=baseline_allocated,cleanup_allocated_bytes=result['cleanup_allocated_bytes'],
        before_workspace_clear_bytes=result['cleanup_before_cublas_workspace_clear_bytes'],workspace_clear_available=clear_workspaces is not None))
    assert result['cleanup_allocated_bytes']<=baseline_allocated+10*1024**2,'GPU allocations retained after full cleanup'
    return result,parameters


def main(out):
    assert (out/'tests_cpu_02.exitcode').read_text().strip()=='0'
    audit=out
    out=out/os.environ.get('NODE_BALANCED_SMOKE_ATTEMPT','gpu_attempt_04');out.mkdir(exist_ok=False)
    assert shutil.disk_usage(out).free>2*1024**3 and shutil.disk_usage('/mnt/datadisk').free>2*1024**3
    assert torch.cuda.is_available()
    free,total=torch.cuda.mem_get_info();assert free>8*1024**3
    write(out/'smoke_preflight.json',dict(gpu=torch.cuda.get_device_name(),free_vram_bytes=free,total_vram_bytes=total,
        root_free_bytes=shutil.disk_usage(out).free,data_free_bytes=shutil.disk_usage('/mnt/datadisk').free,
        maximum_batches=100,maximum_windows=2,no_full_training=True,no_checkpoints=True))
    torch.set_num_threads(2)
    # Scope deterministic kernels to this diagnostic process. The first normal
    # CUDA run showed reduction-order drift even for the unchanged legacy path.
    torch.use_deterministic_algorithms(True)
    overall={};old_forward=training.log;training.log=lambda *a:None
    try:
        for e in [3,5]:
            run,cache=source_for(e);cfg=cfg_for(e,out);paths=sorted(p for p in (cache/'train').iterdir() if p.is_file())
            cap=max(max(int(w.src.max()),int(w.dst.max())) for w in windows(paths))+1
            weights=NodeContributionWeights.fit(windows(paths),num_nodes=cap)
            frequency=json.loads((audit/f'e{e}_train_frequency.json').read_text())
            assert weights.summary()['counts_sha256']==frequency['counts_sha256']
            stats=TimeGapStatistics.load(run/'metadata/time_statistics.json')
            prepared=[data_utils._prepare_window(cfg,str(p),cfg.dataset_view.mode) for p in paths[:2]]
            graphs,_,_,full,capacity=data_utils._eager_load_all_datasets(cfg,prepared,[],[])
            source_stats={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in paths[:2]}
            results={};parameter_sets={}
            for variant in ['legacy','control','balanced']:
                results[variant],parameter_sets[variant]=execute(e,variant,cfg,graphs,full,capacity,stats,weights,out)
            a,b=parameter_sets['legacy'],parameter_sets['control']
            deltas={k:float((a[k]-b[k]).abs().max()) for k in a}
            exact=all(torch.equal(a[k],b[k]) for k in a)
            write(out/f'e{e}_baseline_deltas.json',dict(max_absolute=max(deltas.values()),by_parameter=deltas,bitwise_exact=exact,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))
            assert all(torch.allclose(a[k],b[k],rtol=1e-5,atol=1e-6) for k in a),'default diverged from legacy'
            assert len({r['initial_parameters_sha256'] for r in results.values()})==1
            assert len({r['event_order_sha256'] for r in results.values()})==1
            assert all(r['state']==results['control']['state'] for r in results.values())
            assert all((p.stat().st_size,p.stat().st_mtime_ns)==source_stats[str(p)] for p in paths[:2])
            overall[f'E{e}']=dict(runs=results,baseline_legacy_parameters_bitwise_exact=exact,
                baseline_legacy_max_parameter_error=max(deltas.values()),baseline_tolerance=dict(rtol=1e-5,atol=1e-6),
                initial_parameters_identical=True,batch_order_identical=True,time_and_neighbor_states_identical=True,
                node_weights_fitted_full_train=True,training_weight_fit=weights.summary(),source_windows=[str(p) for p in paths[:2]],
                note='仅训练可运行性和数值检查；未评估攻击性能；未保存任何模型')
            write(out/f'e{e}_smoke_summary.json',overall[f'E{e}'])
            del graphs,full,prepared,weights,parameter_sets;gc.collect()
        write(out/'gpu_smoke.json',dict(status='PASS',results=overall,formal_training_started=False))
    finally:training.log=old_forward


if __name__=='__main__':
    try:main(Path(sys.argv[1]))
    except Exception as exc:
        out=Path(sys.argv[1])/os.environ.get('NODE_BALANCED_SMOKE_ATTEMPT','gpu_attempt_04');write(out/'gpu_smoke_failure.json',dict(status='FAIL',error=repr(exc)))
        raise
