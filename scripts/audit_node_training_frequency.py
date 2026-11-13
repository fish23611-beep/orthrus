"""Only formal train endpoint tensors are read; no model, GT or val/test access."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time
import resource
from types import SimpleNamespace

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
import yaml
from mstc.training_loss import NodeContributionWeights


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def arrays_sha(*arrays):
    h=hashlib.sha256()
    for a in arrays:
        a=np.asarray(a);h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def source_for(e):
    root=Path(f'/mnt/datadisk/yby/mstc/artifacts/u4090_e{e}')
    run=root/f'THEIA_E{e}/runs/orthrus/seed_0'
    caches=list(root.glob('edge_featurization/*/embed_edges/*/edge_embeds'))
    assert len(caches)==1
    return run,caches[0]


def windows(paths, sources=None):
    previous=None
    for i,path in enumerate(paths):
        g=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        src,dst,t=g.src.numpy(),g.dst.numpy(),g.t.numpy()
        assert len(src)==len(dst)==len(t) and len(t)>0
        assert np.all(t[1:]>=t[:-1]) and (previous is None or t[0]>=previous)
        previous=t[-1]
        if sources is not None:
            s=path.stat()
            sources.append(dict(path=str(path),bytes=s.st_size,mtime_ns=s.st_mtime_ns,events=len(src),
                endpoint_time_tensor_sha256=arrays_sha(src,dst,t),first_ns=int(t[0]),last_ns=int(t[-1])))
        yield SimpleNamespace(src=g.src,dst=g.dst,split_name='train')
        if (i+1)%50==0:print(f'训练窗口 {i+1}/{len(paths)}',flush=True)


def main(out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    started=time.monotonic()
    for e in [3,5]:
        run,cache=source_for(e);cfg=yaml.safe_load((run/'config_resolved.yml').read_text())
        assert cfg['dataset_view']['mode']=='host_network_full'
        paths=sorted(p for p in (cache/'train').iterdir() if p.is_file())
        # Use only training IDs for capacity; raw caches already represent the formal target view.
        maxnode=-1
        for w in windows(paths):maxnode=max(maxnode,int(w.src.max()),int(w.dst.max()))
        sources=[];weighting=NodeContributionWeights.fit(windows(paths,sources),num_nodes=maxnode+1)
        counts=weighting.counts;active=counts[counts>0];ordered=np.sort(active)[::-1]
        shares={str(q):float(ordered[:int(np.ceil(q*len(active)))].sum()/active.sum()) for q in [.01,.05,.1]}
        weights=np.empty(weighting.num_events,dtype=np.float64);offset=0;selfloops=0;second=[]
        for w in windows(paths,second):
            n=w.src.numel();weights[offset:offset+n]=weighting.numpy_weights(w.src,w.dst);offset+=n
            selfloops+=int((w.src==w.dst).sum())
        assert sources==second and offset==len(weights)
        assert abs(weights.mean()-1)<1e-12
        quantiles=[0,.001,.01,.1,.25,.5,.75,.9,.99,.999,1]
        square=float(np.dot(weights,weights));ess=float(weights.sum()**2/square)
        sorted_w=np.sort(weights)[::-1]
        result=dict(dataset=f'THEIA_E{e}',train_graphs=cfg['dataset']['train_files'],val_graphs=cfg['dataset']['val_files'],test_graphs=cfg['dataset']['test_files'],
            training_windows=len(paths),**weighting.summary(),top_node_contribution_shares=shares,self_loops=selfloops,
            weight_distribution=dict(mean=float(weights.mean()),quantiles={str(q):float(v) for q,v in zip(quantiles,np.quantile(weights,quantiles))},
                ess=ess,ess_fraction=ess/len(weights),max_weight_fraction_of_total=float(weights.max()/weights.sum()),
                top_event_weight_shares={str(q):float(sorted_w[:int(np.ceil(q*len(weights)))].sum()/weights.sum()) for q in [.001,.01,.05,.1]}),
            raw_training_event_losses_available=False,raw_loss_frequency_groups='不可检验：原训练仅保存Batch标量日志，无逐事件训练loss；未追加全量GNN推理',
            gradient_interpretation='这里只证明事件系数不均及ESS；损失值/梯度方向与参数Jacobians未知，不能等同实际梯度支配',
            no_val_test_frequency_access=True,no_attack_labels=True,source_tensor_hashes=sources,
            config_sha256=hashlib.sha256((run/'config_resolved.yml').read_bytes()).hexdigest(),
            source_tensor_hash_rule='逐源文件src,dst,t的dtype/shape/bytes；mmap只读，不全扫大msg；两次独立扫描完全一致')
        write(out/f'e{e}_train_frequency.json',result)
        print('E',e,{k:v for k,v in result.items() if k not in ['source_tensor_hashes']},flush=True)
        del weights,sorted_w,weighting
    write(out/'frequency_execution.json',dict(status='PASS',seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__=='__main__':main(Path(sys.argv[1]))
