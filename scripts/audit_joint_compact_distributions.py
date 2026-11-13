"""CPU只读紧凑索引：补充训练/验证/测试无标签分布及未见实体统计。"""
import json
import os
from pathlib import Path
import sys
import time
import resource
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from joint_optimization_pipeline import read,save,sha


def main(out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    started=time.monotonic();result={}
    for e in [3,5]:
        cache=next(Path(f'/mnt/datadisk/yby/mstc/artifacts/u4090_e{e}').glob('edge_featurization/*/embed_edges/*/edge_embeds'))
        mp=next((cache/'_compact_index_v2').glob('manifest__*.json'));manifest=read(mp)
        path=cache/'_compact_index_v2'/manifest['compact_file'];before=(path.stat().st_size,path.stat().st_mtime_ns)
        data=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        capacity=manifest['max_node'];seen=np.zeros(capacity,dtype=bool);known_triplets=set();splits={};last_end=None
        for split_index,split in enumerate(['train','val','test']):
            observed=np.zeros(capacity,dtype=bool);node_types=np.full(capacity,-1,dtype=np.int8)
            rel=np.zeros(10,dtype=np.int64);triplets={};events=0;unseen_events=0;first=None;last=None
            for w in (x for x in manifest['window_specs'] if x['split_name']==split):
                start=w['global_offset'];stop=start+w['num_events'];sl=slice(start,stop)
                assert bool((data['split'][sl]==split_index).all())
                src,dst,t,st,dt,er=[data[k][sl].numpy() for k in ['src','dst','t','src_type','dst_type','edge_type_index']]
                assert np.all(t[1:]>=t[:-1]) and (last is None or t[0]>=last)
                if first is None:first=int(t[0])
                last=int(t[-1]);events+=len(t)
                observed[src]=True;observed[dst]=True;node_types[src]=st;node_types[dst]=dt
                rel+=np.bincount(er,minlength=10)
                keys=st.astype(np.int64)*100+er*10+dt
                values,counts=np.unique(keys,return_counts=True)
                for key,n in zip(values,counts):triplets[int(key)]=triplets.get(int(key),0)+int(n)
                if split!='train':unseen_events+=int((~seen[src]|~seen[dst]).sum())
            assert last_end is None or first>last_end,'跨split时间重叠'
            last_end=last
            if split=='train':seen=observed.copy();known_triplets=set(triplets)
            splits[split]=dict(events=events,nodes=int(observed.sum()),first_ns=first,last_ns=last,
                unseen_nodes_vs_train=int((observed&~seen).sum()),unseen_node_fraction_vs_train=float((observed&~seen).sum()/observed.sum()),
                events_with_unseen_endpoint_vs_train=unseen_events,relation_counts=rel.tolist(),
                node_type_counts={str(int(k)):int(np.sum(node_types[observed]==k)) for k in np.unique(node_types[observed])},
                triplet_event_counts={str(k):v for k,v in triplets.items()},
                unseen_triplet_events=sum(n for k,n in triplets.items() if k not in known_triplets))
        assert before==(path.stat().st_size,path.stat().st_mtime_ns)
        statpath=Path(f'/mnt/datadisk/yby/mstc/artifacts/u4090_e{e}/THEIA_E{e}/runs/orthrus/seed_0/metadata/time_statistics.json')
        result[str(e)]=dict(splits=splits,train_time_statistics=read(statpath),train_time_statistics_sha256=sha(statpath),
            compact_manifest=str(mp),compact_manifest_sha256=sha(mp),compact_before_after_unchanged=True,
            chronological_splits=True,labels_read=False,triplet_encoding='100*src_type+10*relation+dst_type',
            note='未见ID不等于语义未见；类型/关系频率为正常训练与验证及混合测试全事件的无标签描述；不是攻击条件统计')
        print('无标签分布PASS',e,{s:{k:v for k,v in d.items() if k in ['events','nodes','unseen_node_fraction_vs_train','unseen_triplet_events']} for s,d in splits.items()},flush=True)
    save(out/'compact_distribution_audit.json',dict(status='PASS',datasets=result,seconds=time.monotonic()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__=='__main__':main(Path(sys.argv[1]))
