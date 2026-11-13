"""E3/E5 seed0 CPU诊断；复用冻结恢复协议，不搜索参数，不改写任何输入。"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ks_2samp
import analyze_e3_postprocessing as e3
import restore_e5_frozen_postprocessing as r
from mstc.metrics import compute_classification_metrics

ACCEPT = ROOT/'analysis/seed0_final_acceptance_20261028'
OLD5 = ROOT/'analysis/e5_seed0_cpu_recovery_20261028/run_01'
BINS = [(1,5),(6,10),(11,100),(101,1000),(1001,10**12)]
QUANTILES = [0, .1, .25, .5, .75, .9, .95, .99, .999, 1]


def read(p):
    return json.loads(Path(p).read_text())


def table(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def dist(x):
    x = np.asarray(x)
    return dict(n=len(x), mean=float(np.mean(x)) if len(x) else None,
                quantiles={str(q):float(v) for q,v in zip(QUANTILES,np.quantile(x,QUANTILES))} if len(x) else {})


def deterministic_ranks(ids, scores):
    ids, scores = np.asarray(ids), np.asarray(scores)
    if len(np.unique(ids)) != len(ids) or not np.isfinite(scores).all():
        raise ValueError('节点ID重复或分数非法')
    order = np.lexsort((ids, -scores))
    ranks = np.empty(len(ids), dtype=np.int64); ranks[order] = np.arange(1,len(ids)+1)
    ascending = np.sort(scores)
    best = len(ids)-np.searchsorted(ascending,scores,side='right')+1
    worst = len(ids)-np.searchsorted(ascending,scores,side='left')
    assert np.array_equal(np.sort(ranks),np.arange(1,len(ids)+1))
    assert np.all((best<=ranks)&(ranks<=worst))
    return order, ranks, best, worst


def budget_metrics(ids, scores, truth, budgets):
    order, ranks, best, worst = deterministic_ranks(ids,scores)
    total = int(np.sum(truth)); rows=[]
    for n in budgets:
        n = min(int(n),len(ids)); chosen=order[:n]
        tp=int(np.sum(truth[chosen])); cutoff=float(scores[chosen[-1]])
        strict=scores>cutoff; tie=scores==cutoff
        slots=n-int(strict.sum()); tie_pos=int(truth[tie].sum()); tie_neg=int(tie.sum())-tie_pos
        base=int(truth[strict].sum())
        rows.append(dict(budget=n,tp=tp,fp=n-tp,recall=tp/total,precision=tp/n,
                         fpr=(n-tp)/(len(ids)-total),score_cutoff=cutoff,
                         cutoff_tie_nodes=int(tie.sum()),cutoff_tie_attacks=tie_pos,
                         tp_tie_min=base+max(0,slots-tie_neg),tp_tie_max=base+min(slots,tie_pos)))
    return rows


class DiagnosticBank(e3.TopK):
    """有界Top5加来源；ties按事件ID、方向排序，保留自环双端贡献。"""
    def __init__(self, ids, maximum=5, threshold=0., relations=32):
        super().__init__(ids,maximum)
        self.threshold=threshold; self.relations=relations
        n=len(ids)
        self.event=np.full((n,5),np.iinfo(np.int64).max,dtype=np.int64)
        self.rel=np.full((n,5),-1,dtype=np.int16)
        self.direction=np.full((n,5),2,dtype=np.int8)
        self.raw=np.zeros((n,5)); self.kind=np.zeros((n,5)); self.temporal=np.zeros((n,5))
        self.level=np.full((n,5),-1,dtype=np.int8)
        self.src_counts=np.zeros(n,dtype=np.int64)
        self.relation_mask=np.zeros(n,dtype=np.uint64)
        self.rel_counts=np.zeros((n,relations),dtype=np.uint32)
        self.rel_tail=np.zeros((n,relations),dtype=np.uint32)
        self.src_max=np.full(n,-np.inf); self.dst_max=np.full(n,-np.inf)
        self.events=0; self.self_loops=0
        self.max_joint_residual=0.; self.max_time_residual=0.

    def add_frame(self, frame, scores, levels=None):
        self.events+=len(frame)
        self.self_loops+=int((frame.srcnode.to_numpy()==frame.dstnode.to_numpy()).sum())
        self.max_joint_residual=max(self.max_joint_residual,float(np.max(np.abs(frame.score_raw-frame.loss_type-.3*frame.loss_time))))
        self.max_time_residual=max(self.max_time_residual,float(np.max(np.abs(frame.loss_time-(frame.loss_time_src+frame.loss_time_dst)/2))))
        if levels is None:
            levels=pd.Categorical(frame.calibration_level,categories=list(r.LEVELS)).codes
        for direction, field in enumerate(['srcnode','dstnode']):
            self.add_contributions(frame[field].to_numpy(),scores,frame,direction,np.asarray(levels))

    def add_contributions(self, ids, scores, f, direction, levels):
        pos=np.searchsorted(self.ids,ids)
        if np.any(pos>=len(self.ids)) or not np.array_equal(self.ids[pos],ids):
            raise ValueError('节点不在全集')
        if not np.isfinite(scores).all(): raise ValueError('非有限事件分数')
        rel=f.edge_type_index.to_numpy()
        if np.any((rel<0)|(rel>=self.relations)): raise ValueError('关系ID超出诊断容量')
        bits=np.left_shift(np.uint64(1),rel.astype(np.uint64))
        np.bitwise_or.at(self.relation_mask,pos,bits)
        np.add.at(self.rel_counts,(pos,rel),1)
        high=scores>self.threshold
        np.add.at(self.rel_tail,(pos[high],rel[high]),1)
        np.maximum.at(self.src_max if direction==0 else self.dst_max,pos,scores)
        if direction==0: np.add.at(self.src_counts,pos,1)
        event=f.event_index.to_numpy()
        order=np.lexsort((event,-scores,pos)); nodes=pos[order]
        unique,start,counts=np.unique(nodes,return_index=True,return_counts=True)
        within=np.arange(len(nodes))-np.repeat(start,counts); keep=within<5
        groups=np.repeat(np.arange(len(unique)),counts)[keep]; indexes=within[keep]
        selected=order[keep]
        payload={'values':scores,'event':event,'rel':rel,'direction':np.full(len(ids),direction),
                 'raw':f.score_raw.to_numpy(),'kind':f.loss_type.to_numpy(),
                 'temporal':f.loss_time.to_numpy(),'level':levels}
        merged={}
        for key,data in payload.items():
            current=getattr(self,key)
            fill=-np.inf if key=='values' else (np.iinfo(np.int64).max if key=='event' else (2 if key=='direction' else -1))
            incoming=np.full((len(unique),5),fill,dtype=current.dtype)
            incoming[groups,indexes]=data[selected]
            merged[key]=np.concatenate([current[unique],incoming],axis=1)
        pick=np.lexsort((merged['direction'],merged['event'],-merged['values']),axis=1)[:,:5]
        for key,data in merged.items():
            getattr(self,key)[unique]=np.take_along_axis(data,pick,axis=1)
        self.counts[unique]+=counts


def mapping_e3(metadata, ids, truth):
    gt,cache=r.load_pickle(metadata/'ground_truth_nodes.pkl')
    um,um_ev=r.load_pickle(metadata/'uuid_to_node_id.pkl')
    inverse,inv_ev=r.load_pickle(metadata/'node_id_to_uuid.pkl')
    rows=[]; ev=[]
    for name in ['node_Browser_Extension_Drakon_Dropper.csv','node_Firefox_Backdoor_Drakon_In_Memory.csv']:
        p=ROOT/'Ground_Truth/darpa/darpa/E3-THEIA'/name
        with p.open(newline='') as f: rows.extend(list(csv.reader(f)))
        ev.append(dict(path=str(p),sha256=e3.sha(p)))
    expected,allgt,uuids=r.map_ground_truth(rows,um,inverse,gt,ids)
    assert np.array_equal(expected,truth)
    return allgt,uuids,[cache,um_ev,inv_ev,*ev]


def metadata_for(e):
    root=Path(f'/mnt/datadisk/yby/mstc/artifacts/u4090_e{e}/graph_construction/THEIA_E{e}/build_graphs')
    found=[p for p in root.glob('*/metadata') if (p/'dataset_manifest.json').is_file()]
    assert len(found)==1
    return found[0]


def analyze(dataset, ids, scores, truth, bank, threshold, val_ids, val_scores, val_counts,
            gt, uuids, k1threshold, score1, out, records):
    pred=scores>threshold; masks={'TP':(truth==1)&pred,'FP':(truth==0)&pred,
                               'FN':(truth==1)&~pred,'TN':(truth==0)&~pred}
    coverage=sum(m.astype(np.int8) for m in masks.values())
    assert np.all(coverage==1)
    metrics=compute_classification_metrics(truth,pred,scores)
    official=read(ACCEPT/f'{dataset.lower()}_acceptance.json')
    for k in ['tp','fp','fn','tn','precision','recall','fpr','mcc','auprc','auroc']:
        assert abs(metrics[k]-official['metrics'][k])<=1e-12,(dataset,k,metrics[k],official['metrics'][k])
    assert len(ids)==official['node_count'] and int(truth.sum())==official['attack_node_count']
    order,ranks,best,worst=deterministic_ranks(ids,scores)
    assert np.array_equal(scores>threshold,ranks<=int(pred.sum()))
    budget=sorted(set([100,500,1000,5000,10000,int(np.ceil(len(ids)*.01)),int(np.ceil(len(ids)*.05)),int(np.ceil(len(ids)*.10))]))
    topn=budget_metrics(ids,scores,truth,budget)
    for row in topn: row['dataset']=dataset
    records['topn'].extend(topn)
    positions=np.searchsorted(bank.ids,ids); counts=bank.counts[positions]
    values=bank.values[positions]; valid=np.isfinite(values); used=valid.sum(axis=1)
    maxscore=values[:,0]; sums=np.where(valid,values,0).sum(axis=1)
    shares=np.divide(maxscore,sums,out=np.zeros(len(ids)),where=sums>0)
    directions=bank.direction[positions]
    dst_top=np.sum(valid&(directions==1),axis=1)/used
    relmask=bank.relation_mask[positions]
    unique_rel=np.fromiter((int(x).bit_count() for x in relmask),np.int16,len(ids))
    top_events=bank.event[positions]
    unique_top=np.array([len(set(row[v].tolist())) for row,v in zip(top_events,valid)])
    attack_positions=np.flatnonzero(truth)
    fn=masks['FN']; near=fn&(scores>=.9*threshold)
    hightail=fn&~near&(ranks<=int(np.ceil(.01*len(ids))))
    low=fn&~near&~hightail
    for i in attack_positions[np.argsort(ranks[attack_positions])]:
        p=positions[i]
        rec=dict(dataset=dataset,node_id=int(ids[i]),uuid=uuids[int(ids[i])],in_test=True,
                 score=float(scores[i]),threshold=threshold,score_minus_threshold=float(scores[i]-threshold),
                 score_over_threshold=float(scores[i]/threshold),rank=int(ranks[i]),
                 rank_top_percent=100*float(ranks[i])/len(ids),rank_tie_best=int(best[i]),rank_tie_worst=int(worst[i]),
                 status='TP' if pred[i] else 'FN',fn_group=('接近阈值' if near[i] else '其余Top1%' if hightail[i] else '其余较低分') if fn[i] else '',
                 contribution_count=int(counts[i]),src_contributions=int(bank.src_counts[p]),
                 dst_contributions=int(counts[i]-bank.src_counts[p]),relation_types=int(unique_rel[i]),
                 top1_score=float(maxscore[i]),top1_minus_top5=float(maxscore[i]-scores[i]),
                 top1_share_of_top5_sum=float(shares[i]),top5_dst_share=float(dst_top[i]),
                 top5_unique_events=int(unique_top[i]),high_event_contributions=int(bank.rel_tail[p].sum()),
                 k1_alert=bool(score1[i]>k1threshold))
        for j in range(5):
            rec[f'top{j+1}_score']=float(values[i,j]) if valid[i,j] else None
        records['attacks'].append(rec)
    for nid in sorted(gt-set(map(int,ids))):
        records['attacks'].append(dict(dataset=dataset,node_id=nid,uuid=uuids[nid],in_test=False,status='未出现在测试事件，排除正式分母'))
    rankdist={}
    for label,mask in [('攻击全部',truth==1),('TP',masks['TP']),('FN',fn)]:
        rankdist[label]=dist(ranks[mask])
        q=rankdist[label]['quantiles']
        records['rank_summary'].append(dict(dataset=dataset,group=label,n=int(mask.sum()),
            rank_min=q.get('0'),rank_median=q.get('0.5'),rank_p75=q.get('0.75'),rank_p90=q.get('0.9'),rank_max=q.get('1'),
            median_top_percent=q.get('0.5',0)/len(ids)*100,
            top01pct=int((mask&(ranks<=np.ceil(.001*len(ids)))).sum()),
            top1pct=int((mask&(ranks<=np.ceil(.01*len(ids)))).sum()),
            top5pct=int((mask&(ranks<=np.ceil(.05*len(ids)))).sum()),
            top10pct=int((mask&(ranks<=np.ceil(.1*len(ids)))).sum())))
    groupstats={}; relation_rows=[]
    for label,mask in masks.items():
        p=positions[mask]; v=valid[mask]
        sub=dict(nodes=int(mask.sum()),score=dist(scores[mask]),score_ratio=dist(scores[mask]/threshold),
            rank=dist(ranks[mask]),contributions=dist(counts[mask]),relation_types=dist(unique_rel[mask]),
            src_fraction=dist(bank.src_counts[p]/counts[mask]),top5_dst_fraction=dist(dst_top[mask]),
            top1_minus_top5=dist(maxscore[mask]-scores[mask]),top1_share=dist(shares[mask]),
            top5_event_score=dist(values[mask][v]),
            max_event_above_formal_threshold=int((mask&(maxscore>threshold)).sum()),
            max_event_above_k1_threshold=int((mask&(maxscore>k1threshold)).sum()),
            one_or_fewer_events_above_threshold=int((bank.rel_tail[p].sum(axis=1)<=1).sum()),
            top1_dominates_ge_half_sum=int((mask&(shares>=.5)).sum()),
            dst_top5_majority=int((mask&(dst_top>.5)).sum()),
            all_top5_dst=int((mask&(dst_top==1)).sum()),
            duplicated_top5_events=int((mask&(unique_top<used)).sum()),
            top5_raw_loss=dist(bank.raw[p][v]),top5_type_loss=dist(bank.kind[p][v]),
            top5_time_loss=dist(bank.temporal[p][v]))
        groupstats[label]=sub
        records['fp'].append(dict(dataset=dataset,analysis='类别总体',group=label,nodes=sub['nodes'],
            contribution_median=sub['contributions']['quantiles'].get('0.5'),
            score_median=sub['score']['quantiles'].get('0.5'),top5_dst_share_median=sub['top5_dst_fraction']['quantiles'].get('0.5'),
            top1_minus_top5_median=sub['top1_minus_top5']['quantiles'].get('0.5'),
            top1_dominates=sub['top1_dominates_ge_half_sum'],max_above_threshold=sub['max_event_above_formal_threshold']))
        rels=bank.rel[p]
        for rel in range(bank.relations):
            sel=v&(rels==rel); n=int(sel.sum()); allc=int(bank.rel_counts[p,rel].sum()); tail=int(bank.rel_tail[p,rel].sum())
            if not allc: continue
            relation_rows.append(dict(dataset=dataset,group=label,edge_type_index=rel,
                all_contributions=allc,above_node_threshold_contributions=tail,
                top5_contributions=n,top5_score_sum=float(values[mask][sel].sum()),
                top5_dst_contributions=int((sel&(directions[mask]==1)).sum())))
    records['relations'].extend(relation_rows)
    strata=[]
    for lo,hi in BINS:
        sel=(counts>=lo)&(counts<=hi); benign=sel&(truth==0)
        n=int(benign.sum()); fp=int((benign&pred).sum())
        vm=(val_counts>=lo)&(val_counts<=hi)
        row=dict(dataset=dataset,analysis='频率分层',group=f'{lo}-{hi}',benign_nodes=n,fp=fp,
            fpr=fp/n if n else None,share_all_fp=fp/metrics['fp'],
            tp=int((sel&masks['TP']).sum()),fn=int((sel&fn).sum()),
            validation_nodes=int(vm.sum()),validation_fpr=float((val_scores[vm]>threshold).mean()) if vm.any() else None,
            k1_fp=int((benign&(score1>k1threshold)).sum()),
            k1_only_fp=int((benign&(score1>k1threshold)&~pred).sum()),
            k5_only_fp=int((benign&(score1<=k1threshold)&pred).sum()))
        strata.append(row); records['fp'].append(row)
    valclean=~np.isin(val_ids,list(gt)); benign=truth==0
    valtail=float((val_scores>threshold).mean()); testtail=float((scores[benign]>threshold).mean())
    drift=dict(validation_assumed_normal=dist(val_scores),validation_excluding_gt_identities=dist(val_scores[valclean]),
        validation_gt_identity_count=int((~valclean).sum()),test_benign=dist(scores[benign]),
        validation_exceedance=valtail,test_benign_exceedance=testtail,
        exceedance_ratio=testtail/valtail if valtail else None,
        ks_statistic=float(ks_2samp(val_scores[valclean],scores[benign],method='asymp').statistic),
        validation_frequency_standardized_to_test_fpr=sum(x['benign_nodes']*x['validation_fpr'] for x in strata if x['validation_fpr'] is not None)/int(benign.sum()))
    percentile_rows=[]
    for q in [.9,.95,.99,.999]:
        t=float(np.quantile(val_scores,q)); alert=scores>t
        percentile_rows.append(dict(dataset=dataset,reference='验证节点分位数（描述性，非新正式阈值）',quantile=q,score=t,
            alerts=int(alert.sum()),attack_covered=int((alert&(truth==1)).sum()),normal_alerts=int((alert&benign).sum()),
            validation_exceedance=float((val_scores>t).mean()),test_benign_exceedance=float((alert&benign).sum()/benign.sum())))
        t=float(np.quantile(scores,q)); alert=scores>t
        percentile_rows.append(dict(dataset=dataset,reference='测试全体节点固定分位数（描述性）',quantile=q,score=t,
            alerts=int(alert.sum()),attack_covered=int((alert&(truth==1)).sum()),normal_alerts=int((alert&benign).sum())))
    records['percentiles'].extend(percentile_rows)
    k1=score1>k1threshold
    k1stats=dict(threshold=k1threshold,metrics=compute_classification_metrics(truth,k1,score1),
        attack_both=int(((truth==1)&k1&pred).sum()),attack_k1_only=int(((truth==1)&k1&~pred).sum()),
        attack_k5_only=int(((truth==1)&~k1&pred).sum()),
        fp_both=int((benign&k1&pred).sum()),fp_k1_only=int((benign&k1&~pred).sum()),fp_k5_only=int((benign&~k1&pred).sum()))
    result=dict(dataset=dataset,nodes=len(ids),attacks=int(truth.sum()),threshold=threshold,metrics=metrics,
        ranks=rankdist,attack_top_percent=dist(ranks[truth==1]/len(ids)*100),
        attack_score=dist(scores[truth==1]),attack_score_ratio=dist(scores[truth==1]/threshold),
        fn_distance_groups=dict(near_ratio_09_to_1=int(near.sum()),other_top1pct=int(hightail.sum()),other_lower=int(low.sum())),
        fn_ratio=dist(scores[fn]/threshold),fn_margin=dist(scores[fn]-threshold),
        fn_top1pct=int((fn&(ranks<=np.ceil(len(ids)*.01))).sum()),
        absent_gt_nodes=sorted(gt-set(map(int,ids))),groups=groupstats,strata=strata,drift=drift,k1=k1stats,
        frequency_score_spearman=float(spearmanr(counts[benign],scores[benign]).statistic),
        events=bank.events,self_loops=bank.self_loops,contributions=int(counts.sum()),
        loss_joint_max_residual=bank.max_joint_residual,loss_time_max_residual=bank.max_time_residual,
        verification=dict(baseline_metrics_exact=True,partition_exact=True,rank_permutation=True,
            alert_rank_prefix_exact=True,node_count_exact=True,attack_count_exact=True,
            scores_sha256=r.digest_arrays(scores),ids_sha256=r.digest_arrays(ids),
            rank_sha256=r.digest_arrays(ranks),labels_sha256=r.digest_arrays(truth)))
    assert int(counts.sum())==2*bank.events
    assert np.array_equal(scores,bank.scores(5)[positions]) or np.max(np.abs(scores-bank.scores(5)[positions]))<1e-12
    e3.write_json(out/f'{dataset.lower()}_diagnostics.json',result)
    print(dataset,'诊断完成',json.dumps(dict(rank=rankdist['攻击全部'],groups=result['fn_distance_groups'],topn=topn[:4]),ensure_ascii=False),flush=True)
    return result


def e3_run(out,records,evidence):
    a=read(ACCEPT/'e3_acceptance.json'); run=Path(a['input_run']); threshold=a['threshold']
    nodes=pd.read_csv(run/'node_scores/node_predictions.csv',float_precision='round_trip').sort_values('node_id')
    ids=nodes.node_id.to_numpy(); truth=nodes.y_true.to_numpy(); scores=nodes.score.to_numpy()
    assert np.array_equal(nodes.y_hat.to_numpy(),(scores>threshold).astype(int))
    gt,uuids,label_ev=mapping_e3(metadata_for(3),ids,truth); evidence.extend(label_ev)
    valpath=run/'evaluation_results/calibration/model_epoch_4/validation_calibrated.csv'
    val=pd.read_csv(valpath,usecols=e3.COLUMNS,float_precision='round_trip')
    val_ids=np.union1d(val.srcnode,val.dstnode); vb=e3.TopK(val_ids,maximum=5)
    for start in range(0,len(val),100000):
        f=val.iloc[start:start+100000]; s=f.score_calibrated.to_numpy()
        vb.update(f.srcnode.to_numpy(),s); vb.update(f.dstnode.to_numpy(),s)
    vs=vb.scores(5); k1threshold=float(np.quantile(vb.scores(1),.999))
    assert float(np.quantile(vs,.999))==threshold
    vcounts=vb.counts.copy(); evidence.append(dict(path=str(valpath),sha256=e3.sha(valpath)))
    del val,vb;gc.collect()
    bank=DiagnosticBank(ids,threshold=threshold)
    event_evidence={}; started=time.monotonic()
    for f in e3.read_chunks(run/'node_scores/event_predictions.csv',100000,event_evidence):
        bank.add_frame(f,f.score_calibrated.to_numpy())
        if bank.events%500000==0: print(f'E3事件 {bank.events:,} 用时 {time.monotonic()-started:.1f}s',flush=True)
    evidence.append(event_evidence)
    expected=read(ROOT/'analysis/e3_seed0_exploratory_20261028/run_02/summary.json')
    assert event_evidence['sha256']==expected['sources']['test']['sha256']
    assert np.max(np.abs(bank.scores(5)-scores))<1e-12
    result=analyze('E3',ids,scores,truth,bank,threshold,val_ids,vs,vcounts,gt,uuids,k1threshold,bank.scores(1),out,records)
    del bank;gc.collect()
    return result


def e5_run(out,records,evidence):
    # 原恢复程序执行全部原始文件哈希、校准前缀、UUID双射和冻结分数校验。
    original_chunks=r.RawSource.chunks; original_make=r.make_scores
    original_freeze=r.freeze_predictions; original_top=r.TopK
    state={}; frame_state={}
    def chunks(self):
        for rows,frame in original_chunks(self):
            if self.audit['expected_events']==14605144:
                arr=np.asarray(rows,dtype=object)
                for name,col in [('loss_type',11),('loss_time_src',12),('loss_time_dst',13),('loss_time',14)]:
                    frame[name]=arr[:,col].astype(float)
                frame_state['frame']=frame;frame_state['direction']=0
            yield rows,frame
    class HookBank(DiagnosticBank):
        def update(self,ids,scores):
            frame=frame_state['frame']; direction=frame_state['direction']
            self.add_contributions(ids,scores,frame,direction,np.zeros(len(frame),dtype=np.int8))
            if direction==0:
                self.events+=len(frame); self.self_loops+=int((frame.srcnode.to_numpy()==frame.dstnode.to_numpy()).sum())
                self.max_joint_residual=max(self.max_joint_residual,float(np.max(np.abs(frame.score_raw-frame.loss_type-.3*frame.loss_time))))
                self.max_time_residual=max(self.max_time_residual,float(np.max(np.abs(frame.loss_time-(frame.loss_time_src+frame.loss_time_dst)/2))))
            frame_state['direction']+=1
    def factory(ids,maximum=5):
        if len(ids)==234896: return original_top(ids,maximum=maximum)
        b=HookBank(ids,maximum=maximum,threshold=read(OLD5/'frozen_thresholds.json')['thresholds']['5'])
        state['bank']=b;return b
    def make(bank):
        result=original_make(bank)
        if 'bank' not in state: state['val']=result
        else: state['test']=result
        return result
    def freeze(path,ids,scores,thresholds):
        predictions=original_freeze(path,ids,scores,thresholds)
        assert read(path)==read(OLD5/'frozen_predictions.json'),'E5全部分数/ID/预测哈希不一致'
        state['frozen_exact']=True
        return predictions
    r.RawSource.chunks=chunks;r.TopK=factory;r.make_scores=make;r.freeze_predictions=freeze
    try:
        run=Path(read(ACCEPT/'e5_acceptance.json')['input_run'])
        args=SimpleNamespace(run=run,audit=ROOT/'analysis/e5_seed0_frozen_k1_20261028/audit_01',metadata=metadata_for(5),
            ground_truth=ROOT/'Ground_Truth/darpa/darpa/E5-THEIA/node_THEIA_1_Firefox_Drakon_APT_BinFmt_Elevate_Inject.csv',
            output=out/'e5_recovery_recheck',chunk_size=20000)
        r.main(args)
        ids,scores,counts=state['test']; val_ids,val_scores,val_counts=state['val']
        seal=args.output/'frozen_predictions.json'; thresholds=read(OLD5/'frozen_thresholds.json')['thresholds']
        predictions={k:(scores[k]>thresholds[str(k)]).astype(np.int8) for k in r.KS}
        for a in predictions.values(): a.flags.writeable=False
        truth,gt,uuids,mapping=r.load_labels_after_seal(seal,ids,scores,predictions,args.metadata,args.ground_truth)
        evidence.extend(mapping['sources'])
        result=analyze('E5',ids,scores[5],truth,state['bank'],thresholds['5'],val_ids,val_scores[5],val_counts,
                       gt,uuids,thresholds['1'],scores[1],out,records)
        assert state['frozen_exact']
        return result
    finally:
        r.RawSource.chunks=original_chunks;r.TopK=original_top;r.make_scores=original_make;r.freeze_predictions=original_freeze


def main(out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert shutil.disk_usage(out.parent).free>1024**3
    # 允许nohup启动器预先放置PID/日志；所有分析文件使用排他创建。
    out.mkdir(exist_ok=True)
    assert not (out/'execution_before.json').exists(), '禁止覆盖既有诊断运行'
    started=time.monotonic()
    evidence=[]; manifest=read(ACCEPT/'artifact_manifest.json')['inputs']
    expected={x['path']:x for x in manifest}
    # 读取所有正式文件stat快照；不读取大型模型或生成中间预测CSV。
    snapshots={}
    for e in [3,5]:
        run=Path(read(ACCEPT/f'e{e}_acceptance.json')['input_run'])
        for p in run.rglob('*'):
            if p.is_file(): snapshots[str(p)]=e3.fingerprint(p)
    prior_paths=[p for d in [ACCEPT,ROOT/'analysis/e3_seed0_exploratory_20261028',ROOT/'analysis/e5_seed0_cpu_recovery_20261028'] for p in d.rglob('*') if p.is_file()]
    prior_hashes={str(p):e3.sha(p) for p in prior_paths if p.stat().st_size<10*1024**2}
    before=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),root_free=shutil.disk_usage('/').free,
        data_free=shutil.disk_usage('/mnt/datadisk').free,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        git_diff_sha256=__import__('hashlib').sha256(subprocess.check_output(['git','diff'])).hexdigest(),
        source_sha256=e3.sha(__file__),formal_stats=snapshots,prior_analysis_hashes=prior_hashes)
    e3.write_json(out/'execution_before.json',before)
    e3.write_json(out/'diagnostic_protocol.json',dict(
        rank_rule='score降序，node_id升序；rank为1起始；top_percent=100*rank/N，越小越异常；保留完整并列区间',
        fn_groups='先score/threshold>=0.9且<1为接近阈值（score==threshold亦归近阈值）；再其余rank<=ceil(1%*N)；其余为较低分，互斥覆盖',
        budget=[100,500,1000,5000,10000,'ceil(1%*N)','ceil(5%*N)','ceil(10%*N)'],
        frequency_bins=BINS,descriptive_percentiles=[.9,.95,.99,.999],
        extreme_definition='Top1占Top5实际有效贡献分数和>=50%；少于5贡献按实际个数均值；自环保留双端',
        formal_threshold_frozen=True,new_threshold_selection=False,test_labels_for_diagnosis_only=True,
        limitations='节点攻击标签不能确认单条事件恶意；无多尺度gate/embedding，不能检验融合表示；E5原程序level另有全量验收，新增top5未保留level归因'))
    records={k:[] for k in ['attacks','rank_summary','topn','fp','relations','percentiles']}
    results={}
    for name,fn in [('E3',e3_run),('E5',e5_run)]: results[name]=fn(out,records,evidence)
    for key,name in [('attacks','attack_node_details.csv'),('rank_summary','attack_rank_summary.csv'),('topn','topn_metrics.csv'),
                     ('fp','false_positive_analysis.csv'),('relations','relation_contribution_summary.csv'),('percentiles','fixed_percentile_diagnostics.csv')]:
        table(out/name,records[key])
    # 当前读入字节哈希与验收清单核对；恢复程序另逐一检查全部E5原始窗口。
    comparisons=[]
    for item in evidence:
        p=item['path']; exp=expected.get(p)
        if exp:
            assert exp['sha256']==item['sha256'],p
        comparisons.append(dict(path=p,sha256=item['sha256'],acceptance_manifest_match=True if exp else None))
    for p in [ACCEPT/'artifact_manifest.json',ACCEPT/'e3_acceptance.json',ACCEPT/'e5_acceptance.json',
              Path(read(ACCEPT/'e3_acceptance.json')['input_run'])/'node_scores/node_predictions.csv']:
        digest=e3.sha(p)
        if str(p) in expected: assert digest==expected[str(p)]['sha256']
        comparisons.append(dict(path=str(p),sha256=digest))
    e3.write_json(out/'input_hash_verification.json',dict(passed=True,sources=comparisons,
        e5_raw_hashes='e5_recovery_recheck：复用RawSource，验证全部val/test窗口完整SHA256和完整校准表SHA256',
        e5_full_score_seal_exact=True))
    assert all(e3.fingerprint(p)==v for p,v in snapshots.items())
    assert all(e3.sha(p)==v for p,v in prior_hashes.items())
    assert before['git_diff_sha256']==__import__('hashlib').sha256(subprocess.check_output(['git','diff'])).hexdigest()
    e3.write_json(out/'analysis_verification.json',dict(status='PASS',formal_inputs_stats_unchanged=True,
        prior_analysis_small_files_hashes_unchanged=True,existing_tracked_diff_unchanged=True,
        resources=dict(elapsed_seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            root_free_after=shutil.disk_usage('/').free,data_free_after=shutil.disk_usage('/mnt/datadisk').free),
        baseline_and_rank_checks={k:v['verification'] for k,v in results.items()},
        artifacts_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())))
    print('全量诊断通过，输出',out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
