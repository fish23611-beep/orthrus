"""Canonical CPU streaming evaluation for a sealed E3/E5 seed0 candidate.

Reuses the verified LOO hierarchical calibration and exact Top5 reducer;
never writes calibrated-event CSVs. Labels are first loaded after score seal.
"""
from __future__ import annotations
import argparse
import csv
import gc
import json
import os
from pathlib import Path
import resource
import sys
import time
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import compare_e3_node_balanced_treatment as paired
import restore_e5_frozen_postprocessing as recovery
import diagnose_e3_e5_low_recall as diag
from mstc.metrics import compute_classification_metrics
from mstc.training_loss import NodeContributionWeights
from detection.checkpoint_policy import validate_selection_manifest
from joint_optimization_pipeline import read,save,sha,now


def require_prediction_seal(path,ids,scores,predictions):
    seal=read(path)
    assert seal['test_labels_read'] is False
    for key,array in [('ids',ids),('scores',scores),('predictions',predictions)]:
        assert not array.flags.writeable and recovery.digest_arrays(array)==seal[key+'_sha256']


def load_labels_after_seal(e,path,ids,scores,predictions):
    require_prediction_seal(path,ids,scores,predictions)
    metadata=diag.metadata_for(e);evidence=[];loaded={}
    for key in ['ground_truth_nodes','uuid_to_node_id','node_id_to_uuid','attack_to_nodes']:
        loaded[key],ev=recovery.load_pickle(metadata/(key+'.pkl'));evidence.append(ev)
    names=['node_Browser_Extension_Drakon_Dropper.csv','node_Firefox_Backdoor_Drakon_In_Memory.csv'] if e==3 else ['node_THEIA_1_Firefox_Drakon_APT_BinFmt_Elevate_Inject.csv']
    rows=[]
    for name in names:
        gt=ROOT/f'Ground_Truth/darpa/darpa/E{e}-THEIA'/name
        with gt.open(newline='') as f:rows.extend(csv.reader(f))
        evidence.append(dict(path=str(gt),sha256=sha(gt)))
    truth,mapped,uuids=recovery.map_ground_truth(rows,loaded['uuid_to_node_id'],loaded['node_id_to_uuid'],loaded['ground_truth_nodes'],ids)
    assert set().union(*(set(map(int,v)) for v in loaded['attack_to_nodes'].values()))==mapped
    return truth,uuids,loaded['attack_to_nodes'],evidence,sorted(mapped-set(ids.tolist()))


def finite_metrics(metrics):
    return {k:None if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v for k,v in metrics.items()}


def group_metrics(truth,pred,scores,mask):
    if not mask.any():return dict(nodes=0)
    # AP/AUROC are subgroup metrics only when both classes exist.
    m=compute_classification_metrics(truth[mask],pred[mask],scores[mask])
    if len(np.unique(truth[mask]))<2:m['auprc']=m['auroc']=None
    return dict(nodes=int(mask.sum()),**finite_metrics(m))


def main(out,e):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    started=time.monotonic();plan=read(out/'prepared_05/plan.json');d=plan['datasets'][str(e)]
    for p,h in plan['source_sha256'].items():assert sha(ROOT/p)==h,p
    assert (out/f'e{e}_training.exitcode').read_text().strip()=='0'
    assert (out/f'e{e}_checkpoint_validation_and_test.exitcode').read_text().strip()=='0'
    run=Path(d['run_dir']);dest=out/f'e{e}_evaluation';dest.mkdir(exist_ok=False)
    import yaml
    from yacs.config import CfgNode
    resolved=CfgNode(yaml.safe_load((run/'config_resolved.yml').read_text()))
    selection=read(run/'checkpoint_selection.json');selected=validate_selection_manifest(selection,resolved)
    assert {p.name for p in (run/'edge_scores/test').iterdir()}=={selected}
    assert resolved.detection.gnn_training.train_loss_weighting=='node_contribution_shrunk'
    assert resolved.calibration.method=='hierarchical_relation'
    assert resolved.calibration.min_triplet_samples==100 and resolved.calibration.min_type_pair_samples==200
    assert resolved.calibration.epsilon==1e-12
    assert resolved.node_aggregation.topk==5 and resolved.node_aggregation.include_dst
    assert resolved.node_threshold.quantile==.999
    assert sha(d['manifest'])==d['manifest_sha256']
    m=read(d['manifest']);capacity=m['max_node'];evidence=[]
    paired.SHARED=Path(f'/mnt/datadisk/yby/mstc/artifacts/u4090_e{e}')
    refs,val=paired.fit_validation(run/'edge_scores/val'/selected,evidence)
    assert abs(val['raw_mean']-selection['per_epoch_validation_objective'][selected])<1e-6
    # Capacity contains no labels. No-event IDs are filtered after full streaming.
    allids=np.arange(capacity,dtype=np.int64);bank=paired.a.TopK(allids,maximum=5)
    types=np.full(capacity,-1,dtype=np.int16);relations=np.zeros(capacity,dtype=np.uint64)
    identity=paired.IdentityDigest()
    for frame in paired.raw_chunks(run/'edge_scores/test'/selected,'test',evidence,identity):
        _,score,_=recovery.query_full(refs,frame)
        rel=frame.edge_type_index.to_numpy();assert rel.max()<64
        bits=np.left_shift(np.uint64(1),rel.astype(np.uint64))
        for endpoint,typecol in [('srcnode','src_type'),('dstnode','dst_type')]:
            ids=frame[endpoint].to_numpy();kind=frame[typecol].to_numpy()
            assert np.all((ids>=0)&(ids<capacity))
            old=types[ids];assert np.all((old==-1)|(old==kind)),'节点类型不一致'
            # Detect conflicting first appearances within this same chunk too.
            pairs=np.unique(np.stack([ids,kind],axis=1),axis=0)
            assert len(np.unique(pairs[:,0]))==len(pairs)
            types[ids]=kind;np.bitwise_or.at(relations,ids,bits);bank.update(ids,score)
    mask=bank.counts>0;ids=allids[mask];scores=bank.scores(5)[mask];counts=bank.counts[mask]
    assert int(counts.sum())==2*identity.rows and np.isfinite(scores).all()
    types=types[mask];relations=relations[mask];pred=scores>val['threshold']
    for array in [ids,scores,pred]:array.flags.writeable=False
    seal=dest/'predictions_frozen.json'
    save(seal,dict(test_labels_read=False,selected_checkpoint=selected,threshold=val['threshold'],
        ids_sha256=recovery.digest_arrays(ids),scores_sha256=recovery.digest_arrays(scores),
        predictions_sha256=recovery.digest_arrays(pred),node_count=len(ids),test_identity=identity.finish()))
    # Keep compact scores and counts; no duplicate event or calibrated tables.
    with (dest/'node_scores.npz').open('xb') as f:np.savez_compressed(f,node_id=ids,score=scores,prediction=pred,test_contributions=counts,node_type=types,relation_mask=relations)
    del refs,bank,allids;gc.collect()
    truth,uuids,attacks,label_evidence,absent=load_labels_after_seal(e,seal,ids,scores,pred);evidence.extend(label_evidence)
    baseline=read(ROOT/f'analysis/seed0_final_acceptance_20261028/e{e}_acceptance.json')
    assert len(ids)==baseline['node_count'] and int(truth.sum())==baseline['attack_node_count']
    if e==5:
        frozen=read(ROOT/'analysis/e5_seed0_cpu_recovery_20261028/run_01/frozen_predictions.json')
        assert recovery.digest_arrays(ids)==frozen['node_ids_sha256']
    else:
        node=pd.read_csv(Path(baseline['input_run'])/'node_scores/node_predictions.csv',usecols=['node_id','y_true']).sort_values('node_id')
        assert np.array_equal(ids,node.node_id.to_numpy()) and np.array_equal(truth,node.y_true.to_numpy())
    metrics=finite_metrics(compute_classification_metrics(truth,pred,scores));_,ranks,best,worst=diag.deterministic_ranks(ids,scores)
    topn=diag.budget_metrics(ids,scores,truth,[1000,5000])
    attack_rows=[dict(node_id=int(ids[i]),uuid=uuids[int(ids[i])],score=float(scores[i]),rank=int(ranks[i]),
        tie_best=int(best[i]),tie_worst=int(worst[i]),predicted=bool(pred[i]),score_minus_threshold=float(scores[i]-val['threshold']),
        test_contributions=int(counts[i]),node_type=int(types[i])) for i in np.flatnonzero(truth)]
    frequency=[]
    from audit_node_training_frequency import windows
    old=read(ROOT/f'analysis/node_balanced_training_20261028/e{e}_train_frequency.json')
    weights=NodeContributionWeights.fit(windows([Path(x['path']) for x in old['source_tensor_hashes']]),num_nodes=capacity)
    ctrain=weights.counts[ids]
    for basis,c in [('train',ctrain),('test',counts)]:
        for low,high in ([(0,0)] if basis=='train' else [])+diag.BINS:
            mask=(c>=low)&(c<=high)
            frequency.append(dict(basis=basis,minimum=low,maximum=high,**group_metrics(truth,pred,scores,mask)))
    groups=[]
    for kind in np.unique(types):groups.append(dict(basis='节点类型',key=int(kind),**group_metrics(truth,pred,scores,types==kind)))
    for rel in range(64):
        mask=(relations & (np.uint64(1)<<np.uint64(rel)))!=0
        if mask.any():groups.append(dict(basis='关联关系类型（节点可重复归组）',key=rel,**group_metrics(truth,pred,scores,mask)))
    for name,nodes in attacks.items():
        mask=np.isin(ids,list(nodes));groups.append(dict(basis='攻击类别',key=str(name),attack_nodes=int((mask&truth.astype(bool)).sum()),tp=int((mask&pred&truth.astype(bool)).sum())))
    for filename,rows in [('attack_nodes.csv',attack_rows),('frequency_metrics.csv',frequency),('type_relation_attack_metrics.csv',groups)]:diag.table(dest/filename,rows)
    runtime=read(run/'runtime.json')
    save(dest/'result.json',dict(status='PASS',dataset=e,created_at=now(),metrics=metrics,topn=topn,
        attack_rank_distribution=diag.dist(ranks[truth==1]),baseline_metrics=baseline['metrics'],threshold=val['threshold'],
        frequency=frequency,type_relation_attack_groups=groups,absent_gt_nodes=absent,
        selected_checkpoint=selected,checkpoint_sha256=sha(run/'checkpoints'/selected/'checkpoint.pt'),
        validation=val,selection=selection,training_runtime=runtime['training'],testing_runtime=runtime['testing'],
        cpu_seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        run_storage_bytes=sum(p.stat().st_size for p in run.rglob('*') if p.is_file()),sources=evidence,
        note='单seed探索性实验；历史测试曾参与诊断；关系组按端点关联定义并非独立事件恶意分类',
        calibrated_csv_copies=0))
    print('CPU canonical评价PASS',e,metrics,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--dataset',type=int,choices=[3,5],required=True)
    a=p.parse_args();main(a.analysis.resolve(),a.dataset)
