"""Exact frozen E3 CPU evaluation/comparison; never creates calibrated CSVs."""
from __future__ import annotations
import argparse
import gc
import hashlib
import io
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
import analyze_e3_postprocessing as a
import restore_e5_frozen_postprocessing as recovery
import diagnose_e3_e5_low_recall as diagnostic
from mstc.metrics import compute_classification_metrics
from mstc.thresholding import compute_node_threshold
from mstc.training_loss import NodeContributionWeights
from audit_node_training_frequency import windows
from e3_node_balanced_pipeline import CONTROL,SHARED,read,save,sha,now

IDENTITY=['event_index','time','srcnode','dstnode','src_type','dst_type','edge_type_index']
RAW_FIELDS=IDENTITY+['score_raw']


class IdentityDigest:
    """One digest per column: independent of streaming chunk boundaries."""
    def __init__(self):self.hashes={k:hashlib.sha256() for k in IDENTITY};self.rows=0
    def add(self,frame):
        for k,h in self.hashes.items():h.update(frame[k].to_numpy(dtype=np.int64).tobytes())
        self.rows+=len(frame)
    def finish(self):return dict(rows=self.rows,columns={k:h.hexdigest() for k,h in self.hashes.items()})


def split_specs(split):
    path=next(SHARED.glob('edge_featurization/THEIA_E3/embed_edges/*/edge_embeds/_compact_index_v2/manifest__*.json'))
    manifest=read(path)
    return [w for w in manifest['window_specs'] if w['split_name']==split]


def csv_name(spec):return Path(spec['relative_path']).name.replace('.TemporalData.simple','.csv')


def raw_chunks(directory,split,evidence,identity,chunk_size=100000):
    directory=Path(directory);specs=split_specs(split)
    paths=sorted(directory.glob('*.csv'))
    # Raw CSV names use the first/last actual event, while cache filenames use
    # graph-window boundaries. Pair chronological windows, then require every
    # global event ID and exact manifest row count; never infer coverage by name.
    assert len(paths)==len(specs),'窗口覆盖与正式manifest不一致'
    previous_time=None
    for i,(spec,path) in enumerate(zip(specs,paths)):
        raw=a.HashReader(path);rows=0;before=a.fingerprint(path)
        with io.BufferedReader(raw) as handle:
            for f in pd.read_csv(handle,usecols=RAW_FIELDS,chunksize=chunk_size,float_precision='round_trip'):
                if f.isna().any().any() or not np.isfinite(f.to_numpy()).all():raise ValueError('非法原始分数或字段')
                for field in IDENTITY:
                    if f[field].dtype.kind not in 'iu' or (f[field]<0).any():raise ValueError('非法事件ID或类型')
                expected=np.arange(spec['global_offset']+rows,spec['global_offset']+rows+len(f))
                assert np.array_equal(expected,f.event_index.to_numpy()),'事件遗漏、重复或顺序错误'
                times=f.time.to_numpy();assert np.all(times[1:]>=times[:-1]) and (previous_time is None or times[0]>=previous_time)
                previous_time=int(times[-1]);rows+=len(f);identity.add(f)
                yield f
        assert rows==spec['num_events'] and before==a.fingerprint(path)
        evidence.append(dict(path=str(path),sha256=raw.digest.hexdigest(),rows=rows,unchanged=True,
            source_window=spec['relative_path'],global_offset=spec['global_offset']))
        if (i+1)%40==0:print(split,'CPU窗口',i+1,'/',len(specs),flush=True)


def fit_validation(directory,evidence):
    identity=IdentityDigest();parts=list(raw_chunks(directory,'val',evidence,identity))
    val=pd.concat(parts,ignore_index=True);refs=a.References(val,val.score_raw.to_numpy());del parts
    ids=np.union1d(val.srcnode,val.dstnode);bank=a.TopK(ids,maximum=5)
    for start in range(0,len(val),100000):
        f=val.iloc[start:start+100000];_,score,_=recovery.query_full(refs,f,loo=True)
        bank.update(f.srcnode.to_numpy(),score);bank.update(f.dstnode.to_numpy(),score)
    scores=bank.scores(5);threshold=compute_node_threshold(scores,quantile=.999)
    summary=dict(threshold=threshold,nodes=len(ids),events=len(val),raw_mean=float(val.score_raw.mean()),
        identity=identity.finish(),node_scores_sha256=recovery.digest_arrays(scores),
        score_quantiles=diagnostic.dist(scores),q=.999,k=5,include_dst=True,loo=True)
    del val,bank;gc.collect()
    return refs,summary


def aggregate_test(directory,ids,refs,evidence):
    identity=IdentityDigest();bank=a.TopK(ids,maximum=5)
    for f in raw_chunks(directory,'test',evidence,identity):
        _,score,_=recovery.query_full(refs,f)
        bank.update(f.srcnode.to_numpy(),score);bank.update(f.dstnode.to_numpy(),score)
    assert np.all(bank.counts>0) and int(bank.counts.sum())==2*identity.rows
    return bank.scores(5),bank.counts.copy(),identity.finish()


def control_identity(evidence):
    path=CONTROL/'node_scores/event_predictions.csv';raw=a.HashReader(path);identity=IdentityDigest()
    with io.BufferedReader(raw) as handle:
        for frame in pd.read_csv(handle,usecols=IDENTITY,chunksize=100000,float_precision='round_trip'):identity.add(frame)
    digest=raw.digest.hexdigest();expected={x['path']:x['sha256'] for x in read(ROOT/'analysis/seed0_final_acceptance_20261028/artifact_manifest.json')['inputs']}
    assert digest==expected[str(path)]
    evidence.append(dict(path=str(path),sha256=digest,rows=identity.rows))
    return identity.finish()


def pair_results(ids,truth,control_scores,treatment_scores,control_threshold,treatment_threshold,test_counts,train_counts):
    predictions={'control':control_scores>control_threshold,'treatment':treatment_scores>treatment_threshold}
    metrics={};ranks={};rank_summaries=[];topn=[]
    for name,scores in [('control',control_scores),('treatment',treatment_scores)]:
        metrics[name]=compute_classification_metrics(truth,predictions[name],scores)
        metrics[name]['threshold']=control_threshold if name=='control' else treatment_threshold
        _,r,lo,hi=diagnostic.deterministic_ranks(ids,scores);ranks[name]=(r,lo,hi)
        budgets=diagnostic.budget_metrics(ids,scores,truth,[1000,5000])
        for row in budgets:row.update(scheme=name)
        topn.extend(budgets)
        masks={'TP':(truth==1)&predictions[name],'FP':(truth==0)&predictions[name],
               'FN':(truth==1)&~predictions[name],'TN':(truth==0)&~predictions[name]}
        assert np.all(sum(m.astype(np.int8) for m in masks.values())==1)
        for label,mask in [('全部攻击',truth==1),('TP',masks['TP']),('FN',masks['FN'])]:
            rank_summaries.append(dict(scheme=name,group=label,**diagnostic.dist(r[mask])))
    frequency=[]
    for basis,counts in [('test',test_counts),('train',train_counts)]:
        for low,high in ([(0,0)] if basis=='train' else [])+diagnostic.BINS:
            group=(counts>=low)&(counts<=high);benign=group&(truth==0);attack=group&(truth==1)
            row=dict(frequency_basis=basis,minimum=low,maximum=high,benign_nodes=int(benign.sum()),attack_nodes=int(attack.sum()))
            for name in predictions:
                fp=int((benign&predictions[name]).sum())
                row[name+'_fp']=fp;row[name+'_fpr']=fp/int(benign.sum()) if benign.any() else None
                row[name+'_tp']=int((attack&predictions[name]).sum())
            frequency.append(row)
    def overlap(mask):
        c=mask&predictions['control'];t=mask&predictions['treatment'];groups={'both':ids[c&t],'control_only':ids[c&~t],
            'treatment_only':ids[t&~c],'neither':ids[mask&~c&~t]}
        return {k:dict(count=len(v),ids_sha256=recovery.digest_arrays(v),node_ids=v.tolist() if len(v)<=1000 else v[:30].tolist(),ids_complete=len(v)<=1000) for k,v in groups.items()}
    attack_details=[]
    for pos in np.flatnonzero(truth==1):
        row=dict(node_id=int(ids[pos]),test_contributions=int(test_counts[pos]),train_contributions=int(train_counts[pos]),
            rank_improvement=int(ranks['control'][0][pos]-ranks['treatment'][0][pos]))
        for name,scores,threshold in [('control',control_scores,control_threshold),('treatment',treatment_scores,treatment_threshold)]:
            row.update({name+'_score':float(scores[pos]),name+'_rank':int(ranks[name][0][pos]),
                name+'_rank_top_percent':float(100*ranks[name][0][pos]/len(ids)),
                name+'_tie_best':int(ranks[name][1][pos]),name+'_tie_worst':int(ranks[name][2][pos]),
                name+'_status':'TP' if predictions[name][pos] else 'FN',name+'_score_minus_threshold':float(scores[pos]-threshold)})
        attack_details.append(row)
    low_frequency={}
    for name,mask in [('test_le5',(truth==1)&(test_counts<=5)),('train_unseen',(truth==1)&(train_counts==0)),
                      ('train_1_to_5',(truth==1)&(train_counts>=1)&(train_counts<=5))]:
        delta=ranks['control'][0][mask]-ranks['treatment'][0][mask]
        low_frequency[name]=dict(nodes=int(mask.sum()),rank_change=diagnostic.dist(delta),
            improved=int((delta>0).sum()),worsened=int((delta<0).sum()),same=int((delta==0).sum()))
    c,t=metrics['control'],metrics['treatment']
    topc={r['budget']:r for r in topn if r['scheme']=='control'};topt={r['budget']:r for r in topn if r['scheme']=='treatment'}
    head=all(topt[n]['tp']>=topc[n]['tp'] for n in [1000,5000]) and any(topt[n]['tp']>topc[n]['tp'] for n in [1000,5000])
    flags=dict(E3_TREATMENT_TRAINING='PASS',E3_TREATMENT_EVALUATION='PASS',BASELINE_COMPARABILITY='PASS',
        HEAD_RANKING_IMPROVED='YES' if head else 'NO',TP_IMPROVED='YES' if t['tp']>c['tp'] else 'NO',
        FP_REDUCED='YES' if t['fp']<c['fp'] else 'NO',AP_IMPROVED='YES' if t['auprc']>c['auprc'] else 'NO',
        MCC_IMPROVED='YES' if t['mcc']>c['mcc'] else 'NO')
    ready=head and t['auprc']>c['auprc'] and t['mcc']>=c['mcc'] and t['tp']>=c['tp'] and t['fp']<=c['fp']
    flags['READY_FOR_CROSS_DATASET_TEST']='YES' if ready else 'NO'
    return dict(metrics=metrics,topn=topn,ranks=rank_summaries,frequency=frequency,attack_details=attack_details,
        attack_alert_overlap=overlap(truth==1),benign_alert_overlap=overlap(truth==0),low_frequency_attacks=low_frequency,flags=flags,
        criteria='头部改善：两个预定预算均不退步且至少一个增加；跨数据集自动保守建议还要求AP增加、MCC/TP不降、FP不增。只作研究建议，不自动启动E5。',
        interpretation_limit='使用已见E3 seed0测试集的探索性干预；频率组可确认，但排名变化不能单独证明低频学习的内部因果机制')


def control_only(out):
    started=time.monotonic();evidence=[]
    node=pd.read_csv(CONTROL/'node_scores/node_predictions.csv',float_precision='round_trip').sort_values('node_id')
    refs,val=fit_validation(CONTROL/'edge_scores/val/model_epoch_4',evidence)
    score,counts,identity=aggregate_test(CONTROL/'edge_scores/test/model_epoch_4',node.node_id.to_numpy(),refs,evidence)
    assert identity==control_identity(evidence)
    threshold=read(ROOT/'analysis/seed0_final_acceptance_20261028/e3_acceptance.json')['threshold']
    assert val['threshold']==threshold
    assert np.array_equal(score,node.score.to_numpy()),'Control全量分数不是逐位相同'
    assert np.array_equal(score>threshold,node.y_hat.to_numpy().astype(bool))
    metrics=compute_classification_metrics(node.y_true,score>threshold,score)
    expected=read(ROOT/'analysis/seed0_final_acceptance_20261028/e3_acceptance.json')['metrics']
    for k in expected:assert abs(metrics[k]-expected[k])<1e-12
    save(out/'cpu_evaluator_control_equivalence.json',dict(status='PASS',full_node_scores_bitwise_exact=True,
        threshold_exact=True,predictions_exact=True,metrics_exact=True,metrics=metrics,validation=val,
        node_count=len(node),attack_count=int(node.y_true.sum()),test_identity=identity,sources=evidence,
        seconds=time.monotonic()-started))
    print('Control CPU全量等价性 PASS',metrics,flush=True)


def main(out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    seal=read(out/'pipeline_code_seal.json')
    for p,h in seal['code_sha256'].items():assert sha(ROOT/p)==h,p
    started=time.monotonic();pre=read(out/'preflight.json');run=Path(pre['run_dir'])
    assert read(out/'cpu_evaluator_control_equivalence.json')['status']=='PASS'
    assert (out/'training.exitcode').read_text().strip()=='0'
    assert (out/'validation_selection_and_best_test.exitcode').read_text().strip()=='0'
    selection=read(run/'checkpoint_selection.json');selected=selection['selected_epoch']
    assert selected==min(selection['per_epoch_validation_objective'],key=selection['per_epoch_validation_objective'].get)
    assert {p.name for p in (run/'edge_scores/test').iterdir()}=={selected}
    ids=pd.read_csv(CONTROL/'node_scores/node_predictions.csv',usecols=['node_id']).node_id.sort_values().to_numpy()
    evidence=[];refs,val=fit_validation(run/'edge_scores/val'/selected,evidence)
    score,counts,identity=aggregate_test(run/'edge_scores/test'/selected,ids,refs,evidence)
    assert identity==read(out/'cpu_evaluator_control_equivalence.json')['test_identity']
    predictions=score>val['threshold']
    save(out/'frozen_treatment_predictions.json',dict(test_labels_used=False,selected_checkpoint=selected,
        threshold=val['threshold'],node_count=len(ids),ids_sha256=recovery.digest_arrays(ids),
        scores_sha256=recovery.digest_arrays(score),predictions_sha256=recovery.digest_arrays(predictions)))
    for array in [ids,score,predictions]:array.flags.writeable=False
    # Label access starts only after the complete Treatment score/prediction seal.
    node=pd.read_csv(CONTROL/'node_scores/node_predictions.csv',float_precision='round_trip').sort_values('node_id')
    assert np.array_equal(ids,node.node_id.to_numpy()) and len(ids)==699295
    truth=node.y_true.to_numpy();assert int(truth.sum())==118
    gt,uuids,label_evidence=diagnostic.mapping_e3(diagnostic.metadata_for(3),ids,truth)
    evidence.extend(label_evidence)
    frequency=read(ROOT/'analysis/node_balanced_training_20261028/e3_train_frequency.json')
    weights=NodeContributionWeights.fit(windows([Path(x['path']) for x in frequency['source_tensor_hashes']]),num_nodes=max(int(ids.max())+1,1258362))
    assert weights.num_events==4063357 and weights.num_nodes==410023
    train_counts=weights.counts[ids]
    control_threshold=read(ROOT/'analysis/seed0_final_acceptance_20261028/e3_acceptance.json')['threshold']
    result=pair_results(ids,truth,node.score.to_numpy(),score,control_threshold,val['threshold'],counts,train_counts)
    for row in result['attack_details']:row['uuid']=uuids[row['node_id']]
    for k,v in read(ROOT/'analysis/seed0_final_acceptance_20261028/e3_acceptance.json')['metrics'].items():assert abs(result['metrics']['control'][k]-v)<1e-12
    for row in result['topn']:
        if row['scheme']=='control':assert row['tp']=={1000:5,5000:12}[row['budget']]
    table_rows=[dict(scheme=k,**v) for k,v in result['metrics'].items()]
    diagnostic.table(out/'metrics_comparison.csv',table_rows)
    diagnostic.table(out/'topn_comparison.csv',result['topn'])
    diagnostic.table(out/'attack_node_comparison.csv',result['attack_details'])
    diagnostic.table(out/'frequency_fpr_comparison.csv',result['frequency'])
    save(out/'attack_rank_distributions.json',result['ranks'])
    del result['attack_details']
    protected=read(out/'protected_inputs.json')
    for p,expected in protected.items():
        s=Path(p).stat();assert s.st_size==expected['bytes'] and s.st_mtime_ns==expected['mtime_ns'],p
    for p,h in pre['code_sha256'].items():assert sha(ROOT/p)==h,p
    assert hashlib.sha256(__import__('subprocess').check_output(['git','diff'],cwd=ROOT)).hexdigest()==pre['git_diff_sha256']
    assert not list(run.rglob('*calibrated*.csv')),'Unexpected duplicate calibrated CSV'
    assert abs(val['raw_mean']-selection['per_epoch_validation_objective'][selected])<1e-6
    runtime=read(run/'runtime.json');train=runtime['training'];test=runtime['testing']
    result.update(status='PASS',finished_at=now(),selected_checkpoint=selected,selection=selection,validation=val,
        checkpoint_sha256=sha(run/'checkpoints'/selected/'checkpoint.pt'),sources=evidence,
        training_runtime=train,testing_runtime=test,storage=dict(own_run_bytes=sum(p.stat().st_size for p in run.rglob('*') if p.is_file()),
            calibrated_csv_copies=0),verification=dict(control_equivalence=True,all_test_event_identity_exact=True,
            node_universe_exact=True,uuid_mapping_exact=True,labels_after_prediction_freeze=True,
            original_formal_inputs_unchanged=True,production_code_unchanged=True),
        cpu_seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    save(out/'paired_comparison.json',result)
    print(json.dumps({'flags':result['flags'],'metrics':result['metrics'],'topn':result['topn']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--verify-control-only',action='store_true')
    args=p.parse_args()
    (control_only if args.verify_control_only else main)(args.analysis.resolve())
