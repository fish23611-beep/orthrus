"""固定原joint score诊断对照，隔离校准影响；不拟合新方案或选择参数。"""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'src')]
import io
import gc
import time
import resource
import numpy as np
import pandas as pd
import diagnose_e3_e5_low_recall as d


def main(out):
    started=time.monotonic();rows=[];details=[];summaries={};evidence=[]
    d.e3.write_json(out/'raw_control_protocol.json',dict(
        objective='仅比较原训练joint score与其正式校准分数，诊断H1/C混淆；保持K=5和双端贡献，不搜索参数',
        saturation='原始分数严格大于所选验证参考最大值：p=1/(n+1)，校准分数达到log(n+1)上限',
        no_new_formal_threshold=True,labels_diagnostic_only=True))
    for e in [3,5]:
        name=f'E{e}';run=Path(d.read(d.ACCEPT/f'e{e}_acceptance.json')['input_run']);epoch={3:4,5:6}[e]
        valpath=run/f'evaluation_results/calibration/model_epoch_{epoch}/validation_calibrated.csv'
        fields=['src_type','dst_type','edge_type_index','score_raw']
        val=pd.read_csv(valpath,usecols=fields,float_precision='round_trip')
        refs=d.e3.References(val,val.score_raw.to_numpy());del val;gc.collect()
        gt,ev=d.r.load_pickle(d.metadata_for(e)/'ground_truth_nodes.pkl'); gt=np.array(sorted(gt),dtype=np.int64)
        evidence.append(ev);evidence.append(dict(path=str(valpath),sha256=d.e3.sha(valpath)))
        if e==3:
            nd=pd.read_csv(run/'node_scores/node_predictions.csv',usecols=['node_id']).sort_values('node_id')
            ids=nd.node_id.to_numpy();paths=[dict(path=str(run/'node_scores/event_predictions.csv'))]
        else:
            audit=d.read(d.ROOT/'analysis/e5_seed0_frozen_k1_20261028/audit_01/test_integrity.json')
            paths=audit['windows'];ids=np.arange(2286056,dtype=np.int64)
        bank=d.e3.TopK(ids,maximum=5)
        min_n=np.full(len(gt),np.iinfo(np.int64).max,dtype=np.int64);max_n=np.zeros(len(gt),dtype=np.int64)
        max_cal=np.zeros(len(gt));saturated=np.zeros(len(gt),dtype=np.int64);contrib=np.zeros(len(gt),dtype=np.int64)
        pair_stats={};loss_stats={k:dict(events=0,type_sum=0.,time_sum=0.,raw_sum=0.) for k in ['attack_incident','other']}
        for win_index,entry in enumerate(paths,1):
            path=Path(entry['path']);raw=d.e3.HashReader(path);count=0
            cols=['srcnode','dstnode','src_type','dst_type','edge_type_index','score_raw','loss_type','loss_time']
            with io.BufferedReader(raw) as handle:
                for f in pd.read_csv(handle,usecols=cols,chunksize=100000,float_precision='round_trip'):
                    score=f.score_raw.to_numpy();bank.update(f.srcnode.to_numpy(),score);bank.update(f.dstnode.to_numpy(),score);count+=len(f)
                    incident=np.isin(f.srcnode,gt)|np.isin(f.dstnode,gt)
                    for label,mask in [('attack_incident',incident),('other',~incident)]:
                        loss_stats[label]['events']+=int(mask.sum())
                        for field,key in [('loss_type','type_sum'),('loss_time','time_sum'),('score_raw','raw_sum')]: loss_stats[label][key]+=float(f[field].to_numpy()[mask].sum())
                    small=f[incident].reset_index(drop=True)
                    for key,idx in small.groupby(['src_type','edge_type_index','dst_type']).indices.items():
                        ref=refs.triplets.get(key);level='triplet'
                        if ref is None or len(ref)<100:
                            ref=refs.pairs.get((key[0],key[2]));level='type_pair'
                            if ref is None or len(ref)<200:ref=refs.global_values;level='global'
                        n=len(ref);part=small.iloc[idx];values=part.score_raw.to_numpy()
                        ge=n-np.searchsorted(ref,values,side='left');cal=-np.log((1.+ge)/(n+1.));sat=values>ref[-1]
                        for side in ['srcnode','dstnode']:
                            ni=part[side].to_numpy();select=np.isin(ni,gt);p=np.searchsorted(gt,ni[select])
                            np.minimum.at(min_n,p,n);np.maximum.at(max_n,p,n);np.maximum.at(max_cal,p,cal[select])
                            np.add.at(contrib,p,1);np.add.at(saturated,p,sat[select].astype(np.int64))
                            for node in np.unique(ni[select]):
                                m=ni==node;k=(int(node),*map(int,key),level,n)
                                old=pair_stats.setdefault(k,[0,0,0.]);old[0]+=int(m.sum());old[1]+=int((m&sat).sum());old[2]=max(old[2],float(cal[m].max()))
            digest=raw.digest.hexdigest()
            if 'sha256'in entry: assert digest==entry['sha256']
            elif e==3: assert digest==d.read(d.ROOT/'analysis/e3_seed0_exploratory_20261028/run_02/summary.json')['sources']['test']['sha256']
            evidence.append(dict(path=str(path),sha256=digest,rows=count))
            if win_index%25==0:print(name,'原joint诊断窗口',win_index,'/',len(paths),flush=True)
        observed=bank.counts>0;node_ids=bank.ids[observed];scores=bank.scores(5)[observed]
        truth=np.isin(node_ids,gt).astype(np.int8)
        order,ranks,best,worst=d.deterministic_ranks(node_ids,scores)
        metrics=d.compute_classification_metrics(truth,np.zeros(len(truth),dtype=np.int8),scores)
        topn=d.budget_metrics(node_ids,scores,truth,[100,500,1000,5000,10000,int(np.ceil(len(node_ids)*.01)),int(np.ceil(len(node_ids)*.05)),int(np.ceil(len(node_ids)*.1))])
        for x in topn:x['dataset']=name;x['score_definition']='原joint score，无校准，Top5双端'
        rows.extend(topn)
        if e==3:
            old=pd.read_csv(d.ROOT/'analysis/e3_seed0_exploratory_20261028/run_02/candidate_metrics.csv')
            old=old[(old.variant=='raw_joint')&(old.direction=='both')&(old.k==5)&(old['quantile']==.999)].iloc[0]
            assert abs(old.auprc-metrics['auprc'])<1e-12 and abs(old.auroc-metrics['auroc'])<1e-12
        for j,nid in enumerate(gt):
            where=np.searchsorted(node_ids,nid);exists=where<len(node_ids) and node_ids[where]==nid
            details.append(dict(dataset=name,node_id=int(nid),in_test=exists,raw_joint_score=float(scores[where]) if exists else None,
                raw_joint_rank=int(ranks[where]) if exists else None,raw_joint_top_percent=float(ranks[where]/len(node_ids)*100) if exists else None,
                reference_n_min=int(min_n[j]) if exists else None,reference_n_max=int(max_n[j]) if exists else None,
                max_calibration_ceiling=float(np.log(max_n[j]+1)) if exists else None,
                max_actual_calibrated_event=float(max_cal[j]) if exists else None,
                saturated_contributions=int(saturated[j]),all_contributions=int(contrib[j])))
        threshold=d.read(d.ACCEPT/f'e{e}_acceptance.json').get('threshold') or d.read(d.ACCEPT/f'e{e}_acceptance.json')['metrics']['threshold']
        summaries[name]=dict(nodes=len(node_ids),attacks=int(truth.sum()),raw_joint_ap=metrics['auprc'],raw_joint_auroc=metrics['auroc'],
            raw_attack_rank=d.dist(ranks[truth==1]),raw_attack_top_percent=d.dist(ranks[truth==1]/len(node_ids)*100),
            attack_nodes_any_saturated=int((saturated>0).sum()),
            attack_nodes_ceiling_below_node_threshold=int(((contrib>0)&(np.log(max_n+1)<=threshold)).sum()),
            loss_sums=loss_stats,hashes_verified=True)
        pair_rows=[dict(dataset=name,node_id=k[0],src_type=k[1],edge_type_index=k[2],dst_type=k[3],level=k[4],reference_n=k[5],
            calibration_ceiling=float(np.log(k[5]+1)),contributions=v[0],saturated=v[1],max_calibrated=v[2]) for k,v in pair_stats.items()]
        d.table(out/f'{name.lower()}_attack_reference_diagnostics.csv',pair_rows)
        print(name,'原joint诊断完成',summaries[name],flush=True)
        del bank,refs;gc.collect()
    d.table(out/'raw_joint_topn_control.csv',rows);d.table(out/'raw_joint_attack_control.csv',details)
    d.e3.write_json(out/'raw_joint_control.json',dict(results=summaries,sources=evidence,
        resource=dict(elapsed_seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)))


if __name__=='__main__':
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    main(Path(sys.argv[1]))
