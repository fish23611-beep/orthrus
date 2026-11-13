"""Single-treatment evaluator identity, coverage and paired-metric contracts."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import compare_e3_node_balanced_treatment as c


def frame(start,n):
    return pd.DataFrame(dict(event_index=np.arange(start,start+n),time=np.arange(start,start+n)*10,
        srcnode=np.arange(n)%2,dstnode=np.arange(n)%3,src_type=np.zeros(n,dtype=int),
        dst_type=np.ones(n,dtype=int),edge_type_index=np.ones(n,dtype=int),score_raw=np.arange(n,dtype=float)))


def test_identity_hash_is_chunk_independent_and_detects_endpoint_changes():
    f=frame(100,13);whole=c.IdentityDigest();whole.add(f)
    parts=c.IdentityDigest()
    for start in range(0,13,3):parts.add(f.iloc[start:start+3])
    assert parts.finish()==whole.finish()
    f.loc[5,'dstnode']=99;changed=c.IdentityDigest();changed.add(f)
    assert changed.finish()!=whole.finish()


def test_raw_stream_exact_window_coverage_and_event_order(tmp_path,monkeypatch):
    specs=[dict(relative_path='test/a.TemporalData.simple',global_offset=100,num_events=3),
           dict(relative_path='test/b.TemporalData.simple',global_offset=103,num_events=2)]
    frame(100,3).to_csv(tmp_path/'a.csv',index=False);frame(103,2).to_csv(tmp_path/'b.csv',index=False)
    monkeypatch.setattr(c,'split_specs',lambda split:specs)
    identity=c.IdentityDigest();evidence=[]
    result=list(c.raw_chunks(tmp_path,'test',evidence,identity,chunk_size=1))
    assert len(result)==5 and identity.rows==5 and len(evidence)==2
    f=frame(103,2);f.loc[1,'event_index']=103;f.to_csv(tmp_path/'b.csv',index=False)
    with pytest.raises(AssertionError,match='事件遗漏'):list(c.raw_chunks(tmp_path,'test',[],c.IdentityDigest(),chunk_size=2))
    frame(105,1).to_csv(tmp_path/'extra.csv',index=False)
    with pytest.raises(AssertionError,match='窗口覆盖'):list(c.raw_chunks(tmp_path,'test',[],c.IdentityDigest()))


def test_paired_heads_status_sets_and_frequency_partitions():
    ids=np.arange(6000);truth=np.zeros(6000,dtype=np.int8);truth[[3000,5500]]=1
    control=(6000-ids).astype(float);treatment=control.copy();treatment[[3000,5500]]=[6002,6001]
    counts=np.where(ids<10,1001,2);train=np.where(ids%2==0,0,3)
    result=c.pair_results(ids,truth,control,treatment,5990.,5990.,counts,train)
    assert result['flags']['HEAD_RANKING_IMPROVED']=='YES'
    assert result['flags']['TP_IMPROVED']=='YES' and result['flags']['AP_IMPROVED']=='YES'
    assert result['attack_alert_overlap']['treatment_only']['count']==2
    for basis in ['train','test']:
        groups=[row for row in result['frequency'] if row['frequency_basis']==basis]
        assert sum(x['benign_nodes'] for x in groups)==5998
        assert sum(x['attack_nodes'] for x in groups)==2
    assert [x['tp'] for x in result['topn'] if x['scheme']=='control']==[0,1]
    assert [x['tp'] for x in result['topn'] if x['scheme']=='treatment']==[2,2]
    identical=c.pair_results(ids,truth,control,control,5990.,5990.,counts,train)
    assert identical['flags']['HEAD_RANKING_IMPROVED']=='NO'
    assert identical['flags']['TP_IMPROVED']=='NO' and identical['flags']['FP_REDUCED']=='NO'
