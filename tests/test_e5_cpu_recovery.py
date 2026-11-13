"""冻结E5恢复协议的合成CPU测试。"""
import importlib.util
from pathlib import Path
import sys
import json
import pickle

import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('e5_recovery',ROOT/'scripts/restore_e5_frozen_postprocessing.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
from mstc.calibration import HierarchicalRelationCalibrator
from mstc.aggregation import NodeScoreAggregator


def events():
    rows=[]
    for src,edge,dst,n in [(0,0,1,105),(0,1,1,100),(2,0,0,3)]:
        rows.extend(dict(src_type=src,edge_type_index=edge,dst_type=dst,score_raw=float(i%7)) for i in range(n))
    return rows


def test_reconstructed_calibrator_references_and_complete_pickle(tmp_path):
    records=events();frame=pd.DataFrame(records)
    cal=HierarchicalRelationCalibrator().fit(records)
    refs=r.References(frame,frame.score_raw.to_numpy())
    assert r.validate_calibrator(cal,refs,frame)['fit_event_sequence_exact']
    p=tmp_path/'cal.pkl';p.write_bytes(pickle.dumps(cal))
    loaded,evidence=r.load_pickle(p)
    assert evidence['eof_complete']
    assert r.validate_calibrator(loaded,refs,frame)['passed']
    p.write_bytes(p.read_bytes()[:-3])
    with pytest.raises((EOFError,pickle.UnpicklingError)):
        r.load_pickle(p)


@pytest.mark.parametrize('chunk_size',[1,17,500])
def test_calibration_chunk_loo_ties_unknown_relation_and_numerical_match(chunk_size):
    records=events();frame=pd.DataFrame(records)
    cal=HierarchicalRelationCalibrator().fit(records)
    refs=r.References(frame,frame.score_raw.to_numpy())
    expected=cal.transform_val_with_loo(records)
    check=r.Equivalence()
    for start in range(0,len(frame),chunk_size):
        part=frame.iloc[start:start+chunk_size]
        p,s,level=r.query_full(refs,part,loo=True)
        exp=expected[start:start+chunk_size]
        check.add(p,s,level,np.array([x['empirical_p'] for x in exp]),np.array([x['score_calibrated'] for x in exp]),np.array([list(r.LEVELS).index(x['calibration_level']) for x in exp]))
    assert check.finish()['passed']
    assert check.max_p_error==check.max_score_error==0
    unknown=pd.DataFrame([dict(src_type=9,edge_type_index=9,dst_type=9,score_raw=100.)])
    p,s,l=r.query_full(refs,unknown)
    exp=cal.calibrate(unknown.to_dict('records'))[0]
    assert p[0]==exp['empirical_p'] and s[0]==exp['score_calibrated'] and l[0]==2


def test_bad_reference_and_nonfinite_saved_values_rejected():
    records=events();frame=pd.DataFrame(records)
    cal=HierarchicalRelationCalibrator().fit(records)
    refs=r.References(frame,frame.score_raw.to_numpy())
    cal.global_scores[0]=-1
    with pytest.raises(ValueError,match='global'):
        r.validate_calibrator(cal,refs,frame)
    with pytest.raises(ValueError,match='非有限'):
        r.Equivalence().add(np.array([1.]),np.array([0.]),np.array([0]),np.array([np.nan]),np.array([0.]),np.array([0]))


def test_k1_k5_self_loop_and_no_event_node_protocol():
    events=[dict(srcnode=1,dstnode=1,score_calibrated=9.),dict(srcnode=1,dstnode=2,score_calibrated=1.)]
    bank=r.TopK(np.arange(4),maximum=5)
    bank.update(np.array([1,1]),np.array([9.,1.]));bank.update(np.array([1,2]),np.array([9.,1.]))
    ids,scores,counts=r.make_scores(bank)
    assert ids.tolist()==[1,2] and counts.tolist()==[3,1]
    for k in r.KS:
        exp=NodeScoreAggregator().aggregate_events(events,topk=k)
        np.testing.assert_array_equal(scores[k],[exp[n] for n in ids])
    empty=r.TopK(np.arange(3),maximum=5)
    ids,scores,counts=r.make_scores(empty)
    assert len(ids)==len(counts)==0 and all(len(a)==0 for a in scores.values())
    with pytest.raises(ValueError,match='empty'):
        r.compute_node_threshold([])


def test_seal_before_label_loader_and_strict_threshold(tmp_path,monkeypatch):
    ids=np.array([1,2]);scores={5:np.array([1.,2.]),1:np.array([3.,4.])};thresholds={5:1.,1:3.}
    path=tmp_path/'seal.json'
    with pytest.raises(FileNotFoundError):
        r.load_labels_after_seal(path,ids,scores,{5:np.zeros(2),1:np.zeros(2)},tmp_path,tmp_path/'gt.csv')
    preds=r.freeze_predictions(path,ids,scores,thresholds)
    assert preds[5].tolist()==preds[1].tolist()==[0,1]
    assert not ids.flags.writeable and all(not a.flags.writeable for a in preds.values())
    r.require_seal(path,ids,scores,preds)
    class LabelRead(Exception):pass
    original=Path.open
    def observed_open(self,*a,**kw):
        if self.name=='gt.csv':
            assert path.is_file() and json.loads(path.read_text())['test_labels_read'] is False
            raise LabelRead
        return original(self,*a,**kw)
    monkeypatch.setattr(Path,'open',observed_open)
    with pytest.raises(LabelRead):
        r.load_labels_after_seal(path,ids,scores,preds,tmp_path,tmp_path/'gt.csv')


def test_uuid_mapping_uses_uuid_not_stale_gt_integer_and_checks_bijection():
    rows=[['attack-uuid','path','9999']]
    truth,mapped,_=r.map_ground_truth(rows,{'attack-uuid':7,'normal-uuid':2},{7:'attack-uuid',2:'normal-uuid'},{7},np.array([2,7]))
    assert truth.tolist()==[0,1] and mapped=={7}
    with pytest.raises(ValueError,match='缓存'):
        r.map_ground_truth(rows,{'attack-uuid':7},{7:'attack-uuid'},{9999},np.array([7]))
    with pytest.raises(ValueError,match='双射'):
        r.map_ground_truth(rows,{'attack-uuid':7,'other':7},{7:'attack-uuid'},{7},np.array([7]))
    with pytest.raises(ValueError,match='完整UUID'):
        r.map_ground_truth(rows,{'attack-uuid':7},{7:'attack-uuid'},{7},np.array([1,7]))


def test_empty_truth_and_attack_overlap_defined():
    truth,mapped,_=r.map_ground_truth([],{}, {},set(),np.array([],dtype=np.int64))
    assert len(truth)==0 and not mapped
    pred={5:np.array([],dtype=np.int8),1:np.array([],dtype=np.int8)}
    groups=r.overlap(np.array([],dtype=np.int64),truth,pred,1)
    assert all(v['count']==0 for v in groups.values())


def test_explicit_tolerances_reject_changes():
    check=r.Equivalence()
    check.add(np.array([.5]),np.array([1.]),np.array([0]),np.array([.5+1e-13]),np.array([1.+1e-10]),np.array([1]))
    assert not check.finish()['passed']
    assert check.p_mismatches==check.score_mismatches==check.level_mismatches==1


@pytest.mark.parametrize('event_ids',[[0,0,1],[1,0,2],[0,2,3]])
def test_raw_stream_duplicate_order_and_gap_across_chunk_boundaries(tmp_path,event_ids):
    import csv
    path=tmp_path/'raw.csv'
    rows=[]
    for i in event_ids:
        row=['0']*20
        row[0],row[1],row[4],row[5],row[8]=str(i),str(1000+i),'src','dst','EVENT_OPEN'
        rows.append(row)
    with path.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(r.RAW_HEADER);writer.writerows(rows)
    audit={'expected_events':3,'windows':[{'path':str(path),'sha256':r.sha(path),'expected_events':3,
            'first_id':event_ids[0],'last_id':event_ids[-1],'expected_global_offset':0}]}
    source=r.RawSource(audit,1)
    with pytest.raises(ValueError,match='不连续、重复或顺序'):
        list(source.chunks())


@pytest.mark.parametrize('malformation',['truncated','nonfinite'])
def test_raw_frame_truncation_and_nonfinite_rejected(malformation):
    row=['0']*20
    row[4],row[5],row[8]='src','dst','EVENT_OPEN'
    if malformation=='truncated':row=row[:12]
    else:row[15]='inf'
    with pytest.raises(ValueError):
        r.numeric_frame([row])
