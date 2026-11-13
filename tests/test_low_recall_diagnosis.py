"""排名并列、固定预算及有来源Top5的CPU契约测试。"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import diagnose_e3_e5_low_recall as d
from mstc.aggregation import NodeScoreAggregator


def test_rank_ties_deterministic_and_hand_calculated_budgets():
    ids=np.array([40,10,30,20]);scores=np.array([3.,3.,1.,2.]);truth=np.array([1,0,1,0])
    order,ranks,best,worst=d.deterministic_ranks(ids,scores)
    assert ids[order].tolist()==[10,40,20,30]
    assert ranks.tolist()==[2,1,4,3]
    assert best.tolist()==[1,1,4,3] and worst.tolist()==[2,2,4,3]
    rows=d.budget_metrics(ids,scores,truth,[1,2,3,4])
    assert [x['tp'] for x in rows]==[0,1,1,2]
    assert [x['recall'] for x in rows]==[0,.5,.5,1]
    assert [x['precision'] for x in rows]==[0,.5,1/3,.5]
    assert rows[0]['tp_tie_min']==0 and rows[0]['tp_tie_max']==1
    assert rows[1]['tp_tie_min']==rows[1]['tp_tie_max']==1


@pytest.mark.parametrize('chunk',[1,2,9])
def test_top5_metadata_chunks_selfloops_and_formal_reducer(chunk):
    frame=pd.DataFrame(dict(event_index=np.arange(9),srcnode=[1]*9,dstnode=[1,2,3,2,3,2,3,2,3],
        edge_type_index=[1,2,1,2,1,2,1,2,1],score_raw=np.arange(9,dtype=float),
        loss_type=np.arange(9,dtype=float),loss_time=np.zeros(9),loss_time_src=np.zeros(9),loss_time_dst=np.zeros(9),
        score_calibrated=[9.,9.,9.,8.,7.,6.,5.,4.,3.],calibration_level=['triplet']*9))
    bank=d.DiagnosticBank(np.array([1,2,3,4]),threshold=7.)
    for start in range(0,len(frame),chunk):
        f=frame.iloc[start:start+chunk];bank.add_frame(f,f.score_calibrated.to_numpy())
    formal=NodeScoreAggregator().aggregate_events(frame.to_dict('records'),topk=5)
    for nid,value in formal.items(): assert bank.scores(5)[nid-1]==value
    assert bank.event[0].tolist()==[0,0,1,2,3]
    assert bank.direction[0].tolist()==[0,1,0,0,0]
    assert bank.counts.tolist()==[10,4,4,0]
    assert bank.src_counts.tolist()==[9,0,0,0]
    assert bank.rel_tail[0].sum()==5
    assert bank.relation_mask[0].item()==6
    assert bank.rel_counts.sum()==18 and bank.events==9 and bank.self_loops==1
    assert bank.max_joint_residual==bank.max_time_residual==0


def test_invalid_ranks_and_unknown_relation_rejected():
    with pytest.raises(ValueError): d.deterministic_ranks([1,1],[1.,2.])
    with pytest.raises(ValueError): d.deterministic_ranks([1,2],[1.,np.nan])
    b=d.DiagnosticBank(np.array([1]),relations=2)
    f=pd.DataFrame(dict(edge_type_index=[2]))
    with pytest.raises(ValueError,match='关系ID'): b.add_contributions(np.array([1]),np.array([1.]),f,0,np.array([0]))
