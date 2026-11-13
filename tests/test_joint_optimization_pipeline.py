"""Resource gates fail closed; prediction seals precede all attack-label reads."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import numpy as np
import pytest
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'src'),str(Path(__file__).resolve().parents[1]/'scripts')]
import joint_optimization_pipeline as p
import evaluate_joint_optimization as e


def gate(data=50,root=5,compute='',util=0):
    def usage(path):return NS(free=(data if path=='/mnt/datadisk' else root)*p.GIB)
    def query(argv,**kw):return compute if 'compute-apps' in argv[1] else f'0, GPU-fake, 20, 24564, {util}\n'
    return usage,query


def test_idle_and_sufficient_gate_passes_both_datasets():
    for dataset in [None,3,5]:
        usage,query=gate();r=p.resource_gate(dataset,usage=usage,query=query)
        assert r['status']=='PASS' and not r['errors']


@pytest.mark.parametrize('kwargs,dataset',[(dict(data=25),3),(dict(data=39),5),(dict(root=2),3),
    (dict(compute='123, GPU-fake, 100\n'),3),(dict(util=98),5)])
def test_disk_busy_gpu_gates_reject(kwargs,dataset):
    usage,query=gate(**kwargs);result=p.resource_gate(dataset,usage=usage,query=query)
    assert result['status']=='BLOCKED' and result['errors']


def test_query_failure_never_means_idle():
    usage,_=gate()
    def fail(*a,**kw):raise OSError('nvidia-smi unavailable')
    assert p.resource_gate(usage=usage,query=fail)['status']=='BLOCKED'


def test_smoke_blocks_before_importing_or_allocating_cuda(monkeypatch,tmp_path):
    monkeypatch.setattr(p,'check_seal',lambda out:{})
    monkeypatch.setattr(p,'resource_gate',lambda :dict(status='BLOCKED',errors=['busy']))
    monkeypatch.setattr(p,'save',lambda *a,**kw:None)
    assert p.smoke(tmp_path)==2 and not (tmp_path/'gpu_smoke').exists()


def test_formal_training_requires_smoke_pass_before_any_process(monkeypatch,tmp_path):
    monkeypatch.setattr(p,'check_seal',lambda out:{})
    p.save(tmp_path/'gpu_smoke.json',dict(status='PENDING'))
    def fail(*a,**kw):pytest.fail('must never launch formal process')
    monkeypatch.setattr(p.subprocess,'Popen',fail)
    with pytest.raises(AssertionError,match='Smoke'):p.run(tmp_path)
    assert not (tmp_path/'FORMAL_STATUS.json').exists()


def test_prediction_seal_detects_mutation_and_blocks_label_access(tmp_path,monkeypatch):
    ids=np.array([1,2]);scores=np.array([.2,3.]);pred=scores>1
    for arr in [ids,scores,pred]:arr.flags.writeable=False
    path=tmp_path/'seal.json'
    p.save(path,dict(test_labels_read=False,ids_sha256=e.recovery.digest_arrays(ids),
        scores_sha256=e.recovery.digest_arrays(scores),predictions_sha256=e.recovery.digest_arrays(pred)))
    e.require_prediction_seal(path,ids,scores,pred)
    scores.flags.writeable=True;scores[0]=99;scores.flags.writeable=False
    monkeypatch.setattr(e.diag,'metadata_for',lambda *a:pytest.fail('GT access before seal validation'))
    with pytest.raises(AssertionError):e.load_labels_after_seal(3,path,ids,scores,pred)


def test_group_metrics_match_hand_counts_and_one_class_marks_unidentified_auc():
    truth=np.array([1,0,1,0]);scores=np.array([4.,3.,2.,1.]);pred=scores>2.5
    m=e.group_metrics(truth,pred,scores,np.ones(4,dtype=bool))
    assert (m['tp'],m['fp'],m['fn'],m['tn'])==(1,1,1,1)
    assert m['precision']==m['recall']==.5
    m=e.group_metrics(truth,pred,scores,truth==1)
    assert m['auprc'] is None and m['auroc'] is None


def test_undefined_subgroup_metrics_are_json_null():
    import json
    truth=np.array([0,0]);pred=np.array([False,False]);scores=np.array([.1,.2])
    m=e.group_metrics(truth,pred,scores,np.ones(2,dtype=bool))
    assert m['recall'] is None and m['mcc'] is None and m['precision'] is None
    json.dumps(m,allow_nan=False)
