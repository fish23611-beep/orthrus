"""Numerical and production-routing checks for the frozen shrinkage objective."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
from mstc.training_loss import ShrunkNodeContributionWeights as Weights, fit_training_weights, weighting_mode
from mstc.experiment_utils import stable_config_hash
from test_node_balanced_training import graph, model


def cfg(mode='node_contribution_shrunk'):
    return NS(model=NS(variant='mstc'), detection=NS(gnn_training=NS(train_loss_weighting=mode)))


def test_hand_formula_mean_and_gradient():
    g=graph();w=Weights.fit([g],num_nodes=4)
    assert w.tau==pytest.approx(8/3)
    c=np.array([4.,2.,2.]);s=(c/(c+w.tau)).sum()
    expected=.5+.5*(4/s)*(1/(w.counts[g.src]+w.tau)+1/(w.counts[g.dst]+w.tau))
    np.testing.assert_allclose(w.numpy_weights(g.src,g.dst),expected,rtol=1e-15)
    torch.testing.assert_close(w.event_weights(g.src,g.dst),torch.tensor(expected))
    assert expected.mean()==pytest.approx(1.)
    loss=torch.tensor([1.,2.,4.,8.],dtype=torch.double,requires_grad=True)
    value=w.reduce(loss,g.src,g.dst);value.backward()
    torch.testing.assert_close(loss.grad,torch.tensor(expected)/4)
    assert value.item()==pytest.approx(np.mean(expected*[1.,2.,4.,8.]))
    assert torch.autograd.gradcheck(lambda x:w.reduce(x.square(),g.src,g.dst),(loss,))


@pytest.mark.parametrize('counts',[[2],[1,1],[4,2,2,0],[10**12,1,1,0],[10**12]+[1]*100,[10**18,2,0]])
def test_extreme_frequencies_global_normalization_bounds_and_sparse_zeros(counts):
    c=np.array(counts,dtype=np.int64);w=Weights(c,int(c.sum())//2)
    active=c>0;inv=w._endpoint_inverse()
    # Endpoint-marginal identity is exact independently of event pairings.
    mean=.5+.5*w.normalization*np.sum(c*inv)/w.num_events
    assert mean==pytest.approx(1.,abs=1e-14)
    assert np.all(inv[~active]==0)
    ids=torch.tensor(np.flatnonzero(active));v=w.event_weights(ids,ids)
    assert bool(torch.isfinite(v).all()) and float(v.min())>.5
    assert float(v.max())<=2.5+1e-14 and w.weight_upper_bound<=2.5+1e-14
    if len(counts)==101:assert w.tau>2*w.num_events/w.num_nodes


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_partition_invariance_no_batch_renormalization(dtype):
    g=graph();w=Weights.fit([g],num_nodes=4);loss=torch.arange(1,5,dtype=dtype)
    all_loss=w.reduce(loss,g.src,g.dst)
    pieces=[w.reduce(loss[s],g.src[s],g.dst[s])*len(loss[s]) for s in [slice(0,1),slice(1,3),slice(3,4)]]
    torch.testing.assert_close(sum(pieces)/4,all_loss)
    assert w.event_weights(g.src[-1:],g.dst[-1:]).item()!=pytest.approx(1.)


@pytest.mark.parametrize('src,dst',[([3],[0]),([4],[0]),([-1],[0]),([.1],[0]),([True],[0]),([float('nan')],[0]),([float('inf')],[0]),([0,1],[0])])
def test_invalid_unknown_zero_frequency_ids(src,dst):
    w=Weights(np.array([4,2,2,0]),4)
    with pytest.raises(ValueError):w.numpy_weights(src,dst)
    with pytest.raises(ValueError):w.event_weights(torch.tensor(src),torch.tensor(dst))


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_loss_rejected(bad):
    w=Weights(np.array([1,1]),1)
    with pytest.raises(ValueError):w.reduce(torch.tensor([bad]),torch.tensor([0]),torch.tensor([1]))


def test_fit_factory_split_guard_hash_and_default():
    g=graph();w=fit_training_weights(cfg(),[g],num_nodes=4)
    assert isinstance(w,Weights) and w.summary()['mode']=='node_contribution_shrunk'
    assert fit_training_weights(cfg('event_mean'),[g],num_nodes=4) is None
    assert type(fit_training_weights(cfg('node_contribution_balanced'),[g],num_nodes=4)) is not Weights
    assert len({stable_config_hash(cfg(m)) for m in ['event_mean','node_contribution_balanced','node_contribution_shrunk']})==3
    for split in ['val','test']:
        g.split_name=split
        with pytest.raises(ValueError,match='Only training'):fit_training_weights(cfg(),[g],num_nodes=4)
    c=cfg();c.model.variant='semantic_mlp'
    with pytest.raises(ValueError):weighting_mode(c)


def test_production_training_routes_new_objective_and_preserves_raw_inference(monkeypatch):
    from detection import orthrus_gnn_training as train
    g=graph();w=Weights.fit([g],num_nodes=4);m=model(train.MSTCOrthrus);m.graph_reindexer=None
    monkeypatch.setattr(train,'batch_loader_factory',lambda *args:iter([g]))
    monkeypatch.setattr(train,'log',lambda *args:None)
    seen=[];original=w.reduce
    def reduce(*args):
        result=original(*args);seen.append(float(result.detach()));return result
    monkeypatch.setattr(w,'reduce',reduce)
    actual=train.train(g,None,m,torch.optim.SGD(m.parameters(),lr=0),cfg(),loss_weighting=w)
    assert actual==pytest.approx(seen[0]) and len(seen)==1
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    with pytest.raises(ValueError,match='fixed'):train.train(g,None,m,None,cfg())
    m.reset_state();m.eval();a=m(g,None,inference=True)
    m.reset_state();b=m(g,None,inference=True,return_event_losses=True)
    for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    assert 'event_loss' not in a and 'loss' not in a


def test_production_main_writes_new_fit_and_rejects_changed_split(monkeypatch,tmp_path):
    from detection import orthrus_gnn_training as train
    g=graph();c=cfg();c._run_dir=str(tmp_path);c._from_weights=False;c._test_mode=False
    c.detection.gnn_training.__dict__.update(_trained_models_dir=str(tmp_path/'ckpt'),num_epochs=1,
        resume_checkpoint=None,encoder=NS(batch_size=4),lr=1e-5,weight_decay=1e-5)
    data=[g]
    monkeypatch.setattr(train,'load_all_datasets',lambda c:(data,[],[],None,4))
    monkeypatch.setattr(train,'get_device',lambda c:torch.device('cpu'))
    monkeypatch.setattr(train,'requires_time_gap_statistics',lambda c:False)
    def build(**kw):
        m=model(train.MSTCOrthrus,time=False);m.graph_reindexer=None;return m
    monkeypatch.setattr(train,'build_model',build)
    for name in ['dump_environment','update_runtime','wandb_log','log','save_model']:
        monkeypatch.setattr(train,name,lambda *args,**kw:None)
    train.main(c)
    path=tmp_path/'metadata/train_loss_weighting.json'
    assert json.loads(path.read_text())==Weights.fit(data,num_nodes=4).summary()
    original=path.read_bytes();train.main(c);assert path.read_bytes()==original
    data[0]=graph([0,0,0,1],[1,1,0,2])
    with pytest.raises(ValueError,match='metadata differs'):train.main(c)
