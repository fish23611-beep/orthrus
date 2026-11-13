"""Fixed train-only weighting, gradients, baseline identity and trainer routing."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from torch import nn
from torch_geometric.data import TemporalData

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mstc.training_loss import NodeContributionWeights as Weights, weighting_mode
from mstc.experiment_utils import stable_config_hash
from mstc.time_gap import TimeGapStatistics
from decoders import EdgeTypeDecoder, TimeGapDecoder
from model import MSTCOrthrus


def graph(src=(0,0,0,2),dst=(1,1,0,2)):
    src=torch.tensor(src,dtype=torch.long);dst=torch.tensor(dst,dtype=torch.long);n=len(src)
    return TemporalData(src=src,dst=dst,t=torch.arange(n,dtype=torch.long)*10**9,
        edge_index=torch.stack([src,dst]),x_src=torch.arange(n*3,dtype=torch.float).reshape(n,3)/10,
        x_dst=torch.ones(n,3),msg=torch.zeros(n,1),
        edge_type=torch.nn.functional.one_hot(torch.arange(n)%2,2).float())


class Encoder(nn.Module):
    def __init__(self):super().__init__();self.proj=nn.Linear(3,3)
    def forward(self,x,**kwargs):return self.proj(x[0]),self.proj(x[1])
    def reset_state(self):pass


def model(cls=MSTCOrthrus,time=True):
    def ce(x,y,inference=False):return torch.nn.functional.cross_entropy(x,y,reduction='none' if inference else 'mean')
    return cls(Encoder(),EdgeTypeDecoder(3,2,ce,.25,2,nn.ReLU()),
        TimeGapDecoder(3,4) if time else None,TimeGapStatistics().fit([graph()]) if time else None)


def test_hand_counts_weights_selfloops_duplicates_global_mean_and_node_objective():
    g=graph();w=Weights.fit([g],num_nodes=4)
    assert w.counts.tolist()==[4,2,2,0]
    np.testing.assert_allclose(w.numpy_weights(g.src,g.dst),[1,1,2/3,4/3])
    weights=w.event_weights(g.src,g.dst)
    assert weights.mean().item()==1
    loss=torch.tensor([1.,2.,3.,4.],dtype=torch.double,requires_grad=True)
    result=w.reduce(loss,g.src,g.dst)
    # node0=(1+2+3+3)/4; node1=(1+2)/2; node2=(4+4)/2.
    assert result.item()==pytest.approx((2.25+1.5+4)/3)
    result.backward();torch.testing.assert_close(loss.grad,weights/4)
    assert w.summary()['endpoint_contributions']==8


@pytest.mark.parametrize('s,d', [([0],[0]),([0],[1]),([0,0],[0,0])])
def test_single_event_or_single_node(s,d):
    g=graph(s,d);w=Weights.fit([g],num_nodes=max(s+d)+1)
    torch.testing.assert_close(w.event_weights(g.src,g.dst),torch.ones(len(s),dtype=torch.double))


@pytest.mark.parametrize('cuts',[[1,1,2],[3,1],[4]])
def test_batch_partition_invariance_and_no_batch_renormalization(cuts):
    g=graph();w=Weights.fit([g],num_nodes=3)
    loss=torch.tensor([.1,3.,7.,.5],dtype=torch.double,requires_grad=True)
    complete=w.reduce(loss,g.src,g.dst);pieces=[];start=0
    for n in cuts:
        sl=slice(start,start+n);pieces.append(w.reduce(loss[sl],g.src[sl],g.dst[sl])*n);start+=n
    torch.testing.assert_close(sum(pieces)/4,complete,rtol=0,atol=1e-15)
    # Last one-event batch weight remains 4/3, rather than being normalized to 1.
    assert w.reduce(loss[-1:],g.src[-1:],g.dst[-1:]).item()==pytest.approx(loss[-1].item()*4/3)


def test_differentiable_weighted_losses_gradcheck():
    g=graph();w=Weights.fit([g],num_nodes=3)
    x=torch.randn(4,dtype=torch.double,requires_grad=True)
    assert torch.autograd.gradcheck(lambda x:w.reduce(x.square(),g.src,g.dst),(x,))


@pytest.mark.parametrize('bad', [[-1],[.1],[float('nan')],[float('inf')],[True]])
def test_bad_endpoint_ids_rejected(bad):
    w=Weights(np.array([1,1]),1)
    with pytest.raises(ValueError):w.numpy_weights(bad,[1])


@pytest.mark.parametrize('counts,events',[([0,0],1),([1.,1.],1),([-1,3],1),([1,1],0),([1,1],True),([1,1],1.5)])
def test_invalid_counts_and_event_totals(counts,events):
    with pytest.raises(ValueError):Weights(np.array(counts),events)


def test_zero_count_unknown_nodes_nonfinite_loss_and_val_fit_rejected():
    w=Weights(np.array([1,1,0]),1)
    for src in [torch.tensor([2]),torch.tensor([3]),torch.tensor([-1]),torch.tensor([0.])]:
        with pytest.raises(ValueError):w.event_weights(src,torch.tensor([1]))
    for bad in [float('nan'),float('inf')]:
        with pytest.raises(ValueError):w.reduce(torch.tensor([bad]),torch.tensor([0]),torch.tensor([1]))
    g=graph();g.split_name='val'
    with pytest.raises(ValueError,match='Only training'):Weights.fit([g],num_nodes=3)
    with pytest.raises(ValueError):Weights.fit([graph()],num_nodes=2)


@pytest.mark.parametrize('time',[False,True])
def test_default_forward_and_gradients_bitwise_match_unmodified_head(time):
    namespace={}
    source=subprocess.check_output(['git','show','HEAD:src/model.py'],cwd=ROOT,text=True)
    exec(compile(source,'legacy_model.py','exec'),namespace)
    old=model(namespace['MSTCOrthrus'],time);new=model(time=time);new.load_state_dict(old.state_dict())
    g=graph();results=[]
    for m in [old,new]:
        torch.manual_seed(89);m.train();out=m(g,None);out['loss'].backward()
        results.append((out,{k:p.grad.clone() if p.grad is not None else None for k,p in m.named_parameters()},torch.get_rng_state(),m.last_seen_per_node))
    for k in results[0][0]:assert torch.equal(results[0][0][k],results[1][0][k])
    for k,grad in results[0][1].items():
        if grad is None:assert results[1][1][k] is None
        else:assert torch.equal(grad,results[1][1][k])
    assert torch.equal(results[0][2],results[1][2]) and results[0][3]==results[1][3]


@pytest.mark.parametrize('time',[False,True])
def test_model_per_event_path_keeps_grad_and_inference_never_weights(time):
    m=model(time=time);g=graph();w=Weights.fit([g],num_nodes=3)
    out=m(g,None,return_event_losses=True)
    assert out['event_loss'].shape==(4,) and out['event_loss'].requires_grad
    weighted=w.reduce(out['event_loss'],g.src,g.dst);weighted.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    if not time:assert out['loss_time'].item()==0
    m.reset_state();m.eval();torch.manual_seed(7);a=m(g,None,inference=True)
    m.reset_state();torch.manual_seed(7);b=m(g,None,inference=True,return_event_losses=True)
    assert 'event_loss' not in a and 'loss' not in a
    for k in a:assert torch.equal(a[k],b[k])
    assert not a['score_raw'].requires_grad


def test_mode_and_checkpoint_hash_legacy_default_and_treatment_identity():
    cfg=NS(model=NS(variant='mstc'),detection=NS(gnn_training=NS(lr=1e-5)))
    assert weighting_mode(cfg)=='event_mean';oldhash=stable_config_hash(cfg)
    cfg.detection.gnn_training.train_loss_weighting='event_mean';assert stable_config_hash(cfg)==oldhash
    cfg.detection.gnn_training.train_loss_weighting='node_contribution_balanced'
    assert weighting_mode(cfg)=='node_contribution_balanced' and stable_config_hash(cfg)!=oldhash
    cfg.model.variant='orthrus_baseline'
    with pytest.raises(ValueError):weighting_mode(cfg)
    cfg.detection.gnn_training.train_loss_weighting='typo'
    with pytest.raises(ValueError):weighting_mode(cfg)


def test_training_loop_routes_fixed_weights_and_logs_count_weighted_objective(monkeypatch):
    from detection import orthrus_gnn_training as train
    cfg=NS(model=NS(variant='mstc'),detection=NS(gnn_training=NS(train_loss_weighting='node_contribution_balanced')))
    g=graph();w=Weights.fit([g],num_nodes=3);m=model(train.MSTCOrthrus);m.graph_reindexer=None
    batches=[graph([0,0],[1,1]),graph([0,2],[0,2])]
    batches[1].t+=2*10**9
    monkeypatch.setattr(train,'batch_loader_factory',lambda *a:iter(batches));monkeypatch.setattr(train,'log',lambda *a:None)
    optimizer=torch.optim.SGD(m.parameters(),lr=0)
    actual=train.train(g,None,m,optimizer,cfg,loss_weighting=w)
    assert np.isfinite(actual) and all(p.grad is not None for p in m.parameters())
    with pytest.raises(ValueError,match='fixed'):train.train(g,None,m,optimizer,cfg)


def test_resolved_configs_recent_multiscale_and_default_task_identity(monkeypatch):
    import config
    args=NS(dataset='THEIA_E3',model='orthrus',config=str(ROOT/'config/experiments/mstc_full.yml'),
        cpu=True,from_weights=False,seed=0,skip_tracing=False,artifact_root='/tmp/node-weighting-test',resume_checkpoint=None,inference_checkpoint=None)
    original_paths=config.set_task_paths
    monkeypatch.setattr(config,'set_task_paths',lambda c:None)
    cfg=config.get_yml_cfg(args)
    assert cfg.detection.gnn_training.train_loss_weighting=='event_mean'
    cfg._shared_artifact_readonly=True
    baseline_cfg=cfg.clone();original_paths(baseline_cfg,create_dirs=False);baseline=baseline_cfg.detection.gnn_training._task_path
    missing_cfg=cfg.clone();del missing_cfg.detection.gnn_training['train_loss_weighting']
    original_paths(missing_cfg,create_dirs=False);assert missing_cfg.detection.gnn_training._task_path==baseline
    treatment_cfg=cfg.clone();treatment_cfg.detection.gnn_training.train_loss_weighting='node_contribution_balanced'
    original_paths(treatment_cfg,create_dirs=False);assert treatment_cfg.detection.gnn_training._task_path!=baseline
    cfg.detection.gnn_training.train_loss_weighting='node_contribution_balanced'
    for mode in ['recent','multiscale']:
        cfg.detection.gnn_training.encoder.context.mode=mode
        assert weighting_mode(cfg)=='node_contribution_balanced'


def test_production_main_fits_only_train_targets_and_checks_resume_metadata(monkeypatch,tmp_path):
    from detection import orthrus_gnn_training as train
    g=graph();g.split_name='train'
    cfg=NS(model=NS(variant='mstc'),_run_dir=str(tmp_path),_from_weights=False,_test_mode=False,
        detection=NS(gnn_training=NS(train_loss_weighting='node_contribution_balanced',
            _trained_models_dir=str(tmp_path/'checkpoints'),num_epochs=1,resume_checkpoint=None,
            encoder=NS(batch_size=2),lr=1e-5,weight_decay=1e-5)))
    dataset=[g]
    monkeypatch.setattr(train,'load_all_datasets',lambda cfg:(dataset,[],[],None,3))
    monkeypatch.setattr(train,'get_device',lambda cfg:torch.device('cpu'))
    monkeypatch.setattr(train,'requires_time_gap_statistics',lambda cfg:False)
    def build(**kw):
        m=model(train.MSTCOrthrus,time=False);m.graph_reindexer=None;return m
    monkeypatch.setattr(train,'build_model',build)
    for name in ['dump_environment','update_runtime','wandb_log','log','save_model']:
        monkeypatch.setattr(train,name,lambda *a,**kw:None)
    train.main(cfg)
    import json
    metadata=tmp_path/'metadata/train_loss_weighting.json'
    summary=json.loads(metadata.read_text())
    assert summary==Weights.fit(dataset,num_nodes=3).summary()
    original=metadata.read_bytes();train.main(cfg);assert metadata.read_bytes()==original
    dataset[0]=graph([0,0,0,1],[1,1,0,2]);dataset[0].split_name='train'
    with pytest.raises(ValueError,match='metadata differs'):train.main(cfg)
