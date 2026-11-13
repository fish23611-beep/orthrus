"""CPU-only train-frequency acceptance; no validation/test labels or GNN."""
from pathlib import Path
import hashlib
import json
import os
import resource
import sys
import time
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from audit_node_training_frequency import windows, write, source_for
from mstc.training_loss import ShrunkNodeContributionWeights


def main(out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    started=time.monotonic();torch.set_num_threads(2)
    for e in [3,5]:
        old=json.loads((ROOT/f'analysis/node_balanced_training_20261028/e{e}_train_frequency.json').read_text())
        paths=[Path(x['path']) for x in old['source_tensor_hashes']]
        assert all((p.stat().st_size,p.stat().st_mtime_ns)==(x['bytes'],x['mtime_ns']) for p,x in zip(paths,old['source_tensor_hashes']))
        counts=np.zeros(0,dtype=np.int64);events=0;sources=[]
        for w in windows(paths,sources):
            s,d=w.src.numpy(),w.dst.numpy();capacity=max(int(s.max()),int(d.max()))+1
            if capacity>len(counts):counts=np.pad(counts,(0,capacity-len(counts)))
            np.add.at(counts,s,1);np.add.at(counts,d,1);events+=len(s)
        assert sources==old['source_tensor_hashes'],'训练端点/时间张量完整性不一致'
        weights=ShrunkNodeContributionWeights(counts,events)
        assert weights.summary()['counts_sha256']==old['counts_sha256']
        n=0;total=0.;square=0.;vmin=float('inf');vmax=0.;hist=np.zeros(100,dtype=np.int64)
        bins=np.linspace(.5,2.5,101)
        for w in windows(paths):
            values=weights.numpy_weights(w.src,w.dst);n+=len(values)
            total+=float(values.sum());square+=float(np.dot(values,values))
            vmin=min(vmin,float(values.min()));vmax=max(vmax,float(values.max()))
            hist+=np.histogram(values,bins=bins)[0]
        assert n==events and abs(total/n-1)<1e-12
        assert vmin>.5 and vmax<=2.5+1e-12 and hist.sum()==events
        write(out/f'e{e}_shrunk_weights.json',dict(status='PASS',dataset=f'THEIA_E{e}',**weights.summary(),
            weight_mean=total/n,weight_min=vmin,weight_max=vmax,ess=total*total/square,ess_fraction=total*total/square/n,
            historical_balanced_ess_fraction=old['weight_distribution']['ess_fraction'],
            historical_balanced_max=old['weight_distribution']['quantiles']['1'],
            histogram=dict(edges=bins.tolist(),counts=hist.tolist()),all_training_endpoint_time_hashes_exact=True,
            source_statistics=str(ROOT/f'analysis/node_balanced_training_20261028/e{e}_train_frequency.json'),
            no_validation_test_or_label_access=True,gradient_contribution_measured=False,model_performance_measured=False))
        print('训练权重验收PASS',e,'tau',weights.tau,'min/max',vmin,vmax,'ESS比例',total*total/square/n,flush=True)
    write(out/'train_weight_audit.json',dict(status='PASS',seconds=time.monotonic()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,python=sys.executable,torch=torch.__version__))


if __name__=='__main__':main(Path(sys.argv[1]))
