"""Read-only checkpoint and interrupted-output acceptance for execution v3."""
from pathlib import Path
import gc
import json
import os
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import joint_optimization_pipeline as sealed


def inventory(root):
    return {str(p.resolve()): dict(size=p.stat().st_size, blocks_bytes=p.stat().st_blocks*512,
                sha256=sealed.sha(p)) for p in sorted(Path(root).rglob('*')) if p.is_file()}


def frozen_cfg(out, e, artifact=None):
    import config
    import yaml
    from experiments.run_experiment import build_parser, build_pipeline_args
    from artifact_paths import resolve_artifact_paths
    from mstc.experiment_utils import stable_config_hash
    plan = sealed.read(out/'prepared_05/plan.json')
    d = plan['datasets'][str(e)]
    argv = d['commands']['training'][3:]
    if artifact is not None:
        argv = list(argv)
        argv[argv.index('--artifact-root')+1] = str(artifact)
    args = build_pipeline_args(build_parser().parse_args(argv))
    saved = yaml.safe_load(Path(d['control_training_config']).read_text())
    previous = {key: os.environ.get('ORTHRUS_DB_'+key.upper()) for key in ['host','user','port']}
    try:
        for key in previous:
            os.environ['ORTHRUS_DB_'+key.upper()] = str(saved['database'][key])
        cfg = config.get_yml_cfg(args)
    finally:
        for key,value in previous.items():
            name = 'ORTHRUS_DB_'+key.upper()
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    resolve_artifact_paths(cfg, [], cli_artifact_root=args.artifact_root, create_dirs=False)
    assert stable_config_hash(cfg) == d['config_hash'], 'Frozen effective config differs'
    return cfg


def checkpoint_audit(out, e=3):
    import torch
    import data_utils
    import factory
    from mstc.time_gap import TimeGapStatistics
    from mstc.experiment_utils import stable_config_hash
    plan = sealed.check_seal(out)
    d = plan['datasets'][str(e)]
    run = Path(d['run_dir'])
    before = inventory(run)
    assert (out/f'e{e}_training.exitcode').read_text().strip() == '0'
    cfg = frozen_cfg(out,e)
    assert cfg.detection.gnn_training.train_loss_weighting == 'node_contribution_shrunk'
    assert cfg.detection.gnn_training.num_epochs == 6 and cfg._seed == 0
    m = sealed.read(d['manifest'])
    weights = sealed.read(run/'metadata/train_loss_weighting.json')
    expected = sealed.read(out/f'e{e}_shrunk_weights.json')
    for key in ['mode','target_events','counts_sha256','alpha','tau','weight_lower_bound','weight_upper_bound','formula_version']:
        assert weights[key] == expected[key], key
    runtime = sealed.read(run/'runtime.json')
    receipt = runtime['training']['execution_receipt']
    assert receipt['status'] == 'completed' and receipt['start_epoch'] == 1 and receipt['end_epoch'] == 6
    assert receipt['batch_size'] == cfg.detection.gnn_training.encoder.batch_size == 512
    specs = [w for w in m['window_specs'] if w['split_name']=='train']
    target = sum(w['num_events'] for w in specs)
    assert target == weights['target_events']
    assert runtime['training']['processed_event_count'] == target*6
    for epoch in range(1,7):
        rows = [w for w in receipt['windows'] if w['epoch']==epoch]
        assert len(rows)==len(specs) and [r['window_index'] for r in rows]==list(range(len(specs)))
        assert [r['event_count'] for r in rows]==[w['num_events'] for w in specs]
        assert all(r['expected_batch_count']==(r['event_count']+511)//512 for r in rows)
    sample = data_utils._prepare_window(cfg, str(Path(d['shared_cache'])/specs[0]['relative_path']), cfg.dataset_view.mode)
    stats = TimeGapStatistics.load(run/'metadata/time_statistics.json')
    model = factory.build_model(sample, torch.device('cpu'), cfg, m['max_node'], stats)
    cp_root = run/'checkpoints'
    assert {p.name for p in cp_root.iterdir()} == {f'model_epoch_{i}' for i in range(1,7)}
    records = []
    for epoch in range(1,7):
        folder = cp_root/f'model_epoch_{epoch}'
        cp = torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
        state = data_utils.load_training_checkpoint(model,str(folder),cfg=cfg,map_location='cpu',restore_rng=False)
        assert state['complete'] and state['epoch']==epoch and state['config_hash']==stable_config_hash(cfg)==d['config_hash']
        assert cp['format_version']==2
        legacy = torch.load(folder/'state_dict.pkl',map_location='cpu',weights_only=False)
        assert cp['model_state_dict'].keys()==legacy.keys()
        assert all(torch.equal(v,legacy[k]) and bool(torch.isfinite(v).all()) for k,v in cp['model_state_dict'].items())
        # Native inference loader has no map_location argument. Only audit
        # tensor placement is adapted; the saved CUDA checkpoint is untouched.
        from unittest.mock import patch
        native_load = torch.load
        def cpu_load(*args, **kwargs):
            if kwargs.get("map_location") is None:
                kwargs["map_location"] = "cpu"
            return native_load(*args, **kwargs)
        with patch.object(torch, "load", cpu_load):
            data_utils.load_model(model,str(folder))
        assert all(torch.equal(v,cp['model_state_dict'][k]) for k,v in model.state_dict().items())
        steps = sorted({int(v['step']) for v in cp['optimizer_state_dict']['state'].values() if 'step' in v})
        batches_per_epoch = sum((w['num_events']+511)//512 for w in specs)
        # Adam increments per parameter only when a gradient exists (the first
        # temporal batch need not touch every head). The maximum must match
        # the complete batch receipt; all parameter counters must be bounded.
        assert steps and max(steps)==epoch*batches_per_epoch
        assert all(0 < step <= epoch*batches_per_epoch for step in steps)
        records.append(dict(epoch=epoch,status='PASS',native_complete_loader=True,native_inference_loader=True,
            config_hash=state['config_hash'],legacy_structured_weights_exact=True,
            cumulative_training_events_from_receipt=epoch*target,optimizer_steps=steps,
            files={p.name:before[str(p.resolve())] for p in [folder/'checkpoint.pt',folder/'state_dict.pkl']}))
        print('Checkpoint acceptance',e,epoch,'PASS',flush=True)
        del cp,legacy
        gc.collect()
    del model,sample
    gc.collect()
    assert inventory(run)==before, 'Original run changed during read-only acceptance'
    return dict(status='PASS',dataset=e,created_at=sealed.now(),run_dir=str(run),checkpoints=records,
        train_events_per_epoch=target,training_events=target*6,training_exitcode=0,
        training_runtime=runtime['training'],train_weighting=weights,original_inventory=before,
        checkpoint_event_count_note='Checkpoint format has no event-count field; receipt and optimizer steps jointly checked',
        frozen_seal_status='PASS',original_files_unchanged=True)


def interrupted_outputs(out):
    import numpy as np
    import pandas as pd
    plan = sealed.read(out/'prepared_05/plan.json')
    d = plan['datasets']['3']
    run = Path(d['run_dir'])
    manifest = sealed.read(d['manifest'])
    specs = [w for w in manifest['window_specs'] if w['split_name']=='val']
    epochs = {}
    for epoch in range(1,7):
        folder = run/f'edge_scores/val/model_epoch_{epoch}'
        paths = sorted(folder.glob('*.csv'))
        evidence = []
        complete = len(paths)==len(specs)
        for i,path in enumerate(paths):
            rows = 0
            with path.open('rb') as f:
                f.seek(-1,2)
                complete = complete and f.read(1)==b'\n'
            for frame in pd.read_csv(path,chunksize=100000,float_precision='round_trip'):
                if i>=len(specs):
                    complete=False
                    continue
                expected = np.arange(specs[i]['global_offset']+rows,specs[i]['global_offset']+rows+len(frame))
                complete = complete and np.array_equal(frame.event_index.to_numpy(),expected)
                complete = complete and bool(np.isfinite(frame.score_raw.to_numpy()).all())
                rows += len(frame)
            complete = complete and i<len(specs) and rows==specs[i]['num_events']
            evidence.append(dict(path=str(path),rows=rows,sha256=sealed.sha(path)))
        epochs[str(epoch)] = dict(status='COMPLETE_VALIDATION_WINDOWS' if complete else 'ABSENT_OR_INCOMPLETE',
                                 csv_count=len(paths),expected_csv_count=len(specs),files=evidence)
    samples = [json.loads(line) for line in (out/'resource_monitor.jsonl').read_text().splitlines()]
    trigger = samples[-1]
    original_bytes=sum(p.stat().st_blocks*512 for p in Path(d['artifact_root']).rglob('*') if p.is_file())
    return dict(created_at=sealed.now(),epochs=epochs,selection_manifest_exists=(run/'checkpoint_selection.json').exists(),
        original_owned_allocated_bytes=original_bytes,old_trigger=trigger,old_budget_bytes=6*sealed.GIB,
        old_trigger_condition_verified=trigger['data_net_growth']>6*sealed.GIB,
        current_tmp_files=[str(p) for p in (Path(d['artifact_root'])/'tmp').rglob('*') if p.is_file()],
        conclusion='Old gate used global free-space delta, which does not establish experiment write ownership.',
        uncertainty='No historical process I/O/owned-footprint high-water trace: cannot exclude deleted temporary peaks or attribute all growth to other users.',
        isolation='Keep original validation outputs unchanged; regenerate a coherent six-epoch selection in fresh run. Frozen deterministic_inference=false; no RNG-boundary shortcut.')
