"""Prepare and validate paired YAML/argv only; never starts formal training."""
from pathlib import Path
import hashlib
import json
import shlex
import sys
from types import SimpleNamespace as NS
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import yaml
import config
from mstc.experiment_utils import sanitized_config


def flatten(value,prefix=''):
    if isinstance(value,dict):
        return {k:v for key,item in value.items() for k,v in flatten(item,f'{prefix}.{key}' if prefix else key).items()}
    return {prefix:value}


def main(out):
    plan=out/'paired_plan';plan.mkdir(exist_ok=False)
    base=yaml.safe_load((ROOT/'config/experiments/mstc_full.yml').read_text())
    # Both paired arms use the existing formal validation-only execution policy;
    # comparisons are new paired runs, not historical E3 all-checkpoint selection.
    base['detection']['evaluation']={'checkpoint_policy':'val_select_then_test_best','deterministic_inference':False}
    configs={}
    for arm,mode in [('control','event_mean'),('treatment','node_contribution_balanced')]:
        candidate=json.loads(json.dumps(base));candidate['detection']['gnn_training']['train_loss_weighting']=mode
        path=plan/f'{arm}.yml'
        with path.open('x') as f:yaml.safe_dump(candidate,f,allow_unicode=True,sort_keys=False)
        configs[arm]=path.resolve()
    expected_cache={};differences={};commands=[]
    for e in [3,5]:
        resolved={}
        for arm,path in configs.items():
            args=NS(dataset=f'THEIA_E{e}',model='orthrus',config=str(path),seed=0,cpu=False,from_weights=False,
                skip_tracing=True,artifact_root=str(out/'future_preview'),shared_artifact_root=f'/mnt/datadisk/yby/mstc/artifacts/u4090_e{e}',
                shared_artifact_readonly=True,resume_checkpoint=None,inference_checkpoint=None)
            cfg=config.get_yml_cfg(args)
            cache=Path(cfg.edge_featurization.embed_edges._edge_embeds_dir)
            assert cache.is_dir(),f'Formal cached features cannot be reused: {cache}'
            expected_cache[f'E{e}']=str(cache)
            resolved[arm]=flatten(sanitized_config(cfg))
        differences[f'E{e}']={k:{arm:resolved[arm].get(k) for arm in resolved} for k in resolved['control'] if resolved['control'][k]!=resolved['treatment'][k]}
        assert set(differences[f'E{e}'])=={'detection.gnn_training.train_loss_weighting'}
        for seed in [0,1,2]:
            for arm,path in configs.items():
                destination=f'/mnt/datadisk/yby/mstc/artifacts/node_balanced_pair_20261028/E{e}/seed_{seed}/{arm}'
                argv=[sys.executable,str(ROOT/'src/experiments/run_experiment.py'),'--dataset',f'THEIA_E{e}',
                    '--config',str(path),'--seed',str(seed),'--artifact-root',destination,
                    '--shared-artifact-root',f'/mnt/datadisk/yby/mstc/artifacts/u4090_e{e}',
                    '--shared-artifact-readonly','--stages','train,test,evaluate']
                commands.append(dict(dataset=f'E{e}',seed=seed,arm=arm,output=destination,
                    estimated_additional_bytes=(6 if e==3 else 20)*1024**3,argv=argv))
    payload=dict(status='PREPARED_NOT_EXECUTED',commands=commands,config_differences=differences,shared_cached_features=expected_cache,
        seeds=[0,1,2],epochs=6,selection='min_val_objective，原始未加权验证score_raw',
        primary_metrics=['Recall@1000','Recall@5000','AP/AUPRC'],secondary_metrics=['AUROC','Precision','Recall','FPR','MCC'],
        stratification='分别按训练节点贡献次数和测试节点贡献次数分层；训练未见节点单独c_train=0组；不把训练权重应用于验证测试分数',
        independent_confirmation='E3/E5现有测试均为探索；候选外部E5-CADETS或E5-CLEARSCOPE本轮未读取，须登记既往使用记录并冻结含攻击的标签及正常校准段；不能把无攻击的未用日期当Recall确认集',
        success_rule='同预算下逐数据集报告三个配对seed差值、均值/范围、AP与FPR，不从测试挑epoch/q/K；跨数据集方向一致后另做独立确认',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/model.py',ROOT/'src/config.py',ROOT/'src/mstc/training_loss.py',ROOT/'src/detection/orthrus_gnn_training.py',ROOT/'src/mstc/experiment_utils.py']})
    with (plan/'commands.json').open('x') as f:json.dump(payload,f,ensure_ascii=False,indent=2)
    lines=['#!/bin/bash','set -euo pipefail','if [[ "${1:-}" != "--execute" ]]; then',
        '  echo "仅展示预注册命令；本轮未开始正式训练。实际执行需显式 --execute。"']
    lines += ['  printf \'%s\\n\' '+shlex.quote(shlex.join(c['argv'])) for c in commands]
    lines += ['  exit 0','fi','cd '+shlex.quote(str(ROOT)),"export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled"]
    for c in commands:
        lines += ['test ! -e '+shlex.quote(c['output']),
            'available_bytes=$(df -B1 --output=avail /mnt/datadisk | tail -1)',
            f'test "$available_bytes" -ge {c["estimated_additional_bytes"]+5*1024**3}',
            shlex.join(c['argv'])]
    with (plan/'run_pairs.sh').open('x') as f:f.write('\n'.join(lines)+'\n')
    print('配对配置与12条命令已准备，均未执行',plan)


if __name__=='__main__':main(Path(sys.argv[1]))
