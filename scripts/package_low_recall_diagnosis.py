"""从已完成的CPU证据生成中文报告并执行最终验收；不重算或选择模型方案。"""
from pathlib import Path
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import hashlib
import json
import subprocess
import numpy as np
import pandas as pd
import diagnose_e3_e5_low_recall as d


def md_table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(map(str,r))+' |' for r in rows]])


def write(path,text):
    with path.open('x',encoding='utf-8') as f:f.write(text.strip()+'\n')


def main(out):
    ds={e:d.read(out/f'{e.lower()}_diagnostics.json') for e in ['E3','E5']}
    raw=d.read(out/'raw_joint_control.json');rawres=raw['results']
    top=pd.read_csv(out/'topn_metrics.csv');att=pd.read_csv(out/'attack_node_details.csv')
    ranks=pd.read_csv(out/'attack_rank_summary.csv');fp=pd.read_csv(out/'false_positive_analysis.csv')
    relations=pd.read_csv(out/'relation_contribution_summary.csv');control=pd.read_csv(out/'raw_joint_attack_control.csv')
    rawtop=pd.read_csv(out/'raw_joint_topn_control.csv')
    before=d.read(out/'execution_before.json');verify=d.read(out/'analysis_verification.json')
    assert verify['status']=='PASS'
    for name in ['run_cpu_02','run_raw_control','tests_cpu']:assert (out/f'{name}.exitcode').read_text().strip()=='0'
    assert '42 passed' in (out/'tests_cpu.log').read_text()
    assert d.read(out/'e5_recovery_recheck/frozen_predictions.json')==d.read(d.OLD5/'frozen_predictions.json')
    # 对交付表再做独立检验：行数、标签、阈值、排名、TopN手算和贡献守恒。
    checks={}
    for e,s in ds.items():
        a=att[(att.dataset==e)&att.in_test].copy();t=top[top.dataset==e]
        assert len(a)==s['attacks'] and a.node_id.is_unique and a.uuid.is_unique
        assert np.array_equal((a.score>s['threshold']).to_numpy(),(a.status=='TP').to_numpy())
        assert int((a.status=='TP').sum())==s['metrics']['tp'] and int((a.status=='FN').sum())==s['metrics']['fn']
        assert np.allclose(a.score-a.threshold,a.score_minus_threshold,rtol=0,atol=2e-15)
        assert np.allclose(a.score/a.threshold,a.score_over_threshold,rtol=0,atol=1e-15)
        assert np.array_equal(a.src_contributions+a.dst_contributions,a.contribution_count)
        assert np.allclose(a['rank']/s['nodes']*100,a.rank_top_percent,rtol=0,atol=2e-14)
        assert np.all((a.rank_tie_best<=a['rank'])&(a['rank']<=a.rank_tie_worst))
        assert a['rank'].is_monotonic_increasing
        for row in t.itertuples():
            hand=int((a['rank']<=row.budget).sum())
            assert hand==row.tp and row.fp==row.budget-hand
            assert abs(row.recall-hand/len(a))<1e-14 and abs(row.precision-hand/row.budget)<1e-14
        rel=relations[relations.dataset==e]
        assert rel.all_contributions.sum()==s['contributions']==2*s['events']
        # supplement从原joint score独立重建出的攻击贡献次数与正式校准扫描一致。
        c=control[(control.dataset==e)&control.in_test].set_index('node_id')
        assert len(c)==len(a)
        assert np.array_equal(c.loc[a.node_id,'all_contributions'].to_numpy(),a.contribution_count.to_numpy())
        assert np.allclose(c.loc[a.node_id,'max_actual_calibrated_event'],a.top1_score,rtol=0,atol=1e-12)
        checks[e]=dict(attack_csv_rows=len(a),baseline_metrics_exact=True,all_node_rank_permutation=True,
            tp_fp_fn_tn_partition_exact=True,attack_table_score_threshold_labels_exact=True,
            all_topn_recomputed_from_attack_ranks=True,rank_percent_ties_exact=True,
            node_contributions_equal_src_plus_dst=True,relation_contributions_equal_twice_events=True,
            independent_raw_scan_contribution_counts_exact=True,independent_calibrated_max_scores_exact=True)
    assert len(att[~att.in_test])==1 and int(att[~att.in_test].iloc[0].node_id)==36763
    # 现有正式run、旧验收包与用户已有tracked修改均保持不变。
    assert all(d.e3.fingerprint(p)==v for p,v in before['formal_stats'].items())
    assert all(d.e3.sha(p)==v for p,v in before['prior_analysis_hashes'].items())
    assert hashlib.sha256(subprocess.check_output(['git','diff'])).hexdigest()==before['git_diff_sha256']
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==before['git_head']
    expected={x['path']:x['sha256'] for x in d.read(d.ACCEPT/'artifact_manifest.json')['inputs']}
    raw_matches=[]
    old3=d.read(d.ROOT/'analysis/e3_seed0_exploratory_20261028/run_02/summary.json')
    for ev in raw['sources']:
        p=ev['path']
        if p in expected:assert expected[p]==ev['sha256'];raw_matches.append(p)
        if p.endswith('model_epoch_4/validation_calibrated.csv'):assert ev['sha256']==old3['sources']['validation']['sha256']
    diff=subprocess.run(['git','diff','--check'],capture_output=True,text=True)
    assert diff.returncode==0
    e3,e5=ds['E3'],ds['E5']
    hypothesis={
      'H1':dict(结论='主要瓶颈：当前异常表征及评分链的低告警预算头部排序不足；不等同于模型完全无信号',
        支持证据=[f"攻击排名中位数：E3={90085}/699295（12.88%），E5={161003}/747452（21.54%）",
          'Top5000只覆盖E3 12/118、E5 6/69；即使Top10%也仅37/118、27/69',
          'FN的score/threshold中位数分别0.2919、0.2878，接近阈值者只有2/109、1/68',
          '固定原joint score对照的Top5000覆盖E3 21/118、E5 3/69；去掉校准仍无高覆盖',
          '多数FN连单个校准事件也未超过节点阈值：E3 95/109，E5 62/68'],
        反对或限制=['AUROC 0.7440/0.7216，AP高于随机正例比例，存在一定排序信号',
          'E3原joint score的Top5000由12提升至21，证明校准会损害部分头部次序',
          '本轮没有中间embedding、训练梯度或独立训练对照，不能将全部损失精确归因于训练目标'],
        尚不可验证=['表征容量、正常训练事件频率偏置和优化收敛各自的因果份额','改变训练目标后是否提高独立测试Recall@N']),
      'H2':dict(结论='存在阈值迁移失配，E3明显、E5较弱；不足以解释两者低Recall的主体',
        支持证据=['正常测试FPR/实际验证超阈率：E3 3.5318倍，E5 1.2218倍',
          'E3正常节点P99.9由验证5.4191升至测试6.6352；E5为7.6564至7.7021',
          'E3原joint AP 0.004637高于校准0.002820，固定Top5000多9个攻击；分层校准影响头部排序',
          'E5三个攻击节点的所涉参考规模最多1766，校准理论上限log(1767)=7.4770低于节点阈值7.6564'],
        反对或限制=['漏检接近阈值者仅1.83%/1.47%；不是大量边界误差',
          'E5正常极尾迁移不大但Recall更低；简单降低阈值仍需数千至数万正常告警',
          'E5上述三个受上限限制节点实际最高事件仅1.7867且未饱和，原joint排名约48%-49%，上限不是其当前漏检的充分解释',
          '原joint对照E5 Top5000反而由6降至3；不能据E3一例否定校准'],
        尚不可验证=['有无干净的独立正常校准时间窗口可用于在线适应','阈值漂移中时间跨度、实体构成、关系构成与依赖事件的因果份额']),
      'H3':dict(结论='存在系统性的频率与目标方向关联，并掩盖少数单事件尖峰；不是全部漏检的解释',
        支持证据=['FP事件贡献中位数E3 205、E5 1605；FN为3、6，TN为2、10',
          '贡献>100的正常节点占E3正常节点0.83%、E5 1.17%，却贡献69.07%、82.52%的FP',
          'FP Top5全为目标方向：E3 2168/2515，E5 775/898（均约86%）',
          '14个E3 FN、6个E5 FN包含超过节点阈值的校准事件，但Top5均值未超过阈值',
          'E5 K1将6-10贡献组FP从6变为710，>1000贡献组从691变为47，显示误报结构转移'],
        反对或限制=['E3 95/109、E5 62/68个FN没有事件超过正式节点阈值；仅改Top5无法凭空产生高分',
          'K1的总体AP在E3提高、E5下降；E3还丢掉6个原TP，只新增5个，不能视为统一修复',
          'Top1占有效Top5和>=50%的FP只有26/2515、91/898；大多数FP不是单个极端事件支配',
          '低事件数下Top1占比较高有结构性原因；攻击节点标签不等于每条相邻事件都是攻击'],
        尚不可验证=['频率条件聚合或方向分离是否能在独立数据提高头部排序','多尺度gate、embedding及事件恶意标签未保存，无法做融合因果归因'])}
    d.e3.write_json(out/'hypothesis_evidence.json',dict(说明='观察事实与机制推测分开；不是相互排斥的统计假设检验',假设=hypothesis,
        综合判定='H1为低Recall主瓶颈；H3为明确次级结构问题；H2主要影响FPR且跨数据集程度不同',
        唯一优先推荐='E：改进节点级异常表征与模型训练目标，提升固定低告警预算下的头部排序',
        独立验证状态='未验证；本轮测试标签仅用于诊断'))

    priority="""
# 下一阶段优化方向排序

唯一优先实施：**E，改进模型训练目标与节点异常表征，目标是提高固定告警预算下的头部排序。**

两套正式评分的攻击排名中位数分别为全体前12.88%和21.54%；Top5000仅覆盖10.17%和8.70%。FN分数/阈值中位数均约0.29。去掉校准的原joint score也只在Top5000覆盖17.80%和4.35%。这是当前优先级高于简单阈值搜索的依据；不是已经证明某个新训练方法有效。

| 顺序 | 方向 | 对应观察及定量证据 | 重训与复用 | 预计计算/新增存储成本 | Recall/FPR权衡 | 独立验证方法 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | E：模型训练/异常表征 | 大多数FN远离阈值；原始和校准两种分数都缺乏低预算覆盖。E3 95/109、E5 62/68个FN没有任何事件高于节点阈值 | 需要重训及新模型推理；复用图、特征、划分、标签映射、CPU评价与诊断代码；旧模型分数不能替代新表征 | 下一轮至少每数据集一个配对训练与推理；不能用本轮CPU耗时估算GPU时间。若3 seeds：每数据集3个对照+3个变更运行，已有合规seed0可减去一次对照。单checkpoint约9.54MiB；保留完整E5新分数需原始test约3.70GiB、val约1.03GiB，再保留校准表还会增加数GiB。优先仅保留最优checkpoint和流式节点摘要；GPU时间/缓存峰值另行预算 | 可能同时提高头部排序和Recall；也可能学到更多罕见正常行为导致FPR升高，不能预估正收益 | 固定一个训练目标改动，其余时间损失、多尺度、校准、K5、q=.999不变；按正常验证目标选epoch。在未参与本轮分析的新时间段/独立数据做多seed配对比较，主指标Recall@1000/@5000、AP，辅以正式Recall/FPR/MCC |
| 2 | B：节点聚合 | 高频正常节点贡献69.07%/82.52% FP；约86% FP的Top5全来自目标方向；14/6个FN有被均值稀释的高事件 | 无需重训，复用所有事件分数；本轮不实施新聚合搜索 | 本轮完整恢复+统计约171秒、峰值3.20GiB RAM；增加一种固定聚合可用同阶CPU扫描，状态随节点数增长；只存摘要通常<20MiB | 能抑制高频误报并保留稀疏尖峰，但K1已显示低频正常节点FP转移，可能丢掉持续攻击信号 | 只预注册一种频率条件/方向分离聚合，验证集定规则和阈值；在新独立集比较固定预算覆盖、频率分层FPR；不可从本轮E3/E5结果选最优K |
| 3 | C：分层关系校准 | E3无校准Top5000攻击数21优于正式12；E5却为3劣于6，影响不一致。E5有3个节点理论上限低于阈值但当前未饱和 | 无需重训，复用val/test原始分数；重新拟合校准需要CPU | 验证事件排序和一次测试扫描；本轮固定raw诊断31.4秒/0.67GiB RAM，正式严格恢复147秒/3.20GiB，可作当前量级参考；摘要输出<20MiB | 改善稀疏关系上限或分层失配可能救回部分节点，也可能恢复常见正常行为的极端分数导致FP增加 | 预注册单一校准改动，在独立正常验证段检查各关系尾概率与正常FPR，再在新攻击数据测AP/Recall@N；同时固定原joint与正式校准对照 |
| 4 | A：验证阈值/漂移适应 | E3正常测试尾超阈为验证3.53倍，E5仅1.22倍；近阈值FN仅2/109、1/68 | 无需重训，可复用节点分数和独立正常窗口；现有测试不提供干净在线正常标签 | 已有节点分数时O(N) CPU/数MB内存；只有摘要则需一次事件重聚合。不能为此次诊断再写全量CSV | 保持FPR稳定通常需要提高E3阈值，可能进一步降Recall；降低阈值会增加大量正常告警，无法改变排名 | 用独立正常时间窗口拟合适应规则，冻结后测试告警率和低预算覆盖；不对本轮测试标签搜索q |
| 5 | D：时间异常损失/多尺度融合 | 时间分量可见但无gate/embedding。E3已有raw_type AP=.004777、cal_type AP=.001396、正式AP=.002820，说明时间与校准交互并非单向；不足以锁定哪个尺度失效 | 真正改变损失/融合需重训；仅重组合已存loss不能验证新表征 | 至少同E类训练推理成本，额外gate/embedding追踪可能显著增加日志；仅保存分层摘要可控 | 去掉时间可能减少时序正常行为FP，也可能漏掉时间型攻击；本轮不能确定收益 | 在E方向检验后再预注册一种结构消融；保存各尺度可用性与gate摘要，在独立数据及多seed比较，不将已有分数重加权称为重训验证 |

## 唯一首轮实施建议的具体边界

改动限定于**训练损失如何汇总节点贡献**，先保持模型结构、时间任务、多尺度融合、事件分数定义和正式后处理全部不变。一个可预注册的单一候选是：用正常训练集的节点贡献次数，把每个节点的平均事件重构损失等权汇总，而非让事件多的节点按事件总量支配目标。形式为 `L = mean_v(mean_{e贡献到v}(loss_type_e + 0.3*loss_time_e))`，权重只由训练集计算并固定；不引入测试标签或本轮测试阈值搜索。

这只是可证伪的首轮训练假设。当前未检查训练梯度或训练节点频率，不能宣称“训练被高频节点支配”已经成立；它也可能放大稀有正常行为、降低E3高频TP的检测。实施前应在正常训练日志中核实实际贡献权重，并预注册成功条件：在独立数据的同等告警预算下，两数据集的Recall@N方向一致改善，同时报告FPR、AP和逐seed差值。只提高AUROC、只降低FP、或在已见seed0测试上调参，不算完成目标。

建议新配对实验至少3 seeds；用验证集选epoch与任何训练超参数。若只能复用已看过的E3/E5测试集，结论必须标为探索性再分析，需要另留新时间段/外部数据作确认。本轮没有实施上述训练改动，也没有产生新模型成功声明。
"""
    write(out/'optimization_priority.md',priority)

    baseline=md_table(['数据集/epoch','N / 攻击节点','TP / FP / FN / TN','Recall','MCC','AP / AUROC','阈值'],[
      [e+'/'+str({'E3':4,'E5':6}[e]),f"{s['nodes']} / {s['attacks']}",f"{s['metrics']['tp']} / {s['metrics']['fp']} / {s['metrics']['fn']} / {s['metrics']['tn']}",
       f"{s['metrics']['recall']:.6f}",f"{s['metrics']['mcc']:.6f}",f"{s['metrics']['auprc']:.6f} / {s['metrics']['auroc']:.6f}",f"{s['threshold']:.12f}"] for e,s in ds.items()])
    ranktable=md_table(['数据集/类别','数量','最小排名','中位数','P75','P90','最大排名'],[
        [x.dataset+'/'+x.group,x.n,f'{x.rank_min:,.0f}',f'{x.rank_median:,.1f}',f'{x.rank_p75:,.1f}',f'{x.rank_p90:,.1f}',f'{x.rank_max:,.0f}'] for x in ranks.itertuples()])
    top_table=md_table(['数据集','预算N','攻击TP','正常FP','Recall@N','Precision@N'],[
        [x.dataset,x.budget,x.tp,x.fp,f'{100*x.recall:.3f}%',f'{100*x.precision:.3f}%'] for x in top[top.budget.isin([100,500,1000,5000])].itertuples()])
    freqtable=md_table(['数据集','节点贡献次数','正常节点数','K5 FP','K5 FPR','K1 FP','K1独有FP'],[
        [e,x['group'],x['benign_nodes'],x['fp'],f"{100*x['fpr']:.4f}%",x['k1_fp'],x['k1_only_fp']] for e,s in ds.items() for x in s['strata']])
    group_table=md_table(['数据集/类别','贡献次数中位数','不同关系数中位数','Top5单事件分数P50/P90/P99','Top1-Top5均值中位数','Top1占和≥50%节点'],[
        [e+'/'+g,f"{v['contributions']['quantiles']['0.5']:.1f}",f"{v['relation_types']['quantiles']['0.5']:.1f}",
         '/'.join(f"{v['top5_event_score']['quantiles'][str(q)]:.3f}" for q in [.5,.9,.99]),
         f"{v['top1_minus_top5']['quantiles']['0.5']:.3f}",f"{v['top1_dominates_ge_half_sum']}/{v['nodes']}"] for e,s in ds.items() for g,v in s['groups'].items()])
    rawtable=md_table(['数据集','正式校准AP/AUROC','原joint AP/AUROC','正式/原joint Top5000攻击数','正式/原joint攻击排名中位数'],[
        [e,f"{s['metrics']['auprc']:.6f}/{s['metrics']['auroc']:.6f}",f"{rawres[e]['raw_joint_ap']:.6f}/{rawres[e]['raw_joint_auroc']:.6f}",
         f"{int(top[(top.dataset==e)&(top.budget==5000)].iloc[0].tp)}/{int(rawtop[(rawtop.dataset==e)&(rawtop.budget==5000)].iloc[0].tp)}",
         f"{s['ranks']['攻击全部']['quantiles']['0.5']:,.0f}/{rawres[e]['raw_attack_rank']['quantiles']['0.5']:,.1f}"] for e,s in ds.items()])
    relation_table=md_table(['数据集/类别','关系编号','全部贡献数','事件分数>T贡献数','进入Top5贡献数','Top5分数和'],[
        [e+'/'+g,int(x.edge_type_index),int(x.all_contributions),int(x.above_node_threshold_contributions),int(x.top5_contributions),f'{x.top5_score_sum:.2f}']
        for e in ['E3','E5'] for g in ['FP','FN'] for x in relations[(relations.dataset==e)&(relations.group==g)].nlargest(2,'top5_score_sum').itertuples()])
    report=f"""
# MSTC-PIDS E3/E5 seed0 低召回率根因诊断

日期：2026-10-28（北京时间）。性质：全量CPU事后诊断；不是独立优化验证。

## 首先回答四个问题

**Q1：攻击节点到底排在哪里？** E3攻击排名中位数为第90,085/699,295（全体前12.88%），E5为第161,003/747,452（前21.54%）；最靠前分别为25和117，最靠后为557,965和646,704。Top5000只覆盖E3 12/118、E5 6/69。存在小部分头部富集，但多数攻击没有进入可承受告警量的极尾。

**Q2：低Recall主要来自排序还是阈值？** 当前主要是**低告警预算下的异常排序能力不足**。FN分数/阈值中位数分别0.2919、0.2878；达到阈值90%的FN只有2/109和1/68。阈值迁移确实存在，正常测试FPR分别为实际验证超阈率的3.53倍和1.22倍，但不能解释多数攻击距离阈值很远。去掉校准的固定原joint对照也没有高覆盖：Top5000仅21/118和3/69。这里的“排序不足”指已观察到的完整评分链；尚不能把其因果份额全部归于某一层神经网络。

**Q3：Top-K是否系统性掩盖攻击或放大正常行为？** 存在明确关联：14个E3 FN、6个E5 FN有单事件分数超过正式节点阈值，均值聚合后未报警；FP贡献次数中位数为205/1605，约86%的FP Top5全来自目标方向。高频节点承受更多取极值机会，可解释误报结构，但多数FP并非单个极端事件支配，且95/109、62/68个FN连单事件都未超过阈值。K1改变误报结构并不稳定改善两数据集整体排序，因此不能直接认定换成max即可修复。

**Q4：下一步优先改什么？** **唯一优先方向为E：改进模型训练目标与节点异常表征，提高固定低告警预算下的头部排序。** 建议先预注册一个正常节点贡献均衡训练目标，保持时间、多尺度结构及正式后处理不变，以独立数据和多seed验证。它是待验证的训练假设，不是本轮已经证明有效的方法。完整排序为E > B > C > A > D，详见[优化方向排序](optimization_priority.md)。

## 1. 正式口径、来源和代码审计

正式结果逐项复核于 `../seed0_final_acceptance_20261028/`，不是从用户给出的四舍五入数值复制。K=5、include_dst=true、hierarchical_relation、验证LOO/测试非LOO、q=.999，严格 `score > threshold`。正式阈值全部复现，无配置变更。

{baseline}

E3连续分数直接读取原 `node_scores/node_predictions.csv`，随后对4,977,126条测试事件独立聚合，分数最大误差<1e-12，预测完全相同。E5复用 `restore_e5_frozen_postprocessing.py`，对3,965,758条验证事件和14,605,144条测试原始事件恢复校准。原测试校准表只有13,535,993条完整记录及一条截断记录；本轮再次验收完整前缀与输入原始CSV哈希，并恢复缺失部分，未运行GNN。747,452个节点的ID、K5/K1连续分数、预测摘要哈希与原冻结包完全一致；K5分数哈希为 `52987264ec81e595c9a586f3f800ba0e5e4d311505cd79fddf16d5266d91f096`。

输入哈希记录在[input_hash_verification.json](input_hash_verification.json)、[raw_joint_control.json](raw_joint_control.json)及 `e5_recovery_recheck/`；E5原窗口由复用的RawSource逐一完整SHA256核验。E3验证校准表与此前全事件分析哈希一致，节点表与事件表与最终验收清单一致。标签按原GT的UUID重新映射，验收UUID正反映射双射、缓存GT与节点标签完全一致；不采用GT CSV中历史整数ID。部分E3元数据没有旧验收清单哈希条目，记录当前SHA256，并通过完整UUID映射与正式节点标签交叉验证，未伪称存在历史哈希对照。

E5另有GT节点36763、UUID `00000000-0000-0000-0000-000000000000`，没有测试事件。单独保留在[攻击节点明细](attack_node_details.csv)的排除行，不计入69个正例、不补零、不作为第70个FN。E3全部118个GT节点均被观察到。

代码依据：`src/model.py` 的原分数为 `loss_type + .3*loss_time`，时间分量为双端平均；`src/mstc/calibration.py` 使用含并列的经验右尾概率并转为 `-log(p)`；`src/mstc/aggregation.py` 将同一事件完整贡献给源和目标节点，再对实际有效的min(K,n)个最高分求均值，不补零。自环在协议中算双份，本次两数据集实际自环数均为0。`src/mstc/thresholding.py` 只使用验证节点分数的numpy分位数，无按节点事件次数或方向校正。代码实现与恢复协议一致，没有发现可以解释低Recall的排名、标签错位或Top5均值实现错误。

## 2. 阶段A：全量攻击排名

排名确定性规则：分数降序，同分按数值node_id升序；1为最高异常。`rank_top_percent=100*rank/N`，越小越异常，**不是CDF百分位越大越异常**。每个攻击节点保留同分块的最好/最坏排名；预算表同时记录并列时TP可达区间。以下分位数采用numpy线性分位数，可能为小数。

{ranktable}

{top_table}

E3只有12/118（10.17%）、E5只有10/69（14.49%）攻击位于Top1%；Top0.1%只有5和1个。相对随机的1%覆盖仍分别富集约10.17和14.49倍，不能说模型毫无异常信号；但89.83%/85.51%的攻击不在最高1%极尾。AP分别约为随机正例比例的16.71/7.37倍，而绝对精度仍极低，说明AUROC不能代表头部告警质量。

扩大到Top5%（34,965/37,373个告警）也仅覆盖26/118、24/69；到Top10%（69,930/74,746个告警）仅覆盖37/118、27/69。全部攻击节点的连续分数、确定排名和并列区间在CSV中逐节点列出，不只保存汇总。

## 3. 阶段B：漏检与阈值距离

分组标准在主扫描前固定、与标签无关：先以score/T>=.9且未报警判为“接近阈值”；再将其余排名<=ceil(1%*N)者归“其余Top1%”；剩余为“其余较低分”。这是描述用互斥分组，不能将最后一组每个节点都解释为统计上显著低分；连续分布另行保留。

| 项目 | E3 | E5 |
| --- | --- | --- |
| FN数量 | 109 | 68 |
| 接近阈值FN | 2（1.83%） | 1（1.47%） |
| 其余Top1% FN | 1 | 8 |
| 其余较低分FN | 106 | 59 |
| FN score/T的P25 / P50 / P75 / P90 | .1009 / .2919 / .3984 / .5785 | .1829 / .2878 / .5629 / .6678 |
| FN score/T最大值 | .9391 | .9171 |
| FN score-T中位数 | -3.8372 | -5.4531 |
| FN排名中位数 | 100,989 | 171,407.5 |

**降低门槛有有限覆盖收益，但不是足量补救。** E3从正式2524个告警（9 TP/2515 FP）扩到Top5000只多3个攻击、增加2473个正常告警（每新增攻击约824个正常告警）；E5从899个（1/898）扩到Top5000多5个攻击、增加4096个正常告警（约819个/新增攻击）。到Top10000，E3多5个攻击却多7471个正常告警；E5多12个攻击、多9089个正常告警。后者的中等预算覆盖提升较明显，但仍漏56/69个攻击。

固定预算给的是按当前分数排序可覆盖的理论节点集合，未把node_id并列裁决改成新的正式阈值。固定分位数的严格超阈告警量及并列差异保存在[固定分位数诊断](fixed_percentile_diagnostics.csv)：例如测试Top5%预算为34,965/37,373，但严格大于测试P95分数只有33,775/36,007个告警，不能把二者混用。所有预定预算的实际TP并列上下界相同，本文结论不依赖同分node_id偏好。

跨数据集共同点是多数FN分数仅阈值约三成；不同点是E3正式告警量已达2524、测试正常尾漂移显著，E5正式告警899、Top1%还能额外覆盖9个攻击，但更深部仍有大量漏检。因此不能统一解释成“q=.999太严格”。

## 4. 阶段C：误报归因与聚合机制

贡献次数是事件对节点的端点贡献次数；双端分别计数。不同关系数指 `edge_type_index` 数量，不等同于含源/目标节点类型的完整关系三元组数。关系三元组参考另外在 `*_attack_reference_diagnostics.csv` 中列出。

{group_table}

{freqtable}

贡献>100的正常节点仅E3 5793/699177（0.83%）、E5 8747/747383（1.17%），却产生1737/2515（69.07%）、741/898（82.52%）FP。>1000组FPR达48.08%/24.43%，贡献<=5组仅0.00497%/0.04012%。频率与正常节点分数Spearman相关为0.2790/0.4353；这是显著的结构关联，未控制关系类型、节点类型与时间跨度，不能当作频率因果效应。E3 FN有79/109贡献<=5；E5则为23/68，频率关联程度不同。

源/目标贡献同样有偏向：FP的源贡献比例中位数0.1210/0.00310，Top5目标占比中位数均为1；全部Top5来自目标端的FP为2168/2515（86.20%）、775/898（86.30%），目标占多数为2255/2515、787/898。TP的Top5目标占比中位数均为.2，但E5 TP仅1个，不能推广为稳定模式。FN的目标占比中位数.6/.8，简单禁用目标贡献可能同时伤害攻击信号。图中部分系统调用关系预处理会反向，因此这里“目标”是保存图事件方向，不能直接等同系统调用中的客体。

关系贡献最高的项目如下；完整TP/FP/FN/TN及所有关系在[关系贡献汇总](relation_contribution_summary.csv)。事件分数>T只是用正式节点阈值做诊断计数，不是正式事件分类器；同一事件可计入两个节点类别，表内不能把端点贡献数当作独立事件数。

{relation_table}

E3 FP中关系2/3贡献了绝大部分Top5分数和（39474.27+35383.57）；E5同样为关系2/3（21049.91+9208.21）。关系体量、端点类型与频率同时变化，缺少事件恶意标签，不能直接称某种关系检测“错误”。

**Top5存在稀释，但也抑制大量正常尖峰。** E3有14/109、E5有6/68个FN的最高事件分数>T；TN也有22030和3109个这样的节点。若只看到攻击尖峰就改max，会忽略这些正常尖峰。Top1占有效Top5分数和>=50%的FP仅26/2515、91/898；多数FP有多个高分贡献，而非单个离群事件。FN分别69/109、9/68满足该比例，但低贡献节点天然有高Top1占比（只有一次贡献时为100%），不能只凭这一比例断言攻击由极端事件主导。

复核既有冻结K1方案：E3阈值8.28462，TP8/FP175，较K5丢掉6个原TP、救回5个FN；E5阈值9.45324，TP3/FP1003，保留原TP并救回2个FN。E5的6-10贡献组FP由6升为710，其中708个为K1独有；>1000组却由691降为47。K1需要更高的验证阈值，因此比较并非把max和Top5套用同一个阈值。E3 AP由.002820升至.004318，E5却由.000680降至.000545，**不能把单数据集FP下降或少数TP增加等同统一改进**。

机制推测：固定Top5对事件更多的节点提供更多抽取高阶次序统计量的机会；双端赋值让少数目标实体汇集不同来源的高分，低频攻击则常只有一两个高分贡献被其他贡献拉低。上述代码语义与观测吻合，但没有固定事件分布的频率干预实验，尚非因果证明。

## 5. 验证正常分布与测试正常分布

| 指标 | E3验证 / 测试正常 | E5验证 / 测试正常 |
| --- | --- | --- |
| P50分数 | .1789 / .5665 | .9719 / 1.2395 |
| P99分数 | 3.7471 / 3.9701 | 4.8705 / 4.9147 |
| P99.9分数 | 5.4191 / 6.6352 | 7.6564 / 7.7021 |
| 正式阈值实际超阈率 | .10185% / .35971% | .09834% / .12015% |
| 测试/验证超阈率倍数 | 3.5318 | 1.2218 |
| 按测试频率构成加权的验证超阈率 | .08941% | .08114% |

验证集只有正常训练协议假设，没有逐事件攻击标签。与测试GT相同身份在验证中有2/1个节点，排除这些身份后的分布也保存在JSON，但身份相同不代表验证期恶意，未据此改变正式阈值。实际尾超阈率受离散与严格>影响，不一定正好为.1%。

即使以测试正常节点的频率组构成重加权验证超阈率，仍低于测试实际.35971%/.12015%，因此**频率组占比改变不能单独解释漂移**；组内分数分布、关系构成与时间跨度仍可能变化。这只是粗频率分层的描述性标准化。KS距离.3355/.2433记录整体分布差异，样本存在图依赖，未把大样本p值当作机制证据。E3中部与极尾并非统一平移，不能仅用平均数判断尾部是否稳定。

## 6. 固定原joint对照、校准上限与时间分量

补充扫描保持K5、双端和原训练分数定义，仅移除校准，直接分析连续排名；没有拟合新正式阈值或搜索分数组合。E3原joint AP/AUROC与既有探索包精确一致；E5原始窗口逐一哈希核验。该对照用于区分H1与校准混淆，不能依据本轮测试选择新正式方案。

{rawtable}

E3校准提高总体AUROC与攻击中位排名，却损失一部分头部覆盖；E5校准总体和Top5000更好，原joint并未救回大量攻击。故“分层校准必然错误”和“只优化阈值即可恢复”均不成立。

经验尾概率有有限参考上限：事件原始分数严格大于参考最大值时p=1/(n+1)，分数上限log(n+1)。E3有12个攻击节点至少出现一次饱和贡献，E5有2个。E5节点32184/32186/32188的全部涉及参考规模最多1766，理论事件上限7.4770低于节点阈值7.6564；但它们实际最高事件仅1.7867、没有饱和，原joint排名约48.36%-49.46%。这是真实的潜在校准盲区，尚不能解释这些节点当前的低原始异常分数。E3没有全部事件理论上限均低于节点阈值的攻击节点。详见[原joint逐攻击控制](raw_joint_attack_control.csv)及三元组参考CSV。

实际保留了loss_type、loss_time_src/dst及其平均、score_raw，以及时间目标/预测桶编号；不含完整logits、连续实际gap、每尺度embedding或gate。全量事件的joint公式最大残差E3 5.01e-7/E5 9.54e-7，时间双端平均残差2.38e-7/4.77e-7，与保存的float32计算精度一致。

正式Top5被选事件的原始损失中，按分数和计算的`.3*time/raw`比例：E3 TP/FP/FN/TN为26.91%/43.24%/36.23%/50.69%；E5为14.80%/23.42%/46.20%/61.49%。这是**按校准分数选出的事件样本**，不能推断时间损失具有相同的校准分数可加贡献，也不能拿它当时间模块消融收益。攻击相邻事件也未必恶意。本轮未验证时间桶预测混淆的攻击因果意义；gate饱和、尺度失活、特征可分性不可由保留产物检验。最小补救是在未来已授权的新训练/推理中只额外保存每尺度可用性、gate与logit/embedding分层摘要；本轮不补跑模型。

## 7. 阶段D/E：假设判定与后续方案

结构化的支持、反对和不可验证项见[hypothesis_evidence.json](hypothesis_evidence.json)。H1证据最强，但只说明实际可观测排序不足，尚无神经表示内部的因果分解；H3在误报频率和方向结构上证据明确，对FN只解释少数尖峰；H2在E3明显影响FPR，在E5程度弱，且没有大量近阈值FN。

**只优先实施E**。B列第二，C第三，A第四，D第五；各项重训需求、复用能力、计算/存储量级、风险和独立验证条件均已在[optimization_priority.md](optimization_priority.md)中列明。本轮不运行新优化、不修改配置、不将已见测试结果反向用于正式阈值选取。正常节点贡献均衡训练是一个需要预注册和证伪的首轮假设，可能伤害高频攻击，不能从本轮频率相关性宣称其一定有效。

## 8. 执行与验收

主CPU任务用nohup运行并保留管理会话；首次后台启动未存活，未进入分析、无退出状态，空日志与PID仍保留。随后 `run_cpu_02.sh` 正常退出0，耗时170.65秒，峰值RSS约3.20GiB；原joint对照 `run_raw_control.sh` 退出0，31.42秒、峰值0.67GiB。日志、PID及exitcode均在本目录。未关闭当前SSH会话。

初始系统盘可用约{before['root_free']/1024**3:.2f}GiB、数据盘约{before['data_free']/1024**3:.2f}GiB；主任务结束约{verify['resources']['root_free_after']/1024**3:.2f}/{verify['resources']['data_free_after']/1024**3:.2f}GiB。最终输出占用在 `final_verification.json` 中记录，并强制小于20MiB。不生成全节点或事件中间CSV。

CPU测试42项通过（6.42秒），其中新增5项验证确定性并列、手算TopN、自环双端及Top5正式聚合一致性。交付表再次按攻击排名手工公式独立核算全部预算，核对188行攻击明细（187个正式攻击节点+1个排除节点）、阈值/标签/排名/贡献守恒，完整TP/FP/FN/TN互斥覆盖、E5全部分数摘要和正式指标一致性。`git diff --check`通过。

本轮新实现位于 `scripts/diagnose_e3_e5_low_recall.py`、`scripts/diagnose_raw_score_control.py`、`scripts/package_low_recall_diagnosis.py`，测试位于 `tests/test_low_recall_diagnosis.py`；原训练、GNN、预处理均未启动。正式run全部文件大小/mtime保持不变，旧分析小文件SHA256逐一不变，已有tracked diff保持不变；无删除、安装依赖、commit或push。

复现请使用新的输出目录执行主脚本和原joint脚本，再运行打包脚本；所有诊断文件排他创建，禁止覆盖已有包。测试命令及执行环境见本目录启动脚本和 `final_verification.json`。原始E3历史checkpoint选择曾观察测试结果、E5为CPU恢复指标而非原评估流水线成功等限制，沿用最终验收包，不因本轮诊断而消除。

## 9. 交付索引与结论边界

- [attack_rank_summary.csv](attack_rank_summary.csv)：E3/E5全部攻击、TP和FN的排名分位数与尾部覆盖。
- [attack_node_details.csv](attack_node_details.csv)：逐攻击节点分数、排名、并列范围、阈值距离、频率、关系数、方向与Top5分数；无事件GT单列。
- [topn_metrics.csv](topn_metrics.csv)：固定预算覆盖、Precision、FPR、截止并列上下界。
- [false_positive_analysis.csv](false_positive_analysis.csv)：四类节点及频率分层、K1误报结构。
- `e3_diagnostics.json` / `e5_diagnostics.json`：完整连续分布、四类节点的Top5/方向/损失统计与指标验收。
- `relation_contribution_summary.csv`、`fixed_percentile_diagnostics.csv`：全关系贡献与固定百分位描述。
- `raw_joint_*` / `*_attack_reference_diagnostics.csv`：无校准对照及参考规模/饱和诊断。
- [hypothesis_evidence.json](hypothesis_evidence.json)、[optimization_priority.md](optimization_priority.md)、`final_verification.json`：假设证据、唯一下一步建议与执行验收。

所有优化判断使用了本轮测试标签，属于诊断研究。不能把已见E3/E5 seed0上的观察、既有探索对照、或下一步候选方案称为独立验证成功；需要新的未参与诊断数据与多seed重复。
"""
    write(out/'报告.md',report)
    scripts=[Path(__file__),Path(d.__file__),Path(__file__).with_name('diagnose_raw_score_control.py'),d.ROOT/'tests/test_low_recall_diagnosis.py']
    size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    assert size<20*1024**2
    final=dict(状态='PASS',日期='2026-10-28',用途='测试标签事后诊断，不是独立优化验证',
        唯一优先推荐='E：改进模型训练目标与节点异常表征',
        验证=checks,排除节点=dict(dataset='E5',node_id=36763,uuid='00000000-0000-0000-0000-000000000000',included_in_denominator=False),
        测试=dict(passed=42,elapsed_seconds=6.42,added_tests=5,log='tests_cpu.log',git_diff_check_exitcode=diff.returncode,
            table_independent_checks='所有攻击行和全部固定预算通过独立公式复算',supplement_source_hashes_matching_acceptance=len(raw_matches)),
        执行=dict(main_pid=(out/'run_cpu_02.pid').read_text().strip(),main_exitcode=0,
            raw_control_pid=(out/'run_raw_control.pid').read_text().strip(),raw_control_exitcode=0,
            first_launcher='首次nohup未存活，未进入分析；无exitcode，不计为成功运行；保留空日志和PID',
            no_training=True,no_gnn_inference=True,no_preprocessing=True,no_dependency_install=True,no_delete=True,no_commit_push=True,
            formal_run_stats_unchanged=True,prior_analysis_hashes_unchanged=True,existing_tracked_diff_unchanged=True),
        资源=dict(main=verify['resources'],raw_control=raw['resource'],output_bytes_before_final_json=size,
            output_limit_bytes=20*1024**2,only_small_csv_json_markdown_and_execution_logs=True),
        来源=dict(input_hash_verification='input_hash_verification.json',e5_all_score_seal_exact=True,
            original_joint_control_hashes_verified=True,git_head=before['git_head']),
        限制=['未独立检验任何新优化方案','无事件恶意标签，节点端点贡献不等于独立恶意事件',
            '无完整logit/连续时间gap/多尺度gate/embedding，无法检验内部表征机制',
            'E3部分元数据无旧验收清单哈希，已当前哈希并以UUID映射/正式标签交叉核验',
            '正式产物以stat未变化及读取输入SHA256验证；未重哈希所有无关大型checkpoint/图缓存'],
        代码哈希={str(p.relative_to(d.ROOT)):d.e3.sha(p) for p in scripts},
        交付哈希={str(p.relative_to(out)):d.e3.sha(p) for p in out.rglob('*') if p.is_file()})
    d.e3.write_json(out/'final_verification.json',final)
    assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<20*1024**2
    print(json.dumps(dict(状态='PASS',输出字节=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),测试通过=42,唯一优先='E'),ensure_ascii=False))


if __name__=='__main__':main(Path(sys.argv[1]))
