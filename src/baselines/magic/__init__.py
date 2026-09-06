"""
MAGIC External Baseline - Unified Protocol Adapter

本模块实现 MAGIC 外部基线的统一协议适配器。

本模块是 M3/M4 实现：
- M3: MAGIC Canonical Input Adapter
- M4: MAGIC Unified Causal Protocol

关键设计原则：
1. 与 DGL 解耦 - Neutral Graph Contract
2. Train-only type vocabulary
3. Deterministic ordering: (timestamp, global_event_index)
4. Split isolation
5. Ground-truth boundary

禁止事项：
- 不实现 MAGIC GAT/GMAE 模型
- 不安装 DGL/Torch 1.12
- 不修改 /opt/magic-upstream
- 不读取 ground truth labels
"""

from __future__ import annotations

# M3: Neutral Graph Contract
from .contracts import (
    SplitType,
    NodeRecord,
    EdgeRecord,
    GraphSnapshot,
    TypeVocabulary,
    NeutralGraphContract,
    SnapshotScores,
    ThresholdConfig,
    UNKNOWN_TYPE,
    MAGIC_REVERSED_EDGE_PREFIXES,
    MAGIC_SIMPLE_GRAPH_POLICY,
)

# M3: MAGIC Input Adapter
from .input_adapter import MAGICInputAdapter, build_magic_input

# M4: Unified Causal Protocol
from .protocol import (
    compute_validation_quantile_threshold,
    merge_snapshot_scores,
    build_causal_snapshots,
    ThresholdProtocol,
)

__all__ = [
    # Contracts
    "SplitType",
    "NodeRecord",
    "EdgeRecord",
    "GraphSnapshot",
    "TypeVocabulary",
    "NeutralGraphContract",
    "SnapshotScores",
    "ThresholdConfig",
    "UNKNOWN_TYPE",
    "MAGIC_REVERSED_EDGE_PREFIXES",
    "MAGIC_SIMPLE_GRAPH_POLICY",
    # M3 Adapter
    "MAGICInputAdapter",
    "build_magic_input",
    # M4 Protocol
    "compute_validation_quantile_threshold",
    "merge_snapshot_scores",
    "build_causal_snapshots",
    "ThresholdProtocol",
]
