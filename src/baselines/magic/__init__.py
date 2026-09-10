"""
MAGIC External Baseline - Unified Protocol Adapter

本模块实现 MAGIC 外部基线的统一协议适配器。

本模块是 M3-M6 实现：
- M3: MAGIC Canonical Input Adapter
- M4: MAGIC Unified Causal Protocol
- M5: MAGIC Raw Score Export
- M6: MAGIC Unified Evaluator Integration

关键设计原则：
1. 与 DGL 解耦 - Neutral Graph Contract
2. Train-only type vocabulary
3. Train-only KNN reference / normalization statistics
4. Deterministic ordering: (timestamp, global_event_index)
5. Split isolation
6. Ground-truth boundary
7. Label-independent scores and predictions

禁止事项：
- 不实现 MAGIC GAT/GMAE 模型
- 不安装 DGL/Torch 1.12
- 不修改 /opt/magic-upstream
- 不读取 ground truth labels (仅在最终 metrics 阶段)
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
    ProfileType,
    ProfileConfig,
    DirectionPolicy,
    DuplicateEdgePolicy,
    LEGACY_UPSTREAM_CONFIG,
    ORTHRUS_UNIFIED_CONFIG,
)

# M3: MAGIC Input Adapter
from .input_adapter import MAGICInputAdapter, build_magic_input

# M3.5: ORTHRUS Predecessor (M9-P3C.2)
from .orthrus_predecessor import (
    OrthrusPredecessor,
    OrthrusPredecessorError,
    MissingRequiredFieldError,
    is_verified_empty_day_marker,
)

# M4: Unified Causal Protocol
from .protocol import (
    compute_validation_quantile_threshold,
    merge_snapshot_scores,
    build_causal_snapshots,
    ThresholdProtocol,
)

# M5: Raw Score Export
from .scoring import (
    MAGICEntityScorer,
    NodeScoreRecord,
    NodeIdentityMap,
    MAGIC_K_NEIGHBORS,
    MAX_TRAIN_REFERENCE_SAMPLES,
    compute_snapshot_scores,
    merge_and_score_snapshots,
)

# M6: Unified Evaluator
from .evaluator import (
    MagicEvaluator,
    MagicRunResult,
    fit_magic_threshold,
    predict_magic,
    compute_magic_metrics,
    load_magic_ground_truth,
    EvaluationUniverseError,
    write_predictions_csv,
    write_metrics_json,
    VALIDATION_BENIGN_STATUS,
    verify_label_independence,
)

# M9-P1: pinned upstream real-backend bridge (DGL remains lazy)
from .real_backend import (
    MAGICEmbeddingBatch,
    MAGICModelConfig,
    MAGICPreparedGraph,
    MAGICRealBackend,
    verify_upstream_identity,
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
    "ProfileType",
    "ProfileConfig",
    "DirectionPolicy",
    "DuplicateEdgePolicy",
    "LEGACY_UPSTREAM_CONFIG",
    "ORTHRUS_UNIFIED_CONFIG",
    # M3 Adapter
    "MAGICInputAdapter",
    "build_magic_input",
    # M3.5 ORTHRUS Predecessor
    "OrthrusPredecessor",
    "OrthrusPredecessorError",
    "MissingRequiredFieldError",
    "is_verified_empty_day_marker",
    # M4 Protocol
    "compute_validation_quantile_threshold",
    "merge_snapshot_scores",
    "build_causal_snapshots",
    "ThresholdProtocol",
    # M5 Scoring
    "MAGICEntityScorer",
    "NodeScoreRecord",
    "NodeIdentityMap",
    "MAGIC_K_NEIGHBORS",
    "MAX_TRAIN_REFERENCE_SAMPLES",
    "compute_snapshot_scores",
    "merge_and_score_snapshots",
    # M6 Evaluator
    "MagicEvaluator",
    "MagicRunResult",
    "fit_magic_threshold",
    "predict_magic",
    "compute_magic_metrics",
    "load_magic_ground_truth",
    "EvaluationUniverseError",
    "write_predictions_csv",
    "write_metrics_json",
    "VALIDATION_BENIGN_STATUS",
    "verify_label_independence",
    # M9-P1 Real Backend
    "MAGICEmbeddingBatch",
    "MAGICModelConfig",
    "MAGICPreparedGraph",
    "MAGICRealBackend",
    "verify_upstream_identity",
]
