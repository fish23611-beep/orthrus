"""
MAGIC Canonical Input Adapter

本模块实现 MAGIC Input Adapter，将 canonical THEIA artifacts 转换为 neutral graph contract。

设计依据：
- 冻结合同: docs/MAGIC_BASELINE_ENVIRONMENT_CONTRACT.md
- 审计报告: docs/MAGIC_BASELINE_INTEGRATION_AUDIT.md
- 当前项目 canonical artifacts: src/data_utils.py
- M9-P3C.1R 审计报告

关键设计原则：
1. Neutral Graph Contract 与 DGL 解耦
2. Train-only type vocabulary
3. Deterministic ordering: (timestamp, global_event_index)
4. Split isolation: train/val/test 显式分离
5. E3/E5 共用同一个 adapter
6. 支持多种 profile：legacy_upstream / orthrus_unified

支持的 Profile：
- legacy_upstream: 上游兼容行为，READ/RECV/LOAD 反转，first-edge dedup
- orthrus_unified: ORTHRUS production graph 协议，保持因果方向，保留所有边

Python 3.8 兼容性：
- 不使用类型注解中的内置类型 (使用 typing)
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
)

from .contracts import (
    SplitType,
    EdgeRecord,
    NeutralGraphContract,
    TypeVocabulary,
    GraphSnapshot,
    UNKNOWN_TYPE,
    ProfileType,
    ProfileConfig,
    PROFILE_CONFIGS,
    DirectionPolicy,
    DuplicateEdgePolicy,
)


class MAGICInputAdapter:
    """
    MAGIC Canonical Input Adapter.

    将 canonical THEIA artifacts 转换为 neutral MAGIC graph specification。
    支持 THEIA_E3 和 THEIA_E5。

    支持两种 profile：
    - legacy_upstream: 上游兼容行为
    - orthrus_unified: ORTHRUS production graph 协议

    核心职责：
    1. 读取 canonical artifacts
    2. 应用 profile 指定的 direction 和 dedup 策略
    3. 构建 train-only type vocabulary
    4. 生成 neutral graph contract

    禁止行为：
    - 不读取 ground truth labels
    - 不扩展 train vocabulary
    - 不假设 validation 是 benign-only
    """

    def __init__(
        self,
        dataset: str,
        train_records: List[Dict],
        val_records: Optional[List[Dict]] = None,
        test_records: Optional[List[Dict]] = None,
        profile: str = "legacy_upstream",
    ):
        """
        Initialize the adapter.

        Args:
            dataset: Dataset name (THEIA_E3 or THEIA_E5)
            train_records: List of canonical edge records for training
            val_records: List of canonical edge records for validation
            test_records: List of canonical edge records for testing
            profile: Adapter profile ("legacy_upstream" or "orthrus_unified")
        """
        self.dataset = dataset
        self.train_records = train_records or []
        self.val_records = val_records or []
        self.test_records = test_records or []

        # Profile configuration
        profile_type = ProfileType(profile) if isinstance(profile, str) else profile
        if profile_type not in PROFILE_CONFIGS:
            raise ValueError(
                f"Unknown profile: {profile}. "
                f"Valid profiles: {[p.value for p in PROFILE_CONFIGS.keys()]}"
            )
        self._profile = PROFILE_CONFIGS[profile_type]
        self._profile_name = profile_type

        # Type vocabulary - fitted on train only
        self._vocabulary: Optional[TypeVocabulary] = None

        # Graph contract
        self._contract: Optional[NeutralGraphContract] = None

        # Split isolation check
        self._fitted = False

    @property
    def profile(self) -> ProfileConfig:
        """Get the current profile configuration."""
        return self._profile

    @property
    def profile_name(self) -> ProfileType:
        """Get the current profile name."""
        return self._profile_name

    def fit(self) -> "MAGICInputAdapter":
        """
        Fit the adapter on training data.

        This builds the train-only type vocabulary.
        Must be called before transform.

        Returns:
            self
        """
        # Extract node types and edge types from training data only
        node_types = {}
        edge_types = {}

        for record in self.train_records:
            src = record.get("src")
            dst = record.get("dst")
            src_type = record.get("src_type")
            dst_type = record.get("dst_type")
            edge_type = record.get("edge_type")

            # Collect node types
            if src:
                node_types[src] = src_type
            if dst:
                node_types[dst] = dst_type

            # Collect edge types
            if src and dst and edge_type is not None:
                edge_types[edge_type] = edge_type

        # Build vocabulary from training data
        self._vocabulary = TypeVocabulary()
        self._vocabulary = self._vocabulary.fit(node_types, edge_types)

        self._fitted = True
        return self

    def transform(
        self,
        records: List[Dict],
        split: SplitType,
    ) -> NeutralGraphContract:
        """
        Transform records to neutral graph contract.

        Args:
            records: List of canonical edge records
            split: The split type (train/val/test)

        Returns:
            NeutralGraphContract

        Raises:
            RuntimeError: If vocabulary is not fitted
        """
        if not self._fitted:
            raise RuntimeError(
                "Vocabulary not fitted. Call fit() before transform()."
            )

        contract = NeutralGraphContract()

        # Sort by (timestamp, global_event_index)
        ordered_records = sorted(records, key=self._record_order_key)

        # Track seen (src, dst) pairs for dedup if profile requires it
        seen_pairs: Set[Tuple[str, str]] = set()

        # Count edges for conservation check
        edge_count_before = len(ordered_records)

        for record in ordered_records:
            # Extract fields from canonical artifact
            src = record.get("src")
            dst = record.get("dst")
            timestamp = record.get("t") or record.get("timestamp", 0)
            src_type = record.get("src_type")
            dst_type = record.get("dst_type")
            edge_type = record.get("edge_type")
            global_event_index = record.get("global_event_index", 0)

            if not src or not dst:
                continue

            # Apply profile-specific direction rule
            adjusted_src, adjusted_dst = self._apply_direction_rule(
                src, dst, edge_type
            )
            is_reversed = (adjusted_src, adjusted_dst) == (dst, src) and (
                src != dst
            )
            adjusted_src_type = dst_type if is_reversed else src_type
            adjusted_dst_type = src_type if is_reversed else dst_type

            pair = (adjusted_src, adjusted_dst)

            # Apply profile-specific dedup policy
            if self._profile.should_dedup():
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

            # Add nodes
            contract.add_node(adjusted_src, adjusted_src_type, split)
            contract.add_node(adjusted_dst, adjusted_dst_type, split)

            # Create edge record
            edge = EdgeRecord(
                src=adjusted_src,
                dst=adjusted_dst,
                edge_type=edge_type,
                timestamp=int(timestamp),
                global_event_index=int(global_event_index),
                split=split,
            )
            contract.add_edge_preserve_index(edge)

        # Track unseen types
        for record in records:
            src_type = record.get("src_type")
            dst_type = record.get("dst_type")
            edge_type = record.get("edge_type")

            # Check node types
            if src_type is not None:
                idx, seen = self._vocabulary.transform_node_type(src_type)
                if not seen and src_type not in contract.unseen_node_types:
                    contract.unseen_node_types[src_type] = 0
                elif not seen:
                    contract.unseen_node_types[src_type] += 1

            if dst_type is not None:
                idx, seen = self._vocabulary.transform_node_type(dst_type)
                if not seen and dst_type not in contract.unseen_node_types:
                    contract.unseen_node_types[dst_type] = 0
                elif not seen:
                    contract.unseen_node_types[dst_type] += 1

            # Check edge type
            if edge_type is not None:
                idx, seen = self._vocabulary.transform_edge_type(edge_type)
                if not seen and edge_type not in contract.unseen_edge_types:
                    contract.unseen_edge_types[edge_type] = 0
                elif not seen:
                    contract.unseen_edge_types[edge_type] += 1

        # Attach vocabulary
        contract.type_vocabulary = self._vocabulary

        # Set duplicate policy based on profile configuration
        # This is the public contract field that backend will use
        contract.duplicate_policy = self._profile.duplicate_policy

        return contract

    def fit_transform(self) -> Tuple[
        NeutralGraphContract,
        NeutralGraphContract,
        NeutralGraphContract,
    ]:
        """
        Fit vocabulary on train and transform all splits.

        Returns:
            Tuple of (train_contract, val_contract, test_contract)
        """
        self.fit()

        train_contract = self.transform(self.train_records, SplitType.TRAIN)
        val_contract = self.transform(self.val_records, SplitType.VALIDATION)
        test_contract = self.transform(self.test_records, SplitType.TEST)

        return train_contract, val_contract, test_contract

    def _record_order_key(self, record: Mapping) -> Tuple[int, int]:
        """Generate sort key for deterministic ordering."""
        timestamp = record.get("t")
        if timestamp is None:
            timestamp = record.get("timestamp", 0)
        return (
            int(timestamp or 0),
            int(record.get("global_event_index", 0)),
        )

    def _apply_direction_rule(
        self,
        src: str,
        dst: str,
        edge_type: str,
    ) -> Tuple[str, str]:
        """
        Apply profile-specific direction rule.

        Args:
            src: Source node
            dst: Destination node
            edge_type: Edge type

        Returns:
            Tuple of (adjusted_src, adjusted_dst)
        """
        if self._profile.should_reverse(edge_type):
            # Reverse the direction
            return dst, src
        return src, dst

    @property
    def vocabulary(self) -> Optional[TypeVocabulary]:
        """Get the fitted vocabulary."""
        return self._vocabulary

    @property
    def is_fitted(self) -> bool:
        """Check if vocabulary is fitted."""
        return self._fitted


def build_magic_input(
    dataset: str,
    canonical_artifacts: Dict[str, List[Dict]],
    profile: str = "legacy_upstream",
) -> Tuple[NeutralGraphContract, NeutralGraphContract, NeutralGraphContract]:
    """
    Build MAGIC input from canonical artifacts.

    Args:
        dataset: Dataset name (THEIA_E3 or THEIA_E5)
        canonical_artifacts: Dict with 'train', 'validation', 'test' keys
            containing lists of canonical edge records
        profile: Adapter profile ("legacy_upstream" or "orthrus_unified")

    Returns:
        Tuple of (train_contract, val_contract, test_contract)
    """
    adapter = MAGICInputAdapter(
        dataset=dataset,
        train_records=canonical_artifacts.get("train", []),
        val_records=canonical_artifacts.get("validation", []),
        test_records=canonical_artifacts.get("test", []),
        profile=profile,
    )
    return adapter.fit_transform()


def validate_no_future_leakage(
    snapshots: List[GraphSnapshot],
    records: List[Dict],
) -> bool:
    """
    Validate that no future events leak into earlier snapshots.

    For each snapshot, verify that only events with timestamp <= snapshot_end
    are included.

    Args:
        snapshots: List of causal snapshots
        records: All records in chronological order

    Returns:
        True if no leakage, False otherwise
    """
    for snapshot in snapshots:
        for record in records:
            timestamp = record.get("t") or record.get("timestamp", 0)
            if timestamp > snapshot.snapshot_end:
                # This record should not be in the snapshot
                src = record.get("src")
                dst = record.get("dst")
                if src in snapshot.nodes or dst in snapshot.nodes:
                    return False
    return True
