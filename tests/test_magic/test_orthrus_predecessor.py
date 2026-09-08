"""
M9-P3C.2 ORTHRUS Predecessor and Profile Adapter Tests

测试 M9-P3C.2 实现：
1. OrthrusPredecessor 基本转换
2. 必需字段验证
3. 确定性排序
4. 平行边保留
5. 边数守恒
6. Adapter profile 行为
7. Direction semantics
8. Train-only vocabulary
9. Dynamic feature dimensions
"""
from __future__ import annotations

import pytest
import torch
import networkx as nx

from src.baselines.magic.orthrus_predecessor import (
    OrthrusPredecessor,
    OrthrusPredecessorError,
    MissingRequiredFieldError,
    is_verified_empty_day_marker,
)
from src.baselines.magic.input_adapter import MAGICInputAdapter
from src.baselines.magic.contracts import (
    SplitType,
    ProfileType,
)


# =============================================================================
# OrthrusPredecessor Basic Tests
# =============================================================================

class TestOrthrusPredecessorBasic:
    """基本转换功能测试."""

    def test_convert_nx_to_canonical_basic(self):
        """测试基本 nx.MultiDiGraph 转换."""
        g = nx.MultiDiGraph()
        g.add_node("A", node_type="subject")
        g.add_node("B", node_type="file")
        g.add_edge("A", "B", label="EVENT_WRITE", time=1000, event_uuid="uuid-1")

        pred = OrthrusPredecessor(dataset="THEIA_E3")
        records = pred.convert_nx_to_canonical(g)

        assert len(records) == 1
        assert records[0]["src"] == "A"
        assert records[0]["dst"] == "B"
        assert records[0]["src_type"] == "subject"
        assert records[0]["dst_type"] == "file"
        assert records[0]["edge_type"] == "EVENT_WRITE"
        assert records[0]["timestamp"] == 1000
        assert records[0]["event_uuid"] == "uuid-1"
        assert records[0]["global_event_index"] == 0

    def test_multiple_edges(self):
        """测试多条边转换."""
        g = nx.MultiDiGraph()
        g.add_node("A", node_type="subject")
        g.add_node("B", node_type="file")
        g.add_edge("A", "B", label="EVENT_WRITE", time=1000, event_uuid="uuid-1")
        g.add_edge("B", "A", label="EVENT_READ", time=2000, event_uuid="uuid-2")

        pred = OrthrusPredecessor(dataset="THEIA_E3")
        records = pred.convert_nx_to_canonical(g)

        assert len(records) == 2
        # 按 timestamp 排序
        assert records[0]["edge_type"] == "EVENT_WRITE"
        assert records[1]["edge_type"] == "EVENT_READ"

    def test_deterministic_order(self):
        """测试确定性排序：相同输入必须产生相同输出."""
        g = nx.MultiDiGraph()
        g.add_node("A", node_type="subject")
        g.add_node("B", node_type="file")
        g.add_edge("A", "B", label="EVENT_WRITE", time=2000, event_uuid="uuid-2")
        g.add_edge("A", "B", label="EVENT_READ", time=1000, event_uuid="uuid-1")

        pred = OrthrusPredecessor(dataset="THEIA_E3")

        # 转换两次
        records1 = pred.convert_nx_to_canonical(g)
        records2 = pred.convert_nx_to_canonical(g)

        # 必须完全一致
        assert records1 == records2
        # 排序正确
        assert records1[0]["edge_type"] == "EVENT_READ"
        assert records1[1]["edge_type"] == "EVENT_WRITE"


# =============================================================================
# OrthrusPredecessor Required Fields Tests
# =============================================================================

class TestOrthrusPredecessorRequiredFields:
    """必需字段验证测试."""

    def test_missing_node_type(self):
        """测试缺少 node_type 字段."""
        g = nx.MultiDiGraph()
        g.add_node("A")  # 缺少 node_type
        g.add_node("B", node_type="file")
        g.add_edge("A", "B", label="EVENT_WRITE", time=1000, event_uuid="uuid-1")

        pred = OrthrusPredecessor(dataset="THEIA_E3")
        with pytest.raises(MissingRequiredFieldError):
            pred.convert_nx_to_canonical(g)

    def test_missing_edge_label(self):
        """测试缺少边 label 字段."""
        g = nx.MultiDiGraph()
        g.add_node("A", node_type="subject")
        g.add_node("B", node_type="file")
        g.add_edge("A", "B", time=1000, event_uuid="uuid-1")  # 缺少 label

        pred = OrthrusPredecessor(dataset="THEIA_E3")
        with pytest.raises(MissingRequiredFieldError):
            pred.convert_nx_to_canonical(g)

    def test_missing_edge_time(self):
        """测试缺少边 time 字段."""
        g = nx.MultiDiGraph()
        g.add_node("A", node_type="subject")
        g.add_node("B", node_type="file")
        g.add_edge("A", "B", label="EVENT_WRITE", event_uuid="uuid-1")  # 缺少 time

        pred = OrthrusPredecessor(dataset="THEIA_E3")
        with pytest.raises(MissingRequiredFieldError):
            pred.convert_nx_to_canonical(g)

    def test_missing_edge_uuid(self):
        """测试缺少边 event_uuid 字段."""
        g = nx.MultiDiGraph()
        g.add_node("A", node_type="subject")
        g.add_node("B", node_type="file")
        g.add_edge("A", "B", label="EVENT_WRITE", time=1000)  # 缺少 event_uuid

        pred = OrthrusPredecessor(dataset="THEIA_E3")
        with pytest.raises(MissingRequiredFieldError):
            pred.convert_nx_to_canonical(g)


# =============================================================================
# OrthrusPredecessor Parallel Edges Tests
# =============================================================================

class TestOrthrusPredecessorParallelEdges:
    """平行边保留测试."""

    def test_preserves_all_multiedges(self):
        """测试保留所有平行边."""
        g = nx.MultiDiGraph()
        g.add_node("A", node_type="subject")
        g.add_node("B", node_type="file")
        # 添加多条平行边
        g.add_edge("A", "B", label="EVENT_OPEN", time=1000, event_uuid="uuid-1")
        g.add_edge("A", "B", label="EVENT_READ", time=2000, event_uuid="uuid-2")
        g.add_edge("A", "B", label="EVENT_WRITE", time=3000, event_uuid="uuid-3")

        pred = OrthrusPredecessor(dataset="THEIA_E3")
        records = pred.convert_nx_to_canonical(g)

        # 必须保留所有 3 条边
        assert len(records) == 3
        edge_types = [r["edge_type"] for r in records]
        assert "EVENT_OPEN" in edge_types
        assert "EVENT_READ" in edge_types
        assert "EVENT_WRITE" in edge_types

    def test_parallel_edges_with_same_timestamp(self):
        """测试同 timestamp 的多条平行边."""
        g = nx.MultiDiGraph()
        g.add_node("A", node_type="subject")
        g.add_node("B", node_type="file")
        g.add_edge("A", "B", label="EVENT_OPEN", time=1000, event_uuid="uuid-1")
        g.add_edge("A", "B", label="EVENT_READ", time=1000, event_uuid="uuid-2")

        pred = OrthrusPredecessor(dataset="THEIA_E3")
        records = pred.convert_nx_to_canonical(g)

        # 必须保留 2 条边
        assert len(records) == 2
        # event_uuid 决定排序
        assert records[0]["event_uuid"] == "uuid-1"
        assert records[1]["event_uuid"] == "uuid-2"

    def test_edge_count_conservation(self):
        """测试边数守恒."""
        g = nx.MultiDiGraph()
        g.add_node("A", node_type="subject")
        g.add_node("B", node_type="file")
        g.add_node("C", node_type="netflow")
        g.add_edge("A", "B", label="EVENT_WRITE", time=1000, event_uuid="uuid-1")
        g.add_edge("B", "C", label="EVENT_SEND", time=2000, event_uuid="uuid-2")
        g.add_edge("C", "A", label="EVENT_RECV", time=3000, event_uuid="uuid-3")

        nx_edge_count = g.number_of_edges()

        pred = OrthrusPredecessor(dataset="THEIA_E3")
        records = pred.convert_nx_to_canonical(g)

        # 边数必须守恒
        assert len(records) == nx_edge_count


# =============================================================================
# Adapter Profile Tests
# =============================================================================

class TestAdapterProfile:
    """Adapter profile 功能测试."""

    def test_legacy_profile_reverses_read(self):
        """测试 legacy profile 反转 READ."""
        records = [
            {"src": "file", "dst": "proc", "edge_type": "EVENT_READ",
             "timestamp": 1000, "global_event_index": 0,
             "src_type": "file", "dst_type": "subject"}
        ]
        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=records,
            profile="legacy_upstream"
        )
        adapter.fit()
        contract = adapter.transform(records, SplitType.TRAIN)

        edges = contract.get_sorted_edges()
        assert len(edges) == 1
        # 应该是反转的：proc -> file
        assert edges[0].src == "proc"
        assert edges[0].dst == "file"

    def test_orthrus_profile_keeps_direction(self):
        """测试 orthrus_unified profile 保持方向."""
        records = [
            {"src": "file", "dst": "proc", "edge_type": "EVENT_READ",
             "timestamp": 1000, "global_event_index": 0,
             "src_type": "file", "dst_type": "subject"}
        ]
        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=records,
            profile="orthrus_unified"
        )
        adapter.fit()
        contract = adapter.transform(records, SplitType.TRAIN)

        edges = contract.get_sorted_edges()
        assert len(edges) == 1
        # 应该保持：file -> proc
        assert edges[0].src == "file"
        assert edges[0].dst == "proc"

    def test_legacy_profile_dedup_first(self):
        """测试 legacy profile first-edge dedup."""
        # 使用两条都会产生相同 (src,dst) pair 的边
        # WRITE 不反转 → (A, B)
        # OPEN 不反转 → (A, B)
        records = [
            {"src": "A", "dst": "B", "edge_type": "EVENT_WRITE",
             "timestamp": 1000, "global_event_index": 0,
             "src_type": "subject", "dst_type": "file"},
            {"src": "A", "dst": "B", "edge_type": "EVENT_OPEN",
             "timestamp": 2000, "global_event_index": 1,
             "src_type": "subject", "dst_type": "file"},
        ]
        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=records,
            profile="legacy_upstream"
        )
        adapter.fit()
        contract = adapter.transform(records, SplitType.TRAIN)

        # 只保留第一条
        assert contract.train_edge_count == 1

    def test_orthrus_profile_preserves_parallel(self):
        """测试 orthrus_unified profile 保留所有平行边."""
        records = [
            {"src": "A", "dst": "B", "edge_type": "EVENT_WRITE",
             "timestamp": 1000, "global_event_index": 0,
             "src_type": "subject", "dst_type": "file"},
            {"src": "A", "dst": "B", "edge_type": "EVENT_READ",
             "timestamp": 2000, "global_event_index": 1,
             "src_type": "subject", "dst_type": "file"},
        ]
        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=records,
            profile="orthrus_unified"
        )
        adapter.fit()
        contract = adapter.transform(records, SplitType.TRAIN)

        # 保留所有 2 条边
        assert contract.train_edge_count == 2


# =============================================================================
# Direction Semantics Tests
# =============================================================================

class TestDirectionSemantics:
    """方向语义测试."""

    def test_orthrus_preserves_read_direction(self):
        """测试 ORTHRUS 保持 READ 方向: file -> subject."""
        records = [
            {"src": "file_node", "dst": "proc_node", "edge_type": "EVENT_READ",
             "timestamp": 1000, "global_event_index": 0,
             "src_type": "file", "dst_type": "subject"}
        ]
        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=records,
            profile="orthrus_unified"
        )
        adapter.fit()
        contract = adapter.transform(records, SplitType.TRAIN)

        edges = contract.get_sorted_edges()
        assert edges[0].src == "file_node"
        assert edges[0].dst == "proc_node"

    def test_orthrus_preserves_recv_direction(self):
        """测试 ORTHRUS 保持 RECV 方向: netflow -> subject."""
        records = [
            {"src": "net_node", "dst": "proc_node", "edge_type": "EVENT_RECVFROM",
             "timestamp": 1000, "global_event_index": 0,
             "src_type": "netflow", "dst_type": "subject"}
        ]
        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=records,
            profile="orthrus_unified"
        )
        adapter.fit()
        contract = adapter.transform(records, SplitType.TRAIN)

        edges = contract.get_sorted_edges()
        assert edges[0].src == "net_node"
        assert edges[0].dst == "proc_node"

    def test_orthrus_preserves_open_direction(self):
        """测试 ORTHRUS 保持 OPEN 方向: file -> subject."""
        records = [
            {"src": "file_node", "dst": "proc_node", "edge_type": "EVENT_OPEN",
             "timestamp": 1000, "global_event_index": 0,
             "src_type": "file", "dst_type": "subject"}
        ]
        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=records,
            profile="orthrus_unified"
        )
        adapter.fit()
        contract = adapter.transform(records, SplitType.TRAIN)

        edges = contract.get_sorted_edges()
        assert edges[0].src == "file_node"
        assert edges[0].dst == "proc_node"

    def test_orthrus_preserves_execute_direction(self):
        """测试 ORTHRUS 保持 EXECUTE 方向: file -> subject."""
        records = [
            {"src": "file_node", "dst": "proc_node", "edge_type": "EVENT_EXECUTE",
             "timestamp": 1000, "global_event_index": 0,
             "src_type": "file", "dst_type": "subject"}
        ]
        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=records,
            profile="orthrus_unified"
        )
        adapter.fit()
        contract = adapter.transform(records, SplitType.TRAIN)

        edges = contract.get_sorted_edges()
        assert edges[0].src == "file_node"
        assert edges[0].dst == "proc_node"

    def test_orthrus_preserves_write_direction(self):
        """测试 ORTHRUS 保持 WRITE 方向: subject -> file."""
        records = [
            {"src": "proc_node", "dst": "file_node", "edge_type": "EVENT_WRITE",
             "timestamp": 1000, "global_event_index": 0,
             "src_type": "subject", "dst_type": "file"}
        ]
        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=records,
            profile="orthrus_unified"
        )
        adapter.fit()
        contract = adapter.transform(records, SplitType.TRAIN)

        edges = contract.get_sorted_edges()
        assert edges[0].src == "proc_node"
        assert edges[0].dst == "file_node"

    def test_legacy_upstream_keeps_reversal_behavior(self):
        """测试 legacy upstream 保持 READ/RECV 反转行为."""
        # READ
        read_records = [
            {"src": "file", "dst": "proc", "edge_type": "EVENT_READ",
             "timestamp": 1000, "global_event_index": 0,
             "src_type": "file", "dst_type": "subject"}
        ]
        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=read_records,
            profile="legacy_upstream"
        )
        adapter.fit()
        contract = adapter.transform(read_records, SplitType.TRAIN)
        edges = contract.get_sorted_edges()
        # READ 被反转
        assert edges[0].src == "proc"
        assert edges[0].dst == "file"

        # RECV
        recv_records = [
            {"src": "net", "dst": "proc", "edge_type": "EVENT_RECVFROM",
             "timestamp": 1000, "global_event_index": 0,
             "src_type": "netflow", "dst_type": "subject"}
        ]
        adapter2 = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=recv_records,
            profile="legacy_upstream"
        )
        adapter2.fit()
        contract2 = adapter2.transform(recv_records, SplitType.TRAIN)
        edges2 = contract2.get_sorted_edges()
        # RECV 被反转
        assert edges2[0].src == "proc"
        assert edges2[0].dst == "net"

    def test_legacy_upstream_keeps_dedup_first(self):
        """测试 legacy upstream 保持 first-edge dedup."""
        records = [
            {"src": "A", "dst": "B", "edge_type": "EVENT_WRITE",
             "timestamp": 1000, "global_event_index": 0,
             "src_type": "subject", "dst_type": "file"},
            {"src": "A", "dst": "B", "edge_type": "EVENT_OPEN",
             "timestamp": 2000, "global_event_index": 1,
             "src_type": "subject", "dst_type": "file"},
        ]
        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=records,
            profile="legacy_upstream"
        )
        adapter.fit()
        contract = adapter.transform(records, SplitType.TRAIN)

        # 只保留第一条
        assert contract.train_edge_count == 1


# =============================================================================
# Vocabulary Tests
# =============================================================================

class TestTrainOnlyVocabulary:
    """Train-only vocabulary 测试."""

    def test_train_only_node_vocab(self):
        """测试 node vocabulary 只来自 train."""
        train_records = [
            {"src": "A", "dst": "B", "edge_type": "EVENT_WRITE",
             "timestamp": 1000, "global_event_index": 0,
             "src_type": "subject", "dst_type": "file"},
        ]
        val_records = [
            {"src": "C", "dst": "D", "edge_type": "EVENT_READ",
             "timestamp": 2000, "global_event_index": 0,
             "src_type": "netflow", "dst_type": "pipe"},  # unseen types
        ]

        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=train_records,
            val_records=val_records,
            profile="orthrus_unified"
        )
        adapter.fit()
        val_contract = adapter.transform(val_records, SplitType.VALIDATION)

        # val 中出现 unseen types
        assert len(val_contract.unseen_node_types) > 0

    def test_train_only_edge_vocab(self):
        """测试 edge vocabulary 只来自 train."""
        train_records = [
            {"src": "A", "dst": "B", "edge_type": "EVENT_WRITE",
             "timestamp": 1000, "global_event_index": 0,
             "src_type": "subject", "dst_type": "file"},
        ]
        val_records = [
            {"src": "C", "dst": "D", "edge_type": "EVENT_CONNECT",
             "timestamp": 2000, "global_event_index": 0,
             "src_type": "subject", "dst_type": "netflow"},  # unseen type
        ]

        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=train_records,
            val_records=val_records,
            profile="orthrus_unified"
        )
        adapter.fit()
        val_contract = adapter.transform(val_records, SplitType.VALIDATION)

        # val 中出现 unseen edge type
        assert len(val_contract.unseen_edge_types) > 0

    def test_val_does_not_expand_vocab(self):
        """测试 val 不扩展 vocabulary."""
        train_records = [
            {"src": "A", "dst": "B", "edge_type": "EVENT_WRITE",
             "timestamp": 1000, "global_event_index": 0,
             "src_type": "subject", "dst_type": "file"},
        ]
        val_records = [
            {"src": "C", "dst": "D", "edge_type": "EVENT_CLONE",
             "timestamp": 2000, "global_event_index": 0,
             "src_type": "subject", "dst_type": "subject"},
        ]

        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=train_records,
            val_records=val_records,
            profile="orthrus_unified"
        )
        adapter.fit()
        val_contract = adapter.transform(val_records, SplitType.VALIDATION)

        # vocabulary 大小不变
        assert adapter.vocabulary.node_feature_dim == 3  # 2 types + UNKNOWN
        assert adapter.vocabulary.edge_feature_dim == 2  # 1 type + UNKNOWN

    def test_test_does_not_expand_vocab(self):
        """测试 test 不扩展 vocabulary."""
        train_records = [
            {"src": "A", "dst": "B", "edge_type": "EVENT_WRITE",
             "timestamp": 1000, "global_event_index": 0,
             "src_type": "subject", "dst_type": "file"},
        ]
        test_records = [
            {"src": "C", "dst": "D", "edge_type": "EVENT_EXECUTE",
             "timestamp": 2000, "global_event_index": 0,
             "src_type": "netflow", "dst_type": "subject"},
        ]

        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=train_records,
            test_records=test_records,
            profile="orthrus_unified"
        )
        adapter.fit()
        test_contract = adapter.transform(test_records, SplitType.TEST)

        # vocabulary 大小不变
        assert adapter.vocabulary.node_feature_dim == 3  # 2 types + UNKNOWN
        assert adapter.vocabulary.edge_feature_dim == 2  # 1 type + UNKNOWN

    def test_dynamic_node_feature_dim(self):
        """测试动态 node feature dimension."""
        train_records = [
            {"src": "A", "dst": "B", "edge_type": "EVENT_WRITE",
             "timestamp": 1000, "global_event_index": 0,
             "src_type": "subject", "dst_type": "file"},
            {"src": "C", "dst": "D", "edge_type": "EVENT_CONNECT",
             "timestamp": 2000, "global_event_index": 1,
             "src_type": "subject", "dst_type": "netflow"},
        ]

        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=train_records,
            profile="orthrus_unified"
        )
        adapter.fit()

        # 3 node types + UNKNOWN = 4
        assert adapter.vocabulary.node_feature_dim == 4

    def test_dynamic_edge_feature_dim(self):
        """测试动态 edge feature dimension."""
        train_records = [
            {"src": "A", "dst": "B", "edge_type": "EVENT_WRITE",
             "timestamp": 1000, "global_event_index": 0,
             "src_type": "subject", "dst_type": "file"},
            {"src": "C", "dst": "D", "edge_type": "EVENT_READ",
             "timestamp": 2000, "global_event_index": 1,
             "src_type": "file", "dst_type": "subject"},
            {"src": "E", "dst": "F", "edge_type": "EVENT_CONNECT",
             "timestamp": 3000, "global_event_index": 2,
             "src_type": "subject", "dst_type": "netflow"},
        ]

        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=train_records,
            profile="orthrus_unified"
        )
        adapter.fit()

        # 3 edge types + UNKNOWN = 4
        assert adapter.vocabulary.edge_feature_dim == 4


# =============================================================================
# Empty Day Tests
# =============================================================================

class TestEmptyDay:
    """Empty day 支持测试."""

    def test_verified_empty_day_marker_detection(self):
        """测试 verified empty day marker 检测."""
        import tempfile
        import os
        import json

        # 创建临时 marker 文件
        with tempfile.TemporaryDirectory() as tmpdir:
            marker_path = os.path.join(tmpdir, ".preprocess_empty_day.json")
            with open(marker_path, "w") as f:
                json.dump({"reason": "no_raw_events"}, f)

            assert is_verified_empty_day_marker(marker_path) is True

    def test_verified_empty_day_marker_rejection(self):
        """测试非 empty day marker 被拒绝."""
        import tempfile
        import os
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            # 普通文件
            regular_path = os.path.join(tmpdir, "regular.json")
            with open(regular_path, "w") as f:
                json.dump({"data": "test"}, f)

            assert is_verified_empty_day_marker(regular_path) is False

            # 不同 reason 的 marker
            wrong_reason_path = os.path.join(tmpdir, ".preprocess_empty_day.json")
            with open(wrong_reason_path, "w") as f:
                json.dump({"reason": "other_reason"}, f)

            assert is_verified_empty_day_marker(wrong_reason_path) is False
