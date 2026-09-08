"""
ORTHRUS Production Graph Predecessor

本模块实现 ORTHRUS nx.MultiDiGraph 到 canonical records 的转换。

设计依据：
- M9-P3C.1R 审计报告
- ORTHRUS production graph semantics

关键设计原则：
1. nx.MultiDiGraph → canonical List[Dict] 单向转换
2. 确定性排序：按 (timestamp, event_uuid)
3. 全量保留：所有 nx edges 必须对应一条 canonical record
4. 必需字段缺失即 fail-fast
5. 不读取 ground truth、不训练、不创建 DGL graph
"""
from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import networkx as nx


class OrthrusPredecessorError(Exception):
    """ORTHRUS Predecessor 错误基类."""
    pass


class MissingRequiredFieldError(OrthrusPredecessorError):
    """必需字段缺失错误."""
    pass


class OrthrusPredecessor:
    """
    ORTHRUS nx.MultiDiGraph → canonical edge records 转换器.

    职责：
    - 读取 ORTHRUS production nx.MultiDiGraph
    - 转换为 canonical edge records
    - 确定性排序并分配 global_event_index
    - 全量保留所有 edges（禁止 silent drop）

    禁止行为：
    - 不训练 vocabulary
    - 不创建 DGL graph
    - 不读取 ground truth
    - 不读取攻击标签
    - 不运行 MAGIC 模型
    """

    REQUIRED_NODE_FIELDS = frozenset({"node_type"})
    REQUIRED_EDGE_FIELDS = frozenset({"label", "time", "event_uuid"})

    def __init__(
        self,
        dataset: Optional[str] = None,
    ):
        """
        初始化 ORTHRUS Predecessor.

        Args:
            dataset: 数据集名称 (THEIA_E3, THEIA_E5 等)，可选
        """
        self.dataset = dataset

    def _validate_node_fields(
        self,
        node_id: str,
        data: Dict[str, Any],
    ) -> None:
        """验证节点必需字段."""
        for field in self.REQUIRED_NODE_FIELDS:
            if field not in data:
                raise MissingRequiredFieldError(
                    f"Node {node_id} missing required field: {field}. "
                    f"Available fields: {list(data.keys())}"
                )

    def _validate_edge_fields(
        self,
        src: str,
        dst: str,
        data: Dict[str, Any],
    ) -> None:
        """验证边必需字段."""
        for field in self.REQUIRED_EDGE_FIELDS:
            if field not in data:
                raise MissingRequiredFieldError(
                    f"Edge ({src} -> {dst}) missing required field: {field}. "
                    f"Available fields: {list(data.keys())}"
                )

    def _get_sort_key(
        self,
        edge_data: Dict[str, Any],
    ) -> Tuple[int, str, str, str, Any]:
        """
        生成边的确定性排序键.

        使用 (timestamp, event_uuid, src, dst, edge_key) 确保确定性。

        Args:
            edge_data: 包含 edge 数据的字典

        Returns:
            排序键元组
        """
        timestamp = int(edge_data.get("time", 0))
        event_uuid = str(edge_data.get("event_uuid", ""))
        src = str(edge_data.get("src", ""))
        dst = str(edge_data.get("dst", ""))
        edge_key = edge_data.get("edge_key", "")
        return (timestamp, event_uuid, src, dst, edge_key)

    def convert_nx_to_canonical(
        self,
        graph: nx.MultiDiGraph,
        *,
        graph_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        将 nx.MultiDiGraph 转换为 canonical edge records.

        必需字段映射：
        - src: nx edge source
        - dst: nx edge destination
        - src_type: graph.nodes[src]["node_type"]
        - dst_type: graph.nodes[dst]["node_type"]
        - edge_type: edge_data["label"]
        - timestamp: int(edge_data["time"])
        - event_uuid: str(edge_data["event_uuid"])
        - global_event_index: 按 (timestamp, event_uuid, ...) 排序后分配

        Args:
            graph: ORTHRUS nx.MultiDiGraph
            graph_name: 图名称（可选，用于日志）

        Returns:
            List[Dict]：Canonical edge records 列表

        Raises:
            MissingRequiredFieldError: 必需字段缺失时

        事件守恒验证：
        - NX_EDGE_COUNT == CANONICAL_RECORD_COUNT
        """
        if not isinstance(graph, nx.MultiDiGraph):
            raise TypeError(
                f"Expected nx.MultiDiGraph, got {type(graph).__name__}"
            )

        # 验证所有节点的必需字段
        for node_id, data in graph.nodes(data=True):
            self._validate_node_fields(node_id, data)

        # 收集所有边数据
        raw_edges: List[Dict[str, Any]] = []

        for src, dst, edge_key, data in graph.edges(keys=True, data=True):
            # 验证边的必需字段
            self._validate_edge_fields(src, dst, data)

            # 构建边的完整数据
            edge_record = {
                "src": src,
                "dst": dst,
                "src_type": graph.nodes[src]["node_type"],
                "dst_type": graph.nodes[dst]["node_type"],
                "edge_type": data["label"],
                "timestamp": int(data["time"]),
                "event_uuid": str(data["event_uuid"]),
                # edge_key 用于确定性排序
                "_edge_key": edge_key,
            }
            raw_edges.append(edge_record)

        # 确定性排序
        sorted_edges = sorted(
            raw_edges,
            key=lambda e: self._get_sort_key(e),
        )

        # 分配 global_event_index
        canonical_records: List[Dict[str, Any]] = []
        for idx, edge in enumerate(sorted_edges):
            record = {
                "src": edge["src"],
                "dst": edge["dst"],
                "src_type": edge["src_type"],
                "dst_type": edge["dst_type"],
                "edge_type": edge["edge_type"],
                "timestamp": edge["timestamp"],
                "event_uuid": edge["event_uuid"],
                "global_event_index": idx,
            }
            canonical_records.append(record)

        # 事件守恒验证
        nx_edge_count = graph.number_of_edges()
        if nx_edge_count != len(canonical_records):
            raise ValueError(
                f"Edge count conservation failed: "
                f"nx.MultiDiGraph has {nx_edge_count} edges, "
                f"but produced {len(canonical_records)} canonical records. "
                f"Check if any edges were silently dropped."
            )

        return canonical_records

    def get_graph_stats(
        self,
        graph: nx.MultiDiGraph,
    ) -> Dict[str, Any]:
        """
        获取图的统计信息（不转换）.

        Args:
            graph: ORTHRUS nx.MultiDiGraph

        Returns:
            Dict 包含统计信息
        """
        # 统计 node types
        node_types: Dict[str, int] = {}
        for _, data in graph.nodes(data=True):
            node_type = data.get("node_type", "unknown")
            node_types[node_type] = node_types.get(node_type, 0) + 1

        # 统计 edge types
        edge_types: Dict[str, int] = {}
        for _, _, _, data in graph.edges(keys=True, data=True):
            edge_type = data.get("label", "unknown")
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

        # 统计平行边
        parallel_edge_pairs = 0
        for u, v in graph.edges():
            if graph.number_of_edges(u, v) > 1:
                parallel_edge_pairs += 1

        return {
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "node_types": node_types,
            "edge_types": edge_types,
            "parallel_edge_pairs": parallel_edge_pairs,
        }


def is_verified_empty_day_marker(path: str) -> bool:
    """
    检查路径是否是 verified empty day marker.

    Args:
        path: 文件路径

    Returns:
        True 如果是 empty day marker
    """
    import os
    import json

    if not os.path.isfile(path):
        return False

    if not os.path.basename(path).startswith(".preprocess_empty_day"):
        return False

    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data.get("reason") == "no_raw_events"
    except (json.JSONDecodeError, IOError):
        return False


__all__ = [
    "OrthrusPredecessor",
    "OrthrusPredecessorError",
    "MissingRequiredFieldError",
    "is_verified_empty_day_marker",
]
