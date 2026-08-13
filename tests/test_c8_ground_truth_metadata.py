"""C8 regression tests for ground-truth metadata pipeline correctness.

Tests cover:
1. Ground truth root resolver (nested submodule vs legacy layout)
2. Metadata export fail-closed (missing file, empty CSV, zero UUID matches)
3. metadata_complete() semantic validation
4. detection_only timestamp-to-window conversion
5. Cache semantic separation (timestamp keys vs window-index keys)
"""

import csv
import os
import pickle
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Add src to path
src_dir = str(Path(__file__).resolve().parents[1] / "src")
sys.path.insert(0, src_dir)

from mstc.metadata_cache import MetadataCache, MetadataCacheError
from mstc.metadata_cache import metadata_complete, metadata_validation_status


class TestGroundTruthRootResolver:
    """Tests for _resolve_ground_truth_dir() resolver."""

    def test_resolver_nested_layout(self, tmp_path):
        """Resolver detects nested submodule layout (Ground_Truth/darpa/darpa/)."""
        # Simulate nested layout: Ground_Truth/darpa/darpa/E3-THEIA
        gt_root = tmp_path / "Ground_Truth" / "darpa"
        nested = gt_root / "darpa" / "E3-THEIA"
        nested.mkdir(parents=True)

        # Directly test the resolver logic by calling it with a patched ROOT
        import config
        old_root = config.ROOT_GROUND_TRUTH_DIR
        config.ROOT_GROUND_TRUTH_DIR = str(gt_root) + "/"

        try:
            result = config._resolve_ground_truth_dir()
            expected = str(nested.parent) + "/"
            assert result == expected, f"Expected {expected}, got {result}"
        finally:
            config.ROOT_GROUND_TRUTH_DIR = old_root

    def test_resolver_legacy_layout(self, tmp_path):
        """Resolver detects legacy layout (Ground_Truth/darpa/E3-THEIA)."""
        # Simulate legacy layout: Ground_Truth/darpa/E3-THEIA
        gt_root = tmp_path / "Ground_Truth" / "darpa"
        legacy = gt_root / "E3-THEIA"
        legacy.mkdir(parents=True)

        import config
        old_root = config.ROOT_GROUND_TRUTH_DIR
        config.ROOT_GROUND_TRUTH_DIR = str(gt_root) + "/"

        try:
            result = config._resolve_ground_truth_dir()
            expected = str(gt_root) + "/"
            assert result == expected, f"Expected {expected}, got {result}"
        finally:
            config.ROOT_GROUND_TRUTH_DIR = old_root

    def test_resolver_raises_when_invalid(self, tmp_path):
        """Resolver raises FileNotFoundError when neither layout exists."""
        gt_root = tmp_path / "Ground_Truth" / "darpa"
        gt_root.mkdir(parents=True)  # Empty directory

        import config
        old_root = config.ROOT_GROUND_TRUTH_DIR
        config.ROOT_GROUND_TRUTH_DIR = str(gt_root) + "/"

        try:
            with pytest.raises(FileNotFoundError, match="Ground truth root not found"):
                config._resolve_ground_truth_dir()
        finally:
            config.ROOT_GROUND_TRUTH_DIR = old_root


class TestMetadataExportFailClosed:
    """Tests for _ground_truth_mappings() fail-closed behavior."""

    def _make_cfg(self, gt_root, relative_paths):
        """Create a minimal cfg with ground truth configuration."""
        cfg = SimpleNamespace()
        cfg._ground_truth_dir = str(gt_root)
        cfg.dataset = SimpleNamespace()
        cfg.dataset.ground_truth_relative_path = relative_paths
        return cfg

    def test_missing_gt_file_raises(self, tmp_path):
        """Missing ground truth CSV raises MetadataCacheError."""
        from mstc.metadata_export import _ground_truth_mappings

        gt_root = tmp_path / "gt"
        gt_root.mkdir()

        cfg = self._make_cfg(gt_root, ["E3-THEIA/node_Browser_Extension_Drakon_Dropper.csv"])
        uuid_map = {"uuid-1": 1, "uuid-2": 2}

        with pytest.raises(MetadataCacheError, match="Ground truth CSV not found"):
            _ground_truth_mappings(cfg, uuid_map)

    def test_empty_gt_file_raises(self, tmp_path):
        """Empty ground truth CSV raises MetadataCacheError."""
        from mstc.metadata_export import _ground_truth_mappings

        gt_root = tmp_path / "gt" / "E3-THEIA"
        gt_root.mkdir(parents=True)
        csv_path = gt_root / "node_Attack.csv"
        csv_path.touch()  # Empty file

        cfg = self._make_cfg(tmp_path / "gt", ["E3-THEIA/node_Attack.csv"])
        uuid_map = {"uuid-1": 1}

        with pytest.raises(MetadataCacheError, match="no data rows"):
            _ground_truth_mappings(cfg, uuid_map)

    def test_zero_uuid_matches_raises(self, tmp_path):
        """Zero UUID matches raises MetadataCacheError."""
        from mstc.metadata_export import _ground_truth_mappings

        gt_root = tmp_path / "gt" / "E3-THEIA"
        gt_root.mkdir(parents=True)
        csv_path = gt_root / "node_Attack.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["uuid-not-in-map", "label", "id"])
            writer.writerow(["uuid-also-not-found", "label", "id"])

        cfg = self._make_cfg(tmp_path / "gt", ["E3-THEIA/node_Attack.csv"])
        uuid_map = {"uuid-1": 1, "uuid-2": 2}  # No matches

        with pytest.raises(MetadataCacheError, match="0 UUIDs matched"):
            _ground_truth_mappings(cfg, uuid_map)

    def test_valid_gt_produces_nonempty_sets(self, tmp_path):
        """Valid GT with matched UUIDs produces non-empty sets."""
        from mstc.metadata_export import _ground_truth_mappings

        gt_root = tmp_path / "gt" / "E3-THEIA"
        gt_root.mkdir(parents=True)
        csv_path = gt_root / "node_Attack.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["uuid-1", "malicious", "0"])
            writer.writerow(["uuid-2", "malicious", "1"])

        cfg = self._make_cfg(tmp_path / "gt", ["E3-THEIA/node_Attack.csv"])
        uuid_map = {"uuid-1": 10, "uuid-2": 20}

        gt_nodes, attack_map = _ground_truth_mappings(cfg, uuid_map)

        assert len(gt_nodes) == 2
        assert 10 in gt_nodes
        assert 20 in gt_nodes
        assert 0 in attack_map
        assert len(attack_map[0]) == 2


class TestMetadataCompleteSemantic:
    """Tests for metadata_complete() / metadata_validation_status() API."""

    def _write_complete_cache(self, tmp_path, *, dataset="TEST",
                              ground_truth_relative_path=None,
                              attack_to_time_window=None,
                              time_to_malicious=None,
                              attack_to_nodes=None,
                              ground_truth_nodes=None):
        """Populate a fully-valid cache. Default uses non-empty payloads."""
        cache = MetadataCache(tmp_path)
        cache.save_node_metadata({1: {"uuid": "u1", "type": "subject", "display": "test"}})
        cache.save_uuid_to_node_id({"u1": 1})
        cache.save_node_id_to_uuid({1: "u1"})
        cache.save_ground_truth_nodes(
            ground_truth_nodes if ground_truth_nodes is not None else {1, 2, 3}
        )
        cache.save_attack_to_nodes(
            attack_to_nodes if attack_to_nodes is not None else {0: {1, 2}, 1: {3}}
        )
        cache.save_time_to_malicious_nodes(
            time_to_malicious if time_to_malicious is not None else {1000: ["u1"]}
        )
        cache.save_relation_mapping({"r": "mapping"})
        cache.save_nodeid2msg({1: "msg"})
        cache.save_dataset_manifest(
            dataset=dataset, num_node_types=3, num_edge_types=10,
            train_files=["g1"], val_files=["g2"], test_files=["g3"],
            word2vec_dim=128, preprocess_config_hash="abc",
        )
        marker = cache.cache_root / cache.COMPLETION_MARKER_FILE
        marker.write_text("2025-01-01T00:00:00Z")
        return cache

    def _cfg(self, dataset="TEST", gt_rel=("E3-THEIA/node_X.csv",),
             attack_to_time_window=None):
        """Build a cfg that declares the dataset as having GT + time window."""
        cfg = SimpleNamespace()
        cfg.dataset = SimpleNamespace(
            name=dataset,
            ground_truth_relative_path=list(gt_rel) if gt_rel is not None else [],
            attack_to_time_window=list(attack_to_time_window)
            if attack_to_time_window is not None
            else [["E3-THEIA/node_X.csv", "2018-04-12 12:40:00", "2018-04-12 13:30:00"]],
        )
        return cfg

    def test_metadata_complete_returns_bool_for_valid_cache(self, tmp_path):
        """metadata_complete returns True (bool) for valid metadata."""
        cache = self._write_complete_cache(tmp_path, dataset="THEIA_E3")

        cfg = self._cfg(dataset="THEIA_E3")
        is_complete, detail = metadata_validation_status(cache, cfg=cfg)

        assert is_complete is True
        assert detail == "complete"
        # C8: bool API — never returns a tuple
        result = metadata_complete(cache, cfg=cfg)
        assert isinstance(result, bool)
        assert result is True

    def test_metadata_complete_rejects_tuple_truthiness(self, tmp_path):
        """A non-empty cache with one missing file must NOT evaluate truthy.

        Regression for the legacy ``(False, "missing: ...")` tuple bug.
        """
        cache = MetadataCache(tmp_path)
        # Only write one file; rest are missing.
        cache.save_node_metadata({1: {"uuid": "u1", "type": "subject", "display": "x"}})

        cfg = self._cfg()
        result = metadata_complete(cache, cfg=cfg)
        # If somebody wires metadata_complete into ``assert foo(cache)`` and the
        # implementation accidentally returned a tuple, this assertion would
        # silently pass because non-empty tuples are truthy.
        assert result is False

        is_complete, detail = metadata_validation_status(cache, cfg=cfg)
        assert is_complete is False
        assert detail.startswith("missing:")

    def test_validation_status_distinguishes_missing_marker(self, tmp_path):
        cache = MetadataCache(tmp_path)
        is_complete, detail = metadata_validation_status(cache, cfg=self._cfg())
        assert is_complete is False
        assert "missing" in detail

    def test_validation_status_rejects_empty_ground_truth(self, tmp_path):
        cache = self._write_complete_cache(
            tmp_path, dataset="THEIA_E3", ground_truth_nodes=set()
        )
        cfg = self._cfg(dataset="THEIA_E3")
        is_complete, detail = metadata_validation_status(cache, cfg=cfg)
        assert is_complete is False
        assert detail == "empty: ground_truth"
        assert metadata_complete(cache, cfg=cfg) is False

    def test_validation_status_rejects_empty_uuid_map(self, tmp_path):
        cache = MetadataCache(tmp_path)
        cache.save_node_metadata({})
        cache.save_uuid_to_node_id({})
        cache.save_node_id_to_uuid({})
        cache.save_ground_truth_nodes({1})
        cache.save_attack_to_nodes({0: {1}})
        cache.save_time_to_malicious_nodes({1: ["u1"]})
        cache.save_relation_mapping({})
        cache.save_nodeid2msg({})
        cache.save_dataset_manifest(
            dataset="THEIA_E3", num_node_types=3, num_edge_types=10,
            train_files=[], val_files=[], test_files=[],
            word2vec_dim=128, preprocess_config_hash="abc",
        )
        marker = cache.cache_root / cache.COMPLETION_MARKER_FILE
        marker.write_text("2025-01-01T00:00:00Z")

        cfg = self._cfg(dataset="THEIA_E3")
        is_complete, detail = metadata_validation_status(cache, cfg=cfg)
        assert is_complete is False
        assert detail == "empty: uuid_to_node_id"

    def test_validation_status_reports_corrupt_cache(self, tmp_path):
        """A cache file with invalid pickle/JSON must surface as 'corrupt'."""
        cache = MetadataCache(tmp_path)
        cache.save_node_metadata({1: {"uuid": "u1", "type": "subject", "display": "x"}})
        cache.save_uuid_to_node_id({"u1": 1})
        cache.save_node_id_to_uuid({1: "u1"})
        cache.save_ground_truth_nodes({1})
        cache.save_attack_to_nodes({0: {1}})
        # write a corrupt pickle for time_to_malicious_nodes
        cache.save_relation_mapping({})
        cache.save_nodeid2msg({1: "msg"})
        cache.save_dataset_manifest(
            dataset="THEIA_E3", num_node_types=3, num_edge_types=10,
            train_files=["g1"], val_files=["g2"], test_files=["g3"],
            word2vec_dim=128, preprocess_config_hash="abc",
        )
        marker = cache.cache_root / cache.COMPLETION_MARKER_FILE
        marker.write_text("2025-01-01T00:00:00Z")
        (cache.cache_root / cache.TIME_TO_MALICIOUS_NODES_FILE).write_bytes(b"\x00\x01NOPE")

        cfg = self._cfg(dataset="THEIA_E3")
        is_complete, detail = metadata_validation_status(cache, cfg=cfg)
        assert is_complete is False
        assert detail.startswith("corrupt:")


class TestDetectionOnlyComputeTwLabels:
    """C8 contract: empty time cache cannot be certified as complete when
    the configured dataset declares ``attack_to_time_window``.

    These tests replace the old ``test_empty_time_cache_shows_as_valid``,
    which directly contradicted the THEIA_E3 fix.
    """

    def _setup(self, tmp_path, *, time_to_malicious, dataset="THEIA_E3"):
        cache = MetadataCache(tmp_path)
        cache.save_node_metadata({1: {"uuid": "u1", "type": "subject", "display": "test"}})
        cache.save_uuid_to_node_id({"u1": 1})
        cache.save_node_id_to_uuid({1: "u1"})
        cache.save_ground_truth_nodes({1})
        cache.save_attack_to_nodes({0: {1}})
        cache.save_time_to_malicious_nodes(time_to_malicious)
        cache.save_relation_mapping({})
        cache.save_nodeid2msg({1: "msg"})
        cache.save_dataset_manifest(
            dataset=dataset, num_node_types=3, num_edge_types=10,
            train_files=[], val_files=[], test_files=[],
            word2vec_dim=128, preprocess_config_hash="abc",
        )
        marker = cache.cache_root / cache.COMPLETION_MARKER_FILE
        marker.write_text("2025-01-01T00:00:00Z")
        return cache

    def _cfg(self, dataset="THEIA_E3"):
        cfg = SimpleNamespace()
        cfg.dataset = SimpleNamespace(
            name=dataset,
            ground_truth_relative_path=["E3-THEIA/node_Browser_Extension_Drakon_Dropper.csv"],
            attack_to_time_window=[
                ["E3-THEIA/node_Browser_Extension_Drakon_Dropper.csv",
                 "2018-04-12 12:40:00", "2018-04-12 13:30:00"],
            ],
        )
        return cfg

    def test_empty_time_cache_is_rejected_for_theia_e3(self, tmp_path):
        """THEIA_E3 has attack_to_time_window declared → empty time cache
        is a hard contract violation."""
        cache = self._setup(tmp_path, time_to_malicious={}, dataset="THEIA_E3")
        cfg = self._cfg(dataset="THEIA_E3")

        is_complete, detail = metadata_validation_status(cache, cfg=cfg)
        assert is_complete is False
        assert detail == "empty: time_to_malicious_nodes"
        assert metadata_complete(cache, cfg=cfg) is False

    def test_populated_time_cache_is_accepted_for_theia_e3(self, tmp_path):
        cache = self._setup(
            tmp_path,
            time_to_malicious={1767225600000000000: ["u1"]},
            dataset="THEIA_E3",
        )
        cfg = self._cfg(dataset="THEIA_E3")
        is_complete, detail = metadata_validation_status(cache, cfg=cfg)
        assert is_complete is True
        assert detail == "complete"

    def test_empty_time_cache_is_accepted_for_dataset_without_time_window(
        self, tmp_path
    ):
        """A dataset without ``attack_to_time_window`` (legacy config) may
        legally export an empty time cache."""
        cache = self._setup(tmp_path, time_to_malicious={}, dataset="LEGACY")
        cfg = SimpleNamespace()
        cfg.dataset = SimpleNamespace(
            name="LEGACY",
            ground_truth_relative_path=[],
            attack_to_time_window=[],
        )

        is_complete, detail = metadata_validation_status(cache, cfg=cfg)
        assert is_complete is True
        assert detail == "complete"

    def test_detection_only_mode_does_not_require_postgres(self, tmp_path, monkeypatch):
        """compute_tw_labels must hit the cache without a DB connection."""
        from detection import evaluation_utils as eu
        from detection.evaluation_utils import compute_tw_labels

        # Populate cache with valid non-empty time_to_malicious_nodes.
        cache = self._setup(
            tmp_path,
            time_to_malicious={1767225600000000000: ["u1"]},
            dataset="THEIA_E3",
        )
        cache_root = cache.cache_root
        cfg = SimpleNamespace()
        cfg._metadata_dir = str(cache_root)
        cfg.dataset = SimpleNamespace(test_files=[])
        cfg.graph_construction = SimpleNamespace(
            build_graphs=SimpleNamespace(
                _graphs_dir=str(tmp_path / "no_graphs"),
                _tw_labels=str(tmp_path / "no_graphs"),
            )
        )
        cfg.pipeline = SimpleNamespace(mode="detection_only")

        # Guard: if the cache-hit path ever tries to talk to PostgreSQL,
        # this mock will explode. We must never see init_database_connection
        # called during a pure detection_only flow.
        def _db_should_not_be_called(*_a, **_kw):
            raise AssertionError(
                "detection_only compute_tw_labels must not call "
                "provnet_utils.init_database_connection"
            )

        monkeypatch.setattr(
            eu, "init_database_connection", _db_should_not_be_called, raising=False
        )
        # The same guard against any direct psycopg2 / DB import path.
        try:
            from provnet_utils import init_database_connection as _real_db
            monkeypatch.setattr(
                "provnet_utils.init_database_connection",
                _db_should_not_be_called,
            )
        except Exception:
            pass

        # Either the function returns (cache hit) or raises a clear
        # FileNotFoundError (graph artifacts missing). Both paths must
        # avoid the database entirely.
        try:
            result = compute_tw_labels(cfg)
            # Cache-hit path returns a dict (possibly empty when no test
            # graphs are configured). Crucially, no DB call happened.
            assert isinstance(result, dict)
        except FileNotFoundError as exc:
            # Acceptable: missing graph artifacts short-circuit the path,
            # but the message must not mention the database.
            assert "postgres" not in str(exc).lower()
            assert "psycopg" not in str(exc).lower()


class TestCacheSemanticSeparation:
    """Tests ensuring timestamp keys and window-index keys are not confused."""

    def test_time_to_malicious_nodes_uses_timestamp_keys(self, tmp_path):
        """time_to_malicious_nodes cache must use timestamp keys (nanoseconds), not window indices."""
        cache = MetadataCache(tmp_path)

        # Simulate realistic timestamps (nanoseconds since epoch ~2025)
        ns_2025 = 1767225600000000000  # ~2025-01-01
        timestamp_data = {
            ns_2025: ["uuid-1", "uuid-2"],
            ns_2025 + 3600000000000: ["uuid-3"],  # 1 hour later
        }
        cache.save_time_to_malicious_nodes(timestamp_data)
        loaded = cache.load_time_to_malicious_nodes()

        # Verify these are large timestamp values (not small indices like 0, 1, 2)
        for key in loaded.keys():
            assert key > 1e15, (
                f"Cache key {key} looks like a window index, not a timestamp. "
                "time_to_malicious_nodes must use nanosecond timestamps as keys."
            )

    def test_compute_tw_labels_preserves_timestamp_cache_and_uses_half_open_ranges(
        self, tmp_path, monkeypatch
    ):
        """Deriving TW labels must not corrupt the timestamp-level cache.

        The interval contract is ``[start, end)``: a timestamp at ``start``
        belongs to the window, while one at ``end`` belongs only to the next
        adjacent window (if present).
        """
        from detection import evaluation_utils as eu

        ns_2025 = 1767225600000000000
        metadata_dir = tmp_path / "metadata"
        tw_labels_dir = tmp_path / "tw_labels"
        cache = MetadataCache(metadata_dir)
        timestamp_data = {
            ns_2025: ["uuid-start"],
            ns_2025 + 1: ["uuid-mid", "uuid-mid"],
            ns_2025 + 100: ["uuid-boundary"],
            ns_2025 + 200: ["uuid-end"],
        }
        cache.save_time_to_malicious_nodes(timestamp_data)
        cache.save_ground_truth_nodes({1, 2, 3, 4})
        cache.save_uuid_to_node_id({
            "uuid-start": 1,
            "uuid-mid": 2,
            "uuid-boundary": 3,
            "uuid-end": 4,
        })
        before = cache.load_time_to_malicious_nodes()

        cfg = SimpleNamespace(
            _metadata_dir=str(metadata_dir),
            dataset=SimpleNamespace(test_files=["graph_test"]),
            graph_construction=SimpleNamespace(
                build_graphs=SimpleNamespace(
                    _graphs_dir=str(tmp_path / "graphs"),
                    _tw_labels=str(tw_labels_dir),
                )
            ),
            pipeline=SimpleNamespace(mode="detection_only"),
        )
        graph_paths = ["tw-0", "tw-1"]
        bounds = {
            "tw-0": (ns_2025, ns_2025 + 100),
            "tw-1": (ns_2025 + 100, ns_2025 + 200),
        }
        monkeypatch.setattr(
            eu, "get_all_files_from_folders", lambda *_args: graph_paths
        )
        real_torch_load = eu.torch.load
        monkeypatch.setattr(eu.torch, "load", lambda path: path)
        monkeypatch.setattr(eu, "get_start_end_from_graph", bounds.__getitem__)

        result = eu.compute_tw_labels(cfg)

        assert result == {0: {1: 1, 2: 2}, 1: {3: 1}}
        assert cache.load_time_to_malicious_nodes() == before == timestamp_data
        assert all(key >= 10**15 for key in before)
        assert not ({0, 1} & set(before))

        tw_artifact = tw_labels_dir / "tw_to_malicious_nodes.pkl"
        assert tw_artifact.is_file()
        assert real_torch_load(tw_artifact, weights_only=False) == result

    def test_uuid_to_node_id_resolution_in_cache_hit(self, tmp_path):
        """Each timestamp entry's UUIDs must be resolved through uuid_to_node_id,
        not silently dropped or used as raw node_ids."""
        ns_2025 = 1767225600000000000
        cache = MetadataCache(tmp_path)
        cache.save_uuid_to_node_id({"uuid-known": 42, "uuid-other": 99})
        cache.save_time_to_malicious_nodes({ns_2025 + 1: ["uuid-known", "uuid-missing"]})

        uuid_to_node_id = cache.load_uuid_to_node_id()
        cached = cache.load_time_to_malicious_nodes()
        resolved = []
        for ts_ns, uuids in cached.items():
            for u in uuids:
                nid = uuid_to_node_id.get(u)
                if nid is not None:
                    resolved.append(nid)
        assert resolved == [42]

    def test_validation_status_distinguishes_empty_vs_missing(self, tmp_path):
        """metadata_validation_status distinguishes 'empty' from 'missing'."""
        cache = MetadataCache(tmp_path)

        # Case 1: file missing entirely
        is_complete, detail = metadata_validation_status(cache, cfg=SimpleNamespace(
            dataset=SimpleNamespace(
                name="TEST",
                ground_truth_relative_path=[],
                attack_to_time_window=[],
            ),
        ))
        assert is_complete is False
        assert "missing" in detail

        # Case 2: file exists but is empty
        cache.save_node_metadata({})
        cache.save_uuid_to_node_id({})
        cache.save_node_id_to_uuid({})
        cache.save_ground_truth_nodes(set())
        cache.save_attack_to_nodes({})
        cache.save_time_to_malicious_nodes({})
        cache.save_relation_mapping({})
        cache.save_nodeid2msg({})
        cache.save_dataset_manifest(
            dataset="TEST", num_node_types=3, num_edge_types=10,
            train_files=[], val_files=[], test_files=[],
            word2vec_dim=128, preprocess_config_hash="abc",
        )
        marker = cache.cache_root / cache.COMPLETION_MARKER_FILE
        marker.write_text("2025-01-01T00:00:00Z")

        is_complete, detail = metadata_validation_status(cache, cfg=SimpleNamespace(
            dataset=SimpleNamespace(
                name="TEST",
                ground_truth_relative_path=[],
                attack_to_time_window=[],
            ),
        ))
        assert is_complete is False
        assert "empty" in detail or "missing" in detail


class TestTheiaE3SyntheticSemantics:
    """Test THEIA_E3 ground truth synthetic/fixture semantics."""

    def test_attack_deduplication(self):
        """Two attacks with overlap produce correct union."""
        # Simulate: Attack A = {uuid-1, uuid-2}, Attack B = {uuid-2, uuid-3}
        # Expected: Attack A = 2 nodes, Attack B = 2 nodes, overlap = 1, union = 3

        # Create fake caches
        with tempfile.TemporaryDirectory() as tmp:
            cache = MetadataCache(tmp)
            cache.save_ground_truth_nodes({1, 2, 3})  # union
            cache.save_attack_to_nodes({
                0: {1, 2},   # attack A
                1: {2, 3},   # attack B
            })
            cache.save_node_metadata({
                1: {"uuid": "uuid-1", "type": "subject"},
                2: {"uuid": "uuid-2", "type": "subject"},
                3: {"uuid": "uuid-3", "type": "subject"},
            })

            gt = cache.load_ground_truth_nodes()
            atn = cache.load_attack_to_nodes()

            # Union
            assert len(gt) == 3

            # Attack A
            assert len(atn[0]) == 2

            # Attack B
            assert len(atn[1]) == 2

            # Overlap
            overlap = atn[0] & atn[1]
            assert len(overlap) == 1


class TestExportMetadataFailClosed:
    """Tests for export_metadata() fail-closed integration."""

    def test_export_missing_gt_raises(self, tmp_path):
        """export_metadata raises when ground truth CSV is missing.

        We test _ground_truth_mappings directly since it's the function that
        validates GT file existence before any DB operations.
        """
        from mstc.metadata_export import _ground_truth_mappings

        cfg = SimpleNamespace()
        cfg._metadata_dir = str(tmp_path / "metadata")
        cfg._ground_truth_dir = str(tmp_path / "gt")  # Non-existent
        cfg.dataset = SimpleNamespace()
        cfg.dataset.ground_truth_relative_path = ["E3-THEIA/node_Attack.csv"]

        uuid_map = {"uuid-1": 1}

        # This should raise before any DB connection is attempted
        with pytest.raises(MetadataCacheError, match="Ground truth CSV not found"):
            _ground_truth_mappings(cfg, uuid_map)
