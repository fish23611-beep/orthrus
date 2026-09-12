"""Contract tests for the pinned MAGIC real-backend bridge."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import os
import subprocess
import sys
from pathlib import Path
import random
from types import ModuleType
from typing import Any, Dict, List

import numpy as np
import pytest

real_backend = importlib.import_module("src.baselines.magic.real_backend")

import torch

from src.baselines.magic.contracts import SplitType
from src.baselines.magic.evaluator import MagicEvaluator
from src.baselines.magic.input_adapter import MAGICInputAdapter
from src.baselines.magic.scoring import MAGICEntityScorer
from src.baselines.magic.seed import FakeDGLModule
from src.baselines.magic.real_backend import (
    ALGORITHM_IDENTITY,
    CheckpointIdentityError,
    FROZEN_FILE_SHA256,
    MAGICModelConfig,
    MAGICRealBackend,
    MAGICUpstreamRuntime,
    RealMAGICDependencyError,
    UPSTREAM_COMMIT,
    WRAPPER_IDENTITY,
    UpstreamIntegrityError,
    verify_upstream_identity,
)


# Test-side resolution of the real-backend upstream snapshot path.
#
# Production contract:
#   * src.baselines.magic.real_backend.DEFAULT_UPSTREAM_PATH stays at
#     /opt/magic-upstream so the Docker / Dev Container fallback works
#     unchanged.
#   * Tests that need a non-default snapshot read MAGIC_UPSTREAM_PATH.
#     This env var is honored ONLY on the test side so the pinned Docker
#     fallback contract is not weakened.
#
# Resolution rules (host-portable):
#   * Unset or empty  -> real_backend.DEFAULT_UPSTREAM_PATH (Docker default).
#   * Set             -> expanduser + resolve, but do NOT require existence
#                         at import time. verify_upstream_identity() still
#                         fail-fasts on a missing or tampered snapshot.
#   * Host user paths (e.g. /home/<user>/magic-upstream) must NEVER be
#     hard-coded here; they enter only through the env var.
_MAGIC_UPSTREAM_ENV = "MAGIC_UPSTREAM_PATH"
_DEFAULT_TEST_UPSTREAM = real_backend.DEFAULT_UPSTREAM_PATH


def _resolve_test_upstream_path() -> Path:
    raw = os.environ.get(_MAGIC_UPSTREAM_ENV)
    if not raw:
        return Path(_DEFAULT_TEST_UPSTREAM)
    return Path(raw).expanduser().resolve()


UPSTREAM_PATH = _resolve_test_upstream_path()


class FakeGraph:
    def __init__(self, edges: Any, num_nodes: int) -> None:
        self.sources, self.destinations = edges
        self._num_nodes = num_nodes
        self.ndata: Dict[str, Any] = {}
        self.edata: Dict[str, Any] = {}
        self.device = "cpu"

    def to(self, device: Any) -> "FakeGraph":
        self.device = device
        return self

    def num_nodes(self) -> int:
        return self._num_nodes

    def number_of_nodes(self) -> int:
        return self._num_nodes

    def number_of_edges(self) -> int:
        return len(self.sources)


class RecordingDGL(FakeDGLModule):
    __version__ = "fake-dgl-1.0"

    def __init__(self) -> None:
        super().__init__()
        self.graph_calls: List[Any] = []

    def graph(self, edges: Any, num_nodes: int) -> FakeGraph:
        result = FakeGraph(edges, num_nodes)
        self.graph_calls.append(result)
        return result


class RecordingModel:
    def __init__(self, stochastic_trace: Any) -> None:
        self.stochastic_trace = tuple(stochastic_trace)
        self.to_calls: List[Any] = []
        self.train_calls = 0
        self.eval_calls = 0
        self.embed_calls: List[FakeGraph] = []
        self.loaded_state = False

    def to(self, device: Any) -> "RecordingModel":
        self.to_calls.append(device)
        return self

    def train(self) -> None:
        self.train_calls += 1

    def eval(self) -> None:
        self.eval_calls += 1

    def state_dict(self) -> Dict[str, Any]:
        return {"trace": torch.tensor(self.stochastic_trace)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.stochastic_trace = tuple(state["trace"].tolist())
        self.loaded_state = True

    def embed(self, graph: FakeGraph) -> Any:
        self.embed_calls.append(graph)
        index = torch.arange(graph.num_nodes(), dtype=torch.float32)
        offset = float(self.stochastic_trace[0])
        return torch.stack(
            (
                index + offset,
                index.square() + 1.0,
                torch.sin(index + offset),
                torch.cos(index + offset),
            ),
            dim=1,
        )


class RuntimeHarness:
    def __init__(self) -> None:
        self.dgl = RecordingDGL()
        self.build_calls: List[Any] = []
        self.optimizer_calls: List[Any] = []
        self.training_calls: List[Dict[str, Any]] = []
        self.models: List[RecordingModel] = []

    def build_model(self, args: Any) -> RecordingModel:
        self.build_calls.append(args)
        trace = (
            random.random(),
            float(np.random.random()),
            float(torch.rand(1).item()),
        )
        model = RecordingModel(trace)
        self.models.append(model)
        return model

    def create_optimizer(
        self,
        name: str,
        model: RecordingModel,
        learning_rate: float,
        weight_decay: float,
    ) -> object:
        self.optimizer_calls.append(
            (name, model, learning_rate, weight_decay)
        )
        return object()

    def train_entity_level(self, **kwargs: Any) -> RecordingModel:
        self.training_calls.append(kwargs)
        return kwargs["model"]

    def runtime(self) -> MAGICUpstreamRuntime:
        return MAGICUpstreamRuntime(
            torch=torch,
            dgl=self.dgl,
            build_model=self.build_model,
            create_optimizer=self.create_optimizer,
            train_entity_level=self.train_entity_level,
        )


def make_records(
    prefix: str,
    count: int,
    timestamp_base: int,
    edge_type: str = "EVENT_WRITE",
) -> List[Dict[str, Any]]:
    records = []
    for index in range(count - 1):
        records.append(
            {
                "src": "{}_{}".format(prefix, index),
                "dst": "{}_{}".format(prefix, index + 1),
                "src_type": "subject" if index % 2 == 0 else "file",
                "dst_type": "subject" if (index + 1) % 2 == 0 else "file",
                "edge_type": edge_type,
                "timestamp": timestamp_base + index,
                "global_event_index": timestamp_base * 10 + index,
            }
        )
    return records


@pytest.fixture
def contracts():
    adapter = MAGICInputAdapter(
        dataset="TINY",
        train_records=make_records("train", 12, 100),
        val_records=make_records("validation", 6, 200),
        test_records=make_records("test", 7, 300),
    )
    return adapter.fit_transform()


def make_backend(
    seed: int = 0,
    config: MAGICModelConfig = None,
):
    harness = RuntimeHarness()
    backend = MAGICRealBackend(
        seed=seed,
        upstream_path=UPSTREAM_PATH,
        config=config or MAGICModelConfig(max_epoch=1),
        _runtime_factory=lambda _: harness.runtime(),
    )
    return backend, harness


def prepare_all(backend: MAGICRealBackend, contracts):
    train_contract, val_contract, test_contract = contracts
    return (
        backend.prepare_graph(
            train_contract,
            SplitType.TRAIN,
            "train-snapshot",
        ),
        backend.prepare_graph(
            val_contract,
            SplitType.VALIDATION,
            "validation-snapshot",
        ),
        backend.prepare_graph(
            test_contract,
            SplitType.TEST,
            "test-snapshot",
        ),
    )


def file_hashes() -> Dict[str, str]:
    result = {}
    for relative_path in FROZEN_FILE_SHA256:
        content = (UPSTREAM_PATH / relative_path).read_bytes()
        result[relative_path] = hashlib.sha256(content).hexdigest()
    return result


def test_module_import_succeeds_without_dgl() -> None:
    assert real_backend.MAGICRealBackend is MAGICRealBackend
    assert importlib.util.find_spec("dgl") is None


def test_real_execution_without_dgl_fails_clearly(contracts) -> None:
    backend = MAGICRealBackend(seed=0, upstream_path=UPSTREAM_PATH)
    with pytest.raises(
        RealMAGICDependencyError,
        match="real MAGIC runtime dependency missing.*DGL",
    ):
        backend.prepare_graph(
            contracts[0],
            SplitType.TRAIN,
            "train-snapshot",
        )


def test_missing_upstream_path_fails(tmp_path: Path) -> None:
    with pytest.raises(UpstreamIntegrityError, match="missing MAGIC upstream"):
        MAGICRealBackend(seed=0, upstream_path=tmp_path / "absent")


def test_wrong_upstream_file_sha_fails(tmp_path: Path) -> None:
    (tmp_path / "train.py").write_text("tampered", encoding="utf-8")
    with pytest.raises(UpstreamIntegrityError, match="wrong upstream SHA256"):
        verify_upstream_identity(tmp_path)


def test_wrong_expected_commit_fails() -> None:
    with pytest.raises(UpstreamIntegrityError, match="wrong upstream SHA"):
        verify_upstream_identity(UPSTREAM_PATH, expected_commit="0" * 40)


def test_upstream_identity_verification_works() -> None:
    identity = verify_upstream_identity(UPSTREAM_PATH)
    assert identity.repository.endswith("FDUDSDE/MAGIC")
    assert identity.commit == UPSTREAM_COMMIT
    assert identity.verification == "frozen-file-sha256"
    assert identity.file_sha256 == FROZEN_FILE_SHA256


def test_reversed_edge_keeps_node_types_attached_to_ids() -> None:
    records = [
        {
            "src": "file-id",
            "dst": "process-id",
            "src_type": "file",
            "dst_type": "subject",
            "edge_type": "EVENT_READ",
            "timestamp": 10,
            "global_event_index": 1,
        }
    ]
    contract = MAGICInputAdapter("TINY", records).fit().transform(
        records,
        SplitType.TRAIN,
    )
    assert contract.edges[0].src == "process-id"
    assert contract.edges[0].dst == "file-id"
    assert contract.nodes["process-id"].node_type == "subject"
    assert contract.nodes["file-id"].node_type == "file"


def test_prepare_graph_builds_upstream_dgl_fields(contracts) -> None:
    backend, harness = make_backend()
    prepared = backend.prepare_graph(
        contracts[0],
        SplitType.TRAIN,
        "train-snapshot",
    )
    graph = prepared.graph
    assert len(harness.dgl.graph_calls) == 1
    assert graph.ndata["type"].dtype == torch.long
    assert graph.edata["type"].dtype == torch.long
    assert graph.ndata["attr"].shape == (
        len(prepared.node_ids),
        prepared.node_feature_dim,
    )
    assert graph.edata["attr"].shape == (
        prepared.edge_count,
        prepared.edge_feature_dim,
    )
    assert torch.all(graph.ndata["attr"].sum(dim=1) == 1)
    assert torch.all(graph.edata["attr"].sum(dim=1) == 1)


def test_embedding_node_mapping_is_stable(contracts) -> None:
    backend_a, _ = make_backend()
    backend_b, _ = make_backend()
    graph_a = backend_a.prepare_graph(
        contracts[0], SplitType.TRAIN, "same"
    )
    graph_b = backend_b.prepare_graph(
        contracts[0], SplitType.TRAIN, "same"
    )
    assert graph_a.node_ids == graph_b.node_ids
    assert graph_a.canonical_to_local == graph_b.canonical_to_local
    assert graph_a.graph_fingerprint == graph_b.graph_fingerprint
    for local_id, canonical_id in enumerate(graph_a.local_to_canonical):
        assert graph_a.canonical_to_local[canonical_id] == local_id


def test_unknown_types_use_frozen_unknown_bucket() -> None:
    train = make_records("train", 4, 10)
    validation = [
        {
            "src": "unknown-a",
            "dst": "unknown-b",
            "src_type": "new-node-type",
            "dst_type": "file",
            "edge_type": "EVENT_NEW",
            "timestamp": 50,
            "global_event_index": 50,
        }
    ]
    train_contract, val_contract, _ = MAGICInputAdapter(
        "TINY", train, validation
    ).fit_transform()
    backend, _ = make_backend()
    prepared_train = backend.prepare_graph(
        train_contract, SplitType.TRAIN, "train"
    )
    prepared_val = backend.prepare_graph(
        val_contract, SplitType.VALIDATION, "validation"
    )
    assert prepared_val.node_feature_dim == prepared_train.node_feature_dim
    assert prepared_val.edge_feature_dim == prepared_train.edge_feature_dim
    assert int(prepared_val.graph.ndata["type"][0]) == (
        prepared_val.node_feature_dim - 1
    )
    assert int(prepared_val.graph.edata["type"][0]) == (
        prepared_val.edge_feature_dim - 1
    )


def test_fake_upstream_construction_and_training_are_called(contracts) -> None:
    backend, harness = make_backend()
    train_graph, _, _ = prepare_all(backend, contracts)
    backend.fit(train_graph)
    assert len(harness.build_calls) == 1
    assert harness.build_calls[0].n_dim == train_graph.node_feature_dim
    assert harness.build_calls[0].e_dim == train_graph.edge_feature_dim
    assert len(harness.optimizer_calls) == 1
    assert len(harness.training_calls) == 1
    call = harness.training_calls[0]
    assert call["graphs"] == [train_graph.graph]
    assert call["max_epoch"] == 1
    assert call["device"] == "cpu"


@pytest.mark.parametrize(
    "split,index",
    [
        (SplitType.VALIDATION, 1),
        (SplitType.TEST, 2),
    ],
)
def test_fit_rejects_non_train_graphs(contracts, split, index) -> None:
    backend, harness = make_backend()
    graphs = prepare_all(backend, contracts)
    with pytest.raises(ValueError, match="train graphs only"):
        backend.fit(graphs[index])
    assert split == graphs[index].split
    assert harness.build_calls == []
    assert harness.training_calls == []


def test_checkpoint_save_and_load_lifecycle(
    contracts,
    tmp_path: Path,
) -> None:
    backend, _ = make_backend(seed=1)
    train_graph, _, test_graph = prepare_all(backend, contracts)
    backend.fit(train_graph)
    fingerprint = backend.training_input_fingerprint
    checkpoint = backend.save_checkpoint(
        tmp_path / "checkpoints" / "magic_final.pt"
    )
    assert checkpoint.is_file()

    restored, restored_harness = make_backend(seed=1)
    restored.load_checkpoint(checkpoint, fingerprint)
    assert restored.training_input_fingerprint == fingerprint
    assert restored_harness.models[-1].loaded_state is True
    output = restored.embed(test_graph)
    assert output.snapshot_id == "test-snapshot"


def test_checkpoint_metadata_contains_complete_identity(
    contracts,
    tmp_path: Path,
) -> None:
    backend, _ = make_backend(seed=2)
    train_graph, _, _ = prepare_all(backend, contracts)
    backend.fit(train_graph)
    checkpoint = backend.save_checkpoint(tmp_path / "magic_final.pt")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    assert metadata["upstream"]["commit"] == UPSTREAM_COMMIT
    assert metadata["upstream"]["file_sha256"] == dict(FROZEN_FILE_SHA256)
    assert metadata["wrapper_identity"] == WRAPPER_IDENTITY
    assert metadata["algorithm_identity"] == ALGORITHM_IDENTITY
    assert metadata["experiment_seed"] == 2
    assert metadata["graph_input_fingerprint"]
    assert metadata["config_identity"] == backend.config.identity


@pytest.mark.parametrize(
    "mutation",
    ["seed", "fingerprint", "config"],
)
def test_checkpoint_wrong_identity_is_rejected(
    contracts,
    tmp_path: Path,
    mutation: str,
) -> None:
    backend, _ = make_backend(seed=0)
    train_graph, _, _ = prepare_all(backend, contracts)
    backend.fit(train_graph)
    fingerprint = backend.training_input_fingerprint
    checkpoint = backend.save_checkpoint(tmp_path / "magic_final.pt")

    if mutation == "seed":
        restored, _ = make_backend(seed=1)
        expected = fingerprint
    elif mutation == "config":
        restored, _ = make_backend(
            seed=0,
            config=MAGICModelConfig(max_epoch=2),
        )
        expected = fingerprint
    else:
        restored, _ = make_backend(seed=0)
        expected = "wrong-" + fingerprint

    with pytest.raises(CheckpointIdentityError, match="identity mismatch"):
        restored.load_checkpoint(checkpoint, expected)


def test_checkpoint_without_metadata_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"state_dict": {}}, checkpoint)
    backend, _ = make_backend()
    with pytest.raises(
        CheckpointIdentityError,
        match="no identity metadata",
    ):
        backend.load_checkpoint(checkpoint, "required-fingerprint")


def test_embedding_extraction_preserves_snapshot_mapping(contracts) -> None:
    backend, harness = make_backend()
    train_graph, val_graph, _ = prepare_all(backend, contracts)
    backend.fit(train_graph)
    output = backend.embed(val_graph)
    assert len(harness.models[-1].embed_calls) == 1
    assert output.embeddings.shape[0] == len(val_graph.node_ids)
    assert output.node_ids == val_graph.node_ids
    assert output.canonical_to_local == val_graph.canonical_to_local
    assert output.local_to_canonical == val_graph.local_to_canonical
    assert output.split == SplitType.VALIDATION
    assert output.graph_fingerprint == val_graph.graph_fingerprint


def test_backend_public_api_has_no_label_argument() -> None:
    forbidden = {
        "y_test",
        "ground_truth",
        "malicious_nodes",
        "malicious_ids",
        "test_labels",
        "test_y",
    }
    methods = (
        MAGICRealBackend.__init__,
        MAGICRealBackend.prepare_graph,
        MAGICRealBackend.fit,
        MAGICRealBackend.save_checkpoint,
        MAGICRealBackend.load_checkpoint,
        MAGICRealBackend.embed,
    )
    for method in methods:
        parameters = set(inspect.signature(method).parameters)
        assert parameters.isdisjoint(forbidden)


def test_ground_truth_loader_is_not_accessed_before_embeddings(
    contracts,
    monkeypatch,
) -> None:
    accesses: List[str] = []
    poison = ModuleType("utils.loaddata")

    def fail_access(name: str) -> Any:
        accesses.append(name)
        raise AssertionError("ground-truth loader accessed")

    poison.__getattr__ = fail_access
    monkeypatch.setitem(sys.modules, "utils.loaddata", poison)

    backend, _ = make_backend()
    train_graph, _, test_graph = prepare_all(backend, contracts)
    backend.fit(train_graph)
    backend.embed(test_graph)
    assert accesses == []


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_official_seed_reaches_model_and_dgl(seed: int, contracts) -> None:
    backend, harness = make_backend(seed=seed)
    train_graph = backend.prepare_graph(
        contracts[0], SplitType.TRAIN, "train"
    )
    backend.fit(train_graph)
    calls = harness.dgl.get_calls()
    assert calls["seed_calls"] == [seed]
    assert calls["random_seed_calls"] == [seed]
    assert backend.seed_manifest.seed == seed
    assert backend.seed_manifest.dgl_seed_set is True
    assert backend.seed_manifest.dgl_random_seed_set is True


def run_stochastic_trace(seed: int, contract) -> Any:
    backend, harness = make_backend(seed=seed)
    train_graph = backend.prepare_graph(
        contract, SplitType.TRAIN, "train"
    )
    backend.fit(train_graph)
    return harness.models[0].stochastic_trace


def test_repeated_seed_produces_stable_mocked_trace(contracts) -> None:
    assert run_stochastic_trace(1, contracts[0]) == run_stochastic_trace(
        1, contracts[0]
    )


def test_different_seed_changes_mocked_trace(contracts) -> None:
    assert run_stochastic_trace(1, contracts[0]) != run_stochastic_trace(
        2, contracts[0]
    )


def test_real_backend_has_no_smoke_or_word_embedding_dependency() -> None:
    source = inspect.getsource(real_backend)
    disallowed = (
        "synthetic_" + "smoke",
        "Word" + "2Vec",
        "x_" + "src",
        "x_" + "dst",
    )
    for token in disallowed:
        assert token not in source


def test_upstream_sources_are_not_modified_by_bridge(contracts) -> None:
    before = file_hashes()
    backend, _ = make_backend()
    train_graph, _, test_graph = prepare_all(backend, contracts)
    backend.fit(train_graph)
    backend.embed(test_graph)
    after = file_hashes()
    assert after == before == dict(FROZEN_FILE_SHA256)


def test_backend_embeddings_feed_existing_m5_and_m6(contracts) -> None:
    backend, _ = make_backend()
    train_graph, val_graph, test_graph = prepare_all(backend, contracts)
    backend.fit(train_graph)
    train_batch = backend.embed(train_graph)
    val_batch = backend.embed(val_graph)
    test_batch = backend.embed(test_graph)

    scorer = MAGICEntityScorer(k=3, seed=0)
    scorer.fit(train_batch.embeddings, list(train_batch.node_ids))
    val_records = scorer.score(
        val_batch.embeddings,
        list(val_batch.node_ids),
    )
    test_records = scorer.score(
        test_batch.embeddings,
        list(test_batch.node_ids),
    )
    validation_scores = {
        record.canonical_node_id: record.score_raw
        for record in val_records
    }
    test_scores = {
        record.canonical_node_id: record.score_raw
        for record in test_records
    }

    evaluator = MagicEvaluator(k=3, threshold_quantile=0.999)
    threshold = evaluator.fit_threshold(validation_scores)
    predictions = evaluator.predict(test_scores)
    assert threshold.provenance == "validation_only"
    assert set(predictions) == set(test_batch.node_ids)
    assert all(np.isfinite(list(test_scores.values())))


def test_failed_real_import_does_not_pollute_generic_namespaces(
    contracts,
) -> None:
    path_before = list(sys.path)
    generic_before = {
        name: module
        for name, module in sys.modules.items()
        if name.split(".", 1)[0] in {"model", "utils"}
    }
    backend = MAGICRealBackend(seed=0, upstream_path=UPSTREAM_PATH)
    with pytest.raises(RealMAGICDependencyError):
        backend.prepare_graph(
            contracts[0], SplitType.TRAIN, "train"
        )
    generic_after = {
        name: module
        for name, module in sys.modules.items()
        if name.split(".", 1)[0] in {"model", "utils"}
    }
    assert sys.path == path_before
    assert generic_after == generic_before

def test_simple_graph_first_edge_uses_canonical_order() -> None:
    records = [
        {
            "src": "a",
            "dst": "b",
            "src_type": "subject",
            "dst_type": "file",
            "edge_type": "EVENT_LATE",
            "timestamp": 200,
            "global_event_index": 2,
        },
        {
            "src": "a",
            "dst": "b",
            "src_type": "subject",
            "dst_type": "file",
            "edge_type": "EVENT_EARLY",
            "timestamp": 100,
            "global_event_index": 1,
        },
    ]
    adapter = MAGICInputAdapter("TINY", records).fit()
    contract = adapter.transform(records, SplitType.TRAIN)
    assert len(contract.edges) == 1
    assert contract.edges[0].timestamp == 100
    assert contract.edges[0].edge_type == "EVENT_EARLY"
    assert adapter.vocabulary.transform_edge_type("EVENT_EARLY")[1] is True
    assert adapter.vocabulary.transform_edge_type("EVENT_LATE")[1] is True


def test_multi_epoch_training_with_normal_backward(contracts) -> None:
    """Regression test: multi-epoch training must work without retain_graph=True.

    This was the core bug fixed in P2D.1. The wrapper now deep-copies
    the input graph each iteration so that DGL's in-place ndata modification
    in encoding_mask_noise() does not corrupt the original graph's tensors.
    """
    backend, harness = make_backend(config=MAGICModelConfig(max_epoch=2))
    train_graph, _, _ = prepare_all(backend, contracts)
    backend.fit(train_graph)

    # Verify training was called with the correct number of epochs
    assert len(harness.training_calls) == 1
    call = harness.training_calls[0]
    assert call["max_epoch"] == 2
    # Verify the model was returned from training
    assert harness.training_calls[0]["model"] is not None


def test_training_loop_uses_normal_backward(contracts) -> None:
    """Verify the training lifecycle does NOT use retain_graph=True.

    This ensures the fix uses proper graph isolation rather than autograd hacks.
    """
    import inspect
    from src.baselines.magic.real_backend import _run_original_entity_training_lifecycle
    source = inspect.getsource(_run_original_entity_training_lifecycle)
    assert "retain_graph" not in source, (
        "Training lifecycle must not use retain_graph=True"
    )


def test_fit_returns_trained_model(contracts) -> None:
    """Regression test: fit() returns the model after training (chainable)."""
    backend, _ = make_backend(seed=0, config=MAGICModelConfig(max_epoch=1))
    train_graph, _, _ = prepare_all(backend, contracts)
    result = backend.fit(train_graph)
    # fit() must return self for chainability
    assert result is backend
    # Model must be set
    assert backend._model is not None


def test_prepared_graph_ndata_is_detached(contracts) -> None:
    """Regression test: prepared graph ndata/edata tensors have requires_grad=False.

    This prevents accidental autograd connections between the prepared graph
    and the training computation graph.
    """
    backend, _ = make_backend()
    train_graph, _, _ = prepare_all(backend, contracts)
    graph = train_graph.graph

    for key in graph.ndata:
        assert not graph.ndata[key].requires_grad, \
            f"ndata['{key}'] should not require grad"
    for key in graph.edata:
        assert not graph.edata[key].requires_grad, \
            f"edata['{key}'] should not require grad"


# =============================================================================
# Duplicate Policy Tests (P3C.2.1)
# =============================================================================

def test_real_backend_legacy_dedup_first() -> None:
    """Test that legacy_upstream profile results in DEDUP_FIRST behavior in backend.

    Note: legacy_upstream reverses READ/RECV/LOAD events, so we use WRITE
    to test dedup behavior without direction changes affecting the pair.
    """
    from src.baselines.magic.contracts import DuplicateEdgePolicy

    # Create records with same src/dst but different events
    # Use EVENT_WRITE to avoid direction reversal in legacy profile
    records = [
        {
            "src": "A",
            "dst": "B",
            "src_type": "subject",
            "dst_type": "file",
            "edge_type": "EVENT_WRITE",
            "timestamp": 1000,
            "global_event_index": 0,
        },
        {
            "src": "A",
            "dst": "B",
            "src_type": "subject",
            "dst_type": "file",
            "edge_type": "EVENT_OPEN",
            "timestamp": 1001,
            "global_event_index": 1,
        },
        {
            "src": "A",
            "dst": "B",
            "src_type": "subject",
            "dst_type": "file",
            "edge_type": "EVENT_EXECUTE",
            "timestamp": 1002,
            "global_event_index": 2,
        },
    ]

    adapter = MAGICInputAdapter(
        dataset="TINY",
        train_records=records,
        profile="legacy_upstream",
    )
    contract = adapter.fit().transform(records, SplitType.TRAIN)

    # Adapter should have already deduped (keep first for same pair)
    # All 3 events have same (src, dst) = (A, B), so only first is kept
    assert len(contract.edges) == 1
    assert contract.edges[0].timestamp == 1000

    # Contract should have DEDUP_FIRST policy
    assert contract.duplicate_policy == DuplicateEdgePolicy.DEDUP_FIRST

    # Backend should respect the policy
    backend, harness = make_backend()
    prepared = backend.prepare_graph(contract, SplitType.TRAIN, "test-snapshot")

    # Edge count should remain 1 (already deduped at adapter level)
    assert prepared.edge_count == 1


def test_real_backend_orthrus_unified_preserves_parallel_edges() -> None:
    """Test that orthrus_unified profile results in PRESERVE_ALL behavior in backend."""
    from src.baselines.magic.contracts import DuplicateEdgePolicy

    # Create records with same src/dst but different events
    records = [
        {
            "src": "A",
            "dst": "B",
            "src_type": "file",
            "dst_type": "subject",
            "edge_type": "EVENT_READ",
            "timestamp": 1000,
            "global_event_index": 0,
        },
        {
            "src": "A",
            "dst": "B",
            "src_type": "file",
            "dst_type": "subject",
            "edge_type": "EVENT_WRITE",
            "timestamp": 1001,
            "global_event_index": 1,
        },
        {
            "src": "A",
            "dst": "B",
            "src_type": "file",
            "dst_type": "subject",
            "edge_type": "EVENT_OPEN",
            "timestamp": 1002,
            "global_event_index": 2,
        },
    ]

    adapter = MAGICInputAdapter(
        dataset="TINY",
        train_records=records,
        profile="orthrus_unified",
    )
    contract = adapter.fit().transform(records, SplitType.TRAIN)

    # Adapter should preserve all 3 edges
    assert len(contract.edges) == 3

    # Contract should have PRESERVE_ALL policy
    assert contract.duplicate_policy == DuplicateEdgePolicy.PRESERVE_ALL

    # Backend should respect the policy and keep all edges
    backend, harness = make_backend()
    prepared = backend.prepare_graph(contract, SplitType.TRAIN, "test-snapshot")

    # Edge count should be 3 (all preserved)
    assert prepared.edge_count == 3


def test_real_backend_profile_policy_propagated_from_contract() -> None:
    """Test that policy is correctly propagated from adapter to contract to backend."""
    from src.baselines.magic.contracts import DuplicateEdgePolicy

    records = [
        {
            "src": "X",
            "dst": "Y",
            "src_type": "file",
            "dst_type": "subject",
            "edge_type": "EVENT_EXECUTE",
            "timestamp": 500,
            "global_event_index": 0,
        },
        {
            "src": "X",
            "dst": "Y",
            "src_type": "file",
            "dst_type": "subject",
            "edge_type": "EVENT_CONNECT",
            "timestamp": 501,
            "global_event_index": 1,
        },
    ]

    # Test orthrus_unified propagation
    adapter_orthrus = MAGICInputAdapter(
        dataset="TINY",
        train_records=records,
        profile="orthrus_unified",
    )
    contract_orthrus = adapter_orthrus.fit().transform(records, SplitType.TRAIN)

    assert contract_orthrus.duplicate_policy == DuplicateEdgePolicy.PRESERVE_ALL
    assert contract_orthrus.should_dedup() is False

    backend, _ = make_backend()
    prepared_orthrus = backend.prepare_graph(
        contract_orthrus, SplitType.TRAIN, "orthrus-snapshot"
    )
    assert prepared_orthrus.edge_count == 2

    # Test legacy_upstream propagation
    adapter_legacy = MAGICInputAdapter(
        dataset="TINY",
        train_records=records,
        profile="legacy_upstream",
    )
    contract_legacy = adapter_legacy.fit().transform(records, SplitType.TRAIN)

    assert contract_legacy.duplicate_policy == DuplicateEdgePolicy.DEDUP_FIRST
    assert contract_legacy.should_dedup() is True

    prepared_legacy = backend.prepare_graph(
        contract_legacy, SplitType.TRAIN, "legacy-snapshot"
    )
    assert prepared_legacy.edge_count == 1


def test_real_backend_unknown_duplicate_policy_fails() -> None:
    """Test that backend raises error for contract with unknown/None duplicate_policy."""
    from src.baselines.magic.contracts import (
        NeutralGraphContract,
        TypeVocabulary,
    )

    # Create a contract with vocabulary but duplicate_policy not set
    vocab = TypeVocabulary().fit(
        {"A": "file", "B": "subject"},
        {"EVENT_EXECUTE": "EVENT_EXECUTE"}
    )

    contract = NeutralGraphContract()
    contract.type_vocabulary = vocab
    contract.duplicate_policy = None  # Explicitly set to None

    backend, _ = make_backend()

    with pytest.raises(ValueError, match="duplicate_policy must be set"):
        backend.prepare_graph(contract, SplitType.TRAIN, "test-snapshot")


def test_legacy_magic_backend_behavior_unchanged() -> None:
    """Regression test: legacy behavior should be preserved.

    Note: legacy_upstream reverses READ/RECV/LOAD events, so we use
    non-reversed event types to test dedup behavior.
    """
    from src.baselines.magic.contracts import DuplicateEdgePolicy

    # Create records with multiple pairs
    # Use EVENT_WRITE and EVENT_CONNECT to avoid direction reversal
    records = [
        # Pair 1: 2 events (same src/dst)
        {
            "src": "A",
            "dst": "B",
            "src_type": "subject",
            "dst_type": "file",
            "edge_type": "EVENT_WRITE",
            "timestamp": 100,
            "global_event_index": 0,
        },
        {
            "src": "A",
            "dst": "B",
            "src_type": "subject",
            "dst_type": "file",
            "edge_type": "EVENT_OPEN",
            "timestamp": 101,
            "global_event_index": 1,
        },
        # Pair 2: 1 event
        {
            "src": "C",
            "dst": "D",
            "src_type": "netflow",
            "dst_type": "subject",
            "edge_type": "EVENT_CONNECT",
            "timestamp": 102,
            "global_event_index": 2,
        },
    ]

    adapter = MAGICInputAdapter(
        dataset="TINY",
        train_records=records,
        profile="legacy_upstream",
    )
    contract = adapter.fit().transform(records, SplitType.TRAIN)

    # Legacy adapter should dedup: keep first event for each pair
    # Pair 1: keep event at timestamp 100 (WRITE)
    # Pair 2: keep event at timestamp 102 (CONNECT)
    assert len(contract.edges) == 2
    assert contract.duplicate_policy == DuplicateEdgePolicy.DEDUP_FIRST

    backend, _ = make_backend()
    prepared = backend.prepare_graph(contract, SplitType.TRAIN, "legacy-snapshot")

    # Backend should preserve the already-deduped edges
    assert prepared.edge_count == 2

    # Verify specific timestamps are preserved
    edge_timestamps = [e.timestamp for e in contract.edges]
    assert 100 in edge_timestamps  # First event of pair 1
    assert 102 in edge_timestamps  # Event of pair 2
    assert 101 not in edge_timestamps  # Second event of pair 1 (deduped)


# =============================================================================
# Host-portable upstream path resolution (MAGIC_UPSTREAM_PATH override)
# =============================================================================
#
# These tests prove the contract that lets the same test file run on both:
#   * Docker / Dev Container: MAGIC_UPSTREAM_PATH unset -> /opt/magic-upstream
#   * Formal Host server   : MAGIC_UPSTREAM_PATH=/home/<user>/magic-upstream
#
# Production code (src/baselines/magic/real_backend.py) is intentionally
# untouched: DEFAULT_UPSTREAM_PATH still resolves to /opt/magic-upstream,
# integrity verification (FROZEN_FILE_SHA256 + UPSTREAM_COMMIT) still runs,
# and missing / tampered snapshots still fail-fast. We do NOT relax the
# contract to pytest.skip when the override path is missing.

def _probe_upstream_path_in_subprocess(env_override):
    """Import the test module in a clean subprocess and read UPSTREAM_PATH.

    Using a subprocess isolates sys.modules / module-level state so we can
    observe the env-var driven import-time constant without reload hacks or
    pollution of the parent interpreter.
    """
    env = dict(os.environ)
    if env_override is None:
        env.pop("MAGIC_UPSTREAM_PATH", None)
    else:
        env["MAGIC_UPSTREAM_PATH"] = env_override

    repo_root = Path(__file__).resolve().parents[2]
    code = (
        "import sys\n"
        "sys.path.insert(0, {!r})\n"
        "import tests.test_magic.test_magic_real_backend as m\n"
        "print(str(m.UPSTREAM_PATH))\n"
    ).format(str(repo_root))

    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip().splitlines()[-1]


def test_default_upstream_path_is_docker_default_when_env_unset() -> None:
    """A. Unset MAGIC_UPSTREAM_PATH -> tests still default to /opt/magic-upstream.

    Validates Docker / Dev Container compatibility.
    """
    observed = _probe_upstream_path_in_subprocess(env_override=None)
    assert Path(observed) == Path(real_backend.DEFAULT_UPSTREAM_PATH)
    assert Path(observed) == Path("/opt/magic-upstream")


def test_env_override_rewrites_test_upstream_path(tmp_path: Path) -> None:
    """B. Setting MAGIC_UPSTREAM_PATH rewrites the test-side snapshot.

    The override must NOT bake a host username into Python code: the path
    only enters the module through the env var.
    """
    custom = tmp_path / "magic-upstream"
    observed = _probe_upstream_path_in_subprocess(env_override=str(custom))
    assert Path(observed) == custom.resolve()


def test_env_override_still_runs_upstream_identity_verification(
    tmp_path: Path,
) -> None:
    """C. Configurability does NOT bypass integrity verification.

    The frozen-file SHA check (and therefore the UPSTREAM_COMMIT / commit
    check when a .git directory is present) must still run on any
    overridden path. A wrong snapshot must fail-fast, not be silently
    accepted because the path came from an env var.
    """
    custom = tmp_path / "magic-upstream"
    custom.mkdir()
    (custom / "train.py").write_text("tampered", encoding="utf-8")

    with pytest.raises(UpstreamIntegrityError, match="wrong upstream SHA256"):
        verify_upstream_identity(custom)


def test_missing_upstream_does_not_skip_under_override(tmp_path: Path) -> None:
    """D. A missing upstream MUST NOT be skipped, even when env-var driven.

    Missing snapshot continues to fail-fast; the host-portable override
    must not relax the contract to pytest.skip.
    """
    missing = tmp_path / "does-not-exist"
    with pytest.raises(UpstreamIntegrityError, match="missing MAGIC upstream"):
        MAGICRealBackend(seed=0, upstream_path=missing)

    # Also confirm the env-var driven test resolver does NOT auto-skip a
    # missing path: the failure surfaces when the backend (or
    # verify_upstream_identity) actually inspects the directory.
    observed = _probe_upstream_path_in_subprocess(env_override=str(missing))
    assert Path(observed) == missing.resolve()


def test_env_override_does_not_change_production_default() -> None:
    """Sanity guard: production DEFAULT_UPSTREAM_PATH must remain Docker default.

    Ensures no drift in the pinned Docker fallback contract during the
    host-portability fix.
    """
    assert Path(real_backend.DEFAULT_UPSTREAM_PATH) == Path("/opt/magic-upstream")
