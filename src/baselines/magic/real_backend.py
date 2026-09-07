"""
Bridge from the canonical MAGIC graph contract to the pinned upstream GAT/GMAE.

The module itself imports neither DGL nor the upstream source tree. Runtime
dependencies are resolved only when graph preparation, training, checkpoint
loading, or embedding execution is requested.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import threading

from .contracts import NeutralGraphContract, SplitType


UPSTREAM_REPOSITORY = "https://github.com/FDUDSDE/MAGIC"
UPSTREAM_COMMIT = "aa0b647eea74b6faa0e52eb444370c4411a32cbe"
DEFAULT_UPSTREAM_PATH = Path("/opt/magic-upstream")
WRAPPER_IDENTITY = "orthrus_magic_real_backend_v1"
ALGORITHM_IDENTITY = "FDUDSDE_MAGIC_GAT_GMAE"

# The five contract-frozen hashes are supplemented with every upstream source
# file imported by the runtime bridge. This prevents a valid top-level snapshot
# from silently loading modified model code.
FROZEN_FILE_SHA256: Mapping[str, str] = {
    "train.py": "9003a2ec19632073d55a1d689b828272bc53496685175fa4c88dbb63fdf028ad",
    "eval.py": "299352ea0387237fd0fcefe806024e1ecf18bd0d82ebdbdf12d31b4fbeed2208",
    "model/eval.py": "4761d6481bd29ddc2789eebf2d4dbb04a76176f23223e0690a1f0a0f5f2d75dc",
    "utils/loaddata.py": "57a84d42d6fa3f13a98e7ef606e9c3087574868bea3d70d62adaeca2bf155924",
    "utils/utils.py": "9e350ba709815437434c27aa4983e28785b4ab673ea3bf75dff6094984757fec",
    "model/autoencoder.py": "a57d771cc065320e4624d934503056cc078ed8b61b09cbf1d307f3ab55171ef7",
    "model/gat.py": "739d71b7699d18ecce59e6621d968967ce44f4b4b633a5b44e687107bf39d1f3",
    "model/loss_func.py": "f6b776336f286e51e67e67c3ba6bb36286a55aea29c2f777125a38d77d105d67",
    "model/train.py": "261914c41158c97478c8ad4305e18f8fb1e01cb4bc4a8be87273dbf89e3272ff",
}

_GENERIC_UPSTREAM_ROOTS = ("model", "utils")
_IMPORT_LOCK = threading.RLock()


class MAGICRealBackendError(RuntimeError):
    """Base error for the real MAGIC bridge."""


class UpstreamIntegrityError(MAGICRealBackendError):
    """The configured upstream snapshot does not match the frozen identity."""


class RealMAGICDependencyError(MAGICRealBackendError):
    """A dependency required only by real MAGIC execution is unavailable."""


class UpstreamImportIsolationError(MAGICRealBackendError):
    """The generic upstream import namespace cannot be used safely."""


class CheckpointIdentityError(MAGICRealBackendError):
    """A checkpoint does not belong to this exact experiment identity."""


@dataclass(frozen=True)
class UpstreamIdentity:
    repository: str
    commit: str
    root: str
    file_sha256: Mapping[str, str]
    verification: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repository": self.repository,
            "commit": self.commit,
            "root": self.root,
            "file_sha256": dict(sorted(self.file_sha256.items())),
            "verification": self.verification,
        }


@dataclass(frozen=True)
class MAGICModelConfig:
    """Entity-level defaults used by the pinned upstream train entry point."""

    num_hidden: int = 64
    num_layers: int = 3
    negative_slope: float = 0.2
    mask_rate: float = 0.5
    alpha_l: float = 3.0
    optimizer: str = "adam"
    learning_rate: float = 0.001
    weight_decay: float = 5e-4
    max_epoch: int = 50

    def __post_init__(self) -> None:
        if self.num_hidden <= 0 or self.num_hidden % 4 != 0:
            raise ValueError("num_hidden must be positive and divisible by four")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0.0 < self.mask_rate < 1.0:
            raise ValueError("mask_rate must be between zero and one")
        if self.max_epoch <= 0:
            raise ValueError("max_epoch must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")

    @property
    def identity(self) -> str:
        return _stable_hash(asdict(self))


@dataclass
class MAGICUpstreamRuntime:
    """Resolved runtime functions and modules; injectable only for contract tests."""

    torch: Any
    dgl: Any
    build_model: Callable[..., Any]
    create_optimizer: Callable[..., Any]
    train_entity_level: Optional[Callable[..., Any]] = None


@dataclass(frozen=True)
class MAGICPreparedGraph:
    graph: Any
    split: SplitType
    snapshot_id: str
    canonical_to_local: Mapping[str, int]
    local_to_canonical: Tuple[str, ...]
    graph_fingerprint: str
    node_feature_dim: int
    edge_feature_dim: int
    edge_count: int

    @property
    def node_ids(self) -> Tuple[str, ...]:
        return self.local_to_canonical


@dataclass(frozen=True)
class MAGICEmbeddingBatch:
    embeddings: Any
    node_ids: Tuple[str, ...]
    canonical_to_local: Mapping[str, int]
    local_to_canonical: Tuple[str, ...]
    split: SplitType
    snapshot_id: str
    graph_fingerprint: str


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_upstream_identity(
    upstream_path: Union[str, os.PathLike] = DEFAULT_UPSTREAM_PATH,
    expected_commit: str = UPSTREAM_COMMIT,
) -> UpstreamIdentity:
    """Fail fast unless the source directory matches the pinned snapshot."""

    if expected_commit != UPSTREAM_COMMIT:
        raise UpstreamIntegrityError(
            "wrong upstream SHA: expected commit must be {}".format(
                UPSTREAM_COMMIT
            )
        )

    root = Path(upstream_path).resolve()
    if not root.is_dir():
        raise UpstreamIntegrityError(
            "missing MAGIC upstream path: {}".format(root)
        )

    actual_hashes: Dict[str, str] = {}
    for relative_path, expected_hash in FROZEN_FILE_SHA256.items():
        source_path = root / relative_path
        if not source_path.is_file():
            raise UpstreamIntegrityError(
                "missing frozen upstream file: {}".format(source_path)
            )
        actual_hash = _sha256_file(source_path)
        actual_hashes[relative_path] = actual_hash
        if actual_hash != expected_hash:
            raise UpstreamIntegrityError(
                "wrong upstream SHA256 for {}: expected {}, got {}".format(
                    relative_path,
                    expected_hash,
                    actual_hash,
                )
            )

    verification = "frozen-file-sha256"
    if (root / ".git").exists():
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        actual_commit = completed.stdout.strip()
        if actual_commit != expected_commit:
            raise UpstreamIntegrityError(
                "wrong upstream git commit: expected {}, got {}".format(
                    expected_commit,
                    actual_commit,
                )
            )
        verification = "git-commit-and-frozen-file-sha256"

    return UpstreamIdentity(
        repository=UPSTREAM_REPOSITORY,
        commit=expected_commit,
        root=str(root),
        file_sha256=actual_hashes,
        verification=verification,
    )


def _module_is_inside(module: Any, root: Path) -> bool:
    filename = getattr(module, "__file__", None)
    if not filename:
        return False
    try:
        return os.path.commonpath(
            [str(Path(filename).resolve()), str(root)]
        ) == str(root)
    except ValueError:
        return False


def _dependency_spec_exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _load_pinned_runtime(root: Path) -> MAGICUpstreamRuntime:
    """Import upstream generic packages transactionally under a process lock."""

    if not _dependency_spec_exists("dgl"):
        raise RealMAGICDependencyError(
            "real MAGIC runtime dependency missing: DGL is not installed"
        )
    if not _dependency_spec_exists("torch"):
        raise RealMAGICDependencyError(
            "real MAGIC runtime dependency missing: PyTorch is not installed"
        )

    try:
        torch_module = importlib.import_module("torch")
        dgl_module = importlib.import_module("dgl")
    except (ImportError, OSError) as exc:
        raise RealMAGICDependencyError(
            "real MAGIC runtime dependency missing or unloadable: {}".format(exc)
        ) from exc

    with _IMPORT_LOCK:
        conflicts = sorted(
            name
            for name in sys.modules
            if name.split(".", 1)[0] in _GENERIC_UPSTREAM_ROOTS
        )
        if conflicts:
            raise UpstreamImportIsolationError(
                "cannot load pinned MAGIC without overwriting existing generic "
                "modules: {}".format(", ".join(conflicts))
            )

        original_path = list(sys.path)
        modules_before = set(sys.modules)
        try:
            sys.path.insert(0, str(root))
            autoencoder = importlib.import_module("model.autoencoder")
            upstream_utils = importlib.import_module("utils.utils")

            if not _module_is_inside(autoencoder, root):
                raise UpstreamImportIsolationError(
                    "loaded model.autoencoder from the wrong source tree"
                )
            if not _module_is_inside(upstream_utils, root):
                raise UpstreamImportIsolationError(
                    "loaded utils.utils from the wrong source tree"
                )

            return MAGICUpstreamRuntime(
                torch=torch_module,
                dgl=dgl_module,
                build_model=autoencoder.build_model,
                create_optimizer=upstream_utils.create_optimizer,
            )
        finally:
            for module_name in list(sys.modules):
                root_name = module_name.split(".", 1)[0]
                if (
                    root_name in _GENERIC_UPSTREAM_ROOTS
                    and module_name not in modules_before
                ):
                    sys.modules.pop(module_name, None)
            sys.path[:] = original_path


def _run_original_entity_training_lifecycle(
    model: Any,
    graphs: Sequence[Any],
    optimizer: Any,
    max_epoch: int,
    device: Any,
    dgl_module: Any,
) -> Any:
    """Mirror the entity-level loop in upstream train.py without its CLI seed."""

    n_train = len(graphs)
    for _ in range(max_epoch):
        for graph in graphs:
            # DGL 1.0.0's g.clone() creates a shallow copy that shares
            # ndata/edata tensor objects with the original. Upstream's
            # encoding_mask_noise() modifies ndata["attr"] in-place on the
            # cloned graph, which corrupts the shared tensor, breaking
            # autograd on the next epoch.
            # We must create a truly independent copy of the graph to match
            # the upstream per-epoch loading lifecycle.
            copied_graph = dgl_module.graph(
                (
                    graph.edges()[0].clone(),
                    graph.edges()[1].clone(),
                ),
                num_nodes=graph.num_nodes(),
            )
            for key, tensor in graph.ndata.items():
                copied_graph.ndata[key] = tensor.clone()
            for key, tensor in graph.edata.items():
                copied_graph.edata[key] = tensor.clone()
            runtime_graph = copied_graph.to(device)
            model.train()
            loss = model(runtime_graph)
            loss /= n_train
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


class MAGICRealBackend:
    """Lifecycle bridge that delegates GAT/GMAE math to the pinned upstream."""

    def __init__(
        self,
        seed: int,
        upstream_path: Union[str, os.PathLike] = DEFAULT_UPSTREAM_PATH,
        config: Optional[MAGICModelConfig] = None,
        device: Any = "cpu",
        _runtime_factory: Optional[
            Callable[[Path], MAGICUpstreamRuntime]
        ] = None,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if seed < 0:
            raise ValueError("seed must be non-negative")

        self.seed = seed
        self.upstream_path = Path(upstream_path).resolve()
        self.config = config or MAGICModelConfig()
        self.device = device
        self._runtime_factory = _runtime_factory
        self._identity = verify_upstream_identity(self.upstream_path)
        self._runtime: Optional[MAGICUpstreamRuntime] = None
        self._model: Optional[Any] = None
        self._seed_manifest: Optional[Any] = None
        self._training_input_fingerprint: Optional[str] = None
        self._model_dimensions: Optional[Tuple[int, int]] = None
        self._checkpoint_metadata: Optional[Dict[str, Any]] = None

    @property
    def upstream_identity(self) -> UpstreamIdentity:
        return self._identity

    @property
    def seed_manifest(self) -> Optional[Any]:
        return self._seed_manifest

    @property
    def training_input_fingerprint(self) -> Optional[str]:
        return self._training_input_fingerprint

    @property
    def checkpoint_metadata(self) -> Optional[Dict[str, Any]]:
        if self._checkpoint_metadata is None:
            return None
        return dict(self._checkpoint_metadata)

    def _ensure_runtime(self) -> MAGICUpstreamRuntime:
        self._identity = verify_upstream_identity(self.upstream_path)
        if self._runtime is None:
            if self._runtime_factory is None:
                self._runtime = _load_pinned_runtime(self.upstream_path)
            else:
                self._runtime = self._runtime_factory(self.upstream_path)
            self._validate_runtime(self._runtime)
        return self._runtime

    @staticmethod
    def _validate_runtime(runtime: MAGICUpstreamRuntime) -> None:
        for attribute in ("torch", "dgl", "build_model", "create_optimizer"):
            if getattr(runtime, attribute, None) is None:
                raise RealMAGICDependencyError(
                    "real MAGIC runtime dependency missing: {}".format(attribute)
                )

    def _set_experiment_seed(self, runtime: MAGICUpstreamRuntime) -> Any:
        # Deliberately lazy: seed.py performs optional Torch/DGL discovery.
        from .seed import MagicSeedController

        controller = MagicSeedController(
            seed=self.seed,
            set_deterministic=True,
            set_cuda=True,
            require_dgl=True,
        )
        manifest = controller.set_all_seeds(dgl_module=runtime.dgl)
        if not manifest.dgl_seed_set:
            raise RealMAGICDependencyError(
                "real MAGIC runtime dependency missing: DGL seed hook unavailable"
            )
        self._seed_manifest = manifest
        return manifest

    @staticmethod
    def _validate_contract_split(
        contract: NeutralGraphContract,
        split: SplitType,
    ) -> None:
        if not isinstance(contract, NeutralGraphContract):
            raise TypeError("contract must be NeutralGraphContract")
        if not isinstance(split, SplitType):
            raise TypeError("split must be SplitType")
        for node in contract.nodes.values():
            if node.split != split:
                raise ValueError("contract contains a node from another split")
        for edge in contract.edges:
            if edge.split != split:
                raise ValueError("contract contains an edge from another split")
        vocabulary = contract.type_vocabulary
        if vocabulary is None or not vocabulary.is_fitted():
            raise ValueError("contract requires a fitted train-only type vocabulary")

    def prepare_graph(
        self,
        contract: NeutralGraphContract,
        split: SplitType,
        snapshot_id: str,
    ) -> MAGICPreparedGraph:
        """Convert a neutral contract to the exact DGL fields used upstream."""

        if not snapshot_id:
            raise ValueError("snapshot_id must be non-empty")
        self._validate_contract_split(contract, split)
        runtime = self._ensure_runtime()
        vocabulary = contract.type_vocabulary
        assert vocabulary is not None

        node_ids: List[str] = []
        canonical_to_local: Dict[str, int] = {}

        def add_node(node_id: str) -> None:
            if node_id not in canonical_to_local:
                canonical_to_local[node_id] = len(node_ids)
                node_ids.append(node_id)

        kept_edges: List[Any] = []
        seen_pairs = set()
        for edge in contract.get_sorted_edges():
            pair = (edge.src, edge.dst)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            if edge.src not in contract.nodes or edge.dst not in contract.nodes:
                raise ValueError("edge references a node missing from contract.nodes")
            add_node(edge.src)
            add_node(edge.dst)
            kept_edges.append(edge)

        for node_id in sorted(contract.nodes):
            add_node(node_id)

        if not node_ids:
            raise ValueError("cannot prepare an empty graph for real MAGIC runtime")

        node_type_ids: List[int] = []
        for node_id in node_ids:
            node_type = contract.nodes[node_id].node_type
            type_id, _ = vocabulary.transform_node_type(node_type)
            node_type_ids.append(type_id)

        edge_type_ids: List[int] = []
        source_ids: List[int] = []
        destination_ids: List[int] = []
        for edge in kept_edges:
            source_ids.append(canonical_to_local[edge.src])
            destination_ids.append(canonical_to_local[edge.dst])
            type_id, _ = vocabulary.transform_edge_type(edge.edge_type)
            edge_type_ids.append(type_id)

        torch_module = runtime.torch
        source_tensor = torch_module.tensor(source_ids, dtype=torch_module.long)
        destination_tensor = torch_module.tensor(
            destination_ids,
            dtype=torch_module.long,
        )
        graph = runtime.dgl.graph(
            (source_tensor, destination_tensor),
            num_nodes=len(node_ids),
        )
        node_types = torch_module.tensor(
            node_type_ids,
            dtype=torch_module.long,
        )
        edge_types = torch_module.tensor(
            edge_type_ids,
            dtype=torch_module.long,
        )
        graph.ndata["type"] = node_types.detach()
        graph.edata["type"] = edge_types.detach()
        graph.ndata["attr"] = (
            torch_module.nn.functional.one_hot(
                node_types,
                num_classes=vocabulary.node_feature_dim,
            ).float().detach()
        )
        graph.edata["attr"] = (
            torch_module.nn.functional.one_hot(
                edge_types,
                num_classes=vocabulary.edge_feature_dim,
            ).float().detach()
        )

        fingerprint_payload = {
            "split": split.value,
            "snapshot_id": snapshot_id,
            "nodes": [
                {
                    "canonical_id": node_id,
                    "local_id": canonical_to_local[node_id],
                    "node_type": contract.nodes[node_id].node_type,
                    "type_id": node_type_ids[canonical_to_local[node_id]],
                }
                for node_id in node_ids
            ],
            "edges": [
                {
                    "src": edge.src,
                    "dst": edge.dst,
                    "edge_type": edge.edge_type,
                    "timestamp": edge.timestamp,
                    "global_event_index": edge.global_event_index,
                }
                for edge in kept_edges
            ],
            "node_feature_dim": vocabulary.node_feature_dim,
            "edge_feature_dim": vocabulary.edge_feature_dim,
            "simple_graph_policy": "first-by-timestamp-and-global-event-index",
            "self_loop_policy": "preserve-input-only",
        }

        return MAGICPreparedGraph(
            graph=graph,
            split=split,
            snapshot_id=snapshot_id,
            canonical_to_local=dict(canonical_to_local),
            local_to_canonical=tuple(node_ids),
            graph_fingerprint=_stable_hash(fingerprint_payload),
            node_feature_dim=vocabulary.node_feature_dim,
            edge_feature_dim=vocabulary.edge_feature_dim,
            edge_count=len(kept_edges),
        )

    def _build_args(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            num_hidden=self.config.num_hidden,
            num_layers=self.config.num_layers,
            negative_slope=self.config.negative_slope,
            mask_rate=self.config.mask_rate,
            alpha_l=self.config.alpha_l,
            n_dim=node_feature_dim,
            e_dim=edge_feature_dim,
        )

    @staticmethod
    def _coerce_graphs(
        train_graphs: Union[
            MAGICPreparedGraph,
            Sequence[MAGICPreparedGraph],
        ],
    ) -> List[MAGICPreparedGraph]:
        if isinstance(train_graphs, MAGICPreparedGraph):
            result = [train_graphs]
        else:
            result = list(train_graphs)
        if not result:
            raise ValueError("fit requires at least one training graph")
        for prepared in result:
            if not isinstance(prepared, MAGICPreparedGraph):
                raise TypeError("fit accepts MAGICPreparedGraph values only")
            if prepared.split != SplitType.TRAIN:
                raise ValueError("fit accepts train graphs only")
        return result

    def fit(
        self,
        train_graphs: Union[
            MAGICPreparedGraph,
            Sequence[MAGICPreparedGraph],
        ],
    ) -> "MAGICRealBackend":
        """Fit only on explicitly train-tagged prepared graphs."""

        prepared_graphs = self._coerce_graphs(train_graphs)
        dimensions = {
            (graph.node_feature_dim, graph.edge_feature_dim)
            for graph in prepared_graphs
        }
        if len(dimensions) != 1:
            raise ValueError("all training graphs must share feature dimensions")
        node_feature_dim, edge_feature_dim = next(iter(dimensions))

        runtime = self._ensure_runtime()
        self._set_experiment_seed(runtime)
        model = runtime.build_model(
            self._build_args(node_feature_dim, edge_feature_dim)
        )
        model = model.to(self.device)
        optimizer = runtime.create_optimizer(
            self.config.optimizer,
            model,
            self.config.learning_rate,
            self.config.weight_decay,
        )
        raw_graphs = [prepared.graph for prepared in prepared_graphs]
        if runtime.train_entity_level is None:
            model = _run_original_entity_training_lifecycle(
                model=model,
                graphs=raw_graphs,
                optimizer=optimizer,
                max_epoch=self.config.max_epoch,
                device=self.device,
                dgl_module=runtime.dgl,
            )
        else:
            model = runtime.train_entity_level(
                model=model,
                graphs=raw_graphs,
                optimizer=optimizer,
                max_epoch=self.config.max_epoch,
                device=self.device,
            )

        self._model = model
        self._model_dimensions = (node_feature_dim, edge_feature_dim)
        self._training_input_fingerprint = _stable_hash(
            [graph.graph_fingerprint for graph in prepared_graphs]
        )
        self._checkpoint_metadata = self._make_checkpoint_metadata()
        return self

    def _make_checkpoint_metadata(self) -> Dict[str, Any]:
        if (
            self._training_input_fingerprint is None
            or self._model_dimensions is None
        ):
            raise MAGICRealBackendError("backend has no fitted training identity")
        return {
            "schema_version": 1,
            "upstream": self._identity.to_dict(),
            "wrapper_identity": WRAPPER_IDENTITY,
            "algorithm_identity": ALGORITHM_IDENTITY,
            "experiment_seed": self.seed,
            "graph_input_fingerprint": self._training_input_fingerprint,
            "config_identity": self.config.identity,
            "config": asdict(self.config),
            "node_feature_dim": self._model_dimensions[0],
            "edge_feature_dim": self._model_dimensions[1],
        }

    def save_checkpoint(
        self,
        checkpoint_path: Union[str, os.PathLike],
    ) -> Path:
        """Save state plus complete experiment identity using runtime Torch."""

        if self._model is None:
            raise MAGICRealBackendError("fit or load_checkpoint must run first")
        runtime = self._ensure_runtime()
        metadata = self._make_checkpoint_metadata()
        destination = Path(checkpoint_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        runtime.torch.save(
            {
                "metadata": metadata,
                "state_dict": self._model.state_dict(),
            },
            str(destination),
        )
        self._checkpoint_metadata = metadata
        return destination

    def _validate_checkpoint_metadata(
        self,
        metadata: Any,
        expected_graph_fingerprint: str,
    ) -> Tuple[int, int]:
        if not isinstance(metadata, dict):
            raise CheckpointIdentityError(
                "checkpoint has no identity metadata"
            )
        required = {
            "upstream",
            "wrapper_identity",
            "algorithm_identity",
            "experiment_seed",
            "graph_input_fingerprint",
            "config_identity",
            "node_feature_dim",
            "edge_feature_dim",
        }
        missing = sorted(required.difference(metadata))
        if missing:
            raise CheckpointIdentityError(
                "checkpoint identity metadata missing: {}".format(
                    ", ".join(missing)
                )
            )

        expected_values = {
            "upstream": self._identity.to_dict(),
            "wrapper_identity": WRAPPER_IDENTITY,
            "algorithm_identity": ALGORITHM_IDENTITY,
            "experiment_seed": self.seed,
            "graph_input_fingerprint": expected_graph_fingerprint,
            "config_identity": self.config.identity,
        }
        for field_name, expected_value in expected_values.items():
            if metadata.get(field_name) != expected_value:
                raise CheckpointIdentityError(
                    "checkpoint identity mismatch for {}".format(field_name)
                )

        node_feature_dim = metadata["node_feature_dim"]
        edge_feature_dim = metadata["edge_feature_dim"]
        if (
            not isinstance(node_feature_dim, int)
            or node_feature_dim <= 0
            or not isinstance(edge_feature_dim, int)
            or edge_feature_dim <= 0
        ):
            raise CheckpointIdentityError(
                "checkpoint feature dimensions are invalid"
            )
        return node_feature_dim, edge_feature_dim

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, os.PathLike],
        expected_graph_fingerprint: str,
    ) -> "MAGICRealBackend":
        """Load only a checkpoint matching this upstream/config/seed/input."""

        if not expected_graph_fingerprint:
            raise CheckpointIdentityError(
                "expected graph/input fingerprint is required"
            )
        source = Path(checkpoint_path)
        if not source.is_file():
            raise FileNotFoundError(str(source))
        runtime = self._ensure_runtime()
        try:
            payload = runtime.torch.load(
                str(source),
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            payload = runtime.torch.load(
                str(source),
                map_location=self.device,
            )
        if not isinstance(payload, dict) or "state_dict" not in payload:
            raise CheckpointIdentityError("invalid MAGIC checkpoint payload")
        metadata = payload.get("metadata")
        dimensions = self._validate_checkpoint_metadata(
            metadata,
            expected_graph_fingerprint,
        )

        self._set_experiment_seed(runtime)
        model = runtime.build_model(self._build_args(*dimensions))
        model.load_state_dict(payload["state_dict"])
        model = model.to(self.device)
        model.eval()

        self._model = model
        self._model_dimensions = dimensions
        self._training_input_fingerprint = expected_graph_fingerprint
        self._checkpoint_metadata = dict(metadata)
        return self

    def embed(
        self,
        prepared_graph: MAGICPreparedGraph,
    ) -> MAGICEmbeddingBatch:
        """Return upstream model.embed output with canonical node identity."""

        if self._model is None or self._model_dimensions is None:
            raise MAGICRealBackendError("fit or load_checkpoint must run first")
        if not isinstance(prepared_graph, MAGICPreparedGraph):
            raise TypeError("prepared_graph must be MAGICPreparedGraph")
        dimensions = (
            prepared_graph.node_feature_dim,
            prepared_graph.edge_feature_dim,
        )
        if dimensions != self._model_dimensions:
            raise ValueError(
                "embedding graph feature dimensions do not match the model"
            )

        runtime = self._ensure_runtime()
        self._model.eval()
        with runtime.torch.no_grad():
            output = self._model.embed(
                prepared_graph.graph.to(self.device)
            )
        embeddings = output.detach().cpu().numpy()
        if len(embeddings.shape) != 2:
            raise MAGICRealBackendError(
                "upstream embed output must be a two-dimensional tensor"
            )
        if embeddings.shape[0] != len(prepared_graph.local_to_canonical):
            raise MAGICRealBackendError(
                "upstream embed output does not match canonical node mapping"
            )

        return MAGICEmbeddingBatch(
            embeddings=embeddings,
            node_ids=prepared_graph.local_to_canonical,
            canonical_to_local=dict(prepared_graph.canonical_to_local),
            local_to_canonical=prepared_graph.local_to_canonical,
            split=prepared_graph.split,
            snapshot_id=prepared_graph.snapshot_id,
            graph_fingerprint=prepared_graph.graph_fingerprint,
        )


__all__ = [
    "ALGORITHM_IDENTITY",
    "CheckpointIdentityError",
    "DEFAULT_UPSTREAM_PATH",
    "FROZEN_FILE_SHA256",
    "MAGICEmbeddingBatch",
    "MAGICModelConfig",
    "MAGICPreparedGraph",
    "MAGICRealBackend",
    "MAGICRealBackendError",
    "MAGICUpstreamRuntime",
    "RealMAGICDependencyError",
    "UPSTREAM_COMMIT",
    "UPSTREAM_REPOSITORY",
    "UpstreamIdentity",
    "UpstreamImportIsolationError",
    "UpstreamIntegrityError",
    "WRAPPER_IDENTITY",
    "verify_upstream_identity",
]
