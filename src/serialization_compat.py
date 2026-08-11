"""
src/serialization_compat.py

PyTorch 2.6 serialization compatibility layer for ORTHRUS-MSTC-PIDS.

Design rationale:
-----------------
PyTorch 2.6 changed torch.load() default from weights_only=False to
weights_only=True. This is a security improvement for untrusted artifacts,
but ORTHRUS's own preprocessing artifacts are project-generated trusted
Python objects (NetworkX MultiDiGraph, PyG TemporalData, etc.) that cannot
be serialized as pure tensors/state_dicts.

Trust boundary:
- weights_only=False can execute arbitrary pickle code
- Only use this for ORTHRUS's own artifact root (preprocessed graphs, TemporalData)
- Never use this for unknown/untrusted artifacts

Usage:
------
    from serialization_compat import load_trusted_torch_artifact

    # Load NetworkX graph (ORTHRUS-generated)
    graph = load_trusted_torch_artifact(
        path,
        expected_type=nx.MultiDiGraph,
    )

    # Load PyG TemporalData (ORTHRUS-generated)
    data = load_trusted_torch_artifact(
        path,
        expected_type=TemporalData,
    ).to("cpu")
"""
from __future__ import annotations

import os
import torch
import warnings
from typing import TYPE_CHECKING, TypeVar, Optional, Type

if TYPE_CHECKING:
    pass

T = TypeVar("T")


def load_trusted_torch_artifact(
    path: str | os.PathLike,
    *,
    map_location: Optional[str] = None,
    expected_type: Optional[Type[T]] = None,
) -> T:
    """
    Load an ORTHRUS-generated artifact with full pickle deserialization.

    Parameters
    ----------
    path : str | os.PathLike
        Path to the torch.save() artifact.
    map_location : str | torch.device | None, optional
        Passed through to torch.load map_location.
    expected_type : Type[T] | None, optional
        If provided, the loaded object will be verified to be an instance
        of this type. A TypeError will be raised if the type does not match.

    Returns
    -------
    T
        The deserialized object, optionally verified against expected_type.

    Raises
    ------
    TypeError
        If expected_type is provided but the loaded object is not an instance.
    FileNotFoundError
        If the artifact path does not exist.
    pickle.UnpicklingError
        If the artifact cannot be unpickled (e.g., corrupted file).

    Security note
    -------------
    This function sets weights_only=False, which allows torch.load to
    execute arbitrary Python code embedded in the pickle stream.

    ONLY use this for ORTHRUS's own preprocessing artifacts from a trusted
    artifact root. Do NOT use this for artifacts from untrusted sources,
    user-provided files, or checkpoints downloaded from the internet.

    Examples
    --------
    Loading a NetworkX MultiDiGraph:

    >>> import networkx as nx
    >>> graph = load_trusted_torch_artifact("graph.pkl", expected_type=nx.MultiDiGraph)

    Loading a PyG TemporalData:

    >>> from torch_geometric.data import TemporalData
    >>> data = load_trusted_torch_artifact("window.TemporalData.simple", expected_type=TemporalData)
    """
    path = os.fspath(path)

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Trusted artifact not found: {path}")

    try:
        obj = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError as exc:
        if "weights_only" in str(exc):
            # Very old PyTorch without weights_only support; fall back to legacy load.
            # Only catch the specific TypeError about weights_only to avoid
            # masking unrelated TypeErrors.
            warnings.warn(
                f"PyTorch version does not support weights_only parameter; "
                f"falling back to legacy torch.load for: {path}",
                UserWarning,
            )
            obj = torch.load(path, map_location=map_location)
        else:
            raise

    if expected_type is not None:
        if not isinstance(obj, expected_type):
            raise TypeError(
                f"Trusted artifact type mismatch: expected {expected_type.__name__}, "
                f"got {type(obj).__name__}. Artifact path: {path}"
            )

    return obj
