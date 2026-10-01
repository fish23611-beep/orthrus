"""Low-overhead, opt-in runtime profiling for MSTC experiments.

The profiler is deliberately independent from model code.  When disabled its
context managers are no-ops and it creates no artifacts.  CUDA events are
resolved in one batch during :meth:`flush`, avoiding a synchronize call for
every measured operation.
"""

from __future__ import annotations

import csv
import json
import math
import os
import resource
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


PROFILE_SCHEMA_VERSION = 1
PROFILE_ENV = "ORTHRUS_PERF_PROFILE"
PROFILE_SAMPLE_ENV = "ORTHRUS_PERF_PROFILE_SAMPLE_EVERY_N_BATCHES"


def _env_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def profile_requested(cli_value: bool = False) -> bool:
    """Return whether profiling was explicitly enabled by CLI or environment."""
    return bool(cli_value) or _env_truthy(os.environ.get(PROFILE_ENV))


def sample_every_from(cli_value: int | None = None) -> int:
    value: Any = cli_value
    if value is None:
        value = os.environ.get(PROFILE_SAMPLE_ENV, "10")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"profiling sample interval must be an integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"profiling sample interval must be positive, got {parsed}")
    return parsed


def _current_rss_mb() -> float | None:
    try:
        with open("/proc/self/statm", encoding="ascii") as handle:
            pages = int(handle.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 ** 2)
    except (OSError, ValueError, IndexError):
        return None


def _peak_rss_mb() -> float | None:
    try:
        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024
    except Exception:
        return None


def _proc_io() -> dict[str, int] | None:
    try:
        result: dict[str, int] = {}
        with open("/proc/self/io", encoding="ascii") as handle:
            for line in handle:
                key, value = line.split(":", 1)
                result[key.strip()] = int(value.strip())
        return result
    except (OSError, ValueError):
        return None


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


class RuntimeProfiler:
    """Aggregate wall-clock and CUDA-event timings without changing semantics."""

    def __init__(
        self,
        *,
        enabled: bool,
        run_root: str | os.PathLike[str] | None = None,
        sample_every_n_batches: int = 10,
        raw_sample_limit: int = 2000,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if sample_every_n_batches <= 0:
            raise ValueError("sample_every_n_batches must be positive")
        if raw_sample_limit < 0:
            raise ValueError("raw_sample_limit must be non-negative")
        self.enabled = bool(enabled)
        self.run_root = Path(run_root) if run_root is not None else None
        self.sample_every_n_batches = int(sample_every_n_batches)
        self.raw_sample_limit = int(raw_sample_limit)
        self.metadata = dict(metadata or {})
        self._samples: dict[str, list[float]] = defaultdict(list)
        self._sums: dict[str, float] = defaultdict(float)
        self._observed_counts: dict[str, int] = defaultdict(int)
        self._parents: dict[str, str] = {}
        self._units: dict[str, str] = {}
        self._counters: dict[str, float] = defaultdict(float)
        self._tags: dict[str, Any] = {}
        self._pending_cuda: list[tuple[str, Any, Any, str | None]] = []
        self._batch_index = 0
        self._detail_active = False
        self._start_wall = time.perf_counter()
        self._start_cpu = time.process_time()
        self._start_io = _proc_io()
        self._flushed = False

    @property
    def detail_active(self) -> bool:
        return self.enabled and self._detail_active

    def add_counter(self, name: str, value: float = 1.0) -> None:
        if self.enabled:
            self._counters[name] += float(value)

    def set_tag(self, name: str, value: Any) -> None:
        if self.enabled:
            self._tags[name] = value

    def record(
        self,
        name: str,
        seconds: float,
        *,
        parent: str | None = None,
        unit: str = "seconds",
    ) -> None:
        if not self.enabled:
            return
        duration = max(0.0, float(seconds))
        self._observed_counts[name] += 1
        self._sums[name] += duration
        if len(self._samples[name]) < self.raw_sample_limit:
            self._samples[name].append(duration)
        if parent:
            self._parents[name] = parent
        self._units[name] = unit

    @contextmanager
    def timer(
        self,
        name: str,
        *,
        detail: bool = False,
        parent: str | None = None,
    ) -> Iterator[None]:
        active = self.enabled and (not detail or self._detail_active)
        if not active:
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, time.perf_counter() - started, parent=parent)

    @contextmanager
    def cuda_timer(
        self,
        name: str,
        *,
        detail: bool = True,
        parent: str | None = None,
    ) -> Iterator[None]:
        active = self.enabled and (not detail or self._detail_active)
        if not active:
            yield
            return
        try:
            import torch
            cuda_ready = torch.cuda.is_available()
        except Exception:
            cuda_ready = False
        if not cuda_ready:
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._pending_cuda.append((name, start, end, parent))

    def begin_batch(self) -> tuple[float, bool] | None:
        """Begin one batch and activate deterministic detailed sampling."""
        if not self.enabled:
            return None
        self._batch_index += 1
        previous = self._detail_active
        self._detail_active = (self._batch_index - 1) % self.sample_every_n_batches == 0
        return time.perf_counter(), previous

    def end_batch(self, token: tuple[float, bool] | None, total_name: str) -> None:
        """Finish a batch begun by :meth:`begin_batch`."""
        if token is None:
            return
        started, previous = token
        self.record(total_name, time.perf_counter() - started)
        self._detail_active = previous

    def profile_batches(
        self,
        batches: Any,
        *,
        prepare_name: str = "training.T_batch_prepare",
    ) -> Iterator[tuple[Any, tuple[float, bool] | None]]:
        """Yield batches with retrieval time included in the sampled batch total."""
        if not self.enabled:
            for batch in batches:
                yield batch, None
            return
        iterator = iter(batches)
        while True:
            token = self.begin_batch()
            prepare_started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                assert token is not None
                _, previous = token
                self._detail_active = previous
                self._batch_index -= 1
                return
            if self.detail_active:
                self.record(
                    prepare_name, time.perf_counter() - prepare_started,
                    parent="training.T_total_batch",
                )
            yield batch, token

    @contextmanager
    def batch(self, total_name: str = "training.batch.total") -> Iterator[None]:
        if not self.enabled:
            yield
            return
        self._batch_index += 1
        previous = self._detail_active
        self._detail_active = (self._batch_index - 1) % self.sample_every_n_batches == 0
        try:
            with self.timer(total_name):
                yield
        finally:
            self._detail_active = previous

    def _resolve_cuda(self) -> None:
        if not self._pending_cuda:
            return
        try:
            import torch
            torch.cuda.synchronize()
            for name, start, end, parent in self._pending_cuda:
                self.record(name, start.elapsed_time(end) / 1000.0, parent=parent)
        except Exception as exc:
            self._tags["cuda_event_resolution_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            self._pending_cuda.clear()

    def _metric_summary(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for name in sorted(self._observed_counts):
            values = sorted(self._samples.get(name, ()))
            observed = int(self._observed_counts[name])
            total = float(self._sums[name])
            count = len(values)
            entry: dict[str, Any] = {
                "count": observed,
                "retained_sample_count": count,
                "samples_truncated": observed > count,
                "total_seconds": total,
                "mean_seconds": total / observed if observed else 0.0,
                "median_seconds": _percentile(values, 0.50),
                "p90_seconds": _percentile(values, 0.90),
                "p95_seconds": _percentile(values, 0.95),
                "max_seconds": max(values) if values else 0.0,
                "unit": self._units.get(name, "seconds"),
                "parent": self._parents.get(name),
                "percentage_of_parent": None,
            }
            result[name] = entry
        for name, entry in result.items():
            parent_name = entry.get("parent")
            parent = result.get(parent_name) if parent_name else None
            if parent and parent["mean_seconds"] > 0:
                # Mean ratios remain meaningful when detail sampling makes child
                # and parent counts differ.
                entry["percentage_of_parent"] = (
                    100.0 * entry["mean_seconds"] / parent["mean_seconds"]
                )
        return result

    def _memory(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "rss_current_mb": _current_rss_mb(),
            "rss_peak_mb": _peak_rss_mb(),
            "peak_gpu_memory_allocated_mb": None,
            "peak_gpu_memory_reserved_mb": None,
        }
        try:
            import torch
            if torch.cuda.is_available():
                result["peak_gpu_memory_allocated_mb"] = (
                    torch.cuda.max_memory_allocated() / (1024 ** 2)
                )
                result["peak_gpu_memory_reserved_mb"] = (
                    torch.cuda.max_memory_reserved() / (1024 ** 2)
                )
        except Exception:
            pass
        return result

    def _io_delta(self) -> dict[str, int] | None:
        end = _proc_io()
        if self._start_io is None or end is None:
            return None
        return {
            key: int(end.get(key, 0) - self._start_io.get(key, 0))
            for key in sorted(set(end) | set(self._start_io))
        }

    def payload(self, *, status: str, error: str | None = None) -> dict[str, Any]:
        metrics = self._metric_summary()
        elapsed = max(0.0, time.perf_counter() - self._start_wall)
        throughput: dict[str, float] = {}
        for prefix in ("training", "evaluation"):
            seconds_entry = metrics.get(f"{prefix}.total")
            seconds = seconds_entry["total_seconds"] if seconds_entry else elapsed
            if seconds > 0:
                for item in ("events", "batches", "windows"):
                    value = self._counters.get(f"{prefix}.{item}")
                    if value is not None:
                        throughput[f"{prefix}.{item}_per_second"] = float(value) / seconds
        return {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "enabled": True,
            "status": status,
            "error": error,
            "sample_every_n_batches": self.sample_every_n_batches,
            "raw_sample_limit": self.raw_sample_limit,
            "wall_seconds": elapsed,
            "cpu_seconds": max(0.0, time.process_time() - self._start_cpu),
            "metadata": self.metadata,
            "tags": dict(sorted(self._tags.items())),
            "counters": dict(sorted(self._counters.items())),
            "throughput": throughput,
            "memory": self._memory(),
            "io_delta": self._io_delta(),
            "metrics": metrics,
            "raw_samples_seconds": {
                name: list(values) for name, values in sorted(self._samples.items())
            },
        }

    def flush(self, *, status: str = "completed", error: str | None = None) -> dict[str, Any] | None:
        if not self.enabled or self.run_root is None:
            return None
        self._resolve_cuda()
        payload = self.payload(status=status, error=error)
        profile_dir = self.run_root / "profiling"
        profile_dir.mkdir(parents=True, exist_ok=True)
        json_path = profile_dir / "perf_profile.json"
        csv_path = profile_dir / "perf_profile.csv"
        summary_path = profile_dir / "perf_summary.json"

        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)

        fieldnames = [
            "stage", "count", "retained_sample_count", "samples_truncated",
            "total_seconds", "mean_seconds", "median_seconds", "p90_seconds",
            "p95_seconds", "max_seconds", "parent", "percentage_of_parent",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for stage, metric in payload["metrics"].items():
                writer.writerow({"stage": stage, **{key: metric.get(key) for key in fieldnames[1:]}})

        summary = {key: value for key, value in payload.items() if key != "raw_samples_seconds"}
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        self._flushed = True
        return payload


_ACTIVE_PROFILER = RuntimeProfiler(enabled=False)


def configure_profiler(
    *,
    enabled: bool,
    run_root: str | os.PathLike[str] | None,
    sample_every_n_batches: int = 10,
    raw_sample_limit: int = 2000,
    metadata: dict[str, Any] | None = None,
) -> RuntimeProfiler:
    global _ACTIVE_PROFILER
    _ACTIVE_PROFILER = RuntimeProfiler(
        enabled=enabled,
        run_root=run_root,
        sample_every_n_batches=sample_every_n_batches,
        raw_sample_limit=raw_sample_limit,
        metadata=metadata,
    )
    return _ACTIVE_PROFILER


def get_profiler() -> RuntimeProfiler:
    return _ACTIVE_PROFILER


def finish_profiler(*, status: str, error: str | None = None) -> dict[str, Any] | None:
    return _ACTIVE_PROFILER.flush(status=status, error=error)
