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
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from runtime.memory_metrics import current_rss_mb, peak_rss_mb


PROFILE_SCHEMA_VERSION = 2
PROFILE_ENV = "ORTHRUS_PERF_PROFILE"
PROFILE_SAMPLE_ENV = "ORTHRUS_PERF_PROFILE_SAMPLE_EVERY_N_BATCHES"
TORCH_TRACE_MAX_ACTIVE_BATCHES = 4
TORCH_TRACE_MAX_WARMUP_BATCHES = 16
TORCH_PROFILE_TOP_LIMIT = 50

_TARGET_OPERATOR_NAMES = (
    "aten::linear",
    "aten::addmm",
    "aten::item",
    "aten::_local_scalar_dense",
    "aten::to",
    "aten::_to_copy",
    "aten::copy_",
)
_EXPLICIT_CUDA_SYNC_NAMES = (
    "cudadevicesynchronize",
    "cudastreamsynchronize",
    "cudaeventsynchronize",
    "cudacontextsynchronize",
)
_CUDA_TRANSFER_RUNTIME_NAMES = (
    "cudamemcpy",
    "memcpy hto",
    "memcpy dto",
    "memcpy dtoh",
    "memcpy htod",
)
_CUDA_LAUNCH_NAMES = (
    "cudalaunchkernel",
    "cudalaunchcooperativekernel",
)


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
    return current_rss_mb()


def _peak_rss_mb() -> float | None:
    return peak_rss_mb()


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


@dataclass
class _TimerFrame:
    name: str
    started: float
    parent: str | None
    child_seconds: float = 0.0


class RuntimeProfiler:
    """Aggregate wall-clock and CUDA-event timings without changing semantics."""

    def __init__(
        self,
        *,
        enabled: bool,
        run_root: str | os.PathLike[str] | None = None,
        basic_timing: bool = False,
        inventory_history_stats: bool = False,
        sample_every_n_batches: int = 10,
        raw_sample_limit: int = 2000,
        metadata: dict[str, Any] | None = None,
        torch_trace_batches: int = 0,
        torch_trace_warmup_batches: int = 0,
        torch_trace_window_start: int = 0,
    ) -> None:
        if sample_every_n_batches <= 0:
            raise ValueError("sample_every_n_batches must be positive")
        if raw_sample_limit < 0:
            raise ValueError("raw_sample_limit must be non-negative")
        if torch_trace_batches < 0:
            raise ValueError("torch_trace_batches must be non-negative")
        if torch_trace_batches > TORCH_TRACE_MAX_ACTIVE_BATCHES:
            raise ValueError(
                "torch_trace_batches must be at most "
                f"{TORCH_TRACE_MAX_ACTIVE_BATCHES}"
            )
        if torch_trace_warmup_batches < 0:
            raise ValueError("torch_trace_warmup_batches must be non-negative")
        if torch_trace_warmup_batches > TORCH_TRACE_MAX_WARMUP_BATCHES:
            raise ValueError(
                "torch_trace_warmup_batches must be at most "
                f"{TORCH_TRACE_MAX_WARMUP_BATCHES}"
            )
        if torch_trace_window_start < 0:
            raise ValueError("torch_trace_window_start must be non-negative")
        self.enabled = bool(enabled)
        self.basic_timing = bool(basic_timing)
        self.inventory_history_stats = bool(inventory_history_stats)
        self.run_root = Path(run_root) if run_root is not None else None
        self.sample_every_n_batches = int(sample_every_n_batches)
        self.raw_sample_limit = int(raw_sample_limit)
        self.torch_trace_batches = int(torch_trace_batches)
        self.torch_trace_warmup_batches = int(torch_trace_warmup_batches)
        self.torch_trace_window_start = int(torch_trace_window_start)
        self.metadata = dict(metadata or {})
        self._samples: dict[str, list[float]] = defaultdict(list)
        self._self_samples: dict[str, list[float]] = defaultdict(list)
        self._sums: dict[str, float] = defaultdict(float)
        self._self_sums: dict[str, float] = defaultdict(float)
        self._observed_counts: dict[str, int] = defaultdict(int)
        self._parents: dict[str, str] = {}
        self._domains: dict[str, str] = {}
        self._counters: dict[str, float] = defaultdict(float)
        self._tags: dict[str, Any] = {}
        self._pending_cuda: list[tuple[str, Any, Any, str | None, dict[str, Any] | None]] = []
        self._timer_stack: list[_TimerFrame] = []
        self._observations: dict[str, list[float]] = defaultdict(list)
        self._observation_units: dict[str, str] = {}
        self._batch_rows: list[dict[str, Any]] = []
        self._current_batch_row: dict[str, Any] | None = None
        self._context: dict[str, Any] = {}
        self._torch_profiler: Any = None
        self._torch_trace_seen = 0
        self._torch_trace_path: str | None = None
        self._torch_profile_summary_path: str | None = None
        self._torch_profile_callback_ran = False
        self._batch_index = 0
        self._detail_active = False
        self._start_wall = time.perf_counter()
        self._start_cpu = time.process_time()
        self._start_io = _proc_io() if self.enabled else None
        self._flushed = False

    @property
    def timing_active(self) -> bool:
        return self.enabled or self.basic_timing

    @property
    def detail_active(self) -> bool:
        return self.enabled and self._detail_active

    def set_context(self, **values: Any) -> None:
        if self.timing_active:
            self._context.update(values)

    def clear_context(self, *names: str) -> None:
        if self.timing_active:
            for name in names:
                self._context.pop(name, None)

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
        self_seconds: float | None = None,
        domain: str = "cpu_wall",
    ) -> None:
        if not self.enabled:
            return
        duration = max(0.0, float(seconds))
        exclusive = duration if self_seconds is None else max(0.0, float(self_seconds))
        self._observed_counts[name] += 1
        self._sums[name] += duration
        self._self_sums[name] += exclusive
        if len(self._samples[name]) < self.raw_sample_limit:
            self._samples[name].append(duration)
            self._self_samples[name].append(exclusive)
        if parent:
            self._parents[name] = parent
        self._domains[name] = domain
        if self.detail_active and self._current_batch_row is not None:
            key = f"time.{name}"
            self._current_batch_row[key] = float(self._current_batch_row.get(key, 0.0)) + duration
        if self._timer_stack and parent == self._timer_stack[-1].name:
            self._timer_stack[-1].child_seconds += duration

    def record_batch_stat(self, name: str, value: Any, *, unit: str = "count") -> None:
        if not self.detail_active or self._current_batch_row is None:
            return
        stored = value if isinstance(value, (bool, int)) else float(value)
        self._current_batch_row[name] = stored
        if isinstance(stored, (int, float)) and not isinstance(stored, bool):
            values = self._observations[name]
            if len(values) < self.raw_sample_limit:
                values.append(float(stored))
            self._observation_units[name] = unit

    def record_basic_batch_stat(self, name: str, value: Any) -> None:
        # Store only scalars already computed by the training path.
        if not self.timing_active or self._current_batch_row is None:
            return
        self._current_batch_row[name] = (
            value if isinstance(value, (bool, int)) else float(value)
        )

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
        inferred_parent = parent or (self._timer_stack[-1].name if self._timer_stack else None)
        frame = _TimerFrame(name, time.perf_counter(), inferred_parent)
        self._timer_stack.append(frame)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - frame.started
            if self._timer_stack.pop() is not frame:
                raise RuntimeError("runtime profiler timer stack corruption")
            self.record(
                name, elapsed, parent=inferred_parent,
                self_seconds=max(0.0, elapsed - frame.child_seconds),
            )

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
            self._pending_cuda.append((name, start, end, parent, self._current_batch_row))

    @contextmanager
    def record_function(self, name: str) -> Iterator[None]:
        """Add a Torch Profiler range only while an opt-in trace is active."""
        if self._torch_profiler is None:
            yield
            return
        import torch
        with torch.profiler.record_function(name):
            yield

    @staticmethod
    def _profile_number(event: Any, *names: str) -> float:
        for name in names:
            value = getattr(event, name, None)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return 0.0

    @classmethod
    def _profile_row(cls, event: Any) -> dict[str, Any]:
        key = str(getattr(event, "key", getattr(event, "name", "<unknown>")))
        shapes = getattr(event, "input_shapes", None)
        return {
            "name": key,
            "calls": int(getattr(event, "count", 1) or 0),
            "self_cpu_time_us": cls._profile_number(event, "self_cpu_time_total"),
            "cpu_total_time_us": cls._profile_number(event, "cpu_time_total"),
            "self_cuda_time_us": cls._profile_number(
                event, "self_cuda_time_total", "self_device_time_total"
            ),
            "cuda_total_time_us": cls._profile_number(
                event, "cuda_time_total", "device_time_total"
            ),
            "input_shapes": shapes if shapes is not None else [],
        }

    @staticmethod
    def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "calls": sum(int(row["calls"]) for row in rows),
            "self_cpu_time_us": sum(float(row["self_cpu_time_us"]) for row in rows),
            "cpu_total_time_us": sum(float(row["cpu_total_time_us"]) for row in rows),
            "self_cuda_time_us": sum(float(row["self_cuda_time_us"]) for row in rows),
            "cuda_total_time_us": sum(float(row["cuda_total_time_us"]) for row in rows),
        }

    @classmethod
    def _rows_matching(
        cls, rows: list[dict[str, Any]], needles: tuple[str, ...]
    ) -> dict[str, Any]:
        lowered = tuple(value.lower() for value in needles)
        matches = [
            row for row in rows
            if any(needle in str(row["name"]).lower() for needle in lowered)
        ]
        return {
            **cls._aggregate_rows(matches),
            "matched_names": sorted({row["name"] for row in matches}),
        }

    def _write_torch_profile_artifacts(self, torch_profiler: Any) -> None:
        if self.run_root is None or self._torch_profile_callback_ran:
            return
        profile_dir = self.run_root / "profiling"
        profile_dir.mkdir(parents=True, exist_ok=True)

        trace_path = profile_dir / "torch_trace.json"
        torch_profiler.export_chrome_trace(str(trace_path))
        self._torch_trace_path = str(trace_path)

        averaged = list(torch_profiler.key_averages(group_by_input_shape=True))
        rows = [self._profile_row(event) for event in averaged]
        operator_rows = [
            row for row in rows
            if row["name"].startswith(
                ("aten::", "autograd::", "torch::autograd::")
            )
        ]
        stage_rows = [row for row in rows if row["name"].startswith("MSTC::")]

        raw_events = list(torch_profiler.events())
        kernel_calls: Counter[str] = Counter()
        kernel_time_us: defaultdict[str, float] = defaultdict(float)
        device_transfer_calls: Counter[str] = Counter()
        device_transfer_time_us: defaultdict[str, float] = defaultdict(float)
        raw_event_rows: list[dict[str, Any]] = []
        for event in raw_events:
            name = str(getattr(event, "name", getattr(event, "key", "<unknown>")))
            device_type = str(getattr(event, "device_type", "")).lower()
            event_row = {
                "name": name,
                "calls": 1,
                "self_cpu_time_us": self._profile_number(event, "self_cpu_time_total"),
                "cpu_total_time_us": self._profile_number(event, "cpu_time_total"),
                "self_cuda_time_us": self._profile_number(
                    event, "self_cuda_time_total", "self_device_time_total"
                ),
                "cuda_total_time_us": self._profile_number(
                    event, "cuda_time_total", "device_time_total"
                ),
            }
            raw_event_rows.append(event_row)
            if "cuda" in device_type:
                device_duration = max(
                    event_row["self_cuda_time_us"], event_row["cuda_total_time_us"]
                )
                if "memcpy" in name.lower() or "memset" in name.lower():
                    device_transfer_calls[name] += 1
                    device_transfer_time_us[name] += device_duration
                else:
                    kernel_calls[name] += 1
                    kernel_time_us[name] += device_duration

        top_kernels = [
            {"name": name, "calls": kernel_calls[name], "device_time_us": duration}
            for name, duration in sorted(
                kernel_time_us.items(), key=lambda item: item[1], reverse=True
            )[:TORCH_PROFILE_TOP_LIMIT]
        ]
        target_operators = {
            name: self._aggregate_rows(
                [row for row in operator_rows if row["name"] == name]
            )
            for name in _TARGET_OPERATOR_NAMES
        }
        autograd_scheduler = self._rows_matching(
            rows, ("autograd::engine::evaluate_function",)
        )
        backward_ops = self._rows_matching(
            rows, ("backward", "accumulategrad")
        )
        explicit_sync = self._rows_matching(raw_event_rows, _EXPLICIT_CUDA_SYNC_NAMES)
        runtime_transfers = self._rows_matching(
            raw_event_rows, _CUDA_TRANSFER_RUNTIME_NAMES
        )
        launches = self._rows_matching(raw_event_rows, _CUDA_LAUNCH_NAMES)

        top_cpu = sorted(
            operator_rows, key=lambda row: row["self_cpu_time_us"], reverse=True
        )[:TORCH_PROFILE_TOP_LIMIT]
        top_cuda = sorted(
            operator_rows, key=lambda row: row["self_cuda_time_us"], reverse=True
        )[:TORCH_PROFILE_TOP_LIMIT]
        summary = {
            "schema_version": 1,
            "timing_unit": "microseconds",
            "capture": {
                "warmup_batches": self.torch_trace_warmup_batches,
                "active_batches": self.torch_trace_batches,
                "observed_steps": self._torch_trace_seen,
                "window_start": self.torch_trace_window_start,
                "chrome_trace": str(trace_path),
                "record_shapes": True,
                "profile_memory": False,
            },
            "target_operators": target_operators,
            "synchronization": {
                "explicit_cuda_runtime_sync": explicit_sync,
                "scalar_extraction_operators": {
                    name: target_operators[name]
                    for name in ("aten::item", "aten::_local_scalar_dense")
                },
            },
            "data_movement": {
                "cuda_runtime_transfer_events": runtime_transfers,
                "cuda_device_transfer_activities": {
                    "calls": sum(device_transfer_calls.values()),
                    "unique_names": len(device_transfer_calls),
                    "device_time_us": sum(device_transfer_time_us.values()),
                    "names": dict(sorted(device_transfer_calls.items())),
                },
                "operators": {
                    name: target_operators[name]
                    for name in ("aten::to", "aten::_to_copy", "aten::copy_")
                },
            },
            "cuda_launch": launches,
            "cuda_kernels": {
                "calls": sum(kernel_calls.values()),
                "unique_names": len(kernel_calls),
                "device_time_us": sum(kernel_time_us.values()),
                "top_by_device_time": top_kernels,
            },
            "autograd": {
                "engine_evaluate_function": autograd_scheduler,
                "backward_named_events": backward_ops,
                "interpretation": (
                    "CPU Autograd events include engine scheduling and do not by "
                    "themselves prove that differentiable tensor math ran on CPU."
                ),
            },
            "top_operators_by_self_cpu_time": top_cpu,
            "top_operators_by_self_cuda_time": top_cuda,
            "stages": stage_rows,
        }
        summary_path = profile_dir / "torch_profile_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        self._torch_profile_summary_path = str(summary_path)

        fields = [
            "name", "calls", "self_cpu_time_us", "cpu_total_time_us",
            "self_cuda_time_us", "cuda_total_time_us", "input_shapes",
        ]
        for filename, selected_rows in (
            ("torch_operator_summary.csv", operator_rows),
            ("torch_stage_summary.csv", stage_rows),
        ):
            with (profile_dir / filename).open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for row in selected_rows:
                    output = dict(row)
                    output["input_shapes"] = json.dumps(
                        output["input_shapes"], separators=(",", ":")
                    )
                    writer.writerow(output)
        self._torch_profile_callback_ran = True

    def _on_torch_trace_ready(self, torch_profiler: Any) -> None:
        try:
            self._write_torch_profile_artifacts(torch_profiler)
        except Exception as exc:
            self._tags["torch_trace_error"] = f"{type(exc).__name__}: {exc}"

    def _start_torch_trace_if_needed(self) -> None:
        if (
            self.torch_trace_batches <= 0
            or int(self._context.get("window_index", -1)) < self.torch_trace_window_start
            or self._torch_profiler is not None
            or self._torch_trace_seen >= (
                self.torch_trace_warmup_batches + self.torch_trace_batches
            )
        ):
            return
        try:
            import torch
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self._torch_profiler = torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(
                    wait=0,
                    warmup=self.torch_trace_warmup_batches,
                    active=self.torch_trace_batches,
                    repeat=1,
                ),
                on_trace_ready=self._on_torch_trace_ready,
                record_shapes=True,
                profile_memory=False,
                with_stack=False,
                with_flops=False,
                with_modules=False,
            )
            self._torch_profiler.start()
        except Exception as exc:
            self._tags["torch_trace_error"] = f"{type(exc).__name__}: {exc}"
            self._torch_trace_seen = (
                self.torch_trace_warmup_batches + self.torch_trace_batches
            )
            self._torch_profiler = None

    def _stop_torch_trace(self) -> None:
        if self._torch_profiler is None:
            return
        try:
            torch_profiler = self._torch_profiler
            torch_profiler.stop()
            if not self._torch_profile_callback_ran:
                self._write_torch_profile_artifacts(torch_profiler)
        except Exception as exc:
            self._tags["torch_trace_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            self._torch_profiler = None

    def _trace_step(self) -> None:
        if self._torch_profiler is None:
            return
        self._torch_trace_seen += 1
        self._torch_profiler.step()
        if self._torch_trace_seen >= (
            self.torch_trace_warmup_batches + self.torch_trace_batches
        ):
            self._stop_torch_trace()

    def begin_batch(self) -> tuple[_TimerFrame, bool] | None:
        """Begin one batch and activate deterministic detailed sampling."""
        if not self.timing_active:
            return None
        self._batch_index += 1
        previous = self._detail_active
        self._detail_active = (
            self.enabled
            and (self._batch_index - 1) % self.sample_every_n_batches == 0
        )
        frame = _TimerFrame("training.T_total_batch", time.perf_counter(), None)
        self._timer_stack.append(frame)
        self._current_batch_row = {
            "batch_index": self._batch_index,
            "wall_time_start_unix": time.time(),
            "detail_sampled": self._detail_active,
            "basic_timing": self.basic_timing,
            **self._context,
        }
        if self._detail_active:
            self._start_torch_trace_if_needed()
            rss = _current_rss_mb()
            if rss is not None:
                self.record_batch_stat("cpu_rss_start_mb", rss, unit="MB")
        return frame, previous

    def end_batch(self, token: tuple[_TimerFrame, bool] | None, total_name: str) -> None:
        """Finish a batch begun by :meth:`begin_batch`."""
        if token is None:
            return
        frame, previous = token
        elapsed = time.perf_counter() - frame.started
        if self._timer_stack.pop() is not frame:
            raise RuntimeError("runtime profiler batch timer stack corruption")
        self.record(total_name, elapsed, self_seconds=max(0.0, elapsed - frame.child_seconds))
        if self._current_batch_row is not None:
            self._current_batch_row["total_seconds"] = elapsed
            self._current_batch_row["wall_time_end_unix"] = time.time()
            if self._detail_active:
                rss = _current_rss_mb()
                if rss is not None:
                    self.record_batch_stat("cpu_rss_end_mb", rss, unit="MB")
            self._batch_rows.append(self._current_batch_row)
        self._current_batch_row = None
        self._resolve_ready_cuda()
        self._trace_step()
        self._detail_active = previous

    def profile_batches(
        self,
        batches: Any,
        *,
        prepare_name: str = "training.T_batch_prepare",
    ) -> Iterator[tuple[Any, tuple[_TimerFrame, bool] | None]]:
        """Yield batches with retrieval time included in the sampled batch total."""
        if not self.timing_active:
            for batch in batches:
                yield batch, None
            return
        iterator = iter(batches)
        while True:
            token = self.begin_batch()
            prepare_started = time.perf_counter()
            try:
                with self.record_function("MSTC::batch_prepare"):
                    batch = next(iterator)
            except StopIteration:
                assert token is not None
                frame, previous = token
                if self._timer_stack.pop() is not frame:
                    raise RuntimeError("runtime profiler batch timer stack corruption")
                self._current_batch_row = None
                self._detail_active = previous
                self._batch_index -= 1
                return
            if self.detail_active:
                elapsed = time.perf_counter() - prepare_started
                self.record(
                    prepare_name, elapsed,
                    parent="training.T_total_batch",
                )
            yield batch, token

    @contextmanager
    def batch(self, total_name: str = "training.batch.total") -> Iterator[None]:
        token = self.begin_batch()
        try:
            yield
        finally:
            self.end_batch(token, total_name)

    def _resolve_ready_cuda(self) -> None:
        if not self._pending_cuda:
            return
        remaining = []
        for name, start, end, parent, batch_row in self._pending_cuda:
            try:
                if not end.query():
                    remaining.append((name, start, end, parent, batch_row))
                    continue
                elapsed = start.elapsed_time(end) / 1000.0
                self.record(
                    name, elapsed, parent=parent, self_seconds=elapsed,
                    domain="cuda_event_interval",
                )
                if batch_row is not None:
                    key = f"time.{name}"
                    batch_row[key] = float(batch_row.get(key, 0.0)) + elapsed
            except Exception:
                remaining.append((name, start, end, parent, batch_row))
        self._pending_cuda = remaining

    def _resolve_cuda(self) -> None:
        if not self._pending_cuda:
            return
        try:
            import torch
            torch.cuda.synchronize()
            for name, start, end, parent, batch_row in self._pending_cuda:
                elapsed = start.elapsed_time(end) / 1000.0
                self.record(
                    name, elapsed, parent=parent, self_seconds=elapsed,
                    domain="cuda_event_interval",
                )
                if batch_row is not None:
                    key = f"time.{name}"
                    batch_row[key] = float(batch_row.get(key, 0.0)) + elapsed
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
            self_values = sorted(self._self_samples.get(name, ()))
            self_total = float(self._self_sums[name])
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
                "self_total_seconds": self_total,
                "self_mean_seconds": self_total / observed if observed else 0.0,
                "self_median_seconds": _percentile(self_values, 0.50),
                "self_p95_seconds": _percentile(self_values, 0.95),
                "timing_domain": self._domains.get(name, "cpu_wall"),
                "parent": self._parents.get(name),
                "percentage_of_parent": None,
            }
            result[name] = entry
        for name, entry in result.items():
            parent_name = entry.get("parent")
            parent = result.get(parent_name) if parent_name else None
            if (
                parent
                and parent["mean_seconds"] > 0
                and entry["count"] == parent["count"]
            ):
                entry["percentage_of_parent"] = (
                    100.0 * entry["mean_seconds"] / parent["mean_seconds"]
                )
        return result

    def _observation_summary(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for name, retained in sorted(self._observations.items()):
            values = sorted(retained)
            result[name] = {
                "retained_sample_count": len(values),
                "median": _percentile(values, 0.50),
                "p90": _percentile(values, 0.90),
                "p95": _percentile(values, 0.95),
                "min": min(values) if values else 0.0,
                "max": max(values) if values else 0.0,
                "unit": self._observation_units.get(name, "count"),
            }
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
            "torch_trace_batches": self.torch_trace_batches,
            "torch_trace_warmup_batches": self.torch_trace_warmup_batches,
            "torch_trace_window_start": self.torch_trace_window_start,
            "torch_trace_path": self._torch_trace_path,
            "torch_profile_summary_path": self._torch_profile_summary_path,
            "wall_seconds": elapsed,
            "cpu_seconds": max(0.0, time.process_time() - self._start_cpu),
            "metadata": self.metadata,
            "tags": dict(sorted(self._tags.items())),
            "counters": dict(sorted(self._counters.items())),
            "throughput": throughput,
            "memory": self._memory(),
            "io_delta": self._io_delta(),
            "metrics": metrics,
            "observations": self._observation_summary(),
            "raw_samples_seconds": {
                name: list(values) for name, values in sorted(self._samples.items())
            },
            "raw_self_samples_seconds": {
                name: list(values) for name, values in sorted(self._self_samples.items())
            },
        }

    def flush(self, *, status: str = "completed", error: str | None = None) -> dict[str, Any] | None:
        if not self.timing_active or self.run_root is None:
            return None
        if not self.enabled:
            timing_dir = self.run_root / "benchmark_timing"
            timing_dir.mkdir(parents=True, exist_ok=True)
            batch_fields = sorted({key for row in self._batch_rows for key in row})
            if batch_fields:
                preferred = [
                    "batch_index", "epoch", "window_index", "window_id",
                    "detail_sampled", "basic_timing", "batch_events", "total_seconds",
                ]
                batch_fields = [key for key in preferred if key in batch_fields] + [
                    key for key in batch_fields if key not in preferred
                ]
                with (timing_dir / "basic_batches.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=batch_fields)
                    writer.writeheader()
                    writer.writerows(self._batch_rows)
                with (timing_dir / "basic_batches.json").open(
                    "w", encoding="utf-8"
                ) as handle:
                    json.dump(
                        self._batch_rows, handle, indent=2,
                        sort_keys=True, allow_nan=False,
                    )
            durations = sorted(
                float(row["total_seconds"])
                for row in self._batch_rows
                if "total_seconds" in row
            )
            summary = {
                "schema_version": 1,
                "detailed_profiler_enabled": False,
                "basic_timing_enabled": True,
                "status": status,
                "error": error,
                "batch_count": len(durations),
                "total_batch_seconds": sum(durations),
                "p50_batch_seconds": _percentile(durations, 0.50),
                "p95_batch_seconds": _percentile(durations, 0.95),
                "timing_semantics": "host perf_counter; no CUDA synchronization",
            }
            with (timing_dir / "basic_summary.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
            self._flushed = True
            return summary
        self._stop_torch_trace()
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
            "p95_seconds", "max_seconds", "self_total_seconds", "self_mean_seconds",
            "self_median_seconds", "self_p95_seconds", "timing_domain", "parent",
            "percentage_of_parent",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for stage, metric in payload["metrics"].items():
                writer.writerow({"stage": stage, **{key: metric.get(key) for key in fieldnames[1:]}})

        batch_fields = sorted({key for row in self._batch_rows for key in row})
        if batch_fields:
            preferred = [
                "batch_index", "epoch", "window_index", "window_id",
                "detail_sampled", "total_seconds",
            ]
            batch_fields = [key for key in preferred if key in batch_fields] + [
                key for key in batch_fields if key not in preferred
            ]
            with (profile_dir / "perf_batches.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=batch_fields)
                writer.writeheader()
                writer.writerows(self._batch_rows)
            with (profile_dir / "perf_batches.json").open("w", encoding="utf-8") as handle:
                json.dump(self._batch_rows, handle, indent=2, sort_keys=True, allow_nan=False)

        summary = {
            key: value for key, value in payload.items()
            if key not in {"raw_samples_seconds", "raw_self_samples_seconds"}
        }
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        self._flushed = True
        return payload


_ACTIVE_PROFILER = RuntimeProfiler(enabled=False)


def configure_profiler(
    *,
    enabled: bool,
    run_root: str | os.PathLike[str] | None,
    basic_timing: bool = False,
    inventory_history_stats: bool = False,
    sample_every_n_batches: int = 10,
    raw_sample_limit: int = 2000,
    metadata: dict[str, Any] | None = None,
    torch_trace_batches: int = 0,
    torch_trace_warmup_batches: int = 0,
    torch_trace_window_start: int = 0,
) -> RuntimeProfiler:
    global _ACTIVE_PROFILER
    _ACTIVE_PROFILER = RuntimeProfiler(
        enabled=enabled,
        run_root=run_root,
        basic_timing=basic_timing,
        inventory_history_stats=inventory_history_stats,
        sample_every_n_batches=sample_every_n_batches,
        raw_sample_limit=raw_sample_limit,
        metadata=metadata,
        torch_trace_batches=torch_trace_batches,
        torch_trace_warmup_batches=torch_trace_warmup_batches,
        torch_trace_window_start=torch_trace_window_start,
    )
    return _ACTIVE_PROFILER


def get_profiler() -> RuntimeProfiler:
    return _ACTIVE_PROFILER


def finish_profiler(*, status: str, error: str | None = None) -> dict[str, Any] | None:
    return _ACTIVE_PROFILER.flush(status=status, error=error)
