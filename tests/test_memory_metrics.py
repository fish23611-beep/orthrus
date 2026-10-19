"""Cross-platform telemetry contracts; Unix cases are mocked on Windows."""
import builtins
import importlib.util
import io
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

from runtime import memory_metrics as memory


MIB = 1024 ** 2


def platform(monkeypatch, name, sysconf=None):
    # Replacing this module's reference avoids mutating os.name for pytest.
    monkeypatch.setattr(memory, "os", NS(name=name, sysconf=sysconf))


def windows_info(monkeypatch, **fields):
    platform(monkeypatch, "nt")
    monkeypatch.setitem(sys.modules, "psutil", NS(
        Process=lambda: NS(memory_info=lambda: NS(**fields))))


@pytest.mark.parametrize("field", ["peak_wset", "peak_rss"])
def test_windows_current_and_kernel_peak_use_bytes(monkeypatch, field):
    windows_info(monkeypatch, rss=12 * MIB, **{field: 80 * MIB})
    assert memory.current_rss_mb() == 12.0
    assert memory.peak_rss_mb() == 80.0


def test_modern_peak_field_takes_precedence(monkeypatch):
    windows_info(monkeypatch, rss=12 * MIB, peak_rss=80 * MIB, peak_wset=70 * MIB)
    assert memory.peak_rss_mb() == 80.0


def test_missing_peak_never_substitutes_current_rss(monkeypatch):
    windows_info(monkeypatch, rss=12 * MIB)
    assert memory.current_rss_mb(legacy_peak_fallback=True) == 12.0
    assert memory.peak_rss_mb() is None


@pytest.mark.parametrize("value", [None, "bad", float("nan"), float("inf"), -1, 0])
def test_invalid_windows_counters_are_unavailable(monkeypatch, value):
    windows_info(monkeypatch, rss=value, peak_wset=value)
    assert memory.current_rss_mb() is None
    assert memory.peak_rss_mb() is None


@pytest.mark.parametrize("error", [ImportError, OSError, RuntimeError])
def test_windows_backend_failure_is_nonfatal(monkeypatch, error):
    platform(monkeypatch, "nt")
    def fail():
        raise error("telemetry unavailable")
    monkeypatch.setitem(sys.modules, "psutil", NS(Process=fail))
    assert memory.current_rss_mb(legacy_peak_fallback=True) is None
    assert memory.peak_rss_mb() is None


def test_windows_does_not_import_resource_or_read_proc():
    # A fresh interpreter proves there is no resource import at module load or call.
    code = '''
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "resource":
        raise AssertionError("Windows imported resource")
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from runtime import memory_metrics as m
from types import SimpleNamespace as NS
import sys
m.os = NS(name="nt")
sys.modules["psutil"] = NS(Process=lambda: NS(memory_info=lambda: NS(rss=1048576, peak_wset=2097152)))
def no_proc(*args, **kwargs):
    raise AssertionError("Windows read /proc")
builtins.open = no_proc
assert m.current_rss_mb() == 1
assert m.peak_rss_mb() == 2
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_linux_current_uses_statm_resident_pages_and_sysconf(monkeypatch):
    def sysconf(key):
        assert key == "SC_PAGE_SIZE"
        return 4096
    platform(monkeypatch, "posix", sysconf)
    def statm(path, **kwargs):
        assert path == "/proc/self/statm"
        assert kwargs == {"encoding": "ascii"}
        return io.StringIO("99999 1536 0 0 0 0 0\n")
    monkeypatch.setattr(builtins, "open", statm)
    monkeypatch.setitem(sys.modules, "psutil", None)
    assert memory.current_rss_mb() == 6.0


@pytest.mark.parametrize("system,value", [("linux", 81920), ("darwin", 80 * MIB)])
def test_unix_peak_keeps_platform_units_and_rusage_self(monkeypatch, system, value):
    platform(monkeypatch, "posix")
    monkeypatch.setattr(memory, "sys", NS(platform=system))
    def rusage(who):
        assert who == sentinel
        return NS(ru_maxrss=value)
    sentinel = object()
    monkeypatch.setitem(sys.modules, "resource", NS(RUSAGE_SELF=sentinel, getrusage=rusage))
    monkeypatch.setitem(sys.modules, "psutil", None)
    assert memory.peak_rss_mb() == 80.0


@pytest.mark.parametrize("content", [None, "", "broken", "1 not-a-number"])
def test_unix_loader_alone_retains_legacy_peak_fallback(monkeypatch, content):
    platform(monkeypatch, "posix", lambda _: 4096)
    def statm(*args, **kwargs):
        if content is None:
            raise OSError("/proc unavailable")
        return io.StringIO(content)
    monkeypatch.setattr(builtins, "open", statm)
    monkeypatch.setattr(memory, "peak_rss_mb", lambda: 80.0)
    assert memory.current_rss_mb() is None
    assert memory.current_rss_mb(legacy_peak_fallback=True) == 80.0


def test_missing_sysconf_is_nonfatal(monkeypatch):
    platform(monkeypatch, "posix")
    monkeypatch.setattr(builtins, "open", lambda *a, **k: io.StringIO("1 2"))
    assert memory.current_rss_mb() is None


def test_unix_resource_unavailable_is_nonfatal(monkeypatch):
    platform(monkeypatch, "posix")
    monkeypatch.setitem(sys.modules, "resource", None)
    assert memory.peak_rss_mb() is None


def test_telemetry_failure_does_not_break_loader_profiler_or_experiment(monkeypatch, tmp_path):
    import data_utils
    from mstc.experiment_utils import peak_cpu_memory_mb
    from runtime.runtime_profiler import RuntimeProfiler
    platform(monkeypatch, "nt")
    def fail():
        raise OSError("counter unavailable")
    monkeypatch.setitem(sys.modules, "psutil", NS(Process=fail))
    telemetry = data_utils._LoaderTelemetry()
    telemetry.observe("test", emit=False)
    assert telemetry.as_dict()["rss_mb_by_phase"] == {"test": None}
    assert telemetry.as_dict()["peak_rss_mb_by_phase"] == {"test": None}
    assert peak_cpu_memory_mb() is None
    profiler = RuntimeProfiler(enabled=True, run_root=tmp_path)
    observed = []
    for batch, token in profiler.profile_batches([1, 2]):
        observed.append(batch)
        profiler.end_batch(token, "training.T_total_batch")
    payload = profiler.flush()
    assert observed == [1, 2]
    assert payload["memory"]["rss_current_mb"] is None
    assert payload["memory"]["rss_peak_mb"] is None


def test_benchmark_fails_explicitly_when_peak_unavailable(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/benchmark_preprocess_memory.py"
    spec = importlib.util.spec_from_file_location("memory_benchmark_probe", path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    monkeypatch.setattr(memory, "peak_rss_mb", lambda: None)
    with pytest.raises(RuntimeError, match="peak RSS unavailable"):
        benchmark.peak_rss_mb()


def test_native_process_counters_are_finite_and_peak_covers_current():
    if os.name == "nt":
        import psutil
        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_rss", None)
        if peak is None:
            peak = info.peak_wset
        assert info.rss > 0 and peak >= info.rss
    current = memory.current_rss_mb()
    peak = memory.peak_rss_mb()
    assert current is not None and math.isfinite(current) and current > 0
    assert peak is not None and math.isfinite(peak) and peak >= current
