"""Best-effort process resident-memory telemetry, measured in MiB.

Unix keeps the frozen /proc and getrusage sources. Windows uses psutil's
WorkingSetSize and PeakWorkingSetSize counters; an unavailable peak stays None.
"""

import math
import os
import sys

_MIB = 1024 ** 2


def _as_mib(value: float, divisor: int = _MIB) -> float | None:
    result = float(value) / divisor
    return result if math.isfinite(result) and result > 0 else None


def current_rss_mb(*, legacy_peak_fallback: bool = False) -> float | None:
    """Return current RSS, with the data loader's opt-in legacy Unix fallback."""
    if os.name == "nt":
        try:
            import psutil
            return _as_mib(psutil.Process().memory_info().rss)
        except Exception:
            return None

    try:
        with open("/proc/self/statm", encoding="ascii") as handle:
            pages = int(handle.read().split()[1])
        return _as_mib(pages * os.sysconf("SC_PAGE_SIZE"))
    except Exception:
        return peak_rss_mb() if legacy_peak_fallback else None


def peak_rss_mb() -> float | None:
    """Return process peak RSS, not a substitute current-RSS sample."""
    if os.name == "nt":
        try:
            import psutil
            info = psutil.Process().memory_info()
            # psutil 8 uses peak_rss; 7 and earlier use peak_wset.
            value = getattr(info, "peak_rss", None)
            if value is None:
                value = getattr(info, "peak_wset", None)
            return _as_mib(value) if value is not None else None
        except Exception:
            return None

    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return _as_mib(value, _MIB if sys.platform == "darwin" else 1024)
    except Exception:
        return None
