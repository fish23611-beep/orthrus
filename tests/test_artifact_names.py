"""Time-window filenames must round-trip without changing time semantics."""
from types import SimpleNamespace

import artifact_names as names


INTERVAL = "2019-05-08 00:00:01.000000000~2019-05-08 00:17:04.000000000"


def test_windows_escapes_colons_and_round_trips(monkeypatch, tmp_path):
    monkeypatch.setattr(names, "os", SimpleNamespace(name="nt"))
    filename = names.time_window_filename(INTERVAL)
    assert ":" not in filename
    assert names.decode_time_window_filename(filename) == INTERVAL
    path = tmp_path / filename
    path.write_text("window", encoding="utf-8")
    assert path.read_text(encoding="utf-8") == "window"


def test_unix_keeps_frozen_names(monkeypatch):
    monkeypatch.setattr(names, "os", SimpleNamespace(name="posix"))
    assert names.time_window_filename(INTERVAL) == INTERVAL
    assert names.decode_time_window_filename(INTERVAL) == INTERVAL


def test_windows_names_keep_window_chronology(monkeypatch):
    monkeypatch.setattr(names, "os", SimpleNamespace(name="nt"))
    intervals = [INTERVAL, INTERVAL.replace("00:00:01", "00:00:02")]
    encoded = [names.time_window_filename(item) for item in intervals]
    assert [names.decode_time_window_filename(item) for item in sorted(encoded)] == intervals
