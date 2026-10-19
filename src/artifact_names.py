"""Filesystem representations of canonical time-window names.

Only Windows escapes colons. Timestamp values, display strings and Unix
artifact names retain their frozen representation. Decode before parsing a
time-window name imported from Windows on either platform.
"""
import os


def time_window_filename(interval: str) -> str:
    return interval.replace(":", "%3A") if os.name == "nt" else interval


def decode_time_window_filename(name: str) -> str:
    return name.replace("%3A", ":")
