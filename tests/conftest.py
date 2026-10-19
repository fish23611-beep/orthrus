"""Portable filesystem fixtures; all behavioral assertions stay in the tests."""
import os
from pathlib import Path
import subprocess

import pytest


@pytest.fixture
def make_path_alias():
    """Use real symlinks, or Windows directory junctions without admin rights.

    For a file alias, a junction to its parent provides the same realpath
    traversal. This still exercises the production path resolver against a
    real filesystem alias; it does not mock resolution or skip assertions.
    """
    def make(link, target, *, target_is_directory=False):
        link, target = Path(link), Path(target)
        try:
            link.symlink_to(target, target_is_directory=target_is_directory)
            return link
        except OSError as exc:
            if os.name != "nt" or getattr(exc, "winerror", None) != 1314:
                raise
        junction = link if target_is_directory else link.with_name(link.name + "_dir")
        directory = target if target_is_directory else target.parent
        def quote(path):
            return "'" + str(path).replace("'", "''") + "'"
        command = ("New-Item -ItemType Junction -ErrorAction Stop -Path "
                   + quote(junction) + " -Target " + quote(directory) + " | Out-Null")
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            check=True, capture_output=True, text=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        return junction if target_is_directory else junction / target.name
    return make
