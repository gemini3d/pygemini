from __future__ import annotations
import os
import shutil
import subprocess
import functools
from pathlib import Path, PurePosixPath, WindowsPath


@functools.lru_cache()
def wsl_available() -> bool:
    """
    heuristic to detect if Windows Subsystem for Linux is available.
    Uses presence of /etc/os-release in the WSL image to say Linux is there.
    This is a de facto file standard across Linux distros.
    """

    has_wsl = False
    if os.name == "nt" and shutil.which("wsl"):
        has_wsl = wsl_file_exist("/etc/os-release")

    return has_wsl


def is_wsl_path(path: Path) -> bool:
    return os.name == "nt" and path.as_posix().startswith(("//wsl$", "//wsl.localhost"))


def wsl_file_exist(file: str | PurePosixPath) -> bool:
    """
    path is specified as if in WSL
    NOT //wsl$/Ubuntu/etc/os-release
    but /etc/os-release
    """
    if os.name != "nt":
        return False

    try:
        return (
            subprocess.run(["wsl", "test", "-f", str(file)], timeout=10).returncode == 0
        )
    except subprocess.TimeoutExpired:
        return False


def wsl_path2win_path(path: PurePosixPath) -> WindowsPath:
    """
    path is specified as if in WSL
    NOT //wsl$/Ubuntu/etc/os-release
    but /etc/os-release
    """

    assert wsl_available(), "This function is for WSL only"

    out = subprocess.check_output(
        ["wsl", "wslpath", "-w", path.as_posix()],
        text=True,
        timeout=10,
    )

    return WindowsPath(out.strip())


def win_path2wsl_path(path: Path | WindowsPath) -> PurePosixPath:
    r"""
    Parameters:

    path: Path
        path is specified as if in Windows.
        Examples: C:\Users\me\file.txt  \\wsl$\Ubuntu\etc\os-release
    """

    assert wsl_available(), "This function is for WSL only"

    out = subprocess.check_output(
        ["wsl", "wslpath", path.as_posix()],
        text=True,
        timeout=10,
    )

    return PurePosixPath(out.strip())


def wsl_tempfile() -> PurePosixPath:
    """
    Returns a temporary file in WSL.
    """

    assert wsl_available(), "This function is for WSL only"

    out = subprocess.check_output(
        ["wsl", "mktemp", "-u"],
        text=True,
        timeout=10,
    )

    return PurePosixPath(out.strip())
