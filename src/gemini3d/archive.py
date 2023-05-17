from __future__ import annotations
from pathlib import Path
import subprocess
import shutil
import sys


def cmake_exe() -> str:
    cmake = shutil.which("cmake")
    if not cmake:
        # try to help if Homebrew or Ports is not on PATH
        if sys.platform == "darwin":
            paths = ["/opt/homebrew/bin", "/usr/local/bin", "/opt/local/bin"]
            for path in paths:
                cmake = shutil.which("cmake", path=path)
                if cmake:
                    break

    if not cmake:
        raise FileNotFoundError("CMake not found.  Try:\n    pip install cmake")

    cmake_version = (
        subprocess.check_output([cmake, "--version"], text=True)
        .split("\n")[0]
        .split(" ")[2]
    )

    print("Using CMake", cmake_version)

    return cmake


def extract(archive: str | Path, out_path: str | Path):
    """
    extract archive file

    To reduce Python package prereqs we use CMake instead of Python
    "zstandard" package. This is also more efficient computationally.

    Parameters
    ----------

    archive: pathlib.Path or str
      .zst file to extract

    out_path: pathlib.Path or str
      directory to extract files and directories to
    """

    archive = Path(archive).expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)

    out_path = Path(out_path).expanduser().resolve()
    # need .resolve() in case intermediate relative dir doesn't exist
    out_path.mkdir(exist_ok=True, parents=True)

    subprocess.check_call([cmake_exe(), "-E", "tar", "xf", str(archive)], cwd=out_path)
