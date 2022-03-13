from __future__ import annotations
from pathlib import Path
import subprocess

from .cmake import cmake_exe


def extract_zst(archive: str | Path, out_path: str | Path):
    """
    extract .zst file

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


extract_tar = extract_zst
extract_zip = extract_zst
