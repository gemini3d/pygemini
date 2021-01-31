from __future__ import annotations
import typing as T
from pathlib import Path
import subprocess
import shutil
import json
import os
import tempfile
import importlib.resources

from .. import PYGEMINI_ROOT
from ..utils import get_cpu_count
from ..web import git_download

__all__ = ["exe", "build", "find_library"]


def exe() -> str:

    cmake = shutil.which("cmake")
    if not cmake:
        raise FileNotFoundError("CMake not found.")

    cmake_version = (
        subprocess.check_output([cmake, "--version"], text=True).split("\n")[0].split(" ")[2]
    )

    print("Using CMake", cmake_version)

    return cmake


def build(
    source_dir: Path,
    build_dir: Path,
    *,
    config_args: list[str] = None,
    build_args: list[str] = None,
    wipe: bool = False,
    env: T.Mapping[str, str] = None,
    run_test: bool = False,
    dryrun: bool = False,
    install: bool = True,
):
    """ build and install with CMake """
    cmake = exe()

    cache_file = build_dir / "CMakeCache.txt"
    if wipe:
        if cache_file.is_file():
            cache_file.unlink()
    # %% Configure
    cmd = [cmake, f"-B{build_dir}", f"-S{source_dir}"]
    if config_args:
        cmd += config_args
    subprocess.check_call(cmd, env=env)
    # %% Build
    cmd = [cmake, "--build", str(build_dir), "--parallel"]
    if build_args:
        cmd += build_args
    if dryrun:
        print("DRYRUN: would have run\n", " ".join(cmd))
        return None

    subprocess.check_call(cmd)

    if run_test:
        Njobs = get_cpu_count()
        subprocess.check_call(
            ["ctest", "--parallel", str(Njobs), "--output-on-failure"], cwd=str(build_dir)
        )

    if install:
        subprocess.check_call([cmake, "--install", str(build_dir)])


def find_library(lib_name: str, lib_path: list[str], env: T.Mapping[str, str]) -> bool:
    """
    check if library exists with CMake

    lib_name must have the appropriate upper and lower case letter as would be used
    directly in CMake.
    """

    cmake = exe()

    with importlib.resources.path("gemini3d.cmake", "FindLAPACK.cmake") as f:
        mod_path = Path(f).parent

    cmake_template = """
cmake_minimum_required(VERSION 3.15)
project(dummy LANGUAGES C Fortran)

"""

    if mod_path.is_dir():
        cmake_template += f'list(APPEND CMAKE_MODULE_PATH "{mod_path.as_posix()}")\n'

    cmake_template += f"find_package({lib_name} REQUIRED)\n"

    build_dir = f"find-{lib_name.split(' ', 1)[0]}"

    # not context_manager to avoid Windows PermissionError on context exit for Git subdirs
    d = tempfile.TemporaryDirectory()
    r = Path(d.name)
    (r / "CMakeLists.txt").write_text(cmake_template)

    cmd = [cmake, "-S", str(r), "-B", str(r / build_dir)] + lib_path
    # use cwd= to avoid spilling temporary files into current directory if ancient CMake used
    # also avoids bugs if there is a CMakeLists.txt in the current directory
    ret = subprocess.run(cmd, env=env, cwd=str(r))

    try:
        d.cleanup()
    except PermissionError:
        pass

    return ret.returncode == 0


def build_gemini3d(target: Path) -> Path:
    """
    build targets from gemini3d program

    Specify environemnt variable GEMINI_ROOT to reuse existing development code
    """
    src_dir = None
    if os.environ.get("GEMINI_ROOT"):
        src_dir = Path(os.environ["GEMINI_ROOT"]).expanduser()

    if not src_dir or not src_dir.is_dir():
        if (
            len(target.parents) >= 2
            and target.parents[1].is_dir()
            and (target.parents[1] / "CMakeLists.txt").is_file()
        ):
            src_dir = target.parents[1]
        else:
            # clone Gemini3D and do a test build

            src_dir = PYGEMINI_ROOT / "gemini-fortran"

            jmeta = json.loads(importlib.resources.read_text("gemini3d", "libraries.json"))
            git_download(src_dir, repo=jmeta["gemini3d"]["git"], tag=jmeta["gemini3d"]["tag"])

    if not src_dir.is_dir():
        raise NotADirectoryError(f"could not find Gemini3D source directory {src_dir}")

    build_dir = src_dir / "build"
    exe = shutil.which(target.name, path=str(build_dir))

    if not exe:
        build_args = ["--target", target.name]

        build(
            src_dir,
            build_dir,
            run_test=False,
            install=False,
            config_args=["-DBUILD_TESTING:BOOL=false"],
            build_args=build_args,
        )
        exe = shutil.which(target.name, path=str(build_dir))

    if not exe:
        raise RuntimeError(f"{target.name} not found in {build_dir}")

    return Path(exe)
