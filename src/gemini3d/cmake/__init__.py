from __future__ import annotations
import typing as T
from pathlib import Path
import subprocess
import shutil
import json
import os
import tempfile
import importlib.resources

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
    """build and install with CMake"""
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
        subprocess.check_call(["ctest", "--output-on-failure"], cwd=str(build_dir))

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


def get_gemini_root() -> Path:
    gem_root = os.environ.get("GEMINI_ROOT")
    if not gem_root:
        gem_root = os.environ.get("GEMINI3D_ROOT")
    if not gem_root:
        raise EnvironmentError(
            "Please set environment variable GEMINI_ROOT to (desired) top-level Gemini3D directory."
            "If Gemini3D is not already there, PyGemini will download and build Gemini3D there."
        )
    return Path(gem_root).expanduser()


def build_gemini3d(targets: list[str]):
    """
    build targets from gemini3d program

    Specify environment variable GEMINI_ROOT to reuse existing development code
    """

    if isinstance(targets, str):
        targets = [targets]

    gem_root = get_gemini_root()

    src_dir = Path(gem_root).expanduser()

    if not (src_dir / "CMakeLists.txt").is_file():
        jmeta = json.loads(importlib.resources.read_text("gemini3d", "libraries.json"))
        git_download(src_dir, repo=jmeta["gemini3d"]["git"], tag=jmeta["gemini3d"]["tag"])

    build_dir = src_dir / "build"

    build(
        src_dir,
        build_dir,
        run_test=False,
        install=False,
        config_args=["-DBUILD_TESTING:BOOL=false"],
        build_args=["--target", *targets],
    )

    for t in targets:
        for n in {"build", "build/Debug", "build/Release"}:
            exe = shutil.which(t, path=str(src_dir / n))
            if exe:
                break
        if not exe:
            raise RuntimeError(f"{t} not found in {build_dir}")
