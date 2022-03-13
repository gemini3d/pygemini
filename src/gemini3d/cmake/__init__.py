from __future__ import annotations
from pathlib import Path
import subprocess
import shutil
import os

__all__ = ["cmake_exe", "build", "find_library"]


def cmake_exe() -> str:

    cmake = shutil.which("cmake")
    if not cmake:
        raise FileNotFoundError("CMake not found.  Try:\n    pip install cmake")

    cmake_version = (
        subprocess.check_output([cmake, "--version"], text=True).split("\n")[0].split(" ")[2]
    )

    print("Using CMake", cmake_version)

    return cmake


def get_gemini_root() -> Path:
    gemini_root = os.environ.get("GEMINI_ROOT")
    if not gemini_root:
        raise EnvironmentError(
            "Please set environment variable GEMINI_ROOT to (desired) top-level Gemini3D directory."
            "If Gemini3D is not already there, PyGemini will download and build Gemini3D there."
        )
    return Path(gemini_root).expanduser()
