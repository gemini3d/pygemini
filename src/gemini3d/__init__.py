"""
PyGemini is the Python interface to the Gemini3D ionospheric model.
"""

from __future__ import annotations
from pathlib import Path

from . import cmake

__version__ = "1.6.0"

PYGEMINI_ROOT = Path(__path__[0])  # type: ignore

WAVELEN = [
    "3371",
    "4278",
    "5200",
    "5577",
    "6300",
    "7320",
    "10400",
    "3466",
    "7774",
    "8446",
    "3726",
    "LBH",
    "1356",
    "1493",
    "1304",
]

SPECIES = ["O+", "ns1", "ns2", "ns3", "N+", "protons", "electrons"]
# must be list, not tuple
LSP = len(SPECIES)

EXE_PATHS = [
    ".",
    "bin",
    "build",
    "build/bin",
    "build/Debug",
    "build/RelWithDebInfo",
    "build/Release",
]


def setup(targets: list[str] = None, gemini_root: Path = None, cmake_args: list[str] = None):
    """
    setup Gemini3D and other Gemini3D executables
    """

    if not targets:
        targets = ["gemini3d.run", "gemini.bin"]

    cmake.build_gemini3d(targets, gemini_root, cmake_args)


def setup_libs(
    prefix: Path, targets: list[str] = None, find: bool = True, cmake_args: list[str] = None
):
    """
    setup Gemini3D external libraries (needed before gemini3d.setup)
    """

    if not targets:
        targets = ["ffilesystem", "glow", "h5fortran", "msis", "mumps"]

    if not cmake_args:
        cmake_args = []
    if find:
        cmake_args += ["-Dfind:BOOL=true"]
    else:
        cmake_args += ["-Dfind:BOOL=false"]

    cmake.build_libs(prefix, targets, cmake_args)
