"""
installs Gemini3D prerequisite libraries for:

* Linux: CentOS, Debian, Ubuntu, Windows Subsystem for Linux
* MacOS: Homebrew
* Windows: MSYS2, Cygwin

assumes GCC/Gfortran
"""

from __future__ import annotations
import subprocess
import sys
from argparse import ArgumentParser
import shutil

PKG = {
    "yum": [
        "epel-release",
        "gcc-gfortran",
        "MUMPS-openmpi-devel",
        "lapack-devel",
        "scalapack-openmpi-devel",
        "openmpi-devel",
        "hdf5-devel",
    ],
    "apt": [
        "gfortran",
        "libmumps-dev",
        "liblapack-dev",
        "libscalapack-mpi-dev",
        "libopenmpi-dev",
        "openmpi-bin",
        "libhdf5-dev",
    ],
    "pacman": ["gcc-fortran", "ninja", "lapack", "openmpi", "hdf5"],
    "brew": ["gcc", "ninja", "cmake", "lapack", "scalapack", "openmpi", "hdf5"],
    "cygwin": ["gcc-fortran", "liblapack-devel", "libopenmpi-devel"],
    "msys": [
        "mingw-w64-x86_64-gcc-fortran",
        "mingw-w64-x86_64-ninja",
        "mingw-w64-x86_64-hdf5",
        "mingw-w64-x86_64-lapack",
        "mingw-w64-x86_64-scalapack",
        "mingw-w64-x86_64-mumps",
    ],
}


def main(package_manager: str):

    cmd: list[str] = []

    if sys.platform == "linux":

        if not package_manager:
            from gemini3d.linux_info import get_package_manager

            package_manager = get_package_manager()

        if package_manager == "yum":
            subprocess.run(["sudo", "yum", "--assumeyes", "updateinfo"])
            cmd = ["sudo", "yum", "install"] + PKG["yum"]
        elif package_manager == "apt":
            subprocess.run(["sudo", "apt", "update", "--yes"])
            cmd = ["sudo", "apt", "install"] + PKG["apt"]
        elif package_manager == "pacman":
            subprocess.run(["sudo", "pacman", "-S", "--needed"] + PKG["pacman"])
        else:
            raise ValueError(
                f"Unknown package manager {package_manager}, try installing the prereqs manually"
            )
    elif sys.platform == "darwin":
        if not shutil.which("brew"):
            raise SystemExit(
                "We assume Homebrew is available, need to manually install a Fortran compiler otherwise."
            )
        cmd = ["brew", "install"] + PKG["brew"]
        # autobuild Mumps, it's much faster
    elif sys.platform == "cygwin":
        cmd = ["setup-x86_64.exe", "-P"] + PKG["cygwin"]
    elif sys.platform == "win32":
        if not shutil.which("pacman"):
            raise SystemExit("Windows Subsystem for Linux or MSYS2 is recommended.")
        # assume MSYS2
        cmd = ["pacman", "-S", "--needed"] + PKG["msys"]
    else:
        raise ValueError(f"unknown platform {sys.platform}")

    print(" ".join(cmd))
    ret = subprocess.run(cmd)

    raise SystemExit(ret.returncode)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument(
        "package_manager",
        help="specify package manager e.g. apt, yum",
        choices=list(PKG.keys()),
        nargs="?",
    )
    P = p.parse_args()

    main(P.package_manager)
