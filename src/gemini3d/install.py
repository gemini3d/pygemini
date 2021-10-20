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
import json
import importlib.resources
import typing as T


def main(package_manager: str, pkgs: dict[str, T.Any]) -> None:

    cmd = []

    if sys.platform == "linux":

        if not package_manager:
            from gemini3d.linux_info import get_package_manager

            package_manager = get_package_manager()

        if package_manager == "yum":
            subprocess.run(["sudo", "yum", "--assumeyes", "updateinfo"])
        elif package_manager == "apt":
            subprocess.run(["sudo", "apt", "update", "--yes"])
        elif package_manager == "pacman":
            subprocess.run(["sudo", "pacman", "-Sy"])
        else:
            raise ValueError(
                f"Unknown package manager {package_manager}, try installing the prereqs manually"
            )
        k = package_manager
        cmd = ["sudo"]
    elif sys.platform == "darwin":
        if shutil.which("brew"):
            k = "brew"
        elif shutil.which("port"):
            k = "port"
        else:
            raise SystemExit(
                "Neither Homebrew or MacPorts is available. Please install a Fortran compiler."
            )
    elif sys.platform == "cygwin":
        k = "cygwin"
    elif sys.platform == "win32":
        if not shutil.which("pacman"):
            raise SystemExit("Windows Subsystem for Linux or MSYS2 is recommended.")
        # assume MSYS2
        k = "msys2"
    else:
        raise ValueError(f"unknown platform {sys.platform}")

    cmd += pkgs[k]["cmd"].split(" ") + pkgs[k]["pkgs"]

    print(" ".join(cmd))
    ret = subprocess.run(cmd)

    raise SystemExit(ret.returncode)


if __name__ == "__main__":
    json_str = importlib.resources.read_text(__package__, "requirements.json")
    pkgs = json.loads(json_str)

    p = ArgumentParser()
    p.add_argument(
        "package_manager",
        help="specify package manager e.g. apt, yum",
        choices=list(pkgs.keys()),
        nargs="?",
    )
    P = p.parse_args()

    main(P.package_manager, pkgs)
