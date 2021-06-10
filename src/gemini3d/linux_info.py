"""
get Linux system info
"""

from __future__ import annotations
from configparser import ConfigParser
from pathlib import Path
import sys


def os_release() -> list[str]:
    """
    reads /etc/os-release with fallback to legacy methods

    Returns
    -------

    linux_names: list of str
        name(s) of operating system detect
    """

    if sys.platform != "linux":
        return []

    fn = Path("/etc/os-release")
    if not fn.is_file():
        if Path("/etc/redhat-release").is_file() or Path("/etc/centos-release").is_file():
            return ["rhel"]
        elif Path("/etc/debian_version").is_file():
            return ["debian"]
        elif Path("/etc/arch-version").is_file():
            return ["arch"]

    return parse_os_release("[all]" + fn.read_text())


def parse_os_release(txt: str) -> list[str]:
    """parse /etc/os-release text"""

    C = ConfigParser(inline_comment_prefixes=("#", ";"))
    C.read_string(txt)
    like = C["all"].get("ID_LIKE", fallback="")
    if not like:
        like = C["all"].get("ID", fallback="")
    like = like.strip('"').strip("'").split()

    return like


def get_package_manager(like: list[str] = None) -> str:
    if not like:
        like = os_release()
    if isinstance(like, str):
        like = [like]

    sl = set(like)

    if {"centos", "rhel", "fedora"} & sl:
        return "yum"
    elif {"debian", "ubuntu"} & sl:
        return "apt"
    elif like == "arch":
        return "pacman"
    else:
        raise ValueError(
            f"Unknown ID_LIKE={like}, please file bug report or manually specify package manager"
        )
