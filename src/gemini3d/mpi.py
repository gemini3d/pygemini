from __future__ import annotations
from pathlib import Path
import math

from .utils import get_cpu_count
from . import read


def count(path: Path, max_cpu: int) -> int:
    """get appropriate MPI image count for problem shape

    Parameters
    ----------
    path: pathlib.Path
        simsize file

    Returns
    -------
    count: int
        detect number of physical CPU
    """

    return max_mpi(read.simsize(path), max_cpu)


def max_mpi(size: tuple[int, ...], max_cpu: int) -> int:
    """
    goal is to find the highest x2 + x3 to maximum CPU core count
    """

    if len(size) != 3:
        raise ValueError("expected x1,x2,x3")

    if not max_cpu:
        max_cpu = get_cpu_count()

    if size[2] == 1:
        # 2D sim
        N = max_gcd(size[1], max_cpu)
    elif size[1] == 1:
        # 2D sim
        N = max_gcd(size[2], max_cpu)
    else:
        # 3D sim
        N = max_gcd2(size[1:], max_cpu)

    return N


def max_gcd(s: int, M: int) -> int:
    """
    find the Greatest Common Factor to evenly partition the simulation grid

    Output range is [M, 1]
    """

    if M < 1:
        raise ValueError("CPU count must be at least one")

    N = 1
    for i in range(M, 1, -1):
        N = max(math.gcd(s, i), N)
        if i < N:
            break

    return N


def max_gcd2(s: tuple[int, ...] | list[int], M: int) -> int:
    """
    find the Greatest Common Factor to evenly partition the simulation grid

    Output range is [M, 1]

    1. find factors of each dimension
    2. choose partition that yields highest CPU count usage
    """

    if len(s) != 2:
        raise ValueError("expected x2,x3")

    if M < 1:
        raise ValueError("CPU count must be at least one")

    f2 = [max_gcd(s[0], m) for m in range(M, 0, -1)]
    f3 = [max_gcd(s[1], m) for m in range(M, 0, -1)]

    N = 1
    for i in f2:
        for j in f3:
            if M >= i * j > N:
                # print(i,j)
                N = i * j
    return N
