from pathlib import Path
import math
import typing as T

from .utils import get_cpu_count
from .fileio import get_simsize


def get_mpi_count(path: Path, max_cpu: int) -> int:
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

    return max_mpi(get_simsize(path), max_cpu)


def max_mpi(size: T.Tuple[int, ...], max_cpu: int) -> int:

    if not max_cpu:
        max_cpu = get_cpu_count()

    if size[2] == 1:
        # 2D sim
        N = max_gcd(size[1], max_cpu)
    else:
        # 3D sim
        N = max_gcd(size[2], max_cpu)

    return N


def max_gcd(s: int, M: int) -> int:

    N = 1
    for i in range(M, 1, -1):
        N = max(math.gcd(s, i), N)
        if i < N:
            break

    return N
