from __future__ import annotations
import math

import numpy as np


def grid1d(dist: float, L: int, parms: list[float] = None) -> np.ndarray:
    """
    generate 1D grid

    Parameters
    ----------

    dist: float
        one-way distance from origin (meters)
    L: int
        number of cells

    Returns
    -------

    x: np.ndarray
        1D vector grid
    """

    xmin = -dist / 2
    xmax = dist / 2

    if not parms:
        x = uniform1d(xmin, xmax, L)
    else:
        x = non_uniform1d(xmax, parms)

    return x


def uniform1d(xmin: float, xmax: float, L: int) -> np.ndarray:

    if L == 1:
        # degenerate dimension
        # add 2 ghost cells on each side
        x = np.linspace(xmin, xmax, L + 4)
    else:
        # exclude the ghost cells when setting extents
        x = np.linspace(xmin, xmax, L)
        d0 = x[1] - x[0]
        d1 = x[-1] - x[-2]
        # now tack on ghost cells so they are outside user-specified region
        x = np.insert(x, 0, [x[0] - 2 * d0, x[0] - d0])
        x = np.append(x, [x[-1] + d1, x[-1] + 2 * d1])

    return x


def non_uniform1d(xmax: float, parms: list[float]) -> np.ndarray:

    degdist = parms[0]  # distance from boundary at which we start to degrade resolution
    dx0 = parms[1]  # min step size for grid
    dxincr = parms[2]  # max step size increase for grid
    ell = parms[3]  # transition length of degradation
    x2 = xmax - degdist

    x = [dx0 / 2.0]
    # start offset from zero so we can have an even number (better for mpi)

    while x[-1] < xmax:
        dx = dx0 + dxincr * (1 / 2 + 1 / 2 * math.tanh((x[-1] - x2) / ell))
        x.append(x[-1] + dx)

    x = np.append(-np.array(x[::-1]), x)

    return x


def altitude_grid(
    alt_min: float, alt_max: float, incl_deg: float, d: tuple[float, float, float, float]
) -> np.ndarray:

    if alt_min < 0 or alt_max < 0:
        raise ValueError("grid values must be positive")
    if alt_max <= alt_min:
        raise ValueError("grid_max must be greater than grid_min")

    alt = [alt_min]

    while alt[-1] < alt_max:
        # dalt=10+9.5*tanh((alt(i-1)-500)/150)
        dalt = d[0] + d[1] * math.tanh((alt[-1] - d[2]) / d[3])
        alt.append(alt[-1] + dalt)

    alt = np.asarray(alt)
    if alt.size < 10:
        raise ValueError("grid too small")

    # %% tilt for magnetic inclination
    z = alt / math.sin(math.radians(incl_deg))

    # %% add two ghost cells each to top and bottom
    dz1 = z[1] - z[0]
    dzn = z[-1] - z[-2]
    z = np.insert(z, 0, [z[0] - 2 * dz1, z[0] - dz1])
    z = np.append(z, [z[-1] + dzn, z[-1] + 2 * dzn])

    return z
