from __future__ import annotations
import argparse
from pathlib import Path
from math import pi, radians
import typing

import numpy as np

from . import read
from . import write


def magcalc(
    direc: Path,
    dang: float,
    Ltheta: int = 40,
    Lphi: int = 40,
    xg: dict[str, typing.Any] | None = None,
):
    """
    Parameters
    ----------

    direc: pathlib.Path
      top-level simulation path

    dang: float, optional
      ANGULAR RANGE TO COVER FOR THE FIELD POINTS - SOURCE POINTS COVER ENTIRE GRID

    Ltheta: int, optional
        number of points in theta

    Lphi: int, optional
        number of points in phi

    xg: dict, optional
      simulation grid
    """

    direc = Path(direc).expanduser()

    cfg = read.config(direc)

    if not xg:
        xg = read.grid(direc)

    flag2D = any(xg["lx"][1:] == 1)

    # %% TABULATE THE SOURCE OR GRID CENTER LOCATION
    if "sourcemlon" not in cfg:
        thdist = xg["theta"].mean()
        phidist = xg["phi"].mean()
    else:
        thdist = pi / 2 - radians(cfg["sourcemlat"])  # zenith angle of source location
        phidist = radians(cfg["sourcemlon"])

    # %% FIELD POINTS OF INTEREST
    # CAN/SHOULD BE DEFINED INDEPENDENT OF SIMULATION GRID
    if flag2D:
        Lphi = 1

    thmin = thdist - radians(dang)
    thmax = thdist + radians(dang)
    phimin = phidist - radians(dang)
    phimax = phidist + radians(dang)

    theta = np.linspace(thmin, thmax, Ltheta)
    phi = phidist if flag2D else np.linspace(phimin, phimax, Lphi)

    r = 6370e3 * np.ones((Ltheta, Lphi))
    # use ground level for altitude for all field points
    phi, theta = np.meshgrid(phi, theta)

    # %% CREATE AN INPUT FILE OF FIELD POINTS
    mag = {
        "r": r,
        "phi": phi,
        "theta": theta,
    }

    write.maggrid(direc / "inputs/magfieldpoints.h5", mag)


if __name__ == "__main__":
    P = argparse.ArgumentParser(
        description="prepare magcalc.bin input from existing simulation output"
    )
    P.add_argument("direc", help="top-level simulation directory to read and write to")
    P.add_argument(
        "dang",
        help="ANGULAR RANGE TO COVER FOR THE field points CALCLUATIONS",
        type=float,
        nargs="?",
        default=1.5,
    )
    P = P.parse_args()

    magcalc(P.direc, P.dang)
