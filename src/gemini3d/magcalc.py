from __future__ import annotations
import argparse
from pathlib import Path
import logging
from math import pi, radians
import typing as T

import numpy as np

from . import read
from . import write


def magcalc(direc: Path, dang: float, xg: dict[str, T.Any] = None):
    """
    Parameters
    ----------

    direc: pathlib.Path
      top-level simulation path

    dang: float, optional
      ANGULAR RANGE TO COVER FOR THE CALCLUATIONS.
      THIS IS FOR THE FIELD POINTS - SOURCE POINTS COVER ENTIRE GRID.

    xg: dict, optional
      simulation grid
    """

    direc = Path(direc).expanduser()
    if not direc.is_dir():
        raise NotADirectoryError(direc)

    cfg = read.config(direc)

    if not xg:
        xg = read.grid(direc)
        lx3 = xg["lx"][2]

    if lx3 == 1:
        flag2D = True
        logging.info("2D meshgrid")
    else:
        flag2D = False
        logging.info("3D meshgrid")

    # %% TABULATE THE SOURCE OR GRID CENTER LOCATION
    if "sourcemlon" not in cfg:
        thdist = xg["theta"].mean()
        phidist = xg["phi"].mean()
    else:
        thdist = pi / 2 - radians(cfg["sourcemlat"])  # zenith angle of source location
        phidist = radians(cfg["sourcemlon"])

    # %% FIELD POINTS OF INTEREST (CAN/SHOULD BE DEFINED INDEPENDENT OF SIMULATION GRID)
    ltheta = 10
    lphi = 1 if flag2D else 10

    thmin = thdist - radians(dang)
    thmax = thdist + radians(dang)
    phimin = phidist - radians(dang)
    phimax = phidist + radians(dang)

    theta = np.linspace(thmin, thmax, ltheta)
    phi = phidist if flag2D else np.linspace(phimin, phimax, lphi)

    r = 6370e3 * np.ones((ltheta, lphi))
    # use ground level for altitude for all field points
    phi, theta = np.meshgrid(phi, theta)

    # %% CREATE AN INPUT FILE OF FIELD POINTS
    mag = {
        "r": r,
        "phi": phi,
        "theta": theta,
    }
    filename = direc / "inputs/magfieldpoints.h5"
    write.maggrid(filename, mag)


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
