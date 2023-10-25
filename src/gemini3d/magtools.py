from __future__ import annotations
from pathlib import Path
from datetime import datetime
import numpy as np
import typing as T
import gemini3d.find as find
import gemini3d.read as read
import gemini3d.write as write
import h5py

RE = 6370e3


def makegrid(
    direc: str,
    dang: float = 1.5,
    ltheta: int = 16,
    lphi: int = 16,
    write_grid: bool = False,
) -> dict[str, T.Any]:
    """
    dang: float
        ANGULAR RANGE TO COVER FOR THE CALCULATIONS
        (THIS IS FOR THE FIELD POINTS - SOURCE POINTS COVER ENTIRE GRID)
    """

    direc = Path(direc).expanduser()
    assert direc.is_dir(), f"{direc} is not a directory"

    # SIMULATION METADATA
    cfg = read.config(direc)

    # WE ALSO NEED TO LOAD THE GRID FILE
    xg = read.grid(direc)
    print("Grid loaded")

    # lx1 = xg.lx(1);
    lx3 = xg["lx"][2]
    # lh=lx1;   %possibly obviated in this version - need to check
    flag2D = lx3 == 1
    if flag2D:
        print("2D meshgrid")
    else:
        print("3D meshgrid")

    # TABULATE THE SOURCE OR GRID CENTER LOCATION
    if "sourcemlon" not in cfg.keys():
        thdist = xg["theta"].mean()
        phidist = xg["phi"].mean()
    else:
        thdist = np.pi / 2 - np.radians(cfg["sourcemlat"])
        # zenith angle of source location
        phidist = np.radians(cfg["sourcemlon"])

    # FIELD POINTS OF INTEREST (CAN/SHOULD BE DEFINED INDEPENDENT OF SIMULATION GRID)
    # ltheta = 40
    lphi = 1 if flag2D else ltheta
    lr = 1

    thmin = thdist - np.radians(dang)
    thmax = thdist + np.radians(dang)
    phimin = phidist - np.radians(dang)
    phimax = phidist + np.radians(dang)

    theta = np.linspace(thmin, thmax, ltheta)
    phi = phidist if flag2D else np.linspace(phimin, phimax, lphi)

    r = RE * np.ones((ltheta, lphi))
    # use ground level for altitude for all field points

    phi, theta = np.meshgrid(phi, theta, indexing="ij")

    # %% CREATE AN INPUT FILE OF FIELD POINTS
    gridsize = np.array([lr, ltheta, lphi], dtype=np.int32)
    mag = {
        "r": r.astype(np.float32).ravel(),
        "phi": phi.astype(np.float32).ravel(),
        "theta": theta.astype(np.float32).ravel(),
        "gridsize": gridsize,
        "lpoints": gridsize.prod(),
    }

    if write_grid:
        filename = direc / "inputs/magfieldpoints.h5"
        print("Writing grid to", filename)
        write.maggrid(filename, mag)

    return mag


def makegrid_full(
    direc: str,
    ltheta: int = 16,
    lphi: int = 16,
    write_grid: bool = False,
) -> dict[str, T.Any]:
    """
    Make a field point to cover the entire mlat/mlon range for a given simulation grid
    """

    direc = Path(direc).expanduser()
    assert direc.is_dir(), f"{direc} is not a directory"

    # SIMULATION METADATA
    cfg = read.config(direc)

    # WE ALSO NEED TO LOAD THE GRID FILE
    xg = read.grid(direc)
    print("Grid loaded")

    # lx1 = xg.lx(1);
    lx3 = xg["lx"][2]
    # lh=lx1;   %possibly obviated in this version - need to check
    flag2D = lx3 == 1
    if flag2D:
        print("2D meshgrid")
    else:
        print("3D meshgrid")

    # TABULATE THE SOURCE OR GRID CENTER LOCATION
    if "sourcemlon" not in cfg.keys():
        # thdist = xg["theta"].mean()
        phidist = xg["phi"].mean()
    else:
        # thdist = np.pi / 2 - np.radians(cfg["sourcemlat"])
        # zenith angle of source location
        phidist = np.radians(cfg["sourcemlon"])

    # FIELD POINTS OF INTEREST (CAN/SHOULD BE DEFINED INDEPENDENT OF SIMULATION GRID)
    # ltheta = 40
    lphi = 1 if flag2D else ltheta
    lr = 1

    # Computer a buffer region around the grid
    thmin = xg["theta"].min()
    thmax = xg["theta"].max()
    dtheta = thmax - thmin
    thmin = thmin - dtheta / 5
    thmax = thmax + dtheta / 5
    phimin = xg["phi"].min()
    phimax = xg["phi"].max()
    dphi = phimax - phimin
    phimin = phimin - dphi / 5
    phimax = phimax + dtheta / 5

    theta = np.linspace(thmin, thmax, ltheta)
    phi = phidist if flag2D else np.linspace(phimin, phimax, lphi)

    r = RE * np.ones((ltheta, lphi))
    # use ground level for altitude for all field points

    phi, theta = np.meshgrid(phi, theta, indexing="ij")

    # %% CREATE AN INPUT FILE OF FIELD POINTS
    gridsize = np.array([lr, ltheta, lphi], dtype=np.int32)
    mag = {
        "r": r.astype(np.float32).ravel(),
        "phi": phi.astype(np.float32).ravel(),
        "theta": theta.astype(np.float32).ravel(),
        "gridsize": gridsize,
        "lpoints": gridsize.prod(),
    }

    if write_grid:
        filename = direc / "inputs/magfieldpoints.h5"
        print("Writing grid to", filename)
        write.maggrid(filename, mag)

    return mag


def magframe(
    filename: str | Path,
    *,
    cfg: dict[str, T.Any] | None = None,
    time: datetime | None = None,
) -> dict[str, T.Any]:
    """
    # example use
    # dat = gemini3d.read.magframe(filename)
    # dat = gemini3d.read.magframe(folder, "time", datetime)
    # dat = gemini3d.read.magframe(filename, "config", cfg)

    Translated from magframe.m
    2022/07/05
    Spencer M Hatch

    Tweaks to deal with pygemini API idiodsyncracies.  Also force
      return with no value if binary files used (should be deprecated
      soon) -MZ.
    2022/7/7
    """
    # make sure to add the default directory where the magnetic fields are to
    # be found
    fn = Path(filename).expanduser()
    direc = fn.parents[1] if fn.is_file() else fn.parent
    basemagdir = direc / "magfields"

    # find the actual filename if only the directory was given
    if not fn.is_file():
        if time is not None:
            fn = find.frame(basemagdir, time)

    # read the config file if one was not provided as input
    if cfg is None:
        cfg = read.config(direc)

    # load and construct the magnetic field point grid
    fnp = direc / "inputs/magfieldpoints.h5"
    assert fnp.is_file(), f"{fnp} not found"

    with h5py.File(fnp, "r") as f:
        lpoints = f["lpoints"][()]
        gridsize = f["gridsize"][:]
        r = f["r"][:]
        theta = f["theta"][:]
        phi = f["phi"][:]

    # Reorganize the field points if the user has specified a grid size
    if (gridsize < 0).any():
        gridsize = (lpoints, 1, 1)
        # flat list if the user has not specified any gridding
        flatlist = True
    else:
        flatlist = False

    lr, ltheta, lphi = gridsize
    r = r.reshape(gridsize)
    theta = theta.reshape(gridsize)
    phi = phi.reshape(gridsize)

    # Sanity check the grid size and total number of grid points
    assert lpoints == np.prod(
        gridsize
    ), "Incompatible data size and grid specification..."

    # Create grid alt, magnetic latitude, and longitude (assume input points
    # have been permuted in this order)...
    mlat = 90 - np.degrees(theta)
    mlon = np.degrees(phi)

    if flatlist:
        # we have a flat list of points
        ilatsort = slice(0, lpoints)
        ilonsort = slice(0, lpoints)
        dat = {"mlat": mlat, "mlon": mlon, "r": r}
    else:
        # we have a grid of points
        ilatsort = mlat[0, 0, :].argsort()
        ilonsort = mlon[0, :, 0].argsort()

        dat = {
            "mlat": mlat[0, 0, ilatsort],
            "mlon": mlon[0, ilonsort, 0],
            "r": r[:, 0, 0],
        }
        # assume already sorted properly

    with h5py.File(filename, "r") as f:
        for k in {"Br", "Btheta", "Bphi"}:
            if flatlist:
                dat[k] = f[f"/magfields/{k}"][:]
            else:
                dat[k] = f[f"/magfields/{k}"][:].reshape((lr, ltheta, lphi))

    return dat
