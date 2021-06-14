from __future__ import annotations
import typing as T
import logging

import xarray
import numpy as np

from ..config import datetime_range


def precip_grid(cfg: dict[str, T.Any], xg: dict[str, T.Any]) -> xarray.Dataset:
    """CREATE PRECIPITATION CHARACTERISTICS data
    grid cells will be interpolated to grid, so 100x100 is arbitrary
    """

    # %% determine what type of grid (cartesian or dipole) we are dealing with
    if (xg["h1"] > 1.01).any():
        flagdip = True
        logging.info("Dipole grid detected")
    else:
        flagdip = False
        logging.info("Cartesian grid detected")

    llon = cfg.get("precip_llon", 100)
    llat = cfg.get("precip_llat", 100)

    lx2 = None
    lx3 = None
    for k in ("lx", "lxs"):
        if k in xg:
            _, lx2, lx3 = xg[k]
            break

    if flagdip:
        # dipole
        if lx2 == 1:
            llat = 1
        elif lx3 == 1:
            llon = 1
    else:
        if lx2 == 1:
            llon = 1
        elif lx3 == 1:
            llat = 1

    if lx2 is None:
        raise ValueError("size data not in Efield grid")

    # %% CREATE PRECIPITATION INPUT DATA
    """
    Qit: energy flux [mW m^-2]
    E0it: characteristic energy [eV]
    NOTE: since Fortran Gemini interpolates between time steps,
    having E0 default to zero is NOT appropriate, as the file before and/or
    after precipitation would interpolate from E0=0 to desired value, which
    is decidedly non-physical.
    We default E0 to NaN so that it's obvious (by Gemini emitting an
    error) that an unexpected input has occurred.
    """

    thetamin = xg["theta"].min()
    thetamax = xg["theta"].max()
    mlatmin = 90 - np.degrees(thetamax)
    mlatmax = 90 - np.degrees(thetamin)
    mlonmin = np.degrees(xg["phi"].min())
    mlonmax = np.degrees(xg["phi"].max())

    # add a 1% buff
    latbuf = 0.01 * (mlatmax - mlatmin)
    lonbuf = 0.01 * (mlonmax - mlonmin)

    time = datetime_range(cfg["time"][0], cfg["time"][0] + cfg["tdur"], cfg["dtprec"])

    pg = xarray.Dataset(
        {
            "Q": (("time", "mlon", "mlat"), np.zeros((len(time), llon, llat))),
            "E0": (("time", "mlon", "mlat"), np.zeros((len(time), llon, llat))),
        },
        coords={
            "time": time,
            "mlat": np.linspace(mlatmin - latbuf, mlatmax + latbuf, llat),
            "mlon": np.linspace(mlonmin - lonbuf, mlonmax + lonbuf, llon),
        },
    )

    # %% disturbance extents
    # avoid divide by zero
    if "precip_latwidth" in cfg:
        pg.attrs["mlat_sigma"] = max(cfg["precip_latwidth"] * (mlatmax - mlatmin), 0.01)
    if "precip_lonwidth" in cfg:
        pg.attrs["mlon_sigma"] = max(cfg["precip_lonwidth"] * (mlonmax - mlonmin), 0.01)

    return pg
