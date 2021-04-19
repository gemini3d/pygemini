from __future__ import annotations
import typing as T
import xarray
import numpy as np

from ..config import datetime_range


def precip_grid(cfg: dict[str, T.Any], xg: dict[str, T.Any]) -> xarray.Dataset:
    """CREATE PRECIPITATION CHARACTERISTICS data
    grid cells will be interpolated to grid, so 100x100 is arbitrary
    """

    lx2 = None
    lx3 = None

    llon = 100
    llat = 100
    # NOTE: cartesian-specific code
    for k in ("lx", "lxs"):
        if k in xg:
            _, lx2, lx3 = xg[k]
            break
    if lx2 == 1:
        llon = 1
    elif lx3 == 1:
        llat = 1

    if lx2 is None:
        raise ValueError("size data not in Efield grid")

    thetamin = xg["theta"].min()
    thetamax = xg["theta"].max()
    mlatmin = 90 - np.degrees(thetamax)
    mlatmax = 90 - np.degrees(thetamin)
    mlonmin = np.degrees(xg["phi"].min())
    mlonmax = np.degrees(xg["phi"].max())

    # add a 1% buff
    latbuf = 0.01 * (mlatmax - mlatmin)
    lonbuf = 0.01 * (mlonmax - mlonmin)

    pg = xarray.Dataset(
        {
            "Q": (("time", "mlon", "mlat"), np.zeros((len(cfg["time"]), llon, llat))),
            "E0": (("time", "mlon", "mlat"), np.zeros((len(cfg["time"]), llon, llat))),
        },
        coords={
            "time": datetime_range(cfg["time"][0], cfg["time"][0] + cfg["tdur"], cfg["dtprec"]),
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
