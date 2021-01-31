from __future__ import annotations
import typing as T
import numpy as np
import xarray

from . import write
from .config import datetime_range


def particles_BCs(cfg: dict[str, T.Any], xg: dict[str, T.Any]):
    """ write particle precipitation to disk """

    pg = precip_grid(cfg, xg)

    # %% CREATE PRECIPITATION INPUT DATA
    # Q: energy flux [mW m^-2]
    # E0: characteristic energy [eV]

    # did user specify on/off time? if not, assume always on.
    t0 = pg.time[0].values

    if "precip_startsec" in cfg:
        t = t0 + np.timedelta64(cfg["precip_startsec"])
        i_on = abs(pg.time - t).argmin().item()
    else:
        i_on = 0

    if "precip_endsec" in cfg:
        t = t0 + np.timedelta64(cfg["precip_endsec"])
        i_off = abs(pg.time - t).argmin().item()
    else:
        i_off = pg.time.size

    # NOTE: in future, E0 could be made time-dependent in config.nml as 1D array
    for i in range(i_on, i_off):
        pg["Q"][i, :, :] = precip_gaussian2d(pg, cfg["Qprecip"], cfg["Qprecip_background"])
        pg["E0"][i, :, :] = cfg["E0precip"]

    # %% CONVERT THE ENERGY TO EV
    # E0 = max(E0,0.100);
    # E0 = E0*1e3;

    # %% SAVE to files
    # LEAVE THE SPATIAL AND TEMPORAL INTERPOLATION TO THE
    # FORTRAN CODE IN CASE DIFFERENT GRIDS NEED TO BE TRIED.
    # THE EFIELD DATA DO NOT NEED TO BE SMOOTHED.

    write.precip(pg, cfg["precdir"], cfg["file_format"])


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


def precip_gaussian2d(pg: xarray.Dataset, Qpeak: float, Qbackground: float) -> np.ndarray:

    mlon_mean = pg.mlon.mean().item()
    mlat_mean = pg.mlat.mean().item()

    if "mlon_sigma" in pg.attrs and "mlat_sigma" in pg.attrs:
        Q = (
            Qpeak
            * np.exp(-((pg.mlon.values[:, None] - mlon_mean) ** 2) / (2 * pg.mlon_sigma ** 2))
            * np.exp(-((pg.mlat.values[None, :] - mlat_mean) ** 2) / (2 * pg.mlat_sigma ** 2))
        )
    elif "mlon_sigma" in pg.attrs:
        Q = Qpeak * np.exp(-((pg.mlon.values[:, None] - mlon_mean) ** 2) / (2 * pg.mlon_sigma ** 2))
    elif "mlat_sigma" in pg.attrs:
        Q = Qpeak * np.exp(-((pg.mlat.values[None, :] - mlat_mean) ** 2) / (2 * pg.mlat_sigma ** 2))
    else:
        raise LookupError("precipation must be defined in latitude, longitude or both")

    Q[Q < Qbackground] = Qbackground

    return Q
