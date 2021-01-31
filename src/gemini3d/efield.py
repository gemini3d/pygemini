"""
generate, read, write Electric Field
"""

from __future__ import annotations
import xarray
import typing as T
import numpy as np
import logging
from scipy.special import erf

from . import write
from .config import datetime_range


def Efield_BCs(cfg: dict[str, T.Any], xg: dict[str, T.Any]) -> xarray.Dataset:
    """ generate E-field """

    # %% READ IN THE SIMULATION INFO
    lx1 = None
    lx2 = None
    lx3 = None
    for k in ("lx", "lxs"):
        if k in xg:
            lx1, lx2, lx3 = xg[k]
            break

    if lx1 is None:
        raise ValueError("size data not in Efield grid")

    # %% CREATE ELECTRIC FIELD DATASET
    # the Efield is interpolated from these, 100 x 100 is arbitrary
    llon = 100
    llat = 100
    # NOTE: cartesian-specific code
    if lx2 == 1:
        llon = 1
    elif lx3 == 1:
        llat = 1

    thetamin = xg["theta"].min()
    thetamax = xg["theta"].max()
    mlatmin = 90 - np.degrees(thetamax)
    mlatmax = 90 - np.degrees(thetamin)
    mlonmin = np.degrees(xg["phi"].min())
    mlonmax = np.degrees(xg["phi"].max())

    # add a 1% buff
    latbuf = 0.01 * (mlatmax - mlatmin)
    lonbuf = 0.01 * (mlonmax - mlonmin)

    E = xarray.Dataset(
        coords={
            "time": datetime_range(cfg["time"][0], cfg["time"][0] + cfg["tdur"], cfg["dtE0"]),
            "mlat": np.linspace(mlatmin - latbuf, mlatmax + latbuf, llat),
            "mlon": np.linspace(mlonmin - lonbuf, mlonmax + lonbuf, llon),
        }
    )

    Nt = E.time.size
    E.attrs["mlonmean"] = E.mlon.mean()
    E.attrs["mlatmean"] = E.mlat.mean()

    # %% WIDTH OF THE DISTURBANCE
    if "Efield_latwidth" in cfg:
        E.attrs["mlatsig"] = cfg["Efield_latwidth"] * (mlatmax - mlatmin)
        E.attrs["sigx3"] = cfg["Efield_latwidth"] * (xg["x3"].max() - xg["x3"].min())
    if "Efield_lonwidth" in cfg:
        E.attrs["mlonsig"] = cfg["Efield_lonwidth"] * (mlonmax - mlonmin)
        E.attrs["sigx2"] = cfg["Efield_lonwidth"] * (xg["x2"].max() - xg["x2"].min())

    # %% CREATE DATA FOR BACKGROUND ELECTRIC FIELDS
    # assign to zero in case not specifically assigned
    if "Exit" in cfg:
        E["Exit"] = (("time", "mlon", "mlat"), cfg["Exit"] * np.ones((Nt, llon, llat)))
    else:
        E["Exit"] = (("time", "mlon", "mlat"), np.zeros((Nt, llon, llat)))
    if "Eyit" in cfg:
        E["Eyit"] = (("time", "mlon", "mlat"), cfg["Eyit"] * np.ones((Nt, llon, llat)))
    else:
        E["Eyit"] = (("time", "mlon", "mlat"), np.zeros((Nt, llon, llat)))

    # %% CREATE DATA FOR BOUNDARY CONDITIONS FOR POTENTIAL SOLUTION

    # if 0 data is interpreted as FAC, else we interpret it as potential
    E["flagdirich"] = (("time",), np.zeros(Nt, dtype=np.int32))
    E["Vminx1it"] = (("time", "mlon", "mlat"), np.zeros((Nt, llon, llat)))
    E["Vmaxx1it"] = (("time", "mlon", "mlat"), np.zeros((Nt, llon, llat)))
    # these are just slices
    E["Vminx2ist"] = (("time", "mlat"), np.zeros((Nt, llat)))
    E["Vmaxx2ist"] = (("time", "mlat"), np.zeros((Nt, llat)))
    E["Vminx3ist"] = (("time", "mlon"), np.zeros((Nt, llon)))
    E["Vmaxx3ist"] = (("time", "mlon"), np.zeros((Nt, llon)))

    # %% synthesize feature
    if "Etarg" in cfg:
        E.attrs["Etarg"] = cfg["Etarg"]
        E = Efield_target(E, xg, lx1, lx2, lx3)
    elif "Jtarg" in cfg:
        E.attrs["Jtarg"] = cfg["Jtarg"]
        E = Jcurrent_target(E)
    else:
        # background only
        pass

    # %% check for NaNs
    # this is also done in Fortran, but just to help ensure results.
    check_finite(E["Exit"], "Exit")
    check_finite(E["Eyit"], "Eyit")
    check_finite(E["Vminx1it"], "Vminx1it")
    check_finite(E["Vmaxx1it"], "Vmaxx1it")
    check_finite(E["Vminx2ist"], "Vminx2ist")
    check_finite(E["Vmaxx2ist"], "Vmaxx2ist")
    check_finite(E["Vminx3ist"], "Vminx3ist")
    check_finite(E["Vmaxx3ist"], "Vmaxx3ist")

    # %% SAVE THESE DATA TO APPROPRIATE FILES
    # LEAVE THE SPATIAL AND TEMPORAL INTERPOLATION TO THE
    # FORTRAN CODE IN CASE DIFFERENT GRIDS NEED TO BE TRIED.
    # THE EFIELD DATA DO NOT TYPICALLY NEED TO BE SMOOTHED.
    write.Efield(E, cfg["E0dir"], cfg["file_format"])

    return E


def Jcurrent_target(E: xarray.Dataset) -> xarray.Dataset:

    S = (
        E["Jtarg"]
        * np.exp(-((E.mlon - E.mlonmean) ** 2) / 2 / E.mlonsig ** 2)
        * np.exp(-((E.mlat - E.mlatmean - 1.5 * E.mlatsig) ** 2) / 2 / E.mlatsig ** 2)
    )

    for t in E.time[6:]:
        E["flagdirich"].loc[t] = 0
        # could have different boundary types for different times
        E["Vmaxx1it"].loc[t] = S - E.Jtarg * np.exp(
            -((E.mlon - E.mlonmean) ** 2) / 2 / E.mlonsig ** 2
        ) * np.exp(-((E.mlat - E.mlatmean + 1.5 * E.mlatsig) ** 2) / 2 / E.mlatsig ** 2)

    return E


def Efield_target(
    E: xarray.Dataset, xg: dict[str, T.Any], lx1: int, lx2: int, lx3: int
) -> xarray.Dataset:
    """
    synthesize a feature
    """

    if E.Etarg > 1:
        logging.warning(f"Etarg units V/m -- is {E['Etarg']} V/m realistic?")

    # NOTE: h2, h3 have ghost cells, so we use lx1 instead of -1 to index
    # pk is a scalar.

    if lx3 == 1:
        # east-west
        S = E.Etarg * E.sigx2 * xg["h2"][lx1, lx2 // 2, 0] * np.sqrt(np.pi) / 2
        taper = erf((E.mlon - E.mlonmean) / E.mlonsig).values[:, None]
    elif lx2 == 1:
        # north-south
        S = E.Etarg * E.sigx3 * xg["h3"][lx1, 0, lx3 // 2] * np.sqrt(np.pi) / 2
        taper = erf((E.mlat - E.mlatmean) / E.mlatsig).values[None, :]
    else:
        # 3D
        S = E.Etarg * E.sigx2 * xg["h2"][lx1, lx2 // 2, 0] * np.sqrt(np.pi) / 2
        taper = (
            erf((E.mlon - E.mlonmean) / E.mlonsig).values[:, None]
            * erf((E.mlat - E.mlatmean) / E.mlatsig).values[None, :]
        )

    assert S.ndim == 0, "S is a scalar"

    for t in E.time:
        E["flagdirich"].loc[t] = 1
        E["Vmaxx1it"].loc[t] = S * taper

    return E


def check_finite(v: np.ndarray, name: str):

    i = ~np.isfinite(v)
    if i.any():
        raise ValueError(f"{np.count_nonzero(i)} NaN in {name} at {i.nonzero()}")
