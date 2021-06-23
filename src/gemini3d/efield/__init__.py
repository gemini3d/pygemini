"""
generate, read, write Electric Field
"""

from __future__ import annotations
import typing as T
import logging

import xarray
import numpy as np

from .. import write
from ..config import datetime_range
from ..utils import str2func


def Efield_BCs(cfg: dict[str, T.Any], xg: dict[str, T.Any]) -> xarray.Dataset:
    """generate E-field

    Set input potential/FAC boundary conditions and write these to a set of
    files that can be used an input to GEMINI.  This is a basic example that
    can make Gaussian shaped potential or FAC inputs using an input width.
    """

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

    # %% For current density boundary conditions we need to determine top v bottom of the grid
    if xg["alt"][0, 0, 0] > xg["alt"][1, 0, 0]:
        # inverted
        gridflag = 1
        logging.info("Detected an inverted grid")
    else:
        # non-inverted or closed
        gridflag = 2
        logging.info("Detected a non-inverted grid")

    # %% determine what type of grid (cartesian or dipole) we are dealing with
    if (xg["h1"] > 1.01).any():
        flagdip = True
        logging.info("Dipole grid detected")
    else:
        flagdip = False
        logging.info("Cartesian grid detected")

    # %% CREATE ELECTRIC FIELD DATASET
    # the Efield is interpolated from these, 100 x 100 is arbitrary
    llon = cfg.get("Efield_llon", 100)
    llat = cfg.get("Efield_llat", 100)
    if flagdip:
        if lx3 == 1:
            llon = 1
        elif lx2 == 1:
            llat = 1
    else:
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
        if flagdip:
            E.attrs["mlatsig"], E.attrs["sigx2"] = Esigma(
                cfg["Efield_latwidth"], mlatmax, mlatmin, xg["x2"]
            )
        else:
            E.attrs["mlatsig"], E.attrs["sigx3"] = Esigma(
                cfg["Efield_latwidth"], mlatmax, mlatmin, xg["x3"]
            )

    if "Efield_lonwidth" in cfg:
        if flagdip:
            E.attrs["mlonsig"], E.attrs["sigx3"] = Esigma(
                cfg["Efield_lonwidth"], mlonmax, mlonmin, xg["x3"]
            )
        else:
            E.attrs["mlonsig"], E.attrs["sigx2"] = Esigma(
                cfg["Efield_lonwidth"], mlonmax, mlonmin, xg["x2"]
            )

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
    func_path = None
    if "Etarg" in cfg:
        E.attrs["Etarg"] = cfg["Etarg"]
        if "Etarg_function" in cfg:
            if (cfg["nml"].parent / (cfg["Etarg_function"] + ".py")).is_file():
                func_path = cfg["nml"].parent
            func = str2func(cfg["Etarg_function"], func_path)
        else:
            func = str2func("gemini3d.efield.Efield_erf")

        E = func(E, xg, lx1, lx2, lx3, gridflag, flagdip)
    elif "Jtarg" in cfg:
        E.attrs["Jtarg"] = cfg["Jtarg"]
        if "Jtarg_function" in cfg:
            if (cfg["nml"].parent / (cfg["Jtarg_function"] + ".py")).is_file():
                func_path = cfg["nml"].parent
            func = str2func(cfg["Jtarg_function"], func_path)
        else:
            func = str2func("gemini3d.efield.Jcurrent_gaussian")

        E = func(E, gridflag, flagdip)
    else:
        # background only
        pass

    # %% check for NaNs
    # this is also done in Fortran, but just to help ensure results.
    check_finite(E["Exit"])
    check_finite(E["Eyit"])
    check_finite(E["Vminx1it"])
    check_finite(E["Vmaxx1it"])
    check_finite(E["Vminx2ist"])
    check_finite(E["Vmaxx2ist"])
    check_finite(E["Vminx3ist"])
    check_finite(E["Vmaxx3ist"])

    # %% SAVE THESE DATA TO APPROPRIATE FILES
    # LEAVE THE SPATIAL AND TEMPORAL INTERPOLATION TO THE
    # FORTRAN CODE IN CASE DIFFERENT GRIDS NEED TO BE TRIED.
    # THE EFIELD DATA DO NOT TYPICALLY NEED TO BE SMOOTHED.
    write.Efield(E, cfg["E0dir"], cfg["file_format"])

    return E


def Esigma(pwidth: float, pmax: float, pmin: float, px: np.ndarray) -> tuple[float, T.Any]:
    """Set width given a fraction of the coordinate an extent"""

    wsig = pwidth * (pmax - pmin)
    xsig = pwidth * (px.max() - px.min())

    return wsig, xsig


def check_finite(v: xarray.DataArray, name: str = None):

    i = np.logical_not(np.isfinite(v))

    if i.any():
        if not name:
            name = str(v.name) if v.name else ""
        raise ValueError(f"{np.count_nonzero(i)} NaN in {name}")
