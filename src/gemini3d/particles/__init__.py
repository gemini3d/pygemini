from __future__ import annotations
import typing as T
import numpy as np

from .. import write
from ..utils import str2func
from .core import get_times
from .grid import precip_grid

# OUTPUT ARGUMENT NEEDS TO BE MADE OPTIONAL TO RUN 'OLD' EXAMPLES!!!!!

# this is loaded dynamically via str2func
from .gaussian2d import gaussian2d


__all__ = ["get_times", "particles_BCs", "gaussian2d"]


def particles_BCs(cfg: dict[str, T.Any], xg: dict[str, T.Any]):
    """write particle precipitation to disk"""

    pg = precip_grid(cfg, xg)

    # %% CREATE PRECIPITATION INPUT DATA
    # Q: energy flux [mW m^-2]
    # E0: characteristic energy [eV]

    # did user specify on/off time? if not, assume always on.
    t0 = pg.time[0].data

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

    assert np.isfinite(cfg["E0precip"]), "E0 precipitation must be finite"
    assert cfg["E0precip"] > 0, "E0 precip must be positive"
    assert cfg["E0precip"] < 100e6, "E0 precip must not be relativistic 100 MeV"

    # NOTE: in future, E0 could be made time-dependent in config.nml as 1D array

    func_path = None
    if "Qprecip_function" in cfg:
        if (cfg["nml"].parent / (cfg["Qprecip_function"] + ".py")).is_file():
            func_path = cfg["nml"].parent
        Qfunc = str2func(cfg["Qprecip_function"], func_path)
    else:
        Qfunc = str2func("gemini3d.particles.gaussian2d")

    Qtmp, E0_temp = Qfunc(pg, cfg["Qprecip"], cfg["Qprecip_background"])
    print("i_on: ", i_on)
    print("i_off: ", i_off)
    print("Qtmp shape: ", Qtmp.shape)
    print("E0tmp shape: ", E0_temp.shape)
    
    for i in range(i_on, i_off):
        if Qtmp.ndim == 3:
            pg["Q"][i, :, :] = Qtmp[i, :, :] # time, lon, lat
        else:
            pg["Q"][i, :, :] = Qtmp
        pg["E0"][i, :, :] = E0_temp

    assert np.isfinite(pg["Q"]).all(), "Q flux must be finite"
    assert (pg["Q"] >= 0).all(), "Q flux must be non-negative"
    if E0 > 0
        assert (pg["E0"] >=0).all(), "E0 characteristic energy must be non-negative"
    else
        continue

    # %% CONVERT THE ENERGY TO EV
    # E0 = max(E0,0.100);
    # E0 = E0*1e3;

    # %% SAVE to files
    # LEAVE THE SPATIAL AND TEMPORAL INTERPOLATION TO THE
    # FORTRAN CODE IN CASE DIFFERENT GRIDS NEED TO BE TRIED.
    # THE EFIELD DATA DO NOT NEED TO BE SMOOTHED.

    write.precip(pg, cfg["precdir"])
