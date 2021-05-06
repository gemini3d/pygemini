"""
using MSIS Fortran exectuable from Python
"""

from __future__ import annotations
import xarray
from pathlib import Path
import numpy as np
import subprocess
import logging
import typing as T
import h5py

from . import cmake


def msis_setup(p: dict[str, T.Any], xg: dict[str, T.Any]) -> xarray.Dataset:
    """
    calls MSIS Fortran executable msis_setup--builds if not present

    [f107a, f107, ap] = activ
    """

    msis_exe = cmake.build_gemini3d(Path("msis_setup"))

    alt_km = xg["alt"] / 1e3
    # % CONVERT DATES/TIMES/INDICES INTO MSIS-FRIENDLY FORMAT
    t0 = p["time"][0]
    doy = int(t0.strftime("%j"))
    UTsec0 = t0.hour * 3600 + t0.minute * 60 + t0.second + t0.microsecond / 1e6
    # censor BELOW-ZERO ALTITUDES SO THAT THEY DON'T GIVE INF
    alt_km[alt_km <= 0] = 1
    # %% CREATE INPUT FILE FOR FORTRAN PROGRAM
    msis_infile = p.get("msis_infile", p["indat_size"].parent / "msis_setup_in.h5")
    msis_outfile = p.get("msis_outfile", p["indat_size"].parent / "msis_setup_out.h5")

    with h5py.File(msis_infile, "w") as f:
        f["/doy"] = doy
        f["/UTsec"] = UTsec0
        f["/f107a"] = p["f107a"]
        f["/f107"] = p["f107"]
        f["/Ap"] = [p["Ap"]] * 7
        # astype(float32) just to save disk I/O time/space
        # we must give full shape to work with Fortran/h5fortran
        # this is how MatGemini does it
        f.create_dataset("/glat", shape=xg["lx"], dtype=np.float32, data=xg["glat"])
        f.create_dataset("/glon", shape=xg["lx"], dtype=np.float32, data=xg["glon"])
        f.create_dataset("/alt", shape=xg["lx"], dtype=np.float32, data=alt_km)
    # %% run MSIS
    args = [str(msis_infile), str(msis_outfile)]

    if "msis_version" in p:
        args.append(str(p["msis_version"]))
    cmd = [str(msis_exe)] + args
    logging.info(" ".join(cmd))
    ret = subprocess.run(cmd, text=True, cwd=msis_exe.parent)

    if ret.returncode == 20:
        raise RuntimeError("Need to compile with 'cmake -Dmsis20=true'")
    if ret.returncode != 0:
        raise RuntimeError(
            f"MSIS failed to run: return code {ret.returncode}. See console for additional error info."
        )

    # %% load MSIS output
    # use disk coordinates for tracability
    with h5py.File(msis_outfile, "r") as f:
        alt1 = f["/alt"][:, 0, 0]
        glat1 = f["/glat"][0, :, 0]
        glon1 = f["/glon"][0, 0, :]
        atmos = xarray.Dataset(coords={"alt_km": alt1, "glat": glat1, "glon": glon1})

        for k in {"nO", "nN2", "nO2", "Tn", "nN", "nH"}:
            atmos[k] = (("alt_km", "glat", "glon"), f[f"/{k}"][:])

    # Mitra, 1968
    atmos["nNO"] = 0.4 * np.exp(-3700. / atmos["Tn"]) * atmos["nO2"] + 5e-7 * atmos["nO"]

    return atmos
