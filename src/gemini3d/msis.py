"""
using MSIS Fortran executable from Python
"""

from __future__ import annotations
from pathlib import Path, PurePosixPath
import subprocess
import logging
import typing as T
import os

import numpy as np
import h5py
import xarray

from . import find
from . import wsl


def get_msis_features(exe: Path) -> dict[str, bool]:
    """
    tell features of msis_setup executable
    """

    if not exe or not exe.is_file():
        raise FileNotFoundError(exe)

    cmd = [str(exe), "-features"]

    out = subprocess.check_output(cmd, text=True, timeout=10)

    return {"msis00": "MSIS00" in out, "msis2": "MSIS2" in out}


def msis_setup(p: dict[str, T.Any], xg: dict[str, T.Any]) -> xarray.Dataset:
    """
    calls MSIS Fortran executable msis_setup

    [f107a, f107, ap] = activ
    """

    msis_exe = find.executable("msis_setup", p.get("gemini_root"))

    alt_km = xg["alt"] / 1e3
    # % CONVERT DATES/TIMES/INDICES INTO MSIS-FRIENDLY FORMAT
    t0 = p["time"][0]
    doy = int(t0.strftime("%j"))
    UTsec0 = t0.hour * 3600 + t0.minute * 60 + t0.second + t0.microsecond / 1e6
    # clip non-positive ALTITUDES SO THAT THEY DON'T GIVE INF
    alt_km = alt_km.clip(min=1)

    # %% CREATE INPUT FILE FOR FORTRAN PROGRAM
    if p.get("indat_size") is not None:
        input_dir = Path(p["indat_size"]).expanduser().resolve(strict=False).parent

    if p.get("msis_infile") is None:
        if input_dir is None:
            raise ValueError("msis_infile OR indat_size must be specified")
        msis_infile = input_dir / "msis_setup_in.h5"
    else:
        msis_infile = Path(p["msis_infile"]).expanduser().resolve(strict=False)

    if p.get("msis_outfile") is None:
        if input_dir is None:
            raise ValueError("msis_outfile OR indat_size must be specified")
        msis_outfile = input_dir / "msis_setup_out.h5"
    else:
        msis_outfile = Path(p["msis_outfile"]).expanduser().resolve(strict=False)

    msis_version = p.get("msis_version", 0)

    if msis_version > 0:
        features = get_msis_features(msis_exe)
        if not features["msis2"]:
            raise EnvironmentError(f"MSIS 2.x requested but not present in {msis_exe}")

    msis_infile.parent.mkdir(exist_ok=True)

    with h5py.File(msis_infile, "w") as f:
        f.create_dataset("/doy", dtype=np.int32, data=doy)
        f.create_dataset("/UTsec", dtype=np.float32, data=UTsec0)
        f.create_dataset("/f107a", dtype=np.float32, data=p["f107a"])
        f.create_dataset("/f107", dtype=np.float32, data=p["f107"])
        f.create_dataset("/Ap", shape=(7,), dtype=np.float32, data=[p["Ap"]] * 7)
        # astype(float32) to save disk I/O time/space
        # we must give full shape to give proper rank/shape to Fortran/h5fortran
        f.create_dataset("/glat", shape=xg["lx"], dtype=np.float32, data=xg["glat"])
        f.create_dataset("/glon", shape=xg["lx"], dtype=np.float32, data=xg["glon"])
        f.create_dataset("/alt", shape=xg["lx"], dtype=np.float32, data=alt_km)
        f.create_dataset("/msis_version", dtype=np.int32, data=msis_version)
    # %% run MSIS
    if os.name == "nt" and isinstance(msis_exe, PurePosixPath):
        cmd = [
            "wsl",
            str(msis_exe),
            str(wsl.win_path2wsl_path(msis_infile)),
            str(wsl.win_path2wsl_path(msis_outfile)),
        ]
    else:
        cmd = [str(msis_exe), str(msis_infile), str(msis_outfile)]

    logging.info(" ".join(cmd))
    subprocess.check_call(cmd, text=True)

    # %% load MSIS output
    # use disk coordinates for tracability
    with h5py.File(msis_outfile, "r") as f:
        alt1 = f["/alt"][:, 0, 0]
        glat1 = f["/glat"][0, :, 0]
        glon1 = f["/glon"][0, 0, :]
        atmos = xarray.Dataset(coords={"alt_km": alt1, "glat": glat1, "glon": glon1})

        for k in {"nO", "nN2", "nO2", "Tn", "nN", "nH"}:
            atmos[k] = (("alt_km", "glat", "glon"), f[f"/{k}"][:])

    # %% sanity check MSIS output
    for v in atmos.data_vars:
        if v.startswith("n"):  # type: ignore
            assert (atmos[v] >= 0).all(), "density cannot be negative: {v}"
        elif v.startswith("T"):  # type: ignore
            assert (
                atmos[v] < 100000
            ).all(), "temperature above 100,000 K unexpected: {v}"

    # Mitra, 1968
    atmos["nNO"] = 0.4 * np.exp(-3700.0 / atmos["Tn"]) * atmos["nO2"] + 5e-7 * atmos["nO"]

    return atmos
