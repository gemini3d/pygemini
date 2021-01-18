"""
using MSIS Fortran exectuable from Python
"""

from pathlib import Path
import numpy as np
import subprocess
import logging
import typing as T
import h5py

from . import cmake


def msis_setup(p: T.Dict[str, T.Any], xg: T.Dict[str, T.Any]) -> np.ndarray:
    """calls MSIS Fortran exectuable
    compiles if not present

    [f107a, f107, ap] = activ
        COLUMNS OF DATA:
          1 - ALT
          2 - HE NUMBER DENSITY(M-3)
          3 - O NUMBER DENSITY(M-3)
          4 - N2 NUMBER DENSITY(M-3)
          5 - O2 NUMBER DENSITY(M-3)
          6 - AR NUMBER DENSITY(M-3)
          7 - TOTAL MASS DENSITY(KG/M3)
          8 - H NUMBER DENSITY(M-3)
          9 - N NUMBER DENSITY(M-3)
          10 - Anomalous oxygen NUMBER DENSITY(M-3)
          11 - TEMPERATURE AT ALT

    """

    msis_exe = cmake.build_gemini3d(Path("msis_setup"))

    # %% SPECIFY SIZES ETC.
    lx1 = xg["lx"][0]
    lx2 = xg["lx"][1]
    lx3 = xg["lx"][2]
    alt = xg["alt"] / 1e3
    glat = xg["glat"]
    glon = xg["glon"]
    lz = lx1 * lx2 * lx3
    # % CONVERT DATES/TIMES/INDICES INTO MSIS-FRIENDLY FORMAT
    t0 = p["time"][0]
    doy = int(t0.strftime("%j"))
    UTsec0 = t0.hour * 3600 + t0.minute * 60 + t0.second + t0.microsecond / 1e6

    logging.debug(f"MSIS00 using DOY: {doy}")
    # %% KLUDGE THE BELOW-ZERO ALTITUDES SO THAT THEY DON'T GIVE INF
    alt[alt <= 0] = 1
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
        f["/glat"] = glat.astype(np.float32)
        f["/glon"] = glon.astype(np.float32)
        f["/alt"] = alt.astype(np.float32)
    args = [str(msis_infile), str(msis_outfile), str(lz)]

    if "msis_version" in p:
        args.append(str(p["msis_version"]))
    cmd = [str(msis_exe)] + args
    logging.info(" ".join(cmd))
    ret = subprocess.run(cmd, text=True, cwd=msis_exe.parent)

    if ret.returncode == 20:
        raise RuntimeError("Need to compile with 'cmake -Dmsis20=true'")
    if ret.returncode != 0:
        raise RuntimeError(f"MSIS failed to run: {ret.stdout}")

    lsp = 7
    natm = np.empty((lsp, lx1, lx2, lx3))

    with h5py.File(msis_outfile, "r") as f:
        nO = natm[0, ...] = f["/nO"][:]
        natm[1, ...] = f["/nN2"][:]
        nO2 = natm[2, ...] = f["/nO2"][:]
        Tn = natm[3, ...] = f["/Tn"][:]
        natm[4, ...] = f["/nN"][:]
        natm[6, ...] = f["/nH"][:]

    # Mitra, 1968
    natm[5, ...] = 0.4 * np.exp(-3700 / Tn) * nO2 + 5e-7 * nO  # nNO

    return natm
