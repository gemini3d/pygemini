"""
Calculate collision frequencies for full gemini grid

Equations from Shunk and Nagy, 2009

NOTE: These collision freqeuences are good for *almost all* ionospheric conditions -- they
  do break down at very high temperatures.  This has become an issue in the past to
  the point that @jdiazpena has implemented some thresholds in the core gemini code
  (note temperature checking here:
  https://github.com/gemini3d/gemini3d/blob/main/src/collisions/collisions.f90
  These checks are *not* currently implemented here as they are only rarely needed;
  really only when temperature exceeds 6000-7000 K (SAID/STEVE-type conditions).
    -MZ

Original script - M. Redden, 2022
M. Redden, 2023
"""

from pathlib import Path
import numpy as np
from datetime import datetime
import xarray as xr

import gemini3d.read
import gemini3d.msis

# Types of collisions
# - Ion-Neutral [NixNn] X
#   - Ion-Netural resonant X
# - Ion-Ion [NixNi]
# - Electron-Neutral [Nn] X
# - Electron-Ion [Ni]
# - ion collision frequency
# - electron collision frequency


def collisionfrequency(
    path: Path,
    time: datetime | None = None,
) -> tuple:
    """
    Parameters
    ----------

    path: pathlib.Path
        filename or directory + time to plot
    time: datetime.datetime, optional
        if path is a directory, time is required
    """

    # Read in GEMINI data
    dat = gemini3d.read.frame(path, time, var={"ne", "Te", "Ti", "ns", "Ts"})

    # ns = dat["ns"].assign_coords(species=["O+", "NO+", "N2+", "O2+", "N+", "H+", "e"])
    Ts = dat["Ts"].assign_coords(species=["O+", "NO+", "N2+", "O2+", "N+", "H+", "e"])

    # Read in MSIS data
    cfg = gemini3d.read.config(path)
    xg = gemini3d.read.grid(path)

    msisdata0 = gemini3d.msis.msis_setup(cfg, xg)

    # Reconfigure MSIS xarray so it can be used in operations with the GEMINI xarray
    msisdata = xr.DataArray(dims=["x1", "x2", "x3"], coords=[dat.x1, dat.x2, dat.x3])
    for k in list(msisdata0.keys()):
        msisdata[k] = (("x1", "x2", "x3"), np.array(msisdata0[k]))

    ###################################
    ### Ion-Neutral Collision Frequency
    ###################################

    nu_in = xr.DataArray(
        dims=["species", "neutral", "x1", "x2", "x3"],
        coords=(
            ["O+", "NO+", "N2+", "O2+", "N+", "H+"],
            ["O", "N2", "O2", "N", "H"],
            dat["x1"],
            dat["x2"],
            dat["x3"],
        ),
    )

    # For individual ion-neutral pairs, the non-resonant collision frequency is given by
    # S&N equation (4.146): nu_in_nonres = C_in * n_n.  C_in values are listed in
    # Table 4.4.  n_n is in units of cm^-3.

    # O+
    nu_in.loc["O+", "N2"] = 6.82e-10 * msisdata["nN2"] * 1e-6
    nu_in.loc["O+", "O2"] = 6.64e-10 * msisdata["nO2"] * 1e-6
    nu_in.loc["O+", "N"] = 4.62e-10 * msisdata["nN"] * 1e-6

    # NO+
    nu_in.loc["NO+", "O"] = 2.44e-10 * msisdata["nO"] * 1e-6
    nu_in.loc["NO+", "N2"] = 4.34e-10 * msisdata["nN2"] * 1e-6
    nu_in.loc["NO+", "O2"] = 4.27e-10 * msisdata["nO2"] * 1e-6
    nu_in.loc["NO+", "N"] = 2.79e-10 * msisdata["nN"] * 1e-6
    nu_in.loc["NO+", "H"] = 0.69e-10 * msisdata["nH"] * 1e-6

    # N2+
    nu_in.loc["N2+", "O"] = 2.58e-10 * msisdata["nO"] * 1e-6
    nu_in.loc["N2+", "O2"] = 4.49e-10 * msisdata["nO2"] * 1e-6
    nu_in.loc["N2+", "N"] = 2.95e-10 * msisdata["nN"] * 1e-6
    nu_in.loc["N2+", "H"] = 0.74e-10 * msisdata["nH"] * 1e-6

    # O2+
    nu_in.loc["N2+", "O"] = 2.31e-10 * msisdata["nO"] * 1e-6
    nu_in.loc["N2+", "N2"] = 4.13e-10 * msisdata["nN2"] * 1e-6
    nu_in.loc["N2+", "N"] = 2.64e-10 * msisdata["nN"] * 1e-6
    nu_in.loc["N2+", "H"] = 0.65e-10 * msisdata["nH"] * 1e-6

    # N+
    nu_in.loc["N+", "O"] = 4.42e-10 * msisdata["nO"] * 1e-6
    nu_in.loc["N+", "N2"] = 7.47e-10 * msisdata["nN2"] * 1e-6
    nu_in.loc["N+", "O2"] = 7.25e-10 * msisdata["nO2"] * 1e-6
    nu_in.loc["N+", "O"] = 1.45e-10 * msisdata["nH"] * 1e-6

    # H+
    nu_in.loc["H+", "N2"] = 33.6e-10 * msisdata["nN2"] * 1e-6
    nu_in.loc["H+", "O2"] = 32.0e-10 * msisdata["nO2"] * 1e-6
    nu_in.loc["H+", "N"] = 26.1e-10 * msisdata["nN"] * 1e-6

    # For individual ion-neutral pairs, the resonant collision frequencies are given by the
    # equations in Table 4.5

    # O+, O
    Tr = (Ts.sel(species="O+") + msisdata["Tn"]) / 2
    nu_in.loc["O+", "O"] = (
        3.67e-11
        * (msisdata["nO"] * 1e-6)
        * np.sqrt(Tr)
        * (1.0 - 0.064 * np.log10(Tr)) ** 2
    )

    # O+, H
    nu_in.loc["O+", "H"] = (
        4.63e-12
        * (msisdata["nH"] * 1e-6)
        * np.sqrt((msisdata["Tn"] + Ts.sel(species="O+")) / 16)
    )

    # N2+, N2
    Tr = (Ts.sel(species="N2+") + msisdata["Tn"]) / 2
    nu_in.loc["N2+", "N2"] = (
        5.14e-11
        * (msisdata["nN2"] * 1e-6)
        * np.sqrt(Tr)
        * (1.0 - 0.069 * np.log10(Tr)) ** 2
    )

    # O2+, O2
    Tr = (Ts.sel(species="O2+") + msisdata["Tn"]) / 2
    nu_in.loc["O2+", "O2"] = (
        2.59e-11
        * (msisdata["nO2"] * 1e-6)
        * np.sqrt(Tr)
        * (1.0 - 0.073 * np.log10(Tr)) ** 2
    )

    # N+, N
    Tr = (Ts.sel(species="N+") + msisdata["Tn"]) / 2
    nu_in.loc["N+", "N"] = (
        3.83e-11
        * (msisdata["nN"] * 1e-6)
        * np.sqrt(Tr)
        * (1.0 - 0.063 * np.log10(Tr)) ** 2
    )

    # H+, H
    Tr = (Ts.sel(species="H+") + msisdata["Tn"]) / 2
    nu_in.loc["H+", "H"] = (
        2.65e-10
        * (msisdata["nH"] * 1e-6)
        * np.sqrt(Tr)
        * (1.0 - 0.083 * np.log10(Tr)) ** 2
    )

    # H+, O
    nu_in.loc["H+", "O"] = (
        6.61e-11
        * (msisdata["nO"] * 1e-6)
        * np.sqrt(Ts.sel(species="H+"))
        * (1.0 - 0.047 * np.log10(Ts.sel(species="H+"))) ** 2
    )

    # Total ion-neutral collision frequency
    # sum all ion and neutral dimensions of nu_in
    nu_in_tot = nu_in.sum(dim=("species", "neutral"))

    ########################################
    ### Electron-Neutral Collision Frequency
    ########################################

    nu_en = xr.DataArray(
        dims=["neutral", "x1", "x2", "x3"],
        coords=(["O", "N2", "O2", "N", "H"], dat["x1"], dat["x2"], dat["x3"]),
    )

    # Electron-neutral collision frequencies are listed in S&N Table 4.6
    nu_en.loc["O"] = (
        8.9e-11
        * (msisdata["nO"] * 1e-6)
        * (1.0 + 5.7e-4 * dat["Te"])
        * np.sqrt(dat["Te"])
    )
    nu_en.loc["N2"] = (
        2.33e-11 * (msisdata["nN2"] * 1e-6) * (1.0 - 1.21e-4 * dat["Te"]) * dat["Te"]
    )
    nu_en.loc["O2"] = (
        1.82e-10
        * (msisdata["nO2"] * 1e-6)
        * (1.0 + 3.6e-2 * np.sqrt(dat["Te"]))
        * np.sqrt(dat["Te"])
    )
    nu_en.loc["H"] = (
        4.5e-9
        * (msisdata["nH"] * 1e-6)
        * (1.0 - 1.35e-4 * dat["Te"])
        * np.sqrt(dat["Te"])
    )

    ### Total Electron-Neutral Collision Frequency
    # Sum over neutral dimension of nu_en
    nu_en_tot = nu_en.sum(dim="neutral")

    ####################################
    ### Electron-Ion Collision Frequency
    ####################################
    # Note: not currently returned

    nu_ei = xr.DataArray(
        dims=["ion", "x1", "x2", "x3"],
        coords=(
            ["O+", "NO+", "N2+", "O2+", "N+", "H+"],
            dat["x1"],
            dat["x2"],
            dat["x3"],
        ),
    )

    # Electron-ion (nu_ei).  Need to loop through all ion species.  Use Schunk and Nagy
    # Equation 4.144.
    for s in nu_ei.coords["ion"].values:
        nu_ei.loc[s] = 54.5 * ((Ts.sel(species=s) * 1e-6) / dat["Te"] ** 1.5)

    return nu_in, nu_en, nu_in_tot, nu_en_tot
