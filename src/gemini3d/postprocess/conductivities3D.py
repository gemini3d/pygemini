#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 15:40:15 2023

@author: redden
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 13:49:51 2022

@author: mer
"""

from pathlib import Path
import numpy as np
from datetime import datetime
import scipy.constants as const
import xarray as xr

import gemini3d.read

from .collisions3D import collisionfrequency


def conductivity(
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

    # kb = 1.38e-23
    # amu = 1.66e-27
    # e = 1.60e-19
    B = 5.0e-5  # Teslas
    # m_e = 9.11e-31
    # m_Oplus = 2.21e-27
    # m_NOplus = 4.15e-27
    # m_N2plus = 3.87e-27
    # m_O2plus = 4.43e-27
    # m_Nplus = 1.94e-27
    # m_Hplus = 1.38e-28
    # mass_ions = [m_Oplus, m_NOplus, m_N2plus, m_O2plus, m_Nplus, m_Hplus]
    m_i = xr.DataArray(
        dims=("species"),
        coords=[("species", ["O+", "NO+", "N2+", "O2+", "N+", "H+"])],
        data=np.array([16.0, 30.0, 28.0, 32.0, 14.0, 1.0]) * const.m_p,
    )
    # O = 16, N2 = 28, O2 = 32, N = 14, NO = 30, H =1

    # Calculate gyrofrequencies
    Omg_i = const.e * B / m_i
    Omg_e = const.e * B / const.m_e

    # Read in GEMINI data
    dat = gemini3d.read.frame(path, time, var={"ne", "Te", "Ti", "ns", "Ts"})

    ns = dat["ns"].assign_coords(species=["O+", "NO+", "N2+", "O2+", "N+", "H+", "e"])
    n_i = ns.drop("e", dim="species")
    n_e = dat["ne"]

    # Get collision frequencies
    nu_in, nu_en, _, _ = collisionfrequency(path, time)

    # Calculate ion and eletron collision frequencies by summing over all neutral species
    nu_i = nu_in.sum(dim=("neutral"))
    nu_e = nu_en.sum(dim=("neutral"))

    # Ion conductivity (Schunk and Nagy equation 5.112)
    # sigma_ion = (n_ion * e_ion**2) / (mass_ion * ion-netrals_coll_freq) (Schunk and Nagy Eq 5.112)
    sigma_i = (n_i * const.e**2) / (m_i * nu_i)

    # Electron conductivity (Schunk and Nagy equation 5.115)
    # sigma_e = (ne * e**2) / (mass_e * elec_coll_freq) (Schunk and Nagy Eq 5.115)
    sigma_e = (n_e * const.e**2) / (const.m_e * nu_e)

    # Pedersen conductivity (S&N equation 5.119)
    sigma_P = (sigma_i * nu_i**2 / (nu_i**2 + Omg_i**2)).sum(
        dim=("species")
    ) + sigma_e * nu_e**2 / (nu_e**2 + Omg_e**2)

    # Hall conductivity (S&N equation 5.120)
    sigma_H = -(sigma_i * nu_i * Omg_i / (nu_i**2 + Omg_i**2)).sum(
        dim=("species")
    ) + sigma_e * nu_e * Omg_e / (nu_e**2 + Omg_e**2)

    return sigma_P, sigma_H
