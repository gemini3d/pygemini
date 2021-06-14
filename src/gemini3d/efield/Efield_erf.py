from __future__ import annotations
import logging
import typing as T

import xarray
import numpy as np
from scipy.special import erf


def Efield_erf(
    E: xarray.Dataset,
    xg: dict[str, T.Any],
    lx1: int,
    lx2: int,
    lx3: int,
    gridflag: int,
    flagdip: bool,
) -> xarray.Dataset:
    """
    synthesize a feature
    """

    if E.Etarg > 1:
        logging.warning(f"Etarg units V/m -- is {E['Etarg']} V/m realistic?")

    # NOTE: h2, h3 have ghost cells, so we use lx1 instead of -1 to index
    # pk is a scalar.

    if flagdip:
        if lx3 == 1:
            # meridional
            S = E.Etarg * E.sigx2 * xg["h2"][lx1, lx2 // 2, 0] * np.sqrt(np.pi) / 2
            taper = erf((E.mlat - E.mlatmean) / E.mlatsig).data[:, None]
        elif lx2 > 1 and lx3 > 1:
            # 3-D
            S = E.Etarg * E.sigx2 * xg["h2"][lx1, lx2 // 2, 0] * np.sqrt(np.pi) / 2
            taper = (
                erf((E.mlon - E.mlonmean) / E.mlonsig).data[None, :]
                * erf((E.mlat - E.mlatmean) / E.mlatsig).data[None, :]
            )
        else:
            raise ValueError("zonal ribbon grid is not yet supported")
    else:
        if lx3 == 1:
            # east-west
            S = E.Etarg * E.sigx2 * xg["h2"][lx1, lx2 // 2, 0] * np.sqrt(np.pi) / 2
            taper = erf((E.mlon - E.mlonmean) / E.mlonsig).data[:, None]
        elif lx2 == 1:
            # north-south
            S = E.Etarg * E.sigx3 * xg["h3"][lx1, 0, lx3 // 2] * np.sqrt(np.pi) / 2
            taper = erf((E.mlat - E.mlatmean) / E.mlatsig).data[None, :]
        else:
            # 3D
            S = E.Etarg * E.sigx2 * xg["h2"][lx1, lx2 // 2, 0] * np.sqrt(np.pi) / 2
            taper = (
                erf((E.mlon - E.mlonmean) / E.mlonsig).data[:, None]
                * erf((E.mlat - E.mlatmean) / E.mlatsig).data[None, :]
            )

    assert S.ndim == 0, "S is a scalar"

    for t in E.time:
        E["flagdirich"].loc[t] = 1

        if gridflag == 1:
            E["Vminx1it"].loc[t] = S * taper
        else:
            E["Vmaxx1it"].loc[t] = S * taper

    return E
