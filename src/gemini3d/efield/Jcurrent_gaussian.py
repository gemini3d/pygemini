import xarray
import numpy as np


def Jcurrent_gaussian(E: xarray.Dataset, gridflag: int, flagdip: bool) -> xarray.Dataset:

    if E.mlon.size > 1:
        shapelon = np.exp(-((E.mlon - E.mlonmean) ** 2) / 2 / E.mlonsig ** 2)
    else:
        shapelon = 1

    if E.mlat.size > 1:
        shapelat = np.exp(-((E.mlat - E.mlatmean - 1.5 * E.mlatsig) ** 2) / 2 / E.mlatsig ** 2)
    else:
        shapelat = 1

    for t in E.time[6:]:
        E["flagdirich"].loc[t] = 0
        # could have different boundary types for different times

        k = "Vminx1it" if gridflag == 1 else "Vmaxx1it"
        E[k].loc[t] = E.Jtarg * shapelon * shapelat

    return E
