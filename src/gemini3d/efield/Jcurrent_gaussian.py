import xarray
import numpy as np


def Jcurrent_gaussian(E: xarray.Dataset, gridflag: int) -> xarray.Dataset:

    S = (
        E["Jtarg"]
        * np.exp(-((E.mlon - E.mlonmean) ** 2) / 2 / E.mlonsig ** 2)
        * np.exp(-((E.mlat - E.mlatmean - 1.5 * E.mlatsig) ** 2) / 2 / E.mlatsig ** 2)
    )

    for t in E.time[6:]:
        E["flagdirich"].loc[t] = 0
        # could have different boundary types for different times
        J = S - E.Jtarg * np.exp(-((E.mlon - E.mlonmean) ** 2) / 2 / E.mlonsig ** 2) * np.exp(
            -((E.mlat - E.mlatmean + 1.5 * E.mlatsig) ** 2) / 2 / E.mlatsig ** 2
        )

        if gridflag == 1:
            E["Vminx1it"].loc[t] = J
        else:
            E["Vmaxx1it"].loc[t] = J

    return E
