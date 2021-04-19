import numpy as np
import xarray


def gaussian2d(pg: xarray.Dataset, Qpeak: float, Qbackground: float) -> np.ndarray:

    mlon_mean = pg.mlon.mean().item()
    mlat_mean = pg.mlat.mean().item()

    if "mlon_sigma" in pg.attrs and "mlat_sigma" in pg.attrs:
        Q = (
            Qpeak
            * np.exp(-((pg.mlon.data[:, None] - mlon_mean) ** 2) / (2 * pg.mlon_sigma ** 2))
            * np.exp(-((pg.mlat.data[None, :] - mlat_mean) ** 2) / (2 * pg.mlat_sigma ** 2))
        )
    elif "mlon_sigma" in pg.attrs:
        Q = Qpeak * np.exp(-((pg.mlon.data[:, None] - mlon_mean) ** 2) / (2 * pg.mlon_sigma ** 2))
    elif "mlat_sigma" in pg.attrs:
        Q = Qpeak * np.exp(-((pg.mlat.data[None, :] - mlat_mean) ** 2) / (2 * pg.mlat_sigma ** 2))
    else:
        raise LookupError("precipation must be defined in latitude, longitude or both")

    Q[Q < Qbackground] = Qbackground

    return Q
