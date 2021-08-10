"""
Various transformations needed to grid model output so it can be easily plotted

@author: zettergm
"""

from __future__ import annotations
import typing as T

import numpy as np
from numpy import pi
import xarray
import scipy.interpolate

from .convert import Re, geog2geomag


def model2magcoords(
    xg: dict[str, T.Any],
    parm: xarray.DataArray,
    lalt: int,
    llon: int,
    llat: int,
    altlims: tuple[float, float] = None,
    mlonlims: tuple[float, float] = None,
    mlatlims: tuple[float, float] = None,
):
    """
    Grid the scalar GEMINI output data in parm onto a regular *geomagnetic* coordinates
    grid.  By default create a linearly spaced output grid based on
    user-provided limits (or grid limits).  Needs to be updated to deal with
    2D input grids; can interpolate from 3D grids to 2D slices.
    """

    # convenience variables
    mlon = np.degrees(xg["phi"])
    mlat = 90 - np.degrees(xg["theta"])
    alt = xg["alt"]
    lx1 = xg["lx"][0]
    lx2 = xg["lx"][1]
    lx3 = xg["lx"][2]
    inds1 = range(2, lx1 + 2)
    inds2 = range(2, lx2 + 2)
    inds3 = range(2, lx3 + 2)
    x1 = xg["x1"][inds1]
    x2 = xg["x2"][inds2]
    x3 = xg["x3"][inds3]

    # set some defaults if not provided by user
    if altlims is None:
        altlims = (alt.min() + 0.0001, alt.max() - 0.0001)
    if mlonlims is None:
        mlonlims = (mlon.min() + 0.0001, mlon.max() - 0.0001)
    if mlatlims is None:
        mlatlims = (mlat.min() + 0.0001, mlat.max() - 0.0001)

    # define uniform grid in magnetic coords.
    alti = np.linspace(altlims[0], altlims[1], lalt)
    mloni = np.linspace(mlonlims[0], mlonlims[1], llon)
    mlati = np.linspace(mlatlims[0], mlatlims[1], llat)
    ALTi, MLONi, MLATi = np.meshgrid(alti, mloni, mlati, indexing="ij")

    # identify the type of grid that we are using
    minh1 = xg["h1"].min()
    maxh1 = xg["h1"].max()
    if abs(minh1 - 1) > 1e-4 or abs(maxh1 - 1) > 1e-4:  # curvilinear, dipole
        flagcurv = 1
    else:  # cartesian
        flagcurv = 0
        # elif others possible...

    # Compute the coordinates of the intended interpolation grid IN THE MODEL SYSTEM/BASIS.
    # There needs to be a separate transformation here for each coordinate system that the model
    # may use...
    if flagcurv == 1:
        x1i, x2i, x3i = geomag2dipole(ALTi, MLONi, MLATi)
    elif flagcurv == 0:
        x1i, x2i, x3i = geomag2UENgeomag(ALTi, MLONi, MLATi)
    else:
        raise ValueError("Unsupported grid type...")

    # Execute plaid interpolation
    # [X1,X2,X3]=np.meshgrid(x1,x2,x3,indexing="ij")
    if parm.ndim == 3:
        # xi=np.zeros((x1i.size,3))
        xi = np.array((x1i.ravel(), x2i.ravel(), x3i.ravel())).transpose()
        parmi = scipy.interpolate.interpn(
            points=(x1, x2, x3),
            values=parm.data,
            xi=xi,
            method="linear",
            bounds_error=False,
            fill_value=np.NaN,
        )
    elif parm.ndim == 2:
        coord1 = x1
        coord1i = x1i
        if parm.shape[1] == 1:
            coord2 = x3
            coord2i = x3i
        else:
            coord2 = x2
            coord2i = x2i
        # fi=scipy.interpolate.interp2d(coord1,coord2, parm.data, kind="linear", \
        #                              bounds_error=False, fill_value=np.NaN)
        # parmi=fi(coord1i.ravel(),coord2i.ravel())
        xi = np.array((coord1i.ravel(), coord2i.ravel())).transpose()
        parmi = scipy.interpolate.interpn(
            points=(coord1, coord2),
            values=parm.data,
            xi=xi,
            method="linear",
            bounds_error=False,
            fill_value=np.NaN,
        )
    else:
        raise ValueError("Can only grid 2D or 3D data, check array dims...")

    parmi = parmi.reshape(lalt, llon, llat)

    return alti, mloni, mlati, parmi


def model2geogcoords(
    xg: dict[str, T.Any],
    parm: xarray.DataArray,
    lalt: int,
    llon: int,
    llat: int,
    altlims: tuple[float, float] = None,
    glonlims: tuple[float, float] = None,
    glatlims: tuple[float, float] = None,
):
    """
    Grid the scalar GEMINI output data in parm onto a regular *geographic* coordinates
    grid.  By default create a linearly spaced output grid based on
    user-provided limits (or grid limits).  Needs to be updated to deal with
    2D input grids; can interpolate from 3D grids to 2D slices.
    """

    # convenience variables
    glat = xg["glat"]
    glon = xg["glon"]
    alt = xg["alt"]
    lx1 = xg["lx"][0]
    lx2 = xg["lx"][1]
    lx3 = xg["lx"][2]
    inds1 = range(2, lx1 + 2)
    inds2 = range(2, lx2 + 2)
    inds3 = range(2, lx3 + 2)
    x1 = xg["x1"][inds1]
    x2 = xg["x2"][inds2]
    x3 = xg["x3"][inds3]

    # set some defaults if not provided by user
    if altlims is None:
        altlims = (alt.min() + 0.0001, alt.max() - 0.0001)
    if glonlims is None:
        glonlims = (glon.min() + 0.0001, glon.max() - 0.0001)
    if glatlims is None:
        glatlims = (glat.min() + 0.0001, glat.max() - 0.0001)

    # define uniform grid in magnetic coords.
    alti = np.linspace(altlims[0], altlims[1], lalt)
    gloni = np.linspace(glonlims[0], glonlims[1], llon)
    glati = np.linspace(glatlims[0], glatlims[1], llat)
    ALTi, GLONi, GLATi = np.meshgrid(alti, gloni, glati, indexing="ij")

    # identify the type of grid that we are using
    minh1 = xg["h1"].min()
    maxh1 = xg["h1"].max()
    if abs(minh1 - 1) > 1e-4 or abs(maxh1 - 1) > 1e-4:  # curvilinear, dipole
        flagcurv = 1
    else:  # cartesian
        flagcurv = 0
        # elif others possible...

    # Compute the coordinates of the intended interpolation grid IN THE MODEL SYSTEM/BASIS.
    # There needs to be a separate transformation here for each coordinate system that the model
    # may use...
    if flagcurv == 1:
        x1i, x2i, x3i = geog2dipole(ALTi, GLONi, GLATi)
    elif flagcurv == 0:
        x1i, x2i, x3i = geog2UENgeog(ALTi, GLONi, GLATi)
    else:
        raise ValueError("Unsupported grid type...")

    # Execute plaid interpolation
    # [X1,X2,X3]=np.meshgrid(x1,x2,x3,indexing="ij")
    if parm.ndim == 3:
        # xi=np.zeros((x1i.size,3))
        xi = np.array((x1i.ravel(), x2i.ravel(), x3i.ravel())).transpose()
        parmi = scipy.interpolate.interpn(
            points=(x1, x2, x3),
            values=parm.data,
            xi=xi,
            method="linear",
            bounds_error=False,
            fill_value=np.NaN,
        )
    elif parm.ndim == 2:
        coord1 = x1
        coord1i = x1i
        if parm.shape[1] == 1:
            coord2 = x3
            coord2i = x3i
        else:
            coord2 = x2
            coord2i = x2i
        # fi=scipy.interpolate.interp2d(coord1,coord2, parm.data, kind="linear", \
        #                              bounds_error=False, fill_value=np.NaN)
        # parmi=fi(coord1i.ravel(),coord2i.ravel())
        xi = np.array((coord1i.ravel(), coord2i.ravel())).transpose()
        parmi = scipy.interpolate.interpn(
            points=(coord1, coord2),
            values=parm.data,
            xi=xi,
            method="linear",
            bounds_error=False,
            fill_value=np.NaN,
        )
    else:
        raise ValueError("Can only grid 2D or 3D data, check array dims...")

    parmi = parmi.reshape(lalt, llon, llat)

    return alti, gloni, glati, parmi


def geomag2dipole(
    alt: np.ndarray, mlon: np.ndarray, mlat: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert geomagnetic coordinates into dipole"""

    theta = pi / 2 - np.radians(mlat)
    phi = np.radians(mlon)
    r = alt + Re
    q = ((Re / r) ** 2) * np.cos(theta)
    p = r / (Re * np.sin(theta) ** 2)

    return q, p, phi


def geog2dipole(
    alt: np.ndarray, glon: np.ndarray, glat: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert geographic coordinates into dipole"""

    phi, theta = geog2geomag(glon, glat)
    mlat = 90 - np.degrees(theta)
    mlon = np.degrees(phi)
    q, p, phi = geomag2dipole(alt, mlon, mlat)

    return q, p, phi


def geomag2UENgeomag(alt, mlon, mlat) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert geomagnetic to UEN geomagnetic coords."""

    theta = pi / 2 - np.radians(mlat)
    phi = np.radians(mlon)
    meantheta = theta.mean()
    meanphi = phi.mean()
    yUEN = -1 * Re * (theta - meantheta)  # north dist. runs backward from zenith angle
    xUEN = Re * np.sin(meantheta) * (phi - meanphi)  # some warping done here (using meantheta)
    zUEN = alt

    return zUEN, xUEN, yUEN


def geog2UENgeog(
    alt, glon, glat, ref_lat: float = None, ref_lon: float = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert geographic to UEN geographic coords."""

    theta = pi / 2 - np.radians(
        glat
    )  # this is the zenith angle referenced from Earth's spin axis, i.e. the geographic (as opposed to magnetic) pole
    phi = np.radians(glon)

    if ref_lon is None:
        refphi = np.mean(phi)
    else:
        refphi = np.radians(ref_lon)

    if ref_lat is None:
        reftheta = np.mean(theta)
    else:
        reftheta = pi / 2 - np.radians(ref_lat)

    yUEN = -1 * Re * (theta - reftheta)  # north dist. runs backward from zenith angle
    xUEN = Re * np.sin(reftheta) * (phi - refphi)  # some warping done here (using meantheta)
    zUEN = alt

    return zUEN, xUEN, yUEN
