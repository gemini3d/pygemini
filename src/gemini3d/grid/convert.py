"""
these are approximately duplicates of coord.py.
Need to compare vs. gemini3d/coord.py and merge/remove.

---

transformations from dipole to spherical

"""

from __future__ import annotations
import math
import numpy as np
from numpy import sin, cos

# define module-scope constants
Re = 6370e3
thetan = math.radians(11)
phin = math.radians(289)
pi = math.pi
tau = math.tau


def objfunr(r: float, parms: tuple[float, float]) -> float:
    """
    Objective function for solving for r; parms must contain q,p data (same for all fns. here)
    """

    q = parms[0]
    p = parms[1]

    return q**2 * (r / Re) ** 4 + 1 / p * (r / Re) - 1


def objfunr_derivative(r: float, parms: tuple[float, float]) -> float:
    """
    r obj. fn. derivative for Newton's method
    """

    q = parms[0]
    p = parms[1]

    return 4 / Re * q**2 * (r / Re) ** 3 + 1 / p / Re


def calc_theta(r: float, parms: tuple[float, float]) -> float:
    """
    compute polar angle once radial distance is found
    FIXME: need to check for hemisphere????
    """

    return np.arccos(parms[0] * (r / Re) ** 2)


def geog2geomag(glon, glat) -> tuple:
    """
    convert geographic to geomagnetic coordinates (see GEMINI document for details)

    Parameters
    -----------

    glon: float or ndarray
        geographic longitude in degrees
    glat: float or ndarray
        geographic latitude in degrees

    Results
    -------

    phi: float or ndarray
        geomagnetic longitude in radians
    theta: float or ndarray
        geomagnetic latitude in radians
    """

    thetag = pi / 2 - np.radians(glat)
    phig = np.radians(glon % 360)

    theta = np.arccos(
        np.cos(thetag) * np.cos(thetan)
        + np.sin(thetag) * np.sin(thetan) * np.cos(phig - phin)
    )
    argtmp = (np.cos(thetag) - np.cos(theta) * np.cos(thetan)) / (
        np.sin(theta) * np.sin(thetan)
    )

    alpha = np.arccos(argtmp.clip(min=-1, max=1))

    phi = np.empty_like(glon, dtype=float)
    i = ((phin > phig) & ((phin - phig) > pi)) | ((phin < phig) & ((phig - phin) < pi))
    phi[i] = pi - alpha[i]
    i = np.logical_not(i)
    phi[i] = alpha[i] + pi

    return phi, theta


def geomag2geog(phi, theta) -> tuple:
    """convert from geomagnetic to geographic

    Parameters
    ----------

    phi: float or ndarray
        geomagnetic longitude in radians
    theta: float or ndarray
        geomagnetic latitude in radians

    Results
    -------

    glon: float or ndarray
        geographic longitude in degrees
    glat: float or ndarray
        geographic latitude in degrees
    """

    phiwrap = phi % tau

    thetag2p = np.arccos(
        np.cos(theta) * np.cos(thetan) - np.sin(theta) * np.sin(thetan) * np.cos(phiwrap)
    )
    beta = np.arccos(
        (np.cos(theta) - np.cos(thetag2p) * np.cos(thetan))
        / (np.sin(thetag2p) * np.sin(thetan))
    )

    phig2 = np.empty_like(phi, dtype=float)

    i = phiwrap > pi
    phig2[i] = phin - beta[i]
    i = np.logical_not(i)
    phig2[i] = phin + beta[i]

    phig2 = phig2 % tau
    thetag2 = pi / 2 - thetag2p

    glat = np.degrees(thetag2)
    glon = np.degrees(phig2)

    return glon, glat


# Rotation about z axis with angle alpha
def Rz(alpha):
    R = np.zeros((3, 3))
    R[0, 0] = cos(alpha)
    R[0, 1] = -sin(alpha)
    R[1, 0] = sin(alpha)
    R[1, 1] = cos(alpha)
    R[2, 2] = 1
    return R


# Rotation about y axis by angle alpha
def Ry(alpha):
    R = np.zeros((3, 3))
    R[0, 0] = cos(alpha)
    R[0, 2] = sin(alpha)
    R[1, 1] = 1
    R[2, 0] = -sin(alpha)
    R[2, 2] = cos(alpha)
    return R


# Rotation matrix to go from ECEF Cartesian geographic to ECEF Cartesian
#   geomagnetic coordinates; note the
#   rotation is done with angles -phin and -thetan so the transpose of the
#   "standard" rotation matrices are used
def Rgg2gm():
    return (Ry(thetan)).transpose() @ (Rz(phin)).transpose()


# Rotation matrix to go from geomagnetic to geographic
def Rgm2gg():
    return Rz(phin) @ Ry(thetan)


# Rotate an ECEF geographic vector into ECEF geomagnetic
def rotvec_gg2gm(e):
    assert e.ndim == 4, "for 4D arrays"
    assert e.shape[3] == 3, "last dimension must be length 3"
    S = e.shape[:3]

    [lx1, lx2, lx3, lcomp] = e.shape
    exflat = e[..., 0].ravel(order="F")
    eyflat = e[..., 1].ravel(order="F")
    ezflat = e[..., 2].ravel(order="F")
    emat = np.row_stack((exflat, eyflat, ezflat))

    egg = Rgg2gm() @ emat

    eggshp = np.empty((lx1, lx2, lx3, 3))
    eggshp[..., 0] = np.reshape(egg[0, :], S, order="F")
    eggshp[..., 1] = np.reshape(egg[1, :], S, order="F")
    eggshp[..., 2] = np.reshape(egg[2, :], S, order="F")

    return eggshp


def rotvec_gg2gm_points(e):
    assert e.ndim == 2, "for 2D arrays"
    assert e.shape[1] == 3, "must have 3 columns"

    return (Rgg2gm() @ np.asfortranarray(e.transpose())).transpose()


# Return a set of unit vectors in the geographic directions; components in ECEF
#   Cartesian geomagnetic
def unitvecs_geographic(xg):
    thetagg = pi / 2 - xg["glat"] * pi / 180
    phigg = xg["glon"] * pi / 180
    lx1 = xg["lx"][0]
    lx2 = xg["lx"][1]
    lx3 = xg["lx"][2]
    ergg = np.empty((lx1, lx2, lx3, 3))
    ethetagg = np.empty((lx1, lx2, lx3, 3))
    ephigg = np.empty((lx1, lx2, lx3, 3))

    # unit vectors in ECEF Cartesian geographic
    ergg[:, :, :, 0] = sin(thetagg) * cos(phigg)
    ergg[:, :, :, 1] = sin(thetagg) * sin(phigg)
    ergg[:, :, :, 2] = cos(thetagg)
    ethetagg[:, :, :, 0] = cos(thetagg) * cos(phigg)
    ethetagg[:, :, :, 1] = cos(thetagg) * sin(phigg)
    ethetagg[:, :, :, 2] = -sin(thetagg)
    ephigg[:, :, :, 0] = -sin(phigg)
    ephigg[:, :, :, 1] = cos(phigg)
    ephigg[:, :, :, 2] = np.zeros(thetagg.shape)

    # rotate into geomagnetic components (as used in grid dictionary)
    egalt = rotvec_gg2gm(ergg)
    eglon = rotvec_gg2gm(ephigg)
    eglat = -1 * rotvec_gg2gm(ethetagg)

    return egalt, eglon, eglat


def unitvecs_geographic_points(glat, glon):
    thetagg = pi / 2 - glat * pi / 180
    phigg = glon * pi / 180
    # lx1 = xg["lx"][0]
    # lx2 = xg["lx"][1]
    # lx3 = xg["lx"][2]
    nvecs = len(glat)
    ergg = np.empty((nvecs, 3))
    ethetagg = np.empty((nvecs, 3))
    ephigg = np.empty((nvecs, 3))

    # unit vectors in ECEF Cartesian geographic
    ergg[:, 0] = sin(thetagg) * cos(phigg)
    ergg[:, 1] = sin(thetagg) * sin(phigg)
    ergg[:, 2] = cos(thetagg)
    ethetagg[:, 0] = cos(thetagg) * cos(phigg)
    ethetagg[:, 1] = cos(thetagg) * sin(phigg)
    ethetagg[:, 2] = -sin(thetagg)
    ephigg[:, 0] = -sin(phigg)
    ephigg[:, 1] = cos(phigg)
    ephigg[:, 2] = np.zeros(thetagg.shape)

    # rotate into geomagnetic components (as used in grid dictionary)
    egalt = rotvec_gg2gm_points(ergg)
    eglon = rotvec_gg2gm_points(ephigg)
    eglat = -1 * rotvec_gg2gm_points(ethetagg)

    return egalt, eglon, eglat
