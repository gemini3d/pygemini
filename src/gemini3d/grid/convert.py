"""
these are approximately duplicates of coord.py.
Need to compare vs. gemini3d/coord.py and merge/remove.

---

transformations from dipole to spherical

"""

from __future__ import annotations
import numpy as np
import math

# define module-scope constants
Re = 6370e3
thetan = np.deg2rad(11)
phin = np.deg2rad(289)
pi = math.pi
tau = math.tau


def objfunr(r: float, parms: tuple[float, float]) -> float:
    """
    Objective function for solving for r; parms must contain q,p data (same for all fns. here)
    """

    q = parms[0]
    p = parms[1]

    return q ** 2 * (r / Re) ** 4 + 1 / p * (r / Re) - 1


def objfunr_derivative(r: float, parms: tuple[float, float]) -> float:
    """
    r obj. fn. derivative for Newton's method
    """

    q = parms[0]
    p = parms[1]

    return 4 / Re * q ** 2 * (r / Re) ** 3 + 1 / p / Re


def calc_theta(r: float, parms: tuple[float, float]) -> float:
    """
    compute polar angle once radial distance is found
    FIXME: need to check for hemisphere????
    """

    return np.arccos(parms[0] * (r / Re) ** 2)


def geog2geomag(glon: np.ndarray, glat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    convert geographic to geomagnetic coordinates (see GEMINI document for details)
    """

    glon = np.atleast_1d(glon)
    glat = np.atleast_1d(glat)

    glonwrap = np.mod(glon, 360)
    thetag = pi / 2 - np.deg2rad(glat)
    phig = np.deg2rad(glonwrap)

    theta = np.arccos(
        np.cos(thetag) * np.cos(thetan) + np.sin(thetag) * np.sin(thetan) * np.cos(phig - phin)
    )
    argtmp = (np.cos(thetag) - np.cos(theta) * np.cos(thetan)) / (np.sin(theta) * np.sin(thetan))
    alpha = np.arccos(np.maximum(np.minimum(argtmp, 1), -1))

    phi = np.zeros(glon.shape)

    condition = np.logical_or(
        (np.logical_and(phin > phig, phin - phig > pi)),
        (np.logical_and(phin < phig, phig - phin < pi)),
    )
    inds = np.where(condition)
    phi[inds] = pi - alpha[inds]
    inds = np.where(np.logical_not(condition))
    phi[inds] = alpha[inds] + pi

    return phi.squeeze()[()], theta.squeeze()[()]


def geomag2geog(phi: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """convert from geomagnetic to geographic"""

    theta = np.atleast_1d(theta)
    phi = np.atleast_1d(phi)

    phiwrap = np.mod(phi, tau)

    thetag2p = np.arccos(
        np.cos(theta) * np.cos(thetan) - np.sin(theta) * np.sin(thetan) * np.cos(phiwrap)
    )
    beta = np.arccos(
        (np.cos(theta) - np.cos(thetag2p) * np.cos(thetan)) / (np.sin(thetag2p) * np.sin(thetan))
    )

    phig2 = np.zeros(phi.shape)

    inds = np.where(phiwrap > pi)
    phig2[inds] = phin - beta[inds]
    inds = np.where(phiwrap <= pi)
    phig2[inds] = phin + beta[inds]

    phig2 = np.mod(phig2, tau)
    thetag2 = pi / 2 - thetag2p

    glat = np.rad2deg(thetag2)
    glon = np.rad2deg(phig2)

    return glon.squeeze()[()], glat.squeeze()[()]
