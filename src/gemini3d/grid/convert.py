"""
these are approximately duplicates of coord.py.
Need to compare vs. gemini3d/coord.py and merge/remove.

---

transformations from dipole to spherical

"""

from __future__ import annotations
import math

import numpy as np

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

    thetag = pi / 2 - np.radians(glat)
    phig = np.radians(glon % 360)

    theta = np.arccos(
        np.cos(thetag) * np.cos(thetan) + np.sin(thetag) * np.sin(thetan) * np.cos(phig - phin)
    )
    argtmp = (np.cos(thetag) - np.cos(theta) * np.cos(thetan)) / (np.sin(theta) * np.sin(thetan))
    alpha = np.arccos(max(min(argtmp, 1), -1))

    phi = np.empty_like(glon, dtype=float)

    i = ((phin > phig) & ((phin - phig) > pi)) | ((phin < phig) & ((phig - phin) < pi))

    phi[i] = pi - alpha[i]
    i = np.logical_not(i)
    phi[i] = alpha[i] + pi

    return phi, theta


def geomag2geog(phi: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """convert from geomagnetic to geographic"""

    phiwrap = phi % tau

    thetag2p = np.arccos(
        np.cos(theta) * np.cos(thetan) - np.sin(theta) * np.sin(thetan) * np.cos(phiwrap)
    )
    beta = np.arccos(
        (np.cos(theta) - np.cos(thetag2p) * np.cos(thetan)) / (np.sin(thetag2p) * np.sin(thetan))
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
