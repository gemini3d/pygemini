from __future__ import annotations
import math

import numpy as np

pi = math.pi
tau = math.tau


def geomag2geog(thetat: np.ndarray, phit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """geomagnetic to geographic"""

    # FIXME: this is for year 1985, see Schmidt spherical harmonic in MatGemini
    thetan = math.radians(11)
    phin = math.radians(289)

    # enforce phit = [0,2pi]
    phit = phit % tau

    thetag2p = np.arccos(
        np.cos(thetat) * np.cos(thetan) - np.sin(thetat) * np.sin(thetan) * np.cos(phit)
    )

    beta = np.arccos(
        (np.cos(thetat) - np.cos(thetag2p) * np.cos(thetan)) / (np.sin(thetag2p) * np.sin(thetan))
    )
    phig2 = np.empty_like(phit, dtype=float)

    i = phit > pi
    phig2[i] = phin - beta[i]
    i = np.logical_not(i)
    phig2[i] = phin + beta[i]

    i = phig2 < 0
    phig2[i] = phig2[i] + tau

    i = phig2 >= tau
    phig2[i] = phig2[i] - tau

    thetag2 = pi / 2 - thetag2p
    lat = np.degrees(thetag2)
    lon = np.degrees(phig2)

    return lat, lon


def geog2geomag(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """geographic to geomagnetic"""

    # FIXME: this is for year 1985, see Schmidt spherical harmonic in MatGemini
    thetan = math.radians(11)
    phin = math.radians(289)

    thetagp = pi / 2 - np.radians(lat)
    phig = np.radians(lon % 360)

    thetat = np.arccos(
        np.cos(thetagp) * np.cos(thetan) + np.sin(thetagp) * np.sin(thetan) * np.cos(phig - phin)
    )
    argtmp = (np.cos(thetagp) - np.cos(thetat) * np.cos(thetan)) / (np.sin(thetat) * np.sin(thetan))
    alpha = np.arccos(max(min(argtmp, 1), -1))
    phit = np.empty_like(lat, dtype=float)

    i = ((phin > phig) & ((phin - phig) > pi)) | ((phin < phig) & ((phig - phin) < pi))

    phit[i] = pi - alpha[i]
    i = np.logical_not(i)
    phit[i] = alpha[i] + pi

    return thetat, phit[()]


def geog2UEN(alt, glon, glat, thetactr, phictr):
    """
    Converts a set of glon,glat into magnetic up, north, east coordinates.
    thetactr and phictr are the magnetic coordinates of the center of the region of interest.
    They can be computed from geog2geomag.
    """

    # %% UPWARD DISTANCE
    Re = 6370e3
    z = alt

    # Convert to geomganetic coordinates
    theta, phi = geog2geomag(glat, glon)

    # Convert to northward distance in meters
    gamma2 = theta - thetactr  # southward magnetic angular distance
    gamma2 = -gamma2  # convert to northward angular distance
    y = gamma2 * Re

    gamma1 = phi - phictr  # eastward angular distance
    x = Re * np.sin(thetactr) * gamma1

    return z, x, y


def UEN2geog(z, x, y, thetactr, phictr):
    """
    converts magnetic up, north, east coordinates into geographic coordinates.
    """

    # UPWARD DISTANCE
    Re = 6370e3
    alt = z

    # Northward angular distance
    gamma2 = y / Re  # must retain the sign of x3
    theta = thetactr - gamma2  # minus because distance north is against theta's direction

    # Eastward angular distance
    gamma1 = x / Re / np.sin(thetactr)
    # must retain the sign of x2, just use theta of center of grid
    phi = phictr + gamma1

    # Now convert the magnetic to geographic using our simple transformation
    glat, glon = geomag2geog(theta, phi)

    return alt, glon, glat
