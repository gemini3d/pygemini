"""
tilted dipole grid generation function
"""

from __future__ import annotations
import typing as T
import math
import numpy as np
from .newton_method import qp2rtheta
from .convert import geog2geomag, geomag2geog, Re


def tilted_dipole3d(cfg: dict[str, T.Any]) -> dict[str, T.Any]:
    """make tilted dipole grid

    Parameters
    -----------

    cfg: dict
        simulation parameters

    Returns
    -------

    xg: dict
        simulation grid
    """

    # arrange the grid data in a dictionary
    xg = {"lx": np.array([cfg["lq"], cfg["lp"], cfg["lphi"]])}  # aggregate array shape variable

    # mesh size *with* ghost cells added in
    lqg = cfg["lq"] + 4
    lpg = cfg["lp"] + 4
    lphig = cfg["lphi"] + 4
    print("tilted_dipole3d called for mesh size of:  ", cfg["lq"], "x", cfg["lp"], "x", cfg["lphi"])

    # phi,theta coordinates at the "center" of the grid
    phid, thetad = geog2geomag(cfg["glon"], cfg["glat"])

    # find the "corners" of the grid in the source hemisphere
    thetax2max = thetad + math.radians(cfg["dtheta"] / 2)
    thetax2min = thetad - math.radians(cfg["dtheta"] / 2)
    if thetad < np.pi / 2:  # northern hemisphere
        pmax: float = (Re + cfg["altmin"]) / Re / np.sin(thetax2min) ** 2
        # bottom left grid point p
        qtmp = (Re / (Re + cfg["altmin"])) ** 2 * np.cos(
            thetax2min
        )  # %bottom left grid q (also bottom right)
        pmin: float = np.sqrt(
            np.cos(thetax2max) / np.sin(thetax2max) ** 4 / qtmp
        )  # bottom right grid p
    else:
        pmax = (Re + cfg["altmin"]) / Re / np.sin(thetax2max) ** 2
        qtmp = (Re / (Re + cfg["altmin"])) ** 2 * np.cos(thetax2max)
        pmin = np.sqrt(np.cos(thetax2max) / np.sin(thetax2min) ** 4 / qtmp)

    # set the L-shell grid, sans ghost cells
    p = np.zeros(lpg)
    p[2:-2] = np.linspace(pmin, pmax, cfg["lp"])  # sans ghost cells

    # find the max zenith angle (theta) for the grid, need to detect grid type and henmisphere
    if cfg["gridflag"] == 0:  # open dipole grid
        if thetad < np.pi / 2:  # northern hemisphere
            thetamax = thetax2min + np.pi / 100
        else:  # southern hemisphere
            thetamax = thetax2max - np.pi / 100
    else:  # closed dipole grid, reflect theta about equator
        if thetad < np.pi / 2:  # northern
            thetamax = np.pi - thetax2min
        else:  # southern
            thetamax = np.pi - thetax2max

    # find the min/max q values for the grid across both hemispheres
    if thetad < np.pi / 2:
        rmin = p[-3] * Re * np.sin(thetax2min) ** 2  # last field line contains min/max r/q vals.
        rmax = p[-3] * Re * np.sin(thetamax) ** 2
        qmin = np.cos(thetax2min) * Re ** 2 / rmin ** 2
        qmax = np.cos(thetamax) * Re ** 2 / rmax ** 2
    else:
        rmin = p[-3] * Re * np.sin(thetamax) ** 2
        rmax = p[-3] * Re * np.sin(thetax2max) ** 2
        qmin = np.cos(thetamax) * Re ** 2 / rmin ** 2
        qmax = np.cos(thetax2max) * Re ** 2 / rmax ** 2

    # define q grid sans ghost cells
    if qmax < qmin:  # unclear whether this checking is necessary so leave for now
        qtmp = qmax
        qmax = qmin
        qmin = qtmp
    q = np.zeros(lqg)
    q[2:-2] = np.linspace(qmin, qmax, cfg["lq"])

    # define phi grid sans ghost cells
    phimin = phid - np.deg2rad(cfg["dphi"] / 2)
    phimax = phid + np.deg2rad(cfg["dphi"] / 2)
    phi = np.zeros(lphig)
    if cfg["lphi"] != 1:
        phi[2:-2] = np.linspace(phimin, phimax, cfg["lphi"])
    else:
        phi[2:-2] = phid

    # assuming uniform spacing in ghost region, add ghost cells
    pstride = p[3] - p[2]
    p[0] = p[2] - 2 * pstride
    p[1] = p[2] - pstride
    p[-2] = p[-3] + pstride
    p[-1] = p[-3] + 2 * pstride
    qstride = q[3] - q[2]
    q[0] = q[2] - 2 * qstride
    q[1] = q[2] - qstride
    q[-2] = q[-3] + qstride
    q[-1] = q[-3] + 2 * qstride
    if cfg["lphi"] != 1:
        phistride = phi[3] - phi[2]
    else:
        phistride = 0.1  # just use some random value if this is a 2D dipole mesh
    phi[0] = phi[2] - 2 * phistride
    phi[1] = phi[2] - phistride
    phi[-2] = phi[-3] + phistride
    phi[-1] = phi[-3] + 2 * phistride

    # %% allocate meridional slice, including ghost cells - this later gets extended into 3D
    r = np.zeros((lqg, lpg))
    theta = np.zeros((lqg, lpg))
    # qtol = 1e-9  # tolerance for declaring "equator"
    print(" tilted_dipole3d:  converting grid centers to r,theta...")

    for iq in range(lqg):
        for ip in range(lpg):
            r[iq, ip], theta[iq, ip] = qp2rtheta(q[iq], p[ip])

    r = np.broadcast_to(r[:, :, None], (*r.shape, lphig))  # just tile for longitude to save time
    theta = np.broadcast_to(theta[:, :, None], (*theta.shape, lphig))
    phispher = np.broadcast_to(phi[None, None, :], (lqg, lpg, phi.size))

    # %% define cell interfaces and convert coordinates
    print(" tilted_dipole3d:  converting q interface values to r,theta...")
    qi = 1 / 2 * (q[1:-2] + q[2:-1])
    rqi = np.zeros((cfg["lq"] + 1, cfg["lp"]))
    thetaqi = np.zeros((cfg["lq"] + 1, cfg["lp"]))
    for iq in range(cfg["lq"] + 1):
        for ip in range(cfg["lp"]):
            rqi[iq, ip], thetaqi[iq, ip] = qp2rtheta(
                qi[iq], p[ip + 2]
            )  # shift by 2 to exclude ghost
    rqi = np.broadcast_to(rqi[:, :, None], (*rqi.shape, cfg["lphi"]))
    thetaqi = np.broadcast_to(thetaqi[:, :, None], (*thetaqi.shape, cfg["lphi"]))

    print(" tilted_dipole3d:  converting p interface values to r,theta...")
    pi = 1 / 2 * (p[1:-2] + p[2:-1])
    rpi = np.zeros((cfg["lq"], cfg["lp"] + 1))
    thetapi = np.zeros((cfg["lq"], cfg["lp"] + 1))
    for iq in range(cfg["lq"]):
        for ip in range(cfg["lp"] + 1):
            rpi[iq, ip], thetapi[iq, ip] = qp2rtheta(
                q[iq + 2], pi[ip]
            )  # shift non interface index by two to exclude ghost
    rpi = np.broadcast_to(rpi[:, :, None], (*rpi.shape, cfg["lphi"]))
    thetapi = np.broadcast_to(thetapi[:, :, None], (*thetapi.shape, cfg["lphi"]))

    # phii = 1 / 2 * (phi[1:-2] + phi[2:-1])

    # metric factors at cell centers and interfaces
    print(" tilted_dipole3d:  calculating metric ceoffs...")
    denom = np.sqrt(1 + 3 * np.cos(theta) ** 2)  # ghost cells need for these
    xg["h1"] = r ** 3 / Re ** 2 / denom
    xg["h2"] = Re * np.sin(theta) ** 3 / denom
    xg["h3"] = r * np.sin(theta)

    xg["h1x3i"] = np.concatenate(
        (
            xg["h1"][2:-2, 2:-2, 2:-2],
            xg["h1"][2:-2, 2:-2, -1].reshape([cfg["lq"], cfg["lp"], 1]),
        ),
        axis=2,
    )
    xg["h2x3i"] = np.concatenate(
        (
            xg["h2"][2:-2, 2:-2, 2:-2],
            xg["h2"][2:-2, 2:-2, -1].reshape([cfg["lq"], cfg["lp"], 1]),
        ),
        axis=2,
    )
    xg["h3x3i"] = np.concatenate(
        (
            xg["h3"][2:-2, 2:-2, 2:-2],
            xg["h3"][2:-2, 2:-2, -1].reshape([cfg["lq"], cfg["lp"], 1]),
        ),
        axis=2,
    )

    denomtmp = np.sqrt(1 + 3 * np.cos(thetaqi) ** 2)
    xg["h1x1i"] = rqi ** 3 / Re ** 2 / denomtmp
    xg["h2x1i"] = Re * np.sin(thetaqi) ** 3 / denomtmp
    xg["h3x1i"] = rqi * np.sin(thetaqi)

    denomtmp = np.sqrt(1 + 3 * np.cos(thetapi) ** 2)
    xg["h1x2i"] = rpi ** 3 / Re ** 2 / denomtmp
    xg["h2x2i"] = Re * np.sin(thetapi) ** 3 / denomtmp
    xg["h3x2i"] = rpi * np.sin(thetapi)

    # spherical unit vectors (expressed in a Cartesian basis), these should not have ghost cells
    print(" tilted_dipole3d:  calculating spherical unit vectors")
    xg["er"] = np.zeros((cfg["lq"], cfg["lp"], cfg["lphi"], 3))
    xg["etheta"] = np.zeros((cfg["lq"], cfg["lp"], cfg["lphi"], 3))
    xg["ephi"] = np.zeros((cfg["lq"], cfg["lp"], cfg["lphi"], 3))
    xg["er"][:, :, :, 0] = np.sin(theta[2:-2, 2:-2, 2:-2]) * np.cos(phispher[2:-2, 2:-2, 2:-2])
    xg["er"][:, :, :, 1] = np.sin(theta[2:-2, 2:-2, 2:-2]) * np.sin(phispher[2:-2, 2:-2, 2:-2])
    xg["er"][:, :, :, 2] = np.cos(theta[2:-2, 2:-2, 2:-2])
    xg["etheta"][:, :, :, 0] = np.cos(theta[2:-2, 2:-2, 2:-2]) * np.cos(phispher[2:-2, 2:-2, 2:-2])
    xg["etheta"][:, :, :, 1] = np.cos(theta[2:-2, 2:-2, 2:-2]) * np.sin(phispher[2:-2, 2:-2, 2:-2])
    xg["etheta"][:, :, :, 2] = -np.sin(theta[2:-2, 2:-2, 2:-2])
    xg["ephi"][:, :, :, 0] = -np.sin(phispher[2:-2, 2:-2, 2:-2])
    xg["ephi"][:, :, :, 1] = np.cos(phispher[2:-2, 2:-2, 2:-2])
    xg["ephi"][:, :, :, 2] = np.zeros((cfg["lq"], cfg["lp"], cfg["lphi"]))

    # now do the dipole unit vectors
    print(" tilted_dipole3d:  calculating dipole unit vectors")
    xg["e1"] = np.zeros((cfg["lq"], cfg["lp"], cfg["lphi"], 3))
    xg["e2"] = np.zeros((cfg["lq"], cfg["lp"], cfg["lphi"], 3))
    xg["e3"] = np.zeros((cfg["lq"], cfg["lp"], cfg["lphi"], 3))
    xg["e1"][:, :, :, 0] = (
        -3
        * np.cos(theta[2:-2, 2:-2, 2:-2])
        * np.sin(theta[2:-2, 2:-2, 2:-2])
        * np.cos(phispher[2:-2, 2:-2, 2:-2])
        / denom[2:-2, 2:-2, 2:-2]
    )
    xg["e1"][:, :, :, 1] = (
        -3
        * np.cos(theta[2:-2, 2:-2, 2:-2])
        * np.sin(theta[2:-2, 2:-2, 2:-2])
        * np.sin(phispher[2:-2, 2:-2, 2:-2])
        / denom[2:-2, 2:-2, 2:-2]
    )
    xg["e1"][:, :, :, 2] = (1 - 3 * np.cos(theta[2:-2, 2:-2, 2:-2]) ** 2) / denom[2:-2, 2:-2, 2:-2]
    xg["e2"][:, :, :, 0] = (
        np.cos(phispher[2:-2, 2:-2, 2:-2])
        * (1 - 3 * np.cos(theta[2:-2, 2:-2, 2:-2]) ** 2)
        / denom[2:-2, 2:-2, 2:-2]
    )
    xg["e2"][:, :, :, 1] = (
        np.sin(phispher[2:-2, 2:-2, 2:-2])
        * (1 - 3 * np.cos(theta[2:-2, 2:-2, 2:-2]) ** 2)
        / denom[2:-2, 2:-2, 2:-2]
    )
    xg["e2"][:, :, :, 2] = (
        3
        * np.sin(theta[2:-2, 2:-2, 2:-2])
        * np.cos(theta[2:-2, 2:-2, 2:-2])
        / denom[2:-2, 2:-2, 2:-2]
    )
    xg["e3"] = xg["ephi"]  # same as in spherical

    # find inclination angle for each field line
    print(" tilted_dipole3d:  calculating average inclination angle for each field line...")
    proj = np.sum(xg["er"] * xg["e1"], axis=3)
    Imat = np.arccos(proj)
    if cfg["gridflag"] == 0:  # open dipole
        xg["I"] = Imat.mean(axis=0)
    else:  # closed dipole
        Imathalf = Imat[:cfg["lq"] // 2, :, :]
        xg["I"] = Imathalf.mean(axis=0)
    xg["I"] = 90 - np.rad2deg(np.minimum(xg["I"], np.pi - xg["I"]))
    # ignore parallel vs. anti-parallel

    # compute gravitational field components, exclude ghost cells
    print(" tilted_dipole3d:  calculating gravitational field over grid...")
    G = 6.67428e-11
    Me = 5.9722e24
    g = G * Me / r[2:-2, 2:-2, 2:-2] ** 2
    proj = np.sum(-xg["er"] * xg["e1"], axis=3)
    xg["gx1"] = g * proj
    proj = np.sum(-xg["er"] * xg["e2"], axis=3)
    xg["gx2"] = g * proj
    xg["gx3"] = np.zeros(xg["gx1"].shape)

    # compute magnetic field strength
    print(" tilted_dipole3d:  calculating magnetic field strength over grid...")
    xg["Bmag"] = (
        (4 * np.pi * 1e-7)
        * 7.94e22
        / 4
        / np.pi
        / (r[2:-2, 2:-2, 2:-2] ** 3)
        * np.sqrt(3 * (np.cos(theta[2:-2, 2:-2, 2:-2])) ** 2 + 1)
    )

    # compute Cartesian coordinates
    xg["z"] = r[2:-2, 2:-2, 2:-2] * np.cos(theta[2:-2, 2:-2, 2:-2])
    xg["x"] = (
        r[2:-2, 2:-2, 2:-2] * np.sin(theta[2:-2, 2:-2, 2:-2]) * np.cos(phispher[2:-2, 2:-2, 2:-2])
    )
    xg["y"] = (
        r[2:-2, 2:-2, 2:-2] * np.sin(theta[2:-2, 2:-2, 2:-2]) * np.sin(phispher[2:-2, 2:-2, 2:-2])
    )

    # determine grid cells that are "null" - i.e. not included in the computations
    inull = np.where(r[2:-2, 2:-2, 2:-2] < Re + 80e3)
    # note addressing of r to produce array sans ghost cells

    xg["nullpts"] = np.zeros((cfg["lq"], cfg["lp"], cfg["lphi"]))
    xg["nullpts"][inull] = 1
    # may need to convert inull to linear index?

    # compute geographic coordinates for the entire grid
    xg["alt"] = r[2:-2, 2:-2, 2:-2] - Re
    [xg["glon"], xg["glat"]] = geomag2geog(phispher[2:-2, 2:-2, 2:-2], theta[2:-2, 2:-2, 2:-2])

    # at this point we are done with the works arrays to put them in the structure and deallocate
    # assign spherical variables to dictionary and clear out work array to save memory
    xg["r"] = r[2:-2, 2:-2, 2:-2]
    xg["theta"] = theta[2:-2, 2:-2, 2:-2]
    xg["phi"] = phispher[2:-2, 2:-2, 2:-2]

    # assign primary coordinates to dictionary, clear out temps
    xg["x1"] = q
    xg["x2"] = p
    xg["x3"] = phi

    # compute and store interface locations; these recomputed in fortran
    xg["x1i"] = 1 / 2 * (xg["x1"][1:-2] + xg["x1"][2:-1])
    xg["x2i"] = 1 / 2 * (xg["x2"][1:-2] + xg["x2"][2:-1])
    xg["x3i"] = 1 / 2 * (xg["x3"][1:-2] + xg["x3"][2:-1])

    # compute and store backward diffs (other diffs recomputed as needed in fortran)
    xg["dx1b"] = xg["x1"][1:] - xg["x1"][:-1]
    xg["dx2b"] = xg["x2"][1:] - xg["x2"][:-1]
    xg["dx3b"] = xg["x3"][1:] - xg["x3"][:-1]

    # compute and store centered diffs
    xg["dx1h"] = xg["x1i"][1:] - xg["x1i"][:-1]
    xg["dx2h"] = xg["x2i"][1:] - xg["x2i"][:-1]
    xg["dx3h"] = xg["x3i"][1:] - xg["x3i"][:-1]

    return xg
