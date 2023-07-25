"""
cartesian grid
"""

from __future__ import annotations
import logging
import typing as T

import numpy as np

from .. import read
from ..coord import geog2geomag, geomag2geog
from .uniform import altitude_grid, grid1d


def cart3d(p: dict[str, T.Any]) -> dict[str, T.Any]:
    """make cartesian grid

    Parameters
    -----------

    p: dict
        simulation parameters

    Returns
    -------

    xg: dict
        simulation grid
    """

    # %%create altitude grid
    # original Matlab params
    # p.alt_min = 80e3;
    # p.alt_max = 1000e3;
    # p.alt_scale = [10e3, 8e3, 500e3, 150e3];

    if {"alt_min", "alt_max", "alt_scale", "Bincl"} <= p.keys():
        # https://docs.python.org/3/library/stdtypes.html#frozenset.issubset
        z = altitude_grid(p["alt_min"], p["alt_max"], p["Bincl"], p["alt_scale"])
    elif "eq_dir" in p and p["eq_dir"].is_file():
        logging.info(f"reusing grid from {p['eq_dir']}")
        xeq = read.grid(p["eq_dir"])
        z = xeq["x1"]
        del xeq
    elif {"alt_min", "alt_max", "lzp"} <= p.keys():
        logging.info("make uniform altitude grid")
        z = np.linspace(p["alt_min"], p["alt_max"], p["lzp"])
        dz = z[1] - z[0]
        z = np.concatenate((z[0] - 2 * dz, z[0] - dz, z, z[-1] + dz, z[-1] + 2 * dz))
    else:
        raise ValueError("must specify altitude grid parameters or grid file to reuse")

    # %% TRANSVERSE GRID (BASED ON SIZE OF CURRENT REGION SPECIFIED ABOVE)
    # EAST
    if "x2parms" in p:
        x = grid1d(p["xdist"], p["lxp"], p["x2parms"])
    else:
        x = grid1d(p["xdist"], p["lxp"])

    # NORTH
    if "x3parms" in p:
        y = grid1d(p["ydist"], p["lyp"], p["x3parms"])
    else:
        y = grid1d(p["ydist"], p["lyp"])

    # %% COMPUTE CELL WALL LOCATIONS
    lx2 = x.size
    xi = np.empty(lx2 + 1)
    xi[1:-1] = 1 / 2 * (x[1:] + x[:-1])
    xi[0] = x[0] - 1 / 2 * (x[1] - x[0])
    xi[-1] = x[-1] + 1 / 2 * (x[-1] - x[-2])

    lx3 = y.size
    yi = np.empty(lx3 + 1)
    yi[1:-1] = 1 / 2 * (y[1:] + y[:-1])
    yi[0] = y[0] - 1 / 2 * (y[1] - y[0])
    yi[-1] = y[-1] + 1 / 2 * (y[-1] - y[-2])

    lx1 = z.size
    zi = np.empty(lx1 + 1)
    zi[1:-1] = 1 / 2 * (z[1:] + z[:-1])
    zi[0] = z[0] - 1 / 2 * (z[1] - z[0])
    zi[-1] = z[-1] + 1 / 2 * (z[-1] - z[-2])

    # %% GRAVITATIONAL FIELD COMPONENTS IN DIPOLE SYSTEM
    Re = 6370e3
    G = 6.67428e-11
    Me = 5.9722e24
    r = z + Re
    g = G * Me / r**2
    gz = np.broadcast_to(-g[:, None, None], (g.size, lx2, lx3))
    assert gz.shape == (lx1, lx2, lx3)

    print("Grid size:  ", lx1, "x", lx2, "x", lx3)

    # DISTANCE EW AND NS (FROM ENU (or UEN in our case - cyclic permuted) COORD. SYSTEM)
    # #NEED TO BE CONVERTED TO DIPOLE SPHERICAL AND THEN
    # GLAT/GLONG - BASICALLY HERE WE ARE MAPPING THE CARTESIAN GRID ONTO THE
    # SURFACE OF A SPHERE THEN CONVERTING TO GEOGRAPHIC.
    # get the magnetic coordinates of the grid center, based on user input
    thetactr, phictr = geog2geomag(p["glat"], p["glon"])

    # %% Center of earth distance
    r = Re + z
    r = np.broadcast_to(r[:, None, None], (r.size, lx2, lx3))
    assert r.shape == (lx1, lx2, lx3)

    # %% Northward angular distance
    gamma2 = y / Re
    # must retain the sign of x3
    theta = thetactr - gamma2
    # minus because distance north is against theta's direction
    theta = np.broadcast_to(theta[None, None, :], (lx1, lx2, theta.size))
    assert theta.shape == (lx1, lx2, lx3)

    # %% Eastward angular distance
    # gamma1=x/Re;     %must retain the sign of x2
    gamma1 = x / Re / np.sin(thetactr)
    # must retain the sign of x2, just use theta of center of grid
    phi = phictr + gamma1
    phi = np.broadcast_to(phi[None, :, None], (lx1, phi.size, lx3))
    assert phi.shape == (lx1, lx2, lx3)

    # %% COMPUTE THE GEOGRAPHIC COORDINATES OF EACH GRID POINT
    glatgrid, glongrid = geomag2geog(theta, phi)

    # %% COMPUTE ECEF CARTESIAN IN CASE THEY ARE NEEDED
    xECEF = r * np.sin(theta) * np.cos(phi)
    yECEF = r * np.sin(theta) * np.sin(phi)
    zECEF = r * np.cos(theta)

    # %% COMPUTE SPHERICAL ECEF UNIT VECTORS - CARTESIAN-ECEF COMPONENTS
    er = np.empty((lx1, lx2, lx3, 3))
    etheta = np.empty_like(er)
    ephi = np.empty_like(er)

    er[:, :, :, 0] = np.sin(theta) * np.cos(phi)
    # xECEF-component of er
    er[:, :, :, 1] = np.sin(theta) * np.sin(phi)
    # yECEF
    er[:, :, :, 2] = np.cos(theta)
    # zECEF
    etheta[:, :, :, 0] = np.cos(theta) * np.cos(phi)
    etheta[:, :, :, 1] = np.cos(theta) * np.sin(phi)
    etheta[:, :, :, 2] = -np.sin(theta)
    ephi[:, :, :, 0] = -np.sin(phi)
    ephi[:, :, :, 1] = np.cos(phi)
    ephi[:, :, :, 2] = 0

    # %% UEN UNIT VECTORS IN ECEF COMPONENTS
    e1 = er
    # up is the same direction as from ctr of earth
    e2 = ephi
    # e2 is same as ephi
    e3 = -etheta
    # etheta is positive south, e3 is pos. north

    # %% STORE RESULTS IN GRID DATA STRUCTURE
    xg = {
        "x1": z,
        "x2": x,
        "x3": y,
        "x1i": zi,
        "x2i": xi,
        "x3i": yi,
    }

    lx = (xg["x1"].size, xg["x2"].size, xg["x3"].size)
    xg["lx"] = np.array(lx)

    xg["dx1f"] = np.append(xg["x1"][1:] - xg["x1"][:-1], xg["x1"][-1] - xg["x1"][-2])
    # FWD DIFF
    xg["dx1b"] = np.insert(xg["x1"][1:] - xg["x1"][:-1], 0, xg["x1"][1] - xg["x1"][0])
    # BACK DIFF
    xg["dx1h"] = xg["x1i"][1:-1] - xg["x1i"][:-2]
    # MIDPOINT DIFFS

    xg["dx2f"] = np.append(xg["x2"][1:] - xg["x2"][:-1], xg["x2"][-1] - xg["x2"][-2])
    # FWD DIFF
    xg["dx2b"] = np.insert(xg["x2"][1:] - xg["x2"][:-1], 0, xg["x2"][1] - xg["x2"][0])
    # BACK DIFF
    xg["dx2h"] = xg["x2i"][1:-1] - xg["x2i"][:-2]
    # MIDPOINT DIFFS

    xg["dx3f"] = np.append(xg["x3"][1:] - xg["x3"][:-1], xg["x3"][-1] - xg["x3"][-2])
    # FWD DIFF
    xg["dx3b"] = np.insert(xg["x3"][1:] - xg["x3"][:-1], 0, xg["x3"][1] - xg["x3"][0])
    # BACK DIFF
    xg["dx3h"] = xg["x3i"][1:-1] - xg["x3i"][:-2]
    # MIDPOINT DIFFS

    xg["h1"] = np.ones(lx)
    xg["h2"] = np.ones(lx)
    xg["h3"] = np.ones(lx)
    xg["h1x1i"] = np.ones((lx[0] + 1, lx[1], lx[2]))
    xg["h2x1i"] = np.ones((lx[0] + 1, lx[1], lx[2]))
    xg["h3x1i"] = np.ones((lx[0] + 1, lx[1], lx[2]))
    xg["h1x2i"] = np.ones((lx[0], lx[1] + 1, lx[2]))
    xg["h2x2i"] = np.ones((lx[0], lx[1] + 1, lx[2]))
    xg["h3x2i"] = np.ones((lx[0], lx[1] + 1, lx[2]))
    xg["h1x3i"] = np.ones((lx[0], lx[1], lx[2] + 1))
    xg["h2x3i"] = np.ones((lx[0], lx[1], lx[2] + 1))
    xg["h3x3i"] = np.ones((lx[0], lx[1], lx[2] + 1))

    # %% Cartesian, ECEF representation of curvilinar coordinates
    xg["e1"] = e1
    xg["e2"] = e2
    xg["e3"] = e3

    # %% ECEF spherical coordinates
    xg["r"] = r
    xg["theta"] = theta
    xg["phi"] = phi
    # xg.rx1i=[]; xg.thetax1i=[];
    # xg.rx2i=[]; xg.thetax2i=[];

    # %% These are cartesian representations of the ECEF, spherical unit vectors
    xg["er"] = er
    xg["etheta"] = etheta
    xg["ephi"] = ephi

    xg["I"] = np.broadcast_to(p["Bincl"], (lx2, lx3))

    # %% Cartesian ECEF coordinates
    xg["x"] = xECEF
    xg["z"] = zECEF
    xg["y"] = yECEF
    xg["alt"] = xg["r"] - Re
    # since we need a 3D array use xg.r here...

    xg["gx1"] = gz
    xg["gx2"] = np.zeros(lx)
    xg["gx3"] = np.zeros(lx)

    xg["Bmag"] = np.broadcast_to(-50000e-9, xg["lx"])
    # minus for northern hemisphere...

    xg["glat"] = glatgrid
    xg["glon"] = glongrid

    # xg['xp']=x; xg['zp']=z;

    # xg['inull']=[];
    xg["nullpts"] = np.zeros(lx)

    # %% TRIM DATA STRUCTURE TO BE THE SIZE FORTRAN EXPECTS
    # note: xgf is xg == True
    xgf = xg

    # indices corresponding to non-ghost cells for 1 dimension
    i1 = slice(2, lx[0] - 2)
    i2 = slice(2, lx[1] - 2)
    i3 = slice(2, lx[2] - 2)

    # any dx variable will not need to first element (backward diff of two ghost cells)
    idx1 = slice(1, lx[0])
    idx2 = slice(1, lx[1])
    idx3 = slice(1, lx[2])

    # x1-interface variables need only non-ghost cell values (left interface) plus one
    ix1i = slice(2, lx[0] - 1)
    ix2i = slice(2, lx[1] - 1)
    ix3i = slice(2, lx[2] - 1)

    # remove ghost cells
    # now that indices have been define we can go ahead and make this change
    xgf["lx"] = xgf["lx"] - 4

    xgf["dx1b"] = xgf["dx1b"][idx1]
    xgf["dx2b"] = xgf["dx2b"][idx2]
    xgf["dx3b"] = xgf["dx3b"][idx3]

    xgf["x1i"] = xgf["x1i"][ix1i]
    xgf["x2i"] = xgf["x2i"][ix2i]
    xgf["x3i"] = xgf["x3i"][ix3i]

    xgf["dx1h"] = xgf["dx1h"][i1]
    xgf["dx2h"] = xgf["dx2h"][i2]
    xgf["dx3h"] = xgf["dx3h"][i3]

    xgf["h1x1i"] = xgf["h1x1i"][ix1i, i2, i3]
    xgf["h2x1i"] = xgf["h2x1i"][ix1i, i2, i3]
    xgf["h3x1i"] = xgf["h3x1i"][ix1i, i2, i3]

    xgf["h1x2i"] = xgf["h1x2i"][i1, ix2i, i3]
    xgf["h2x2i"] = xgf["h2x2i"][i1, ix2i, i3]
    xgf["h3x2i"] = xgf["h3x2i"][i1, ix2i, i3]

    xgf["h1x3i"] = xgf["h1x3i"][i1, i2, ix3i]
    xgf["h2x3i"] = xgf["h2x3i"][i1, i2, ix3i]
    xgf["h3x3i"] = xgf["h3x3i"][i1, i2, ix3i]

    xgf["gx1"] = xgf["gx1"][i1, i2, i3]
    xgf["gx2"] = xgf["gx2"][i1, i2, i3]
    xgf["gx3"] = xgf["gx3"][i1, i2, i3]

    xgf["glat"] = xgf["glat"][i1, i2, i3]
    xgf["glon"] = xgf["glon"][i1, i2, i3]
    xgf["alt"] = xgf["alt"][i1, i2, i3]

    xgf["Bmag"] = xgf["Bmag"][i1, i2, i3]

    xgf["I"] = xgf["I"][i2, i3]

    xgf["nullpts"] = xgf["nullpts"][i1, i2, i3]

    xgf["e1"] = xgf["e1"][i1, i2, i3, :]
    xgf["e2"] = xgf["e2"][i1, i2, i3, :]
    xgf["e3"] = xgf["e3"][i1, i2, i3, :]

    xgf["er"] = xgf["er"][i1, i2, i3, :]
    xgf["etheta"] = xgf["etheta"][i1, i2, i3, :]
    xgf["ephi"] = xgf["ephi"][i1, i2, i3, :]

    xgf["r"] = xgf["r"][i1, i2, i3]
    xgf["theta"] = xgf["theta"][i1, i2, i3]
    xgf["phi"] = xgf["phi"][i1, i2, i3]

    xgf["x"] = xgf["x"][i1, i2, i3]
    xgf["y"] = xgf["y"][i1, i2, i3]
    xgf["z"] = xgf["z"][i1, i2, i3]

    xgf["glonctr"] = p["glon"]
    xgf["glatctr"] = p["glat"]

    return xgf
