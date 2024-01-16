"""
plasma functions
"""

from __future__ import annotations
import typing as T
import logging

import numpy as np
import xarray
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d, interpn, RegularGridInterpolator

from . import read
from . import LSP, SPECIES
from . import write
from .web import url_retrieve
from .archive import extract
from .msis import msis_setup

# CONSTANTS
KB = 1.38e-23
AMU = 1.67e-27


def equilibrium_resample(p: dict[str, T.Any], xg: dict[str, T.Any]):
    """
    read and interpolate equilibrium simulation data, writing new
    interpolated grid.
    """

    # %% download equilibrium data if needed and specified
    if not p["eq_dir"].is_dir():
        if "eq_url" not in p:
            raise FileNotFoundError(
                f"{p['eq_dir']} not found and eq_url not specified in {p['nml']}"
            )
        url_retrieve(p["eq_url"], p["eq_archive"])
        extract(p["eq_archive"], p["eq_dir"])

    # %% READ Equilibrium SIMULATION INFO
    peq = read.config(p["eq_dir"])

    # %% END FRAME time of equilibrium simulation
    # this will be the starting time of the new simulation
    t_eq_end = peq["time"][-1]

    # %% LOAD THE last equilibrium frame
    dat = read.frame(p["eq_dir"], t_eq_end)
    if not dat:
        raise FileNotFoundError(f"{p['eq_dir']} does not have data for {t_eq_end}")

    # %% sanity check equilibrium simulation input to interpolation
    check_density(dat["ns"])
    check_drift(dat["vs1"])
    check_temperature(dat["Ts"])

    # %% DO THE INTERPOLATION
    xg_in = read.grid(p["eq_dir"])

    dat_interp = model_resample(xg_in, dat, xg)

    # %% sanity check interpolated variables
    check_density(dat_interp["ns"])
    check_drift(dat_interp["vs1"])
    check_temperature(dat_interp["Ts"])

    # %% WRITE OUT THE GRID
    write.grid(p, xg)

    write.state(p["indat_file"], dat_interp)


def model_resample(
    xgin: dict[str, T.Any], dat: xarray.Dataset, xg: dict[str, T.Any]
) -> xarray.Dataset:
    """resample a grid
    usually used to upsample an equilibrium simulation grid

    Parameters
    ----------

    xgin: dict
        original grid (usually equilibrium sim grid)
    dat: xarray.Dataset
        data to interpolate

    Returns
    -------

    dat_interp: xarray.Dataset
        interpolated data
    """

    # %% NEW GRID SIZES
    lx1, lx2, lx3 = xg["lx"]

    # %% ALLOCATIONS

    dat_interp = xarray.Dataset(
        coords={
            "species": SPECIES,
            "x1": xg["x1"][2:-2],
            "x2": xg["x2"][2:-2],
            "x3": xg["x3"][2:-2],
        }
    )
    for k in {"ns", "vs1", "Ts"}:
        dat_interp[k] = (
            ("species", "x1", "x2", "x3"),
            np.empty((LSP, lx1, lx2, lx3), dtype=np.float32),
        )

    # %% INTERPOLATE ONTO NEWER GRID
    """
    Note that float64 upcasting is used to match fast internal
    Cython code. The coordinates and values need to same type
    to avoid false bounds errors due to IEEE754 rounding.
    """
    X2 = xgin["x2"][2:-2].astype(np.float64)
    X1 = xgin["x1"][2:-2].astype(np.float64)
    X3 = xgin["x3"][2:-2].astype(np.float64)
    x1i = xg["x1"][2:-2]
    x2i = xg["x2"][2:-2]
    x3i = xg["x3"][2:-2]

    if lx3 > 1 and lx2 > 1:
        # 3-D
        logging.info("interpolating grid for 3-D simulation")
        # X2, X1, X3 = np.meshgrid(xgin['x2'][2:-2], xgin['x1'][2:-2], xgin['x3'][2:-2])
        X2i, X1i, X3i = np.meshgrid(x2i, x1i, x3i)
        assert X2i.shape == tuple(xg["lx"])

        for i in range(LSP):
            for k in {"ns", "vs1", "Ts"}:
                # the .data is to avoid OutOfMemoryError
                dat_interp[k][i, :, :, :] = interpn(
                    points=(X1, X2, X3),
                    values=dat[k][i, :, :, :].data.astype(np.float64),
                    xi=(X1i, X2i, X3i),
                    bounds_error=False,
                    fill_value=None,
                )

    elif lx3 == 1:
        # 2-D east-west
        logging.info("interpolating grid for 2-D simulation in x1, x2")
        # [X2,X1]=meshgrid(xgin.x2(3:end-2),xgin.x1(3:end-2));
        # [X2i,X1i]=meshgrid(xg.x2(3:end-2),xg.x1(3:end-2));
        X1i, X2i = np.meshgrid(x1i, x2i, indexing="ij")
        for i in range(LSP):
            for k in {"ns", "vs1", "Ts"}:
                f = RegularGridInterpolator(
                    (X1, X2),
                    dat[k][i, :, :, 0].data.astype(np.float64),
                    bounds_error=False,
                    fill_value=None,
                )
                dat_interp[k][i, :, :, :] = f((X1i, X2i))[:, :, None]

    elif lx2 == 1:
        # 2-D north-south
        logging.info("interpolating grid for 2-D simulation in x1, x3")
        # original grid, a priori the first 2 and last 2 values are ghost cells
        # on each axis
        #
        # Detect old non-padded grid and workaround
        if np.isclose(xgin["x3"][0], xg["x3"][2], atol=1):
            # old sim, no external ghost cells.
            # Instead of discarding good cells,keep them and say there are
            # new ghost cells outside the grid
            X3 = np.linspace(xgin["x3"][0], xgin["x3"][-1], xgin["lx"][2])
        else:
            # new sim, external ghost cells
            X3 = xgin["x3"][2:-2]

        X1 = xgin["x1"][2:-2]
        # new grid
        x3i = xg["x3"][2:-2]
        x1i = xg["x1"][2:-2]
        X1i, X3i = np.meshgrid(x1i, x3i, indexing="ij")
        # for each species
        for i in range(LSP):
            for k in {"ns", "vs1", "Ts"}:
                f = RegularGridInterpolator(
                    (X1, X3),
                    dat[k][i, :, 0, :].data.astype(np.float64),
                    bounds_error=False,
                    fill_value=None,
                )
                dat_interp[k][i, :, :, :] = f((X1i, X3i))[:, None, :]

    else:
        raise ValueError("Not sure if this is 2-D or 3-D simulation")

    dat_interp.attrs["time"] = dat.time

    return dat_interp


def check_density(n: xarray.DataArray):
    if not np.isfinite(n).all():
        raise ValueError("non-finite density")
    if (n < 0).any():
        raise ValueError("negative density")
    if n.max() < 1e6:
        raise ValueError("too small maximum density")


def check_drift(v: xarray.DataArray):
    if not np.isfinite(v).all():
        raise ValueError("non-finite drift")
    if (abs(v) > 10e3).any():
        raise ValueError("excessive drift velocity")


def check_temperature(Ts: xarray.DataArray):
    if not np.isfinite(Ts).all():
        raise ValueError("non-finite temperature")
    if (Ts < 0).any():
        raise ValueError("negative temperature")
    if Ts.max() < 500:
        raise ValueError("too cold maximum temperature")


def equilibrium_state(p: dict[str, T.Any], xg: dict[str, T.Any]) -> xarray.Dataset:
    """
    generate (arbitrary) initial conditions for a grid.
    NOTE: only works on symmmetric closed grids!

    [f107a, f107, ap] = activ
    """

    # %% MAKE UP SOME INITIAL CONDITIONS FOR FORTRAN CODE
    mindens = 1e-100

    def Oplus(ns):
        ns[0, :, ix2, ix3] = rho * ne
        zref = 900e3
        i = alt[:, ix2, ix3] > zref
        if any(i):
            iord = alt[:, ix2, ix3].argsort()
            altsort = alt[iord, ix2, ix3]
            nsort = ns[0, :, ix2, ix3]
            nsort = nsort[iord]

            ms = 16 * AMU
            H = KB * 2 * Tn[inds, ix2, ix3] / ms / g[inds, ix2, ix3]
            z = alt[i, ix2, ix3]
            lz = z.size
            iord = z.argsort()
            z = z[iord]
            #     z=[z; 2*z(lz)-z(lz-1)];
            z = np.insert(z, 0, zref)
            integrand = 1 / H[iord]
            integrand = np.append(integrand, integrand[-1])

            # this cumtrapz() does NOT get initial=0, since matlab user code strips first element here
            redheight = cumtrapz(integrand, z)
            f = interp1d(altsort, nsort)
            n1top = f(zref) * np.exp(-redheight)
            n1sort = np.zeros(lz)
            for iz in range(lz):
                n1sort[iord[iz]] = n1top[iz]

            ns[0, i, ix2, ix3] = n1sort

        return ns

    def molecular_density(ns, xgr, inds):
        """MOLECULAR DENSITIES

        Parameters
        ----------

        ns: np.ndarray
            4D by species number density
        xgr: np.ndarray
            xg["r"]
        inds: np.ndarray
            boolean vector

        Returns
        -------

        ns: np.ndarray
            4D by species number density

        """

        i = np.setdiff1d(range(lx1), inds.nonzero()[0])

        nmolc = np.zeros(lx1)
        nmolc[i] = (1 - rho[i]) * ne[i]

        if any(inds):
            if xgr.ndim == 3:
                cond = xgr[0, 0, 0] > xgr[1, 0, 0]
            elif xgr.ndim == 2:
                cond = xgr[0, 0] > xgr[1, 0]
            else:
                raise ValueError(
                    "xg['r'] expected to be 3D, possibly with degenerate 2nd or 3rd dimension"
                )

            iref = i[0] if cond else i[-1]

            n0 = nmolc[iref]
            ms = 30.5 * AMU
            H = KB * Tn[inds, ix2, ix3] / ms / g[inds, ix2, ix3]
            z = alt[inds, ix2, ix3]
            lz = z.size
            iord = z.argsort()
            z = z[iord]
            # z = np.append(z, 2 * z[-1] - z[-2])
            z = np.insert(z, 0, alt[iref, ix2, ix3])
            integrand = 1 / H[iord]
            integrand = np.append(integrand, integrand[-1])
            # this cumtrapz() does NOT get initial=0, since matlab user code strips first element here
            redheight = cumtrapz(integrand, z)
            nmolctop = n0 * np.exp(-redheight)
            nmolcsort = np.zeros(lz)
            for iz in range(lz):
                nmolcsort[iord[iz]] = nmolctop[iz]

            nmolc[inds] = nmolcsort

        ns[1, :, ix2, ix3] = 1 / 3 * nmolc
        ns[2, :, ix2, ix3] = 1 / 3 * nmolc
        ns[3, :, ix2, ix3] = 1 / 3 * nmolc

        # %% PROTONS
        ns[5, inds, ix2, ix3] = (1 - rho[inds]) * ne[inds]
        z = alt[i, ix2, ix3]
        if any(inds):
            iref = inds.nonzero()[0][-1] if cond else inds.nonzero()[0][0]
            n0 = ns[5, iref, ix2, ix3]
        else:
            iref = alt[:, ix2, ix3].argmax()
            n0 = 1e6

        ns[5, i, ix2, ix3] = chapmana(z, n0, alt[iref, ix2, ix3], Hf.mean().item())

        return ns

    # %% SLICE THE FIELD IN HALF IF WE ARE CLOSED
    atmos = msis_setup(p, xg)

    closeddip: bool = abs(xg["r"][0, 0, 0] - xg["r"][-1, 0, 0]) < 50e3
    # logical flag marking the grid as closed dipole
    if closeddip:
        # closed dipole grid
        i = xg["alt"][:, 0, 0].argmax()
        # i = xg["lx"][0] // 2
        # FIXME:  needs to work with asymmetric grid...
        alt = xg["alt"][:i, :, :]
        lx1 = i
        lx2 = xg["lx"][1]
        lx3 = xg["lx"][2]
        Tn = atmos["Tn"][:i, :, :]
        g = abs(xg["gx1"][:i, :, :])
        g = g.clip(min=1)
        for ix3 in range(lx3):
            for ix2 in range(lx2):
                ialt = abs(g[:, ix2, ix3] - 1).argmin()
                if ialt != i:
                    g[ialt:i, ix2, ix3] = 1

    else:
        alt = xg["alt"]
        lx1, lx2, lx3 = xg["lx"]
        Tn = atmos["Tn"]
        g = abs(xg["gx1"])

    ns = np.zeros((7, lx1, lx2, lx3), dtype=np.float32)
    for ix3 in range(lx3):
        for ix2 in range(lx2):
            Hf = KB * Tn[:, ix2, ix3] / AMU / 16 / g[:, ix2, ix3]
            z0f = 325e3
            He = 2 * KB * Tn[:, ix2, ix3] / AMU / 30 / g[:, ix2, ix3]
            z0e = 120e3

            ne = chapmana(alt[:, ix2, ix3], p["nmf"], z0f, Hf) + chapmana(
                alt[:, ix2, ix3], p["nme"], z0e, He
            )
            rho = 1 / 2 * np.tanh((alt[:, ix2, ix3] - 200e3) / 45e3) - 1 / 2 * np.tanh(
                (alt[:, ix2, ix3] - 1000e3) / 200e3
            )

            inds = alt[:, ix2, ix3] > z0f
            if any(inds):
                ms = rho[inds] * 16 * AMU + (1 - rho[inds]) * AMU
                # topside composition only
                H = KB * 2 * Tn[inds, ix2, ix3] / ms / g[inds, ix2, ix3]
                z = alt[inds, ix2, ix3]
                lz = z.size
                iord = z.argsort()
                z = z[iord]
                #     z=[z; 2*z(lz)-z(lz-1)];
                z = np.insert(z, 0, z0f)
                integrand = 1 / H[iord]
                integrand = np.append(integrand, integrand[-1])
                # initial=0 is to match Matlab cumtrapz()
                redheight = cumtrapz(integrand, z, initial=0)
                netop = p["nmf"] * np.exp(-redheight)
                nesort = np.zeros(lz)
                for iz in range(lz):
                    nesort[iord[iz]] = netop[iz]

                ne[inds] = nesort

            ns = Oplus(ns)

            # N+
            ns[4, :, ix2, ix3] = 1e-4 * ns[0, :, ix2, ix3]

            ns = molecular_density(ns, xg["r"], inds)

    ns[:6, :, :, :][ns[:6, :, :, :] < mindens] = mindens
    ns[6, :, :, :] = ns[:6, :, :, :].sum(axis=0)

    vsx1 = np.zeros((7, lx1, lx2, lx3), dtype=np.float32)

    Ts = np.broadcast_to(Tn, [7, lx1, lx2, lx3])

    if closeddip:
        # closed dipole grid
        # FIXME:  This code only works for symmetric grids...

        if 2 * lx1 == xg["lx"][0]:
            ns = np.concatenate((ns, ns[:, ::-1, :, :]), 1)
            Ts = np.concatenate((Ts, Ts[:, ::-1, :, :]), 1)
            vsx1 = np.concatenate((vsx1, vsx1[:, ::-1, :, :]), 1)
        else:
            nsp = ns[:, -1, :, :][:, None, :, :]
            tsp = Ts[:, -1, :, :][:, None, :, :]
            vsp = vsx1[:, -1, :, :][:, None, :, :]
            if 2 * lx1 == xg["lx"][0] - 1:
                # off by one, fill with end point
                ns = np.concatenate((ns, nsp, ns[:, ::-1, :, :]), 1)
                Ts = np.concatenate((Ts, tsp, Ts[:, ::-1, :, :]), 1)
                vsx1 = np.concatenate((vsx1, vsp, vsx1[:, ::-1, :, :]), 1)
            elif 2 * lx1 == xg["lx"][0] - 2:
                # off by two, fill with end point
                ns = np.concatenate((ns, nsp, nsp, ns[:, ::-1, :, :]), 1)
                Ts = np.concatenate((Ts, tsp, tsp, Ts[:, ::-1, :, :]), 1)
                vsx1 = np.concatenate((vsx1, vsp, vsp, vsx1[:, ::-1, :, :]), 1)
            else:
                raise ValueError(
                    f"closed dipole lalt: 2*lx1={2 * lx1} != grid lx1: {xg['lx'][0]}"
                )

    dat = xarray.Dataset(
        {
            "ns": (("species", "x1", "x2", "x3"), ns),
            "vs1": (("species", "x1", "x2", "x3"), vsx1),
            "Ts": (("species", "x1", "x2", "x3"), Ts),
        },
        coords={
            "species": SPECIES,
            "x1": xg["x1"][2:-2],
            "x2": xg["x2"][2:-2],
            "x3": xg["x3"][2:-2],
        },
        attrs={"time": p["time"][0]},
    )

    return dat


def chapmana(z, nm: float, z0: float, H):
    """
    create Chapman profile

    Parameters
    ----------

    z : np.ndarray
        altitude [meters]

    nm : float
        number density scale factor

    z0 : float
        peak altitude [meters]

    H : float or np.ndarray
        scale height vs. altitude z [meters]
    """

    if np.any(H < 0.1):
        raise ValueError(f"scale height H is unrealistically small:  {H}")

    zref = (z - z0) / H
    ne = nm * np.exp(0.5 * (1 - zref - np.exp(-zref)))

    ne[ne < 1] = 1

    return ne
