"""
plasma functions
"""

import typing as T
import numpy as np
import os
import logging
import subprocess
import importlib.resources
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d, interp2d, interpn

from . import read
from . import write
from .build import cmake_build
from .web import url_retrieve, extract_zip

DictArray = T.Dict[str, T.Any]
# CONSTANTS
KB = 1.38e-23
AMU = 1.67e-27


def equilibrium_resample(p: T.Dict[str, T.Any], xg: T.Dict[str, T.Any]):
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
        url_retrieve(p["eq_url"], p["eq_zip"])
        extract_zip(p["eq_zip"], p["eq_dir"])

    # %% READ Equilibrium SIMULATION INFO
    peq = read.config(p["eq_dir"])
    if not peq:
        raise FileNotFoundError(
            f"equilibrium directory {p['eq_dir']} does not appear to contain config.nml"
        )

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
    if not xg_in:
        raise FileNotFoundError(f"{p['eq_dir']} does not have an input simulation grid.")

    nsi, vs1i, Tsi = model_resample(xg_in, ns=dat["ns"], vs=dat["vs1"], Ts=dat["Ts"], xg=xg)

    # %% sanity check interpolated variables
    check_density(nsi)
    check_drift(vs1i)
    check_temperature(Tsi)

    # %% WRITE OUT THE GRID
    write.grid(p, xg)

    write.state(
        p["indat_file"], t_eq_end, ns=nsi, vs=vs1i, Ts=Tsi, file_format=p.get("file_format")
    )


def model_resample(
    xgin: DictArray, ns: np.ndarray, vs: np.ndarray, Ts: np.ndarray, xg: DictArray
) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """resample a grid
    usually used to upsample an equilibrium simulation grid

    Parameters
    ----------

    xgin: dict
        original grid (usually equilibrium sim grid)
    ns: dict
        number density of species(4D)
    vs: dict
        velocity (4D)
    Ts: dict
        temperature of species (4D)

    Returns
    -------

    nsi: dict
        interpolated number density of species(4D)
    vsi: dict
        interpolated velocity (4D)
    Tsi: dict
        interpolated temperature of species (4D)
    """

    # %% NEW GRID SIZES
    lx1, lx2, lx3 = xg["lx"]
    lsp = ns.shape[0]

    # %% ALLOCATIONS
    nsi = np.empty((lsp, lx1, lx2, lx3), dtype=np.float32)
    vsi = np.empty_like(nsi)
    Tsi = np.empty_like(nsi)

    # %% INTERPOLATE ONTO NEWER GRID
    # to avoid IEEE754 rounding issues leading to bounds error,
    # cast the arrays to the same precision,
    # preferring float32 to save disk space and IO time
    X2 = xgin["x2"][2:-2].astype(np.float32)
    X1 = xgin["x1"][2:-2].astype(np.float32)
    X3 = xgin["x3"][2:-2].astype(np.float32)
    x1i = xg["x1"][2:-2].astype(np.float32)
    x2i = xg["x2"][2:-2].astype(np.float32)
    x3i = xg["x3"][2:-2].astype(np.float32)

    if lx3 > 1 and lx2 > 1:
        # 3-D
        logging.info("interpolating grid for 3-D simulation")
        # X2, X1, X3 = np.meshgrid(xgin['x2'][2:-2], xgin['x1'][2:-2], xgin['x3'][2:-2])
        X2i, X1i, X3i = np.meshgrid(x2i, x1i, x3i)
        assert X2i.shape == tuple(xg["lx"])

        for i in range(lsp):
            # the .values are to avoid OutOfMemoryError
            nsi[i, :, :, :] = interpn(
                points=(X1, X2, X3),
                values=ns[i, :, :, :].values,
                xi=(X1i, X2i, X3i),
                bounds_error=True,
            )
            vsi[i, :, :, :] = interpn(
                points=(X1, X2, X3),
                values=vs[i, :, :, :].values,
                xi=(X1i, X2i, X3i),
                bounds_error=True,
            )
            Tsi[i, :, :, :] = interpn(
                points=(X1, X2, X3),
                values=Ts[i, :, :, :].values,
                xi=(X1i, X2i, X3i),
                bounds_error=True,
            )
    elif lx3 == 1:
        # 2-D east-west
        logging.info("interpolating grid for 2-D simulation in x1, x2")
        # [X2,X1]=meshgrid(xgin.x2(3:end-2),xgin.x1(3:end-2));
        # [X2i,X1i]=meshgrid(xg.x2(3:end-2),xg.x1(3:end-2));
        for i in range(lsp):
            f = interp2d(X2, X1, ns[i, :, :, :], bounds_error=True)
            nsi[i, :, :, :] = f(x2i, x1i)[:, :, None]

            f = interp2d(X2, X1, vs[i, :, :, :], bounds_error=True)
            vsi[i, :, :, :] = f(x2i, x1i)[:, :, None]

            f = interp2d(X2, X1, Ts[i, :, :, :], bounds_error=True)
            Tsi[i, :, :, :] = f(x2i, x1i)[:, :, None]
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
        x3i = xg["x3"][2:-2].astype(np.float32)
        x1i = xg["x1"][2:-2].astype(np.float32)

        # for each species
        for i in range(lsp):
            f = interp2d(X3, X1, ns[i, :, :, :], bounds_error=True)
            nsi[i, :, :, :] = f(x3i, x1i)[:, None, :]

            f = interp2d(X3, X1, vs[i, :, :, :], bounds_error=True)
            vsi[i, :, :, :] = f(x3i, x1i)[:, None, :]

            f = interp2d(X3, X1, Ts[i, :, :, :], bounds_error=True)
            Tsi[i, :, :, :] = f(x3i, x1i)[:, None, :]

    else:
        raise ValueError("Not sure if this is 2-D or 3-D simulation")

    return nsi, vsi, Tsi


def check_density(n: np.ndarray):

    if not np.isfinite(n).all():
        raise ValueError("non-finite density")
    if (n < 0).any():
        raise ValueError("negative density")
    if n.max() < 1e6:
        raise ValueError("too small maximum density")


def check_drift(v: np.ndarray):

    if not np.isfinite(v).all():
        raise ValueError("non-finite drift")
    if (abs(v) > 10e3).any():
        raise ValueError("excessive drift velocity")


def check_temperature(T: np.ndarray):

    if not np.isfinite(T).all():
        raise ValueError("non-finite temperature")
    if (T < 0).any():
        raise ValueError("negative temperature")
    if T.max() < 500:
        raise ValueError("too cold maximum temperature")


def equilibrium_state(
    p: T.Dict[str, T.Any], xg: DictArray
) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    generate (arbitrary) initial conditions for a grid.
    NOTE: only works on symmmetric closed grids!

    [f107a, f107, ap] = activ
    """

    # %% MAKE UP SOME INITIAL CONDITIONS FOR FORTRAN CODE
    mindens = 1e-100

    def Oplus(ns: np.ndarray) -> np.ndarray:
        ns[0, :, ix2, ix3] = rho * ne
        zref = 900e3
        i = alt[:, ix2, ix3] > zref
        if any(i):
            iord = np.argsort(alt[:, ix2, ix3])
            altsort = alt[iord, ix2, ix3]
            nsort = ns[0, :, ix2, ix3]
            nsort = nsort[iord]

            ms = 16 * AMU
            H = KB * 2 * Tn[inds, ix2, ix3] / ms / g[inds, ix2, ix3]
            z = alt[i, ix2, ix3]
            lz = z.size
            iord = np.argsort(z)
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

    def molecular_density(ns: np.ndarray, xgr: np.ndarray, inds: np.ndarray) -> np.ndarray:
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

        cond: bool = None

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
            iord = np.argsort(z)
            z = z[iord]
            z = np.append(z, 2 * z[-1] - z[-2])
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

        ns[5, i, ix2, ix3] = chapmana(z, n0, alt[iref, ix2, ix3], Hf.mean())

        return ns

    # %% SLICE THE FIELD IN HALF IF WE ARE CLOSED
    natm = msis_setup(p, xg)

    closeddip = abs(xg["r"][0, 0, 0] - xg["r"][-1, 0, 0]) < 50e3
    # logical flag marking the grid as closed dipole
    if closeddip:
        # closed dipole grid
        #    [~,ialtmax]=max(xg.alt(:,1,1))
        #    lalt=ialtmax
        lalt = xg["lx"][0] // 2
        # FIXME:  needs to work with asymmetric grid...
        alt = xg["alt"][:lalt, :, :]
        lx1 = lalt
        lx2 = xg["lx"][1]
        lx3 = xg["lx"][2]
        Tn = natm[3, :lalt, :, :]
        g = abs(xg["gx1"][:lalt, :, :])
        g = max(g, 1)
        for ix3 in range(lx3):
            for ix2 in range(lx2):
                ialt = abs(g[:, ix2, ix3] - 1).argmin()
                if ialt != lalt:
                    g[ialt:lalt, ix2, ix3] = 1

    else:
        alt = xg["alt"]
        lx1, lx2, lx3 = xg["lx"]
        Tn = natm[3, :, :, :]
        g = abs(xg["gx1"])

    ns = np.zeros((7, lx1, lx2, lx3))
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
                iord = np.argsort(z)
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

    vsx1 = np.zeros((7, lx1, lx2, lx3))
    Ts = np.tile(Tn[None, :, :, :], [7, 1, 1, 1])

    if closeddip:
        # closed dipole grid
        # FIXME:  This code only works for symmetric grids...
        if 2 * lx1 == xg["lx"][0]:
            ns = np.concatenate((ns, ns[:, ::-1, :, :]), 1)
            Ts = np.concatenate((Ts, Ts[:, ::-1, :, :]), 1)
            vsx1 = np.concatenate((vsx1, vsx1[:, ::-1, :, :]), 1)
        else:
            ns = np.concatenate((ns, ns[:, lx1, :, :], ns[:, ::-1, :, :]), 1)
            Ts = np.concatenate((Ts, Ts[:, lx1, :, :], Ts[:, ::-1, :, :]), 1)
            vsx1 = np.concatenate((vsx1, vsx1[:, lx1, :, :], vsx1[:, ::-1, :, :]), 1)

    return ns, Ts, vsx1


def chapmana(z: np.ndarray, nm: float, z0: float, H: float) -> np.ndarray:
    zref = (z - z0) / H
    ne = nm * np.exp(0.5 * (1 - zref - np.exp(-zref)))

    ne[ne < 1] = 1

    return ne


def msis_setup(p: DictArray, xg: DictArray) -> np.ndarray:
    """calls MSIS Fortran exectuable
    compiles if not present

    [f107a, f107, ap] = activ
        COLUMNS OF DATA:
          1 - ALT
          2 - HE NUMBER DENSITY(M-3)
          3 - O NUMBER DENSITY(M-3)
          4 - N2 NUMBER DENSITY(M-3)
          5 - O2 NUMBER DENSITY(M-3)
          6 - AR NUMBER DENSITY(M-3)
          7 - TOTAL MASS DENSITY(KG/M3)
          8 - H NUMBER DENSITY(M-3)
          9 - N NUMBER DENSITY(M-3)
          10 - Anomalous oxygen NUMBER DENSITY(M-3)
          11 - TEMPERATURE AT ALT

    """

    msis_stem = "msis_setup"
    msis_name = msis_stem
    if os.name == "nt":
        msis_name += ".exe"

    if not importlib.resources.is_resource(__package__, msis_name):
        with importlib.resources.path(__package__, "CMakeLists.txt") as setup:
            cmake_build(
                setup.parent,
                setup.parent / "build",
                config_args=["-DBUILD_TESTING:BOOL=false"],
                build_args=["--target", msis_stem],
            )

    # %% SPECIFY SIZES ETC.
    lx1 = xg["lx"][0]
    lx2 = xg["lx"][1]
    lx3 = xg["lx"][2]
    alt = xg["alt"] / 1e3
    glat = xg["glat"]
    glon = xg["glon"]
    lz = lx1 * lx2 * lx3
    # % CONVERT DATES/TIMES/INDICES INTO MSIS-FRIENDLY FORMAT
    t0 = p["time"][0]
    doy = int(t0.strftime("%j"))
    UTsec0 = t0.hour * 3600 + t0.minute * 60 + t0.second + t0.microsecond / 1e6

    logging.debug(f"MSIS00 using DOY: {doy}")
    yearshort = t0.year % 100
    iyd = yearshort * 1000 + doy
    # %% KLUDGE THE BELOW-ZERO ALTITUDES SO THAT THEY DON'T GIVE INF
    alt[alt <= 0] = 1
    # %% CREATE INPUT FILE FOR FORTRAN PROGRAM
    # don't use NamedTemporaryFile because PermissionError on Windows
    # file_in = tempfile.gettempdir() + "/msis_setup_input.dat"

    # with open(file_in, "w") as f:
    #     np.array(iyd).astype(np.int32).tofile(f)
    #     np.array(UTsec0).astype(np.int32).tofile(f)
    #     np.asarray([p["f107a"], p["f107"], p["Ap"], p["Ap"]]).astype(np.float32).tofile(f)
    #     np.array(lz).astype(np.int32).tofile(f)
    #     np.array(glat).astype(np.float32).tofile(f)
    #     np.array(glon).astype(np.float32).tofile(f)
    #     np.array(alt).astype(np.float32).tofile(f)

    invals = (
        f"{iyd}\n{int(UTsec0)}\n{p['f107a']} {p['f107']} {p['Ap']} {p['Ap']}\n{lz}\n"
        + " ".join(map(str, glat.ravel(order="C")))
        + "\n"
        + " ".join(map(str, glon.ravel(order="C")))
        + "\n"
        + " ".join(map(str, alt.ravel(order="C")))
    )
    # %% CALL MSIS
    # the "-" means to use stdin, stdout

    with importlib.resources.path(__package__, msis_name) as exe:
        cmd = [str(exe), "-", "-", str(lz)]
        logging.info(" ".join(cmd))
        ret = subprocess.check_output(cmd, input=invals, text=True)

    Nread = lz * 11

    # old code, from before we used stdout
    # fout_size = Path(file_out).stat().st_size
    # if fout_size != Nread * 4:
    #     raise RuntimeError(f"expected {file_out} size {Nread*4} but got {fout_size}")

    msisdat = np.fromstring(ret, np.float32, Nread, sep=" ").reshape((11, lz), order="F")

    # %% ORGANIZE
    # altitude is a useful sanity check as it's very regular and obvious.
    alt_km = msisdat[0, :].reshape((lx1, lx2, lx3))
    if not np.allclose(alt_km, alt, atol=0.02):  # atol due to precision of stdout ~0.01 km
        raise ValueError("was msis_driver output parsed correctly?")

    nO = msisdat[2, :].reshape((lx1, lx2, lx3))
    nN2 = msisdat[3, :].reshape((lx1, lx2, lx3))
    nO2 = msisdat[4, :].reshape((lx1, lx2, lx3))
    Tn = msisdat[10, :].reshape((lx1, lx2, lx3))
    nN = msisdat[8, :].reshape((lx1, lx2, lx3))

    nNO = 0.4 * np.exp(-3700 / Tn) * nO2 + 5e-7 * nO
    # Mitra, 1968
    nH = msisdat[7, :].reshape((lx1, lx2, lx3))
    natm = np.stack((nO, nN2, nO2, Tn, nN, nNO, nH), 0)

    return natm
