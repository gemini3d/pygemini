import typing as T
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

from .. import LSP
from ..utils import datetime2ymd_hourdec

try:
    import h5py
except (ImportError, AttributeError):
    # must be ImportError not ModuleNotFoundError for botched HDF5 linkage
    h5py = None


def state(time: datetime, ns: np.ndarray, vs: np.ndarray, Ts: np.ndarray, fn: Path):
    """
    write STATE VARIABLE initial conditions

    NOTE THAT WE don't write ANY OF THE ELECTRODYNAMIC
    VARIABLES SINCE THEY ARE NOT NEEDED TO START THINGS
    UP IN THE FORTRAN CODE.

    INPUT ARRAYS SHOULD BE TRIMMED TO THE CORRECT SIZE
    I.E. THEY SHOULD NOT INCLUDE GHOST CELLS

    NOTE: The .transpose() reverses the dimension order.
    The HDF Group never implemented the intended H5T_array_create(..., perm)
    and it's deprecated.
    Fortran, including the HDF Group Fortran interfaces and h5fortran as well as
    Matlab read/write HDF5 in Fortran order. h5py read/write HDF5 in C order so we
    need the .transpose() for h5py
    """

    if h5py is None:
        raise ImportError("pip install h5py")

    logging.info(f"state: {fn}")

    with h5py.File(fn, "w") as f:
        f["/time/ymd"] = [time.year, time.month, time.day]
        f["/time/UTsec"] = np.float32(
            time.hour * 3600 + time.minute * 60 + time.second + time.microsecond / 1e6
        )

        p4 = (0, 3, 2, 1)
        # we have to reverse axes order and put lsp at the last dim

        f.create_dataset(
            "/nsall",
            data=ns.transpose(p4),
            dtype=np.float32,
            compression="gzip",
            compression_opts=1,
            shuffle=True,
            fletcher32=True,
        )
        f.create_dataset(
            "/vs1all",
            data=vs.transpose(p4),
            dtype=np.float32,
            compression="gzip",
            compression_opts=1,
            shuffle=True,
            fletcher32=True,
        )
        f.create_dataset(
            "/Tsall",
            data=Ts.transpose(p4),
            dtype=np.float32,
            compression="gzip",
            compression_opts=1,
            shuffle=True,
            fletcher32=True,
        )


def data(dat: T.Dict[str, T.Any], outfn: Path):
    """
    write simulation data
    e.g. for converting a file format from a simulation
    """

    if h5py is None:
        raise ImportError("pip install h5py")

    lxs = dat["lxs"]

    with h5py.File(outfn, "w") as h:
        for k in ["ns", "vs1", "Ts"]:
            if k not in dat:
                continue

            h.create_dataset(
                k,
                data=dat[k][1].astype(np.float32),
                chunks=(1, *lxs[1:], LSP),
                compression="gzip",
                compression_opts=1,
            )

        for k in ["ne", "v1", "Ti", "Te", "J1", "J2", "J3", "v2", "v3"]:
            if k not in dat:
                continue

            h.create_dataset(
                k,
                data=dat[k][1].astype(np.float32),
                chunks=(1, *lxs[1:]),
                compression="gzip",
                compression_opts=1,
            )

        if "Phitop" in dat:
            h.create_dataset(
                "Phitop",
                data=dat["Phitop"][1],
                compression="gzip",
                compression_opts=1,
            )


def grid(size_fn: Path, grid_fn: Path, xg: T.Dict[str, T.Any]):
    """writes grid to disk

    Parameters
    ----------

    size_fn: pathlib.Path
        file to write
    grid_fn: pathlib.Path
        file to write
    xg: dict
        grid values

    NOTE: The .transpose() reverses the dimension order.
    The HDF Group never implemented the intended H5T_array_create(..., perm)
    and it's deprecated.
    Fortran, including the HDF Group Fortran interfaces and h5fortran as well as
    Matlab read/write HDF5 in Fortran order. h5py read/write HDF5 in C order so we
    need the .transpose() for h5py
    """

    if h5py is None:
        raise ImportError("pip install h5py")

    if "lx" not in xg:
        xg["lx"] = np.array([xg["x1"].shape, xg["x2"].shape, xg["x3"].shape])

    logging.info(f"write_grid: {size_fn}")
    with h5py.File(size_fn, "w") as h:
        h["/lx"] = xg["lx"]

    logging.info(f"write_grid: {grid_fn}")
    with h5py.File(grid_fn, "w") as h:
        for i in (1, 2, 3):
            for k in (
                f"x{i}",
                f"x{i}i",
                f"dx{i}b",
                f"dx{i}h",
                f"h{i}",
                f"h{i}x1i",
                f"h{i}x2i",
                f"h{i}x3i",
                f"gx{i}",
                f"e{i}",
            ):
                if k not in xg:
                    logging.info(f"SKIP: {k}")
                    continue

                if xg[k].ndim >= 2:
                    h.create_dataset(
                        f"/{k}",
                        data=xg[k].transpose(),
                        dtype=np.float32,
                        compression="gzip",
                        compression_opts=1,
                        shuffle=True,
                        fletcher32=True,
                    )
                else:
                    h[f"/{k}"] = xg[k].astype(np.float32)

        for k in (
            "alt",
            "glat",
            "glon",
            "Bmag",
            "I",
            "nullpts",
            "er",
            "etheta",
            "ephi",
            "r",
            "theta",
            "phi",
            "x",
            "y",
            "z",
        ):
            if k not in xg:
                logging.info(f"SKIP: {k}")
                continue

            if xg[k].ndim >= 2:
                h.create_dataset(
                    f"/{k}",
                    data=xg[k].transpose(),
                    dtype=np.float32,
                    compression="gzip",
                    compression_opts=1,
                    shuffle=True,
                    fletcher32=True,
                )
            else:
                h[f"/{k}"] = xg[k].astype(np.float32)


def Efield(outdir: Path, E: T.Dict[str, np.ndarray]):
    """
    write Efield to disk
    """

    if h5py is None:
        raise ImportError("pip install h5py")

    with h5py.File(outdir / "simsize.h5", "w") as f:
        f["/llon"] = E["llon"]
        f["/llat"] = E["llat"]

    with h5py.File(outdir / "simgrid.h5", "w") as f:
        f["/mlon"] = E["mlon"].astype(np.float32)
        f["/mlat"] = E["mlat"].astype(np.float32)

    for i, t in enumerate(E["time"]):
        fn = outdir / (datetime2ymd_hourdec(t) + ".h5")

        # FOR EACH FRAME WRITE A BC TYPE AND THEN OUTPUT BACKGROUND AND BCs
        with h5py.File(fn, "w") as f:
            f["/flagdirich"] = E["flagdirich"][i].astype(np.int32)
            f["/time/ymd"] = [t.year, t.month, t.day]
            f["/time/UTsec"] = t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6

            for k in ("Exit", "Eyit", "Vminx1it", "Vmaxx1it"):
                f.create_dataset(
                    f"/{k}",
                    data=E[k][i, :, :].transpose(),
                    dtype=np.float32,
                    compression="gzip",
                    compression_opts=1,
                    shuffle=True,
                    fletcher32=True,
                )
            for k in ("Vminx2ist", "Vmaxx2ist", "Vminx3ist", "Vmaxx3ist"):
                f[f"/{k}"] = E[k][i, :].astype(np.float32)


def precip(outdir: Path, precip: T.Dict[str, T.Any]):

    if h5py is None:
        raise ImportError("pip install h5py")

    with h5py.File(outdir / "simsize.h5", "w") as f:
        f["/llon"] = precip["llon"]
        f["/llat"] = precip["llat"]

    with h5py.File(outdir / "simgrid.h5", "w") as f:
        f["/mlon"] = precip["mlon"].astype(np.float32)
        f["/mlat"] = precip["mlat"].astype(np.float32)

    for i, t in enumerate(precip["time"]):
        fn = outdir / (datetime2ymd_hourdec(t) + ".h5")

        with h5py.File(fn, "w") as f:
            for k in ("Q", "E0"):
                f.create_dataset(
                    f"/{k}p",
                    data=precip[k][i, :, :].transpose(),
                    dtype=np.float32,
                    compression="gzip",
                    compression_opts=1,
                    shuffle=True,
                    fletcher32=True,
                )
