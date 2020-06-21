#!/usr/bin/env python3
"""
convert Gemini data to HDF5 .h5
"""
import h5py
from pathlib import Path
import argparse
import typing as T
from numpy import float32

import gemini3d

LSP = 7
CLVL = 6


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("indir", help="Gemini .dat file directory")
    p.add_argument("-o", "--outdir", help="directory to write HDF5 files")
    p.add_argument(
        "-flagoutput", help="manually specify flagoutput, for if config.nml is missing", type=int
    )
    P = p.parse_args()

    indir = Path(P.indir).expanduser()
    if P.outdir:
        outdir = Path(P.outdir).expanduser()
    elif indir.is_file():
        outdir = indir.parent
    elif indir.is_dir():
        outdir = indir
    else:
        raise FileNotFoundError(indir)

    if indir.is_file():
        infiles = [indir]
    elif indir.is_dir():
        infiles = sorted(indir.glob("*.dat"))
    else:
        raise FileNotFoundError(indir)

    if not infiles:
        raise FileNotFoundError(f"no files to convert in {indir}")

    lxs = gemini3d.get_simsize(indir)

    cfg = None
    if P.flagoutput is not None:
        cfg = {"flagoutput": P.flagoutput}

    for infile in infiles:
        outfile = outdir / (infile.stem + ".h5")
        print(infile, "=>", outfile)

        dat = gemini3d.readdata(infile, cfg=cfg)
        if "lxs" not in dat:
            dat["lxs"] = lxs

        write_hdf5(dat, outfile)


def write_hdf5(dat: T.Dict[str, T.Any], outfn: Path):
    lxs = dat["lxs"]

    with h5py.File(outfn, "w") as h:
        for k in ["ns", "vs1", "Ts"]:
            if k not in dat:
                continue

            h.create_dataset(
                k,
                data=dat[k][1].astype(float32),
                chunks=(1, *lxs[1:], LSP),
                compression="gzip",
                compression_opts=CLVL,
            )

        for k in ["ne", "v1", "Ti", "Te", "J1", "J2", "J3", "v2", "v3"]:
            if k not in dat:
                continue

            h.create_dataset(
                k,
                data=dat[k][1].astype(float32),
                chunks=(1, *lxs[1:]),
                compression="gzip",
                compression_opts=CLVL,
            )

        if "Phitop" in dat:
            h.create_dataset(
                "Phitop", data=dat["Phitop"][1], compression="gzip", compression_opts=CLVL,
            )


if __name__ == "__main__":
    cli()
