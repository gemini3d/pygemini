#!/usr/bin/env python3
"""
convert Gemini3D old raw binary neutral data to HDF5 .h5
requires "simsize.dat" file to be present in the same directory at the neutral .dat files
"""

from pathlib import Path
import argparse
import h5py
import typing

import gemini3d.raw.read as raw_read

import gemini3d.write as write

p = argparse.ArgumentParser()
p.add_argument("indir", help="Gemini .dat file directory")
p.add_argument("outdir", help="directory to write HDF5 files")
P = p.parse_args()

indir = Path(P.indir).expanduser()
outdir = Path(P.outdir).expanduser()

infiles: typing.Iterable[Path]
if indir.is_file():
    infiles = [indir]
    indir = indir.parent
elif indir.is_dir():
    infiles = indir.glob("*.dat")
else:
    raise FileNotFoundError(indir)

outdir.mkdir(parents=True, exist_ok=True)

# %% convert simsize
lx = raw_read.simsize(indir)
print(f"{indir} lx:", lx)
with h5py.File(outdir / "simsize.h5", "w") as f:
    f["lx1"] = lx[0]
    f["lx2"] = lx[1]
# %% convert data
i = 0
for infile in infiles:
    if infile.stem in {"simsize", "simgrid", "initial_conditions"}:
        continue

    outfile = outdir / (f"{infile.stem}.h5")
    print(infile, "=>", outfile)
    i += 1

    dat = raw_read.neutral2(infile)

    dat["dn0all"] = dat["dn0all"].transpose()
    dat["dnN2all"] = dat["dnN2all"].transpose()
    dat["dnO2all"] = dat["dnO2all"].transpose()
    dat["dvnrhoall"] = dat["dvnrhoall"].transpose()
    dat["dvnzall"] = dat["dvnzall"].transpose()
    dat["dTnall"] = dat["dTnall"].transpose()

    write.neutral2(dat, outfile)

if i == 0:
    raise FileNotFoundError(f"no .dat files found in {indir}")

print(f"DONE: converted {i} files in {indir} to {outdir}")
