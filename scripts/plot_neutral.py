#!/usr/bin/env python3
"""
plot individual neutral HDF5 files
"""

from pathlib import Path
import argparse

import h5py
import matplotlib.pyplot as plt


a = argparse.ArgumentParser()
a.add_argument("file", help="neutral data file to plot")
p = a.parse_args()

file = Path(p.file).expanduser().resolve(True)

with h5py.File(file, "r") as f:
    for k in {"dn0all", "dnN2all", "dnO2all", "dvnrhoall", "dvnzall", "dTnall"}:
        fg = plt.figure(constrained_layout=True)
        ax = fg.gca()

        ax.set_title(k)
        ax.pcolormesh(f[k][:].transpose(), shading="nearest")

plt.show()
