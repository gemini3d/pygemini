#!/usr/bin/env python3

"""
example of using msis_setup
"""

from datetime import datetime
from pathlib import Path
import numpy as np
import logging

import gemini3d.msis as gm


logging.basicConfig(level=logging.INFO)

tmp_path = Path("~/temp")

print("msis_setup: files under directory", tmp_path)

cfg = {
    "time": [datetime(2015, 1, 2, 12)],
    "f107": 100.0,
    "f107a": 100.0,
    "Ap": 4,
    "msis_version": 0,
}

lx = (4, 2, 3)

glon, alt, glat = np.meshgrid(
    np.linspace(-147, -145, lx[1]),
    np.linspace(100e3, 200e3, lx[0]),
    np.linspace(65, 66, lx[2]),
)
xg = {
    "glat": glat,
    "glon": glon,
    "lx": lx,
    "alt": alt,
}


atmos = gm.msis_setup(cfg, xg)

print(atmos)
