#!/usr/bin/env python3
"""
3D translucent visualization -- can be more helpful than slices for some visualizations

https://docs.enthought.com/mayavi/mayavi/mlab_case_studies.html
https://docs.enthought.com/mayavi/mayavi/auto/mlab_decorations.html
"""

import argparse

from mayavi import mlab

import gemini3d.read


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("flag", help="output flag", type=int)
    p.add_argument("fn", help=".dat filename to load directly")
    P = p.parse_args()

    cfg = {"flagoutput": P.flag}

    var = {"ne", "v1", "Ti", "Te", "J1", "J2", "J3", "v2", "v3"}

    dat = gemini3d.read.frame(P.fn, var=var, cfg=cfg)

    mlab.USE_LOD_ACTOR = True  # this didn't help RAM or performance

    for p in var:
        if p in dat:
            mlab.pipeline.volume(mlab.pipeline.scalar_field(dat[p][1]))
            mlab.title(p)
            mlab.colorbar()
            mlab.show()
