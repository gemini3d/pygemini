import argparse
from matplotlib.pyplot import show

from . import plot_3d, plot_all, grid


def cli():

    p = argparse.ArgumentParser()
    p.add_argument("direc", help="directory to plot")
    p.add_argument("--mayavi", help="do 3D Mayavi plots", action="store_true")
    p.add_argument("-var", help="plot these variables", nargs="+")
    p.add_argument("-save", help="save plot format", default="png")
    p = p.parse_args()

    grid(p.direc)
    show()

    if p.mayavi:
        plot_3d(p.direc, p.var, saveplot_fmt=p.save)
    else:
        plot_all(p.direc, p.var, saveplot_fmt=p.save)


if __name__ == "__main__":
    cli()
