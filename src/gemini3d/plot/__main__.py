import argparse
from matplotlib.pyplot import show

from . import plot_all, grid, precip


def cli():

    p = argparse.ArgumentParser()
    p.add_argument("direc", help="directory to plot")
    p.add_argument(
        "which",
        help="which plots to make",
        choices=["all", "grid", "precip"],
        nargs="?",
        default="all",
    )
    p.add_argument("--mayavi", help="do 3D Mayavi plots", action="store_true")
    p.add_argument("-var", help="plot these variables", nargs="+")
    p.add_argument("-save", help="save plot format", default="png")
    p = p.parse_args()

    if p.mayavi:
        from . import plot_3d

        plot_3d(p.direc, p.var, saveplot_fmt=p.save)
        return

    if p.which == "all":
        plot_all(p.direc, p.var, saveplot_fmt=p.save)
    elif p.which == "grid":
        grid(p.direc)
    elif p.which == "precip":
        precip(p.direc)

    show()


if __name__ == "__main__":
    cli()
