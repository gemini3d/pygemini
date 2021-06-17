import argparse

from . import plot_all, grid, input


def cli():

    p = argparse.ArgumentParser()
    p.add_argument("direc", help="directory to plot")
    p.add_argument(
        "which",
        help="which plots to make",
        choices=["all", "grid", "Efield", "precip", "input"],
        nargs="+",
        default="all",
    )
    p.add_argument("--mayavi", help="do 3D Mayavi plots", action="store_true")
    p.add_argument("-var", help="plot these variables", nargs="+")
    p.add_argument(
        "-save", help="save plot format", choices=["png", "svg", "eps", "pdf"], default="png"
    )
    p = p.parse_args()

    if p.mayavi:
        from .render import plot3d_all

        plot3d_all(p.direc, p.var, saveplot_fmt=p.save)
        return

    if "all" in p.which:
        plot_all(p.direc, p.var, saveplot_fmt=p.save)
    if "grid" in p.which:
        grid.grid(p.direc)
    if "Efield" in p.which:
        input.Efield(p.direc)
    if "precip" in p.which:
        input.precip(p.direc)
    if "input" in p.which:
        input.plot_all(p.direc, p.var, saveplot_fmt=p.save)


if __name__ == "__main__":
    cli()
