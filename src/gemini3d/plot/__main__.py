import argparse

from . import plot_all, grid, inputs


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
    p.add_argument("-var", help="plot these variables", nargs="+", default=[])
    p.add_argument(
        "-save",
        help="save plot format",
        choices=["png", "svg", "eps", "pdf"],
        default="png",
    )
    p = p.parse_args()

    var = set(p.var)

    if p.mayavi:
        from .render import plot3d_all

        plot3d_all(p.direc, var, saveplot_fmt=p.save)
        return

    if "all" in p.which:
        plot_all(p.direc, var, saveplot_fmt=p.save)
    if "grid" in p.which:
        grid.grid(p.direc)
    if "Efield" in p.which:
        inputs.Efield(p.direc)
    if "precip" in p.which:
        inputs.precip(p.direc)
    if "input" in p.which:
        inputs.plot_all(p.direc, var, saveplot_fmt=p.save)


if __name__ == "__main__":
    cli()
