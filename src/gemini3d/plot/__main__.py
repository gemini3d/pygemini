import argparse

from . import plot_all, grid, Efield, precip, plot_input


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
        from . import plot_3d

        plot_3d(p.direc, p.var, saveplot_fmt=p.save)
        return

    if "all" in p.which:
        plot_all(p.direc, p.var, saveplot_fmt=p.save)
    if "grid" in p.which:
        grid(p.direc)
    if "Efield" in p.which:
        Efield(p.direc)
    if "precip" in p.which:
        precip(p.direc)
    if "input" in p.which:
        plot_input(p.direc, p.var, saveplot_fmt=p.save)


if __name__ == "__main__":
    cli()
