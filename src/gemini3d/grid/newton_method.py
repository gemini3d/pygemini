"""
Module containing root finding functions based on Newtons' method.
These adaptations meant for dipole to spherical coordinate transformation
"""

from __future__ import annotations

from .convert import calc_theta, Re
from .convert import objfunr as f, objfunr_derivative as fprime


def qp2rtheta(q: float, p: float, verbose: bool = False) -> tuple[float, float]:
    """
    Convert q,p to r,theta coordinates, apply sensible bounding (for dipole conversion problem)
    and restart on root finding
    """

    tol = 1e-9
    maxit = 100

    r = 0.0
    converged = False
    ir0 = 0
    parms = (q, p)

    while (not converged) and ir0 < 400 and (r <= 0 or r > 100 * Re):
        # repeat with different starting points until converged
        # may need to verify we get the correct root?
        r0 = ir0 * (0.25 * Re)
        r, it, converged = newton_exact(f, fprime, r0, parms, maxit, tol, verbose)
        ir0 += 1

    theta = calc_theta(r, parms)

    return r, theta


def newton_exact(
    f, fprime, x0: float, parms: tuple[float, float], maxit: int, tol: float, verbose: bool = False
) -> tuple[float, int, bool]:
    """
    1D Newton solver, tweaked for dipole coordinate conversion problem, parms arg
    corresponds to fixed (vs. iterations) function parameters
    """

    derivtol = 1e-18

    # Guard against starting at an inflection point
    if abs(fprime(x0, parms)) < derivtol:
        raise ValueError("starting near inflection point, please change initial guess!")

    # Newton iteration main loop
    it = 1
    root = x0
    fval = f(root, parms)
    converged = False
    while not converged and it <= maxit:
        derivative = fprime(root, parms)
        if abs(derivative) < derivtol:
            raise ValueError(
                "derivative near zero, terminating iterations with failure"
                "to converge (try a different starting point)!"
            )
            return root, it, converged
        else:
            root = root - fval / derivative
            fval = f(root, parms)
            if verbose:
                print(
                    "Iteration ", it, "; root ", root, "; fval ", fval, "; derivative ", derivative
                )
            it += 1
            converged = abs(fval) < tol
    it -= 1

    return root, it, converged
