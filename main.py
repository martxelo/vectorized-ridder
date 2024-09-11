import datetime

import numpy as np
from scipy.optimize import ridder, toms748
from scipy.optimize._chandrupatla import _chandrupatla

from ridder import vec_ridder


def fun(x, params):
    """
    Dummy function to use. It is prepared to have the zero
    exactly at params, so you can check if the solution
    is correct afterwards.
    """

    return (x - params)**3 * (1 + x**2) * np.cos(x)


def main():

    # number of different roots
    n = 100_000
    print(f"calculating {n} roots")

    params = np.linspace(-0.5, 0.5, n)
    
    # fixed brackets. No need to be the same value for all indices
    a = -np.ones(n)
    b = np.ones(n)

    # Measure the time for vec_rider
    t0 = datetime.datetime.now()
    _ = vec_ridder(fun, a, b, args=(params,))
    t1 = datetime.datetime.now()
    print(f"vec_ridder time = {(t1 - t0).total_seconds():.2f}s")

    # Measure the time for chandrupatla
    t0 = datetime.datetime.now()
    _ = _chandrupatla(fun, a, b, args=(params,))
    t1 = datetime.datetime.now()
    print(f"chandrupatla time = {(t1 - t0).total_seconds():.2f}s")

    # measure the time for normal ridder many times
    a, b = -1, 1
    t0 = datetime.datetime.now()
    for i in range(n):
        _ = ridder(fun, a, b, args=(params[i],))
    t1 = datetime.datetime.now()
    print(f"non vectorized ridder = {(t1 - t0).total_seconds():.2f}s")


if __name__ == "__main__":
    main()
