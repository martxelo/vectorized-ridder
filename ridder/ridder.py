import numpy as np
from scipy.optimize import RootResults

_iter = 100
_xtol = 2e-12
_rtol = 4 * np.finfo(float).eps


def _update_bracket(a, b, c, x, fa, fb, fc, fx):
    """Update the bracket for vec_ridder"""

    # update bracket
    cond1 = np.sign(fc) == np.sign(fx)
    cond2 = np.sign(fa) != np.sign(fx)

    b = np.where(cond1 & cond2, x, b)
    fb = np.where(cond1 & cond2, fx, fb)
    a = np.where(cond1 & ~cond2, x, a)
    fa = np.where(cond1 & ~cond2, fx, fa)
    
    b = np.where(~cond1, x, b)
    fb = np.where(~cond1, fx, fb)
    a = np.where(~cond1, c, a)
    fa = np.where(~cond1, fc, fa)

    return a, b, c, x, fa, fb, fc, fx


def vec_ridder(f, a, b, args=(),
               xtol=_xtol, rtol=_rtol, maxiter=_iter,
               full_output=False, disp=True):
    """
    Find the roots for a vector function using Ridder's method

    Parameters
    ----------
    f : function
        Python function returning a number or a 1D array. f must be continuous, and f(a) and
        f(b) must have opposite signs.
    a : ndarray, shape(n,)
        One end of the bracketing interval [a,b].
    b : ndarray, shape(n,)
        The other end of the bracketing interval [a,b].
    xtol : number, optional
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter must be positive.
    rtol : number, optional
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter cannot be smaller than its default value of
        ``4*np.finfo(float).eps``.
    maxiter : int, optional
        If convergence is not achieved in `maxiter` iterations, an error is
        raised. Must be >= 0.
    args : tuple, optional
        Extra arguments to be used in the function call.
    full_output : bool, optional
        If `full_output` is False, the root is returned. If `full_output` is
        True, the return value is ``(x, r)``, where `x` is the root, and `r` is
        a `RootResults` object.
    disp : bool, optional
        If True, raise RuntimeError if the algorithm didn't converge.
        Otherwise, the convergence status is recorded in any `RootResults`
        return object.

    Returns
    -------
    root : float
        Root of `f` between `a` and `b`.
    r : `RootResults` (present if ``full_output = True``)
        Object containing information about the convergence.
        In particular, ``r.converged`` is True if the routine converged.

    See Also
    --------
    brentq, brenth, bisect, newton : 1-D root-finding
    fixed_point : scalar fixed-point finder

    Notes
    -----
    Uses [Ridders1979]_ method to find a root of the function `f` between the
    arguments `a` and `b`. Ridders' method is faster than bisection, but not
    generally as fast as the Brent routines. [Ridders1979]_ provides the
    classic description and source of the algorithm. A description can also be
    found in any recent edition of Numerical Recipes.

    The routine used here diverges slightly from standard presentations in
    order to be a bit more careful of tolerance.

    References
    ----------
    .. [Ridders1979]
       Ridders, C. F. J. "A New Algorithm for Computing a
       Single Root of a Real Continuous Function."
       IEEE Trans. Circuits Systems 26, 979-980, 1979.

    Examples
    --------

    >>> def f(x, params):
    ...     return (x - params) * (1 + x**2)

    >>> from ridder import vec_ridder

    >>> params = np.array([-0.5, 0.2])
    >>> a = np.array([-1, 0])
    >>> b = np.array([0, 0.3])

    >>> root = vec_ridder(f, a, b)
    >>> root
    array([-0.5,  0.2])

    """
    if xtol <= 0:
        raise ValueError("xtol too small (%g <= 0)" % xtol)
    if rtol < _rtol / 4:
        raise ValueError(f"rtol too small ({rtol:g} < {_rtol/4:g})")
    if maxiter < 1:
        raise ValueError("maxiter must be greater than 0")
    if not np.all(np.isfinite(a)):
        raise ValueError("a is not finite %s" % a)
    if not np.all(np.isfinite(b)):
        raise ValueError("b is not finite %s" % b)
    if np.any(a >= b):
        raise ValueError(f"a and b are not an interval [{a}, {b}]")

    if not isinstance(args, tuple):
        args = (args,)

    # evaluate function in bracket limits
    fa = f(a, *args)
    fb = f(b, *args)
    
    # check different sign
    if np.any((fa * fb) > 0):
        raise ValueError(f"f(a) and f(b) have the same sign")
    
    for iterations in range(maxiter):

        # middle point
        c = 0.5 * (a + b)
        fc = f(c, *args)

        # one step
        s = np.sqrt(fc**2 - fa * fb) + 1e-12
        x = c + (c - a) * np.sign(fa) * fc / s
        fx = f(x, *args)

        # update bracket
        a, b, c, x, fa, fb, fc, fx = _update_bracket(a, b, c, x,
                                                     fa, fb, fc, fx)

        # check convergence
        if np.allclose(a, b, atol=xtol, rtol=rtol):
            flag = 0
            break

    else:
        flag = -2
        if disp:
            msg = f"Failed to converge after {iterations+1} iterations"
            raise RuntimeError(msg)
    
    if full_output:
        x = 0.5 * (a + b)
        results =  RootResults(root=x,
                               iterations=iterations + 1,
                               function_calls=2*iterations + 4,
                               flag=flag, method="vec_ridder")
        return x, results
        
    return 0.5 * (a + b)


        

