"""
integrators.py
---------------------------
Core numerical integration engine for the Computational Physics Lab.

All physics systems feed their derivative functions through here so that
integration strategy is decoupled from physics logic.  Swapping solvers
(e.g. RK45 -> DOP853) or adding event detection only ever requires
changing this one file.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, Sequence


def solve_system(
    derivative_fn: Callable[[float, Sequence[float], dict], Sequence[float]],
    t_span: tuple,
    initial_state: Sequence[float],
    parameters: dict,
    *,
    max_step: float = np.inf,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    method: str = "RK45",
    dense_output: bool = False,
    events=None,
) -> tuple:
    """Integrate a system of first-order ODEs over a time span.

    This is a thin, opinionated wrapper around ``scipy.integrate.solve_ivp``
    that enforces a consistent calling convention across all physics systems
    in the lab:

        dy/dt = derivative_fn(t, y, parameters)

    How it works (RK45 default)
    ----------------------------
    scipy's RK45 is an explicit Runge-Kutta method of order 5 with an
    embedded order-4 error estimator (the "Dormand-Prince" pair). At each
    step the solver:
      1. Evaluates the derivative at several trial points within the step.
      2. Computes a 5th-order solution and a 4th-order error estimate.
      3. Accepts or rejects the step based on rtol/atol tolerances
         and adapts the step size automatically.

    Parameters
    ----------
    derivative_fn : callable
        f(t, y, params) -> dy/dt  --  pure function, no side-effects.
    t_span : (float, float)
        (t_start, t_end) integration window [seconds].
    initial_state : array-like
        Starting state vector y0.
    parameters : dict
        Physical constants forwarded verbatim to ``derivative_fn``.
    max_step : float, optional
        Upper bound on the adaptive step size.
    rtol / atol : float, optional
        Relative and absolute error tolerances passed to the solver.
    method : str, optional
        Any solver accepted by solve_ivp ('RK45', 'DOP853', 'Radau', ...).
    dense_output : bool, optional
        If True the solver builds a continuous interpolant.
    events : callable or list, optional
        Optional event function(s) passed straight to solve_ivp.

    Returns
    -------
    t : np.ndarray, shape (N,)
        Time points chosen by the adaptive integrator.
    y : np.ndarray, shape (N, state_dim)
        State vectors at each time point -- rows are time steps,
        columns are state variables.

    Raises
    ------
    RuntimeError
        If the integrator fails to converge.
    """
    def _rhs(t: float, y: np.ndarray) -> np.ndarray:
        return derivative_fn(t, y, parameters)

    result = solve_ivp(
        _rhs,
        t_span,
        initial_state,
        method=method,
        max_step=max_step,
        rtol=rtol,
        atol=atol,
        dense_output=dense_output,
        events=events,
    )

    if not result.success:
        raise RuntimeError(
            f"Integration failed: {result.message}\n"
            f"  method={method}, t_span={t_span}, "
            f"  last t={result.t[-1] if len(result.t) else 'n/a'}"
        )

    # Transpose from (state_dim, N) to (N, state_dim)
    return result.t, result.y.T
