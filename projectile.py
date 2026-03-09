"""
projectile.py
--------------
Projectile motion simulation under uniform gravity.

Physics model
-------------
State vector: y = [x, y, vx, vy]

    dx/dt  =  vx
    dy/dt  =  vy
    dvx/dt =  0
    dvy/dt = -g
"""

import math
from dataclasses import dataclass

import numpy as np

from integrators import solve_system


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ProjectileResult:
    """Structured result returned by ``simulate``."""
    time:  np.ndarray  # shape (N,)   -- seconds
    x:     np.ndarray  # shape (N,)   -- metres
    y:     np.ndarray  # shape (N,)   -- metres
    vx:    np.ndarray  # shape (N,)   -- m/s
    vy:    np.ndarray  # shape (N,)   -- m/s

    @property
    def range(self) -> float:
        """Horizontal distance at landing [m]."""
        return float(self.x[-1])

    @property
    def max_height(self) -> float:
        """Peak altitude reached during flight [m]."""
        return float(np.max(self.y))

    @property
    def flight_time(self) -> float:
        """Total time of flight [s]."""
        return float(self.time[-1])


# ---------------------------------------------------------------------------
# Derivative function
# ---------------------------------------------------------------------------

def _derivatives(t: float, state: np.ndarray, params: dict) -> np.ndarray:
    """First-order ODE RHS for ideal projectile motion.

    Parameters
    ----------
    t : float
        Current time (unused -- equations are autonomous).
    state : np.ndarray
        [x, y, vx, vy]
    params : dict
        Must contain 'g' (gravitational acceleration, m/s^2).

    Returns
    -------
    np.ndarray
        [dx/dt, dy/dt, dvx/dt, dvy/dt]
    """
    _x, _y, vx, vy = state
    g = params["g"]
    return np.array([vx, vy, 0.0, -g])


# ---------------------------------------------------------------------------
# Ground-hit event
# ---------------------------------------------------------------------------

def _ground_event(t: float, state: np.ndarray, params: dict) -> float:
    """Event: triggers when the projectile returns to y = 0."""
    return state[1]

_ground_event.terminal  = True
_ground_event.direction = -1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate(
    speed: float,
    angle_deg: float,
    *,
    gravity: float = 9.81,
    t_max: float = 1000.0,
    max_step: float = 0.01,
) -> ProjectileResult:
    """Simulate the trajectory of a projectile launched from the origin.

    Parameters
    ----------
    speed : float
        Launch speed [m/s]. Must be > 0.
    angle_deg : float
        Launch angle above the horizontal [degrees]. Range: [0, 90].
    gravity : float, optional
        Gravitational acceleration [m/s^2]. Default 9.81.
    t_max : float, optional
        Safety upper bound on integration time [s].
    max_step : float, optional
        Maximum step size [s] for smooth trajectory curves.

    Returns
    -------
    ProjectileResult
    """
    if speed <= 0:
        raise ValueError(f"speed must be positive, got {speed}")
    if not (0.0 <= angle_deg <= 90.0):
        raise ValueError(f"angle_deg must be in [0, 90], got {angle_deg}")

    angle_rad = math.radians(angle_deg)
    vx0 = speed * math.cos(angle_rad)
    vy0 = speed * math.sin(angle_rad)

    initial_state = np.array([0.0, 0.0, vx0, vy0])
    parameters    = {"g": gravity}

    def ground_event(t, state):
        return _ground_event(t, state, parameters)

    ground_event.terminal  = True
    ground_event.direction = -1

    time, states = solve_system(
        _derivatives,
        t_span=(0.0, t_max),
        initial_state=initial_state,
        parameters=parameters,
        max_step=max_step,
        events=ground_event,
    )

    x, y, vx, vy = states[:, 0], states[:, 1], states[:, 2], states[:, 3]

    return ProjectileResult(time=time, x=x, y=y, vx=vx, vy=vy)
