"""
plots.py
---------
Reusable Matplotlib plotting functions for the Computational Physics Lab.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_trajectory(
    x: np.ndarray,
    y: np.ndarray,
    *,
    label: str = "Trajectory",
    title: str = "Projectile Trajectory",
    ax: plt.Axes = None,
    show: bool = True,
) -> plt.Axes:
    """Plot the spatial path of a projectile.

    Parameters
    ----------
    x, y : np.ndarray
        Horizontal and vertical positions [m].
    label : str, optional
        Legend entry for this trajectory.
    title : str, optional
        Figure title.
    ax : plt.Axes, optional
        Existing axes to draw on (enables multi-curve overlays).
    show : bool, optional
        Call plt.show() at the end. Set False when composing subplots.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(x, y, lw=2, label=label)
    ax.axhline(0, color="sienna", lw=1.2, ls="--", label="Ground")
    ax.fill_between(x, y, 0, where=(y >= 0), alpha=0.08, color="steelblue")

    ax.set_xlabel("Horizontal distance (m)", fontsize=12)
    ax.set_ylabel("Height (m)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.grid(True, linestyle=":", alpha=0.5)

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_velocity(
    time: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    *,
    title: str = "Velocity Components",
    ax: plt.Axes = None,
    show: bool = True,
) -> plt.Axes:
    """Plot horizontal and vertical velocity components over time.

    Parameters
    ----------
    time : np.ndarray
        Time array [s].
    vx, vy : np.ndarray
        Horizontal and vertical velocity components [m/s].
    title : str, optional
        Figure title.
    ax : plt.Axes, optional
        Existing axes to draw on.
    show : bool, optional
        Call plt.show() at the end.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))

    speed = np.hypot(vx, vy)

    ax.plot(time, vx,    lw=2, label="vx (horizontal)", color="steelblue")
    ax.plot(time, vy,    lw=2, label="vy (vertical)",   color="tomato")
    ax.plot(time, speed, lw=2, label="|v| (speed)",     color="mediumseagreen",
            linestyle="--")
    ax.axhline(0, color="grey", lw=0.8, ls=":")

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Velocity (m/s)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)

    if show:
        plt.tight_layout()
        plt.show()

    return ax
