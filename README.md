# Physim (Physics - Simulator)

# Computational Physics Lab
A Simulation-based (for now, Only) classical physics exploration tool-
A small Python project for exploring classical physics systems using numerical simulation and symbolic calculations.
It allows basic experiments such as computing trajectories, solving for unknown parameters, and visualizing motion.

## Tech Stack

Python · NumPy · SciPy · Matplotlib · SymPy

## Features

* ODE integration using `scipy.integrate.solve_ivp` (RK45)
* Projectile motion simulation with ground-impact event detection
* Accepts symbolic expressions as inputs (e.g. `2*x + 5`)
* Uses SymPy to solve for unknown variables when enough constraints are given
* Displays both symbolic formulas and evaluated numeric values
* Simple terminal interface where variables can be left blank to compute them
* Basic parameter sweep plotting when one symbol remains free
* Trajectory and velocity plots using Matplotlib

## Systems Implemented

* [x] Projectile Motion
* [ ] Pendulum (planned)
* [ ] Spring-Mass (planned)
* [ ] Orbital Motion (planned)
* [ ] QFT/Quantum Related Simulations (Maybe)

## File Structure

integrators.py  — numerical ODE integration wrapper around solve_ivp
projectile.py   — projectile motion equations and event detection
solver.py       — SymPy-based symbolic equation handling
plots.py        — plotting utilities using Matplotlib
main.py         — terminal interface for running simulations

This project was developed as a learning exercise in computational physics.
AI tools were occasionally used for coding,suggestions and debugging during development.
