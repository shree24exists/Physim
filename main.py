"""
main.py
--------
Interactive terminal interface for the Computational Physics Lab.

Run:  python main.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sympy as sp
import numpy as np
import matplotlib
matplotlib.use('TkAgg') if 'TkAgg' in matplotlib.rcsetup.all_backends else None
import matplotlib.pyplot as plt

from solver      import ProjectileSolver, parse_expr
from projectile  import simulate
from plots       import plot_trajectory, plot_velocity


# ── ANSI colours ──────────────────────────────────────────────────────────
BOLD  = "\033[1m";  RESET = "\033[0m"
CYAN  = "\033[96m"; GREEN = "\033[92m"
YELLOW= "\033[93m"; RED   = "\033[91m"
DIM   = "\033[2m"

def hdr(text):  print(f"\n{BOLD}{CYAN}{text}{RESET}")
def ok(text):   print(f"{GREEN}{text}{RESET}")
def warn(text): print(f"{YELLOW}{text}{RESET}")
def err(text):  print(f"{RED}{text}{RESET}")
def dim(text):  print(f"{DIM}{text}{RESET}")


# ── Variable metadata ─────────────────────────────────────────────────────
VARS = [
    ('speed',       'Launch speed',             'm/s',  '50'),
    ('angle',       'Launch angle',             'deg',  '45'),
    ('gravity',     'Gravitational accel.',     'm/s²', '9.81'),
    ('vx0',         'Horizontal velocity (vx₀)','m/s',  ''),
    ('vy0',         'Vertical velocity (vy₀)',  'm/s',  ''),
    ('range_val',   'Range (horizontal)',        'm',    ''),
    ('max_height',  'Maximum height',            'm',    ''),
    ('flight_time', 'Flight time',              's',    ''),
]


# ── Symbol collection pass ────────────────────────────────────────────────

def collect_user_symbols(raw_inputs: dict) -> dict:
    """
    Scan all raw strings, collect any identifiers that look like
    user-defined symbols (not known math/sympy names).

    Returns dict  name -> sympy.Symbol
    """
    known = set(dir(sp)) | {'sqrt','sin','cos','tan','pi','exp','log','abs',
                             'g','v','theta','t'}
    found = {}
    for val in raw_inputs.values():
        if val is None:
            continue
        # tokenise: grab word tokens
        import re
        for tok in re.findall(r'[A-Za-z_]\w*', val):
            if tok not in known and tok not in found:
                found[tok] = sp.Symbol(tok, real=True)
    return found


# ── Input collection ──────────────────────────────────────────────────────

def collect_inputs() -> tuple[dict, dict]:
    """
    Interactive prompt for all 8 variables.

    Returns
    -------
    raw_inputs   : dict  name -> str | None
    user_symbols : dict  name -> sympy.Symbol
    """
    hdr("━━  Computational Physics Lab  ━━  Projectile Motion  ━━")
    print(f"\n{DIM}Enter a number, a symbolic expression (e.g.  2*X + 5,  sqrt(g)),")
    print(f"or leave blank to have it computed.{RESET}")
    print(f"{DIM}Known constants available: pi, g (will be treated as symbol if")
    print(f"gravity is left blank), sqrt, sin, cos, tan, exp, log{RESET}\n")

    raw_inputs = {}
    for name, label, unit, default in VARS:
        hint = f"  [{default}]" if default else "  [blank = compute]"
        prompt = f"  {BOLD}{label}{RESET} ({unit}){hint}: "
        val = input(prompt).strip()
        if val == '' and default:
            val = default
        raw_inputs[name] = val if val else None

    user_symbols = collect_user_symbols(raw_inputs)
    if user_symbols:
        print(f"\n{CYAN}  Detected user symbols: "
              f"{', '.join(user_symbols.keys())}{RESET}")

    return raw_inputs, user_symbols


# ── Parse raw strings → sympy expressions ────────────────────────────────

def parse_inputs(raw: dict, user_syms: dict) -> dict:
    parsed = {}
    for name, val in raw.items():
        if val is None:
            parsed[name] = None
        else:
            try:
                parsed[name] = parse_expr(val, user_syms)
            except ValueError as e:
                err(f"  Parse error for '{name}': {e}")
                parsed[name] = None
    return parsed


# ── Display symbolic results ──────────────────────────────────────────────

def display_symbolic(result: dict):
    hdr("━━  Symbolic Expressions  ━━")

    sym = result['symbolic']
    num = result['numeric']
    solved = result['solved_for']

    # Show what was solved for
    if solved:
        print(f"\n  {BOLD}Solved for:{RESET}")
        for sym_obj, val in solved.items():
            try:
                num_val = float(val.evalf())
                print(f"    {CYAN}{sym_obj}{RESET} = {sp.pretty(val)}  "
                      f"≈ {GREEN}{num_val:.6g}{RESET}")
            except Exception:
                print(f"    {CYAN}{sym_obj}{RESET} = {sp.pretty(val)}")

    # Show trajectory functions
    print(f"\n  {BOLD}Trajectory functions:{RESET}")
    print(f"    x(t) = {sp.pretty(sym['x_t'])}")
    print(f"    y(t) = {sp.pretty(sym['y_t'])}")

    # Table of all variables
    hdr("━━  All Variables  ━━")
    col_w = 16
    print(f"  {'Variable':<18} {'Symbolic Expression':<38} {'Numeric Value'}")
    print(f"  {'-'*18} {'-'*38} {'-'*20}")

    display_map = [
        ('speed',       'Speed (m/s)'),
        ('angle',       'Angle (°)'),
        ('gravity',     'Gravity (m/s²)'),
        ('vx0',         'vx₀ (m/s)'),
        ('vy0',         'vy₀ (m/s)'),
        ('range_val',   'Range (m)'),
        ('max_height',  'Max Height (m)'),
        ('flight_time', 'Flight Time (s)'),
    ]

    for key, label in display_map:
        expr    = sym[key]
        num_val = num[key]
        expr_str = sp.pretty(expr, use_unicode=True)
        # truncate long symbolic expressions for the table
        if '\n' in expr_str or len(expr_str) > 36:
            expr_str = sp.sstr(expr)[:36] + '…'
        if num_val is not None:
            num_str = f"{GREEN}{num_val:.6g}{RESET}"
        else:
            free = ', '.join(str(s) for s in expr.free_symbols)
            num_str = f"{YELLOW}free ({free}){RESET}"
        print(f"  {label:<18} {expr_str:<38} {num_str}")

    # Warnings
    for w in result['warnings']:
        print()
        warn(w)


# ── Parameter sweep when one symbol remains free ─────────────────────────

def maybe_sweep(result: dict, user_syms: dict):
    """If exactly one user symbol is still free, offer a parameter sweep."""
    free = result['free_syms'] & set(user_syms.values())
    if len(free) != 1:
        return
    free_sym = next(iter(free))

    print(f"\n{CYAN}  One free symbol detected: {BOLD}{free_sym}{RESET}")
    do_sweep = input(f"  Sweep {free_sym} to see how outputs vary? [Y/n]: ").strip().lower()
    if do_sweep == 'n':
        return

    try:
        lo  = float(input(f"    {free_sym} min value: ").strip())
        hi  = float(input(f"    {free_sym} max value: ").strip())
        pts = int(input(f"    Number of points [50]: ").strip() or 50)
    except ValueError:
        err("  Invalid sweep range."); return

    vals   = np.linspace(lo, hi, pts)
    ranges, heights, times = [], [], []

    sym = result['symbolic']
    for v in vals:
        sub = {free_sym: v}
        try:
            ranges.append( float(sym['range_val'].subs(sub).evalf()))
            heights.append(float(sym['max_height'].subs(sub).evalf()))
            times.append(  float(sym['flight_time'].subs(sub).evalf()))
        except Exception:
            ranges.append(np.nan); heights.append(np.nan); times.append(np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"Parameter Sweep — {free_sym}", fontsize=13, fontweight='bold')

    for ax, data, ylabel, color in zip(
        axes,
        [ranges, heights, times],
        ['Range (m)', 'Max Height (m)', 'Flight Time (s)'],
        ['steelblue', 'tomato', 'mediumseagreen']
    ):
        ax.plot(vals, data, lw=2, color=color)
        ax.set_xlabel(str(free_sym), fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.show()


# ── Numeric simulation + plots ────────────────────────────────────────────

def run_numeric(num: dict):
    """Run the numeric simulation if all required values are resolved."""
    required = ['speed', 'angle', 'gravity']
    missing  = [k for k in required if num.get(k) is None]

    if missing:
        warn(f"\n  Skipping numeric simulation — "
             f"unresolved: {', '.join(missing)}")
        return

    speed   = num['speed']
    angle   = num['angle']
    gravity = num['gravity']

    if speed <= 0:
        err("  speed must be positive — skipping simulation."); return
    if not (0 <= angle <= 90):
        err("  angle must be in [0, 90]° — skipping simulation."); return

    hdr("━━  Numeric Simulation  ━━")
    ok(f"  Running:  speed={speed:.4g} m/s,  angle={angle:.4g}°,  "
       f"g={gravity:.4g} m/s²")

    res = simulate(speed=speed, angle_deg=angle, gravity=gravity)

    print(f"\n  {'Range':<20}: {res.range:.4f} m")
    print(f"  {'Max height':<20}: {res.max_height:.4f} m")
    print(f"  {'Flight time':<20}: {res.flight_time:.4f} s")

    plot_trajectory(res.x, res.y,
                    title=f"Trajectory — {speed:.3g} m/s @ {angle:.3g}°")
    plot_velocity(res.time, res.vx, res.vy,
                  title=f"Velocity — {speed:.3g} m/s @ {angle:.3g}°")


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    try:
        raw_inputs, user_syms = collect_inputs()
    except (KeyboardInterrupt, EOFError):
        print("\n  Exiting."); return

    parsed  = parse_inputs(raw_inputs, user_syms)
    solver  = ProjectileSolver(parsed, user_syms)

    hdr("━━  Solving  ━━")
    print("  Running symbolic solver…")
    result  = solver.solve()

    display_symbolic(result)
    maybe_sweep(result, user_syms)
    run_numeric(result['numeric'])

    hdr("━━  Done  ━━\n")


if __name__ == '__main__':
    main()
