"""
solver.py
----------
Symbolic physics engine for the Computational Physics Lab.

Workflow
--------
1. User supplies any subset of the 7 projectile variables as numbers
   or symbolic expressions (e.g. "2*X + 5", "sqrt(g)").
2. SymPy derives closed-form expressions for ALL outputs in terms of
   whatever free symbols remain.
3. If the system is over-determined or exactly determined the engine
   solves / differentiates / integrates to reduce every expression to
   a single numerical value.
4. Returns both the symbolic formulas and the resolved numeric values
   (where possible) so the caller can run the numeric simulation.

Variables recognised
--------------------
  speed        : launch speed  [m/s]
  angle        : launch angle  [deg]
  gravity      : g             [m/s²]
  vx0          : horizontal velocity component  [m/s]
  vy0          : vertical   velocity component  [m/s]
  range_val    : horizontal range at landing    [m]
  max_height   : peak altitude                  [m]
  flight_time  : total time of flight           [s]
"""

import sympy as sp
from sympy import symbols, sqrt, sin, cos, tan, pi, solve, simplify, Rational
from typing import Optional


# ---------------------------------------------------------------------------
# Global symbols used in physics expressions
# ---------------------------------------------------------------------------
t_sym  = symbols('t',       positive=True)
g_sym  = symbols('g',       positive=True)
v_sym  = symbols('v',       positive=True)        # speed
th_sym = symbols('theta',   positive=True)        # angle in radians
vx_sym = symbols('v_x',     real=True)
vy_sym = symbols('v_y',     real=True)

# Derived closed-form expressions (ideal projectile, launched from origin)
# These are the "ground truth" symbolic physics
_ANGLE_RAD  = th_sym                              # we store angle in radians internally
_VX0_EXPR   = v_sym  * cos(_ANGLE_RAD)
_VY0_EXPR   = v_sym  * sin(_ANGLE_RAD)
_X_T        = _VX0_EXPR * t_sym                                    # x(t)
_Y_T        = _VY0_EXPR * t_sym - Rational(1,2)*g_sym*t_sym**2    # y(t)
_FLIGHT_T   = 2 * _VY0_EXPR / g_sym                               # T
_RANGE_EXPR = simplify(_X_T.subs(t_sym, _FLIGHT_T))               # R
_HMAX_EXPR  = simplify(_VY0_EXPR**2 / (2*g_sym))                  # H


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _make_namespace() -> dict:
    """Return a safe eval namespace with math + sympy symbols."""
    import sympy as _sp
    ns = {k: getattr(_sp, k) for k in dir(_sp) if not k.startswith('_')}
    ns.update({
        'sqrt': sp.sqrt, 'sin': sp.sin, 'cos': sp.cos,
        'tan': sp.tan,   'pi':  sp.pi,  'exp': sp.exp,
        'log': sp.log,   'abs': sp.Abs,
    })
    return ns


def parse_expr(raw: str, extra_syms: dict) -> Optional[sp.Expr]:
    """
    Parse a user string into a SymPy expression.

    Parameters
    ----------
    raw        : string typed by the user, e.g. "2*X + 5" or "45"
    extra_syms : dict mapping name -> sympy Symbol for user-defined symbols

    Returns
    -------
    SymPy expression, or None if the string is blank.
    """
    raw = raw.strip()
    if not raw:
        return None
    ns = _make_namespace()
    ns.update(extra_syms)
    try:
        return sp.sympify(raw, locals=ns)
    except Exception as e:
        raise ValueError(f"Cannot parse '{raw}': {e}")


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

class ProjectileSolver:
    """
    Symbolic solver for ideal projectile motion.

    After construction, call ``.solve()`` which:
      - substitutes user values into the physics expressions
      - attempts to resolve all remaining free symbols numerically
      - returns a result dict with symbolic + numeric forms of every variable
    """

    # Names exposed to the user
    VAR_NAMES = ['speed', 'angle', 'gravity', 'vx0', 'vy0',
                 'range_val', 'max_height', 'flight_time']

    def __init__(self, inputs: dict, user_symbols: dict):
        """
        Parameters
        ----------
        inputs       : dict  name -> SymPy expression (or None if blank)
        user_symbols : dict  name -> sympy.Symbol  for user-defined unknowns
        """
        self.inputs       = inputs          # raw user expressions
        self.user_symbols = user_symbols    # e.g. {'X': Symbol('X')}

    # ------------------------------------------------------------------
    # Internal: build the equation system
    # ------------------------------------------------------------------

    def _build_equations(self, subs: dict) -> list:
        """
        Build constraint equations by equating user-supplied expressions
        to the physics definitions.

        subs : partial substitution dict  sympy_sym -> value
        """
        eqs = []

        def _sub(expr):
            return simplify(expr.subs(subs))

        ui = self.inputs

        # --- speed / angle / gravity directly constrain v, theta, g ---
        if ui.get('speed') is not None:
            eqs.append(sp.Eq(v_sym,  ui['speed']))
        if ui.get('angle') is not None:
            # user gives degrees; convert to radians
            eqs.append(sp.Eq(th_sym, ui['angle'] * pi / 180))
        if ui.get('gravity') is not None:
            eqs.append(sp.Eq(g_sym,  ui['gravity']))

        # --- velocity components ---
        if ui.get('vx0') is not None:
            eqs.append(sp.Eq(_VX0_EXPR, ui['vx0']))
        if ui.get('vy0') is not None:
            eqs.append(sp.Eq(_VY0_EXPR, ui['vy0']))

        # --- output observables ---
        if ui.get('range_val') is not None:
            eqs.append(sp.Eq(_RANGE_EXPR, ui['range_val']))
        if ui.get('max_height') is not None:
            eqs.append(sp.Eq(_HMAX_EXPR,  ui['max_height']))
        if ui.get('flight_time') is not None:
            eqs.append(sp.Eq(_FLIGHT_T,   ui['flight_time']))

        return eqs

    # ------------------------------------------------------------------
    # Public: solve
    # ------------------------------------------------------------------

    def solve(self) -> dict:
        """
        Attempt to solve the system and return a comprehensive result dict.

        Returns
        -------
        dict with keys:
          'symbolic'  : dict  var_name -> sympy expression (simplified)
          'numeric'   : dict  var_name -> float (None if unresolved)
          'formulas'  : dict  var_name -> pretty string
          'free_syms' : set of still-unresolved symbols
          'x_t'       : symbolic x(t)
          'y_t'       : symbolic y(t)
          'solved_for': dict  symbol -> value  (what the solver found)
          'warnings'  : list of warning strings
        """
        warnings = []
        all_syms = {v_sym, th_sym, g_sym}        # core physics unknowns
        user_sym_set = set(self.user_symbols.values())

        # ── Step 1: build equations ────────────────────────────────────
        equations = self._build_equations({})

        # ── Step 2: collect all unknowns to solve for ──────────────────
        # Physics unknowns (v, theta, g) + user symbols (X, Y, ...)
        unknowns = list(all_syms | user_sym_set)

        # ── Step 3: solve the system ───────────────────────────────────
        solved_for = {}
        if equations:
            try:
                solutions = solve(equations, unknowns, dict=True, positive=True)
                if solutions:
                    # Take the first physically meaningful solution
                    for sol in solutions:
                        # Filter: speeds and g must be positive
                        valid = True
                        for sym, val in sol.items():
                            if sym in (v_sym, g_sym):
                                try:
                                    if float(val.evalf()) <= 0:
                                        valid = False
                                        break
                                except Exception:
                                    pass
                        if valid:
                            solved_for = sol
                            break
                    if not solved_for:
                        solved_for = solutions[0]
                elif len(unknowns) > 0:
                    warnings.append(
                        "⚠  No solution found. Check that your inputs are "
                        "physically consistent."
                    )
            except Exception as e:
                warnings.append(f"⚠  Solver error: {e}")

        # ── Step 4: substitute solved values into physics expressions ──
        def _resolve(expr: sp.Expr) -> sp.Expr:
            """Substitute solved_for into expr, then try to evaluate."""
            e = expr.subs(solved_for)
            return simplify(e)

        # Build symbolic results for every variable
        sym_results = {
            'speed'       : _resolve(v_sym),
            'angle'       : _resolve(th_sym * 180 / pi),   # back to degrees
            'gravity'     : _resolve(g_sym),
            'vx0'         : _resolve(_VX0_EXPR),
            'vy0'         : _resolve(_VY0_EXPR),
            'range_val'   : _resolve(_RANGE_EXPR),
            'max_height'  : _resolve(_HMAX_EXPR),
            'flight_time' : _resolve(_FLIGHT_T),
            'x_t'         : _resolve(_X_T),
            'y_t'         : _resolve(_Y_T),
        }

        # ── Step 5: attempt numeric evaluation ────────────────────────
        # x_t / y_t are trajectory functions — they always contain t,
        # which is a free parameter, not an unknown. Exclude them from
        # the "unresolved" check; check only the scalar outputs.
        SCALAR_KEYS = {'speed', 'angle', 'gravity', 'vx0', 'vy0',
                       'range_val', 'max_height', 'flight_time'}

        num_results = {}
        free_syms   = set()

        for name, expr in sym_results.items():
            fs = expr.free_symbols - {t_sym}   # t is always fine in x_t/y_t
            if name in SCALAR_KEYS:
                if fs:
                    free_syms |= fs
                    num_results[name] = None
                else:
                    try:
                        num_results[name] = float(expr.evalf())
                    except Exception:
                        num_results[name] = None
            else:
                # trajectory functions: numeric only if no non-t free symbols
                non_t = expr.free_symbols - {t_sym}
                num_results[name] = None if non_t else expr  # keep as expr for plotting

        # ── Step 6: warn about still-unresolved symbols ────────────────
        if free_syms:
            sym_names = ', '.join(str(s) for s in sorted(free_syms, key=str))
            warnings.append(
                f"⚠  Cannot fully resolve: {sym_names} is still free.\n"
                f"   Provide one more equation/value that constrains {sym_names} "
                f"to get numeric results."
            )

        # Pretty formula strings
        formulas = {k: sp.pretty(v, use_unicode=True)
                    for k, v in sym_results.items()}

        return {
            'symbolic'   : sym_results,
            'numeric'    : num_results,
            'formulas'   : formulas,
            'free_syms'  : free_syms,
            'x_t'        : sym_results['x_t'],
            'y_t'        : sym_results['y_t'],
            'solved_for' : solved_for,
            'warnings'   : warnings,
        }
