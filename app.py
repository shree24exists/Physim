"""
app.py
-------
Streamlit web app for the Computational Physics Lab.
Place this file in the same folder as:
    integrators.py  projectile.py  plots.py  solver.py

Run with:  streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import sympy as sp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import io

from solver     import ProjectileSolver, parse_expr
from projectile import simulate

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Computational Physics Lab",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Dark theme CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Space+Grotesk:wght@300;400;600;700&display=swap');

  /* Base */
  html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
  }
  .stApp {
    background-color: #0d0f14;
    color: #e2e8f0;
  }

  /* Hide default streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }

  /* Top banner */
  .lab-header {
    background: linear-gradient(135deg, #0d0f14 0%, #131720 50%, #0d0f14 100%);
    border-bottom: 1px solid #1e2535;
    padding: 2rem 0 1.5rem 0;
    margin-bottom: 2rem;
    text-align: center;
  }
  .lab-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #e2e8f0;
    letter-spacing: 0.05em;
    margin: 0;
  }
  .lab-title span { color: #38bdf8; }
  .lab-subtitle {
    font-size: 0.85rem;
    color: #4a5568;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.4rem;
  }

  /* Section headers */
  .section-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    color: #38bdf8;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    border-bottom: 1px solid #1e2535;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
  }

  /* Input panel */
  .input-panel {
    background: #131720;
    border: 1px solid #1e2535;
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
  }

  /* Input labels */
  .stTextInput label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #94a3b8 !important;
    letter-spacing: 0.05em !important;
  }

  /* Input boxes */
  .stTextInput input {
    background-color: #0d0f14 !important;
    border: 1px solid #2d3748 !important;
    border-radius: 4px !important;
    color: #38bdf8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9rem !important;
  }
  .stTextInput input:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 1px #38bdf8 !important;
  }
  .stTextInput input::placeholder {
    color: #2d3748 !important;
  }

  /* Solve button */
  .stButton > button {
    background: #38bdf8 !important;
    color: #0d0f14 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
    transition: background 0.2s !important;
  }
  .stButton > button:hover {
    background: #7dd3fc !important;
  }

  /* Results table */
  .results-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    margin-bottom: 1.5rem;
  }
  .results-table th {
    color: #4a5568;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-size: 0.65rem;
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid #1e2535;
    text-align: left;
  }
  .results-table td {
    padding: 0.55rem 0.75rem;
    border-bottom: 1px solid #131720;
    color: #94a3b8;
  }
  .results-table tr:hover td { background: #131720; }
  .results-table .var-name { color: #e2e8f0; font-weight: 600; }
  .results-table .sym-expr { color: #a78bfa; }
  .results-table .num-val  { color: #34d399; font-weight: 600; }
  .results-table .free-val { color: #f59e0b; }

  /* Formula box */
  .formula-box {
    background: #131720;
    border: 1px solid #1e2535;
    border-left: 3px solid #38bdf8;
    border-radius: 4px;
    padding: 1rem 1.25rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #a78bfa;
    margin-bottom: 0.75rem;
    line-height: 1.8;
  }
  .formula-label {
    color: #4a5568;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
  }

  /* Solved-for box */
  .solved-box {
    background: #0f2027;
    border: 1px solid #164e63;
    border-radius: 6px;
    padding: 1rem 1.25rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    margin-bottom: 1.5rem;
  }
  .solved-sym  { color: #38bdf8; }
  .solved-val  { color: #34d399; }

  /* Warning / error */
  .warn-box {
    background: #1a1200;
    border: 1px solid #92400e;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #fbbf24;
    margin-bottom: 1rem;
  }
  .err-box {
    background: #1a0a0a;
    border: 1px solid #7f1d1d;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #f87171;
    margin-bottom: 1rem;
  }

  /* Hint text */
  .hint {
    font-size: 0.7rem;
    color: #2d3748;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 0.25rem;
  }

  /* Numeric summary cards */
  .metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
  }
  .metric-card {
    flex: 1;
    background: #131720;
    border: 1px solid #1e2535;
    border-radius: 6px;
    padding: 1rem;
    text-align: center;
  }
  .metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #4a5568;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
  }
  .metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: #34d399;
  }
  .metric-unit {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #4a5568;
    margin-top: 0.2rem;
  }

  /* Sweep slider */
  .stSlider label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #94a3b8 !important;
  }
  .stNumberInput label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #94a3b8 !important;
  }

  /* Divider */
  hr { border-color: #1e2535 !important; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    background: #0d0f14;
    border-bottom: 1px solid #1e2535;
    gap: 0;
  }
  .stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #4a5568 !important;
    padding: 0.6rem 1.2rem !important;
    border-bottom: 2px solid transparent !important;
  }
  .stTabs [aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
    background: transparent !important;
  }
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="lab-header">
  <div class="lab-title">⟨ <span>Computational</span> Physics Lab ⟩</div>
  <div class="lab-subtitle">Projectile Motion · Symbolic + Numeric Simulation</div>
</div>
""", unsafe_allow_html=True)


# ── Helper: collect user symbols ───────────────────────────────────────────
def collect_user_symbols(raw_inputs: dict) -> dict:
    import re
    known = set(dir(sp)) | {'sqrt','sin','cos','tan','pi','exp','log','abs','g','v','theta','t'}
    found = {}
    for val in raw_inputs.values():
        if not val:
            continue
        for tok in re.findall(r'[A-Za-z_]\w*', val):
            if tok not in known and tok not in found:
                found[tok] = sp.Symbol(tok, real=True)
    return found


# ── Helper: matplotlib figure → streamlit ─────────────────────────────────
def fig_to_st(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150,
                facecolor='#0d0f14', edgecolor='none',
                bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf


# ── Helper: styled plot ────────────────────────────────────────────────────
def make_trajectory_fig(x, y, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0d0f14')
    ax.set_facecolor('#131720')

    ax.plot(x, y, lw=2.5, color='#38bdf8', label='Trajectory')
    ax.axhline(0, color='#92400e', lw=1.2, ls='--', label='Ground')
    ax.fill_between(x, y, 0, where=(y >= 0), alpha=0.08, color='#38bdf8')

    # Mark peak
    idx_peak = np.argmax(y)
    ax.plot(x[idx_peak], y[idx_peak], 'o', color='#34d399', ms=7,
            label=f'Peak ({x[idx_peak]:.1f}, {y[idx_peak]:.1f})')

    ax.set_xlabel('Horizontal distance (m)', color='#4a5568', fontsize=10)
    ax.set_ylabel('Height (m)', color='#4a5568', fontsize=10)
    ax.set_title(title, color='#e2e8f0', fontsize=12, fontweight='bold', pad=12)
    ax.tick_params(colors='#4a5568', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#1e2535')
    ax.legend(fontsize=8, facecolor='#131720', edgecolor='#1e2535',
              labelcolor='#94a3b8')
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.grid(True, linestyle=':', alpha=0.3, color='#2d3748')
    fig.tight_layout()
    return fig


def make_velocity_fig(time, vx, vy, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0d0f14')
    ax.set_facecolor('#131720')

    speed = np.hypot(vx, vy)
    ax.plot(time, vx,    lw=2,   color='#38bdf8',  label='vx (horizontal)')
    ax.plot(time, vy,    lw=2,   color='#f87171',  label='vy (vertical)')
    ax.plot(time, speed, lw=2,   color='#34d399',  label='|v| (speed)', ls='--')
    ax.axhline(0, color='#2d3748', lw=0.8, ls=':')

    ax.set_xlabel('Time (s)', color='#4a5568', fontsize=10)
    ax.set_ylabel('Velocity (m/s)', color='#4a5568', fontsize=10)
    ax.set_title(title, color='#e2e8f0', fontsize=12, fontweight='bold', pad=12)
    ax.tick_params(colors='#4a5568', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#1e2535')
    ax.legend(fontsize=8, facecolor='#131720', edgecolor='#1e2535',
              labelcolor='#94a3b8')
    ax.grid(True, linestyle=':', alpha=0.3, color='#2d3748')
    fig.tight_layout()
    return fig


def make_sweep_fig(vals, ranges, heights, times, free_sym):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    fig.patch.set_facecolor('#0d0f14')
    sym_name = str(free_sym)

    data   = [ranges,       heights,         times]
    labels = ['Range (m)', 'Max Height (m)', 'Flight Time (s)']
    colors = ['#38bdf8',    '#f87171',        '#34d399']

    for ax, d, label, color in zip(axes, data, labels, colors):
        ax.set_facecolor('#131720')
        ax.plot(vals, d, lw=2.5, color=color)
        ax.fill_between(vals, d, alpha=0.08, color=color)
        ax.set_xlabel(sym_name, color='#4a5568', fontsize=9)
        ax.set_ylabel(label, color='#4a5568', fontsize=9)
        ax.tick_params(colors='#4a5568', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#1e2535')
        ax.grid(True, linestyle=':', alpha=0.3, color='#2d3748')

    fig.suptitle(f'Parameter Sweep — {sym_name}',
                 color='#e2e8f0', fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig


# ── Layout: two columns ────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1.6], gap="large")


# ══════════════════════════════════════════════════════
# LEFT COLUMN — Inputs
# ══════════════════════════════════════════════════════
with left_col:
    st.markdown('<div class="section-header">// Input Variables</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="hint" style="margin-bottom:1rem">Leave blank to compute. Use expressions like <code style="color:#38bdf8">2*x + 5</code> or <code style="color:#38bdf8">sqrt(g)</code></div>',
                unsafe_allow_html=True)

    VARS = [
        ('speed',       'Launch Speed',              'm/s',  '50'),
        ('angle',       'Launch Angle',              'deg',  '45'),
        ('gravity',     'Gravity',                   'm/s²', '9.81'),
        ('vx0',         'Horizontal Velocity  vx₀',  'm/s',  ''),
        ('vy0',         'Vertical Velocity  vy₀',    'm/s',  ''),
        ('range_val',   'Range',                     'm',    ''),
        ('max_height',  'Maximum Height',            'm',    ''),
        ('flight_time', 'Flight Time',               's',    ''),
    ]

    raw_inputs = {}
    cols_a, cols_b = st.columns(2)

    for i, (name, label, unit, default) in enumerate(VARS):
        col = cols_a if i % 2 == 0 else cols_b
        with col:
            val = st.text_input(
                f"{label}  ({unit})",
                value=default,
                placeholder="blank = compute",
                key=f"input_{name}",
            )
            raw_inputs[name] = val.strip() if val.strip() else None

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    solve_clicked = st.button("⟶  SOLVE + SIMULATE", key="solve_btn")

    # Hint about vx0/vy0
    st.markdown("""
    <div class="hint" style="margin-top:0.75rem">
      ⚠ Don't mix speed/angle with vx₀/vy₀ — use one group or the other.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# RIGHT COLUMN — Results
# ══════════════════════════════════════════════════════
with right_col:
    if not solve_clicked:
        st.markdown("""
        <div style="
          height: 300px;
          display: flex;
          align-items: center;
          justify-content: center;
          border: 1px dashed #1e2535;
          border-radius: 8px;
          color: #2d3748;
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.8rem;
          letter-spacing: 0.1em;
        ">
          AWAITING INPUT →
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Parse inputs ───────────────────────────────────────────────
        user_syms = collect_user_symbols(raw_inputs)
        parsed = {}
        parse_errors = []
        for name, val in raw_inputs.items():
            if val is None:
                parsed[name] = None
            else:
                try:
                    parsed[name] = parse_expr(val, user_syms)
                except ValueError as e:
                    parse_errors.append(f"{name}: {e}")
                    parsed[name] = None

        if parse_errors:
            for pe in parse_errors:
                st.markdown(f'<div class="err-box">⚠ Parse error — {pe}</div>',
                            unsafe_allow_html=True)

        # ── Detected symbols ───────────────────────────────────────────
        if user_syms:
            syms_str = ',  '.join(
                f'<span class="solved-sym">{k}</span>' for k in user_syms
            )
            st.markdown(
                f'<div class="solved-box">Detected symbols: {syms_str}</div>',
                unsafe_allow_html=True)

        # ── Run solver ─────────────────────────────────────────────────
        solver = ProjectileSolver(parsed, user_syms)
        result = solver.solve()
        sym    = result['symbolic']
        num    = result['numeric']

        # ── Warnings ───────────────────────────────────────────────────
        for w in result['warnings']:
            st.markdown(f'<div class="warn-box">{w}</div>',
                        unsafe_allow_html=True)

        # ── Solved-for ─────────────────────────────────────────────────
        if result['solved_for']:
            lines = []
            for sym_obj, val in result['solved_for'].items():
                try:
                    nv = float(val.evalf())
                    lines.append(
                        f'<span class="solved-sym">{sym_obj}</span>'
                        f'  =  {sp.sstr(val)}'
                        f'  ≈  <span class="solved-val">{nv:.6g}</span>'
                    )
                except Exception:
                    lines.append(
                        f'<span class="solved-sym">{sym_obj}</span>'
                        f'  =  {sp.sstr(val)}'
                    )
            st.markdown(
                '<div class="solved-box">'
                '<div class="formula-label">Solved for</div>'
                + '<br>'.join(lines) +
                '</div>',
                unsafe_allow_html=True)

        # ── Tabs ───────────────────────────────────────────────────────
        tab_sym, tab_plots, tab_sweep = st.tabs([
            "📐  Symbolic",
            "📊  Plots",
            "🔁  Sweep",
        ])

        # ────────────────────────────────
        # TAB 1 — Symbolic
        # ────────────────────────────────
        with tab_sym:
            # Trajectory formulas
            st.markdown('<div class="section-header">// Trajectory Functions</div>',
                        unsafe_allow_html=True)
            x_t_str = sp.pretty(sym['x_t'], use_unicode=True)
            y_t_str = sp.pretty(sym['y_t'], use_unicode=True)
            st.markdown(f"""
            <div class="formula-box">
              <div class="formula-label">Position</div>
              x(t) = {sp.sstr(sym['x_t'])}<br>
              y(t) = {sp.sstr(sym['y_t'])}
            </div>
            """, unsafe_allow_html=True)

            # Variable table
            st.markdown('<div class="section-header" style="margin-top:1rem">// All Variables</div>',
                        unsafe_allow_html=True)

            TABLE_VARS = [
                ('speed',       'Speed',       'm/s'),
                ('angle',       'Angle',       '°'),
                ('gravity',     'Gravity',     'm/s²'),
                ('vx0',         'vx₀',         'm/s'),
                ('vy0',         'vy₀',         'm/s'),
                ('range_val',   'Range',       'm'),
                ('max_height',  'Max Height',  'm'),
                ('flight_time', 'Flight Time', 's'),
            ]

            rows = ""
            for key, label, unit in TABLE_VARS:
                expr    = sym[key]
                num_val = num[key]
                expr_s  = sp.sstr(expr)
                if len(expr_s) > 32:
                    expr_s = expr_s[:32] + '…'

                if num_val is not None:
                    num_s = f'<span class="num-val">{num_val:.5g}</span>'
                else:
                    free = ', '.join(str(s) for s in expr.free_symbols)
                    num_s = f'<span class="free-val">free ({free})</span>'

                rows += f"""
                <tr>
                  <td class="var-name">{label}</td>
                  <td style="color:#4a5568;font-size:0.7rem">{unit}</td>
                  <td class="sym-expr">{expr_s}</td>
                  <td>{num_s}</td>
                </tr>"""

            st.markdown(f"""
            <table class="results-table">
              <thead>
                <tr>
                  <th>Variable</th><th>Unit</th>
                  <th>Symbolic Expression</th><th>Numeric Value</th>
                </tr>
              </thead>
              <tbody>{rows}</tbody>
            </table>
            """, unsafe_allow_html=True)

        # ────────────────────────────────
        # TAB 2 — Plots
        # ────────────────────────────────
        with tab_plots:
            req = ['speed', 'angle', 'gravity']
            missing = [k for k in req if num.get(k) is None]

            if missing:
                st.markdown(
                    f'<div class="warn-box">Cannot plot — unresolved: '
                    f'{", ".join(missing)}</div>',
                    unsafe_allow_html=True)
            else:
                speed_n   = num['speed']
                angle_n   = num['angle']
                gravity_n = num['gravity']

                if speed_n <= 0 or not (0 <= angle_n <= 90):
                    st.markdown(
                        '<div class="err-box">Speed must be > 0 and angle in [0, 90]°</div>',
                        unsafe_allow_html=True)
                else:
                    res = simulate(speed=speed_n, angle_deg=angle_n, gravity=gravity_n)

                    # Metric cards
                    st.markdown(f"""
                    <div class="metric-row">
                      <div class="metric-card">
                        <div class="metric-label">Range</div>
                        <div class="metric-value">{res.range:.2f}</div>
                        <div class="metric-unit">metres</div>
                      </div>
                      <div class="metric-card">
                        <div class="metric-label">Max Height</div>
                        <div class="metric-value">{res.max_height:.2f}</div>
                        <div class="metric-unit">metres</div>
                      </div>
                      <div class="metric-card">
                        <div class="metric-label">Flight Time</div>
                        <div class="metric-value">{res.flight_time:.2f}</div>
                        <div class="metric-unit">seconds</div>
                      </div>
                      <div class="metric-card">
                        <div class="metric-label">Peak at t</div>
                        <div class="metric-value">{res.flight_time/2:.2f}</div>
                        <div class="metric-unit">seconds (T/2)</div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    title_str = f"{speed_n:.3g} m/s @ {angle_n:.3g}°"

                    fig_t = make_trajectory_fig(res.x, res.y,
                                f"Trajectory — {title_str}")
                    st.image(fig_to_st(fig_t), use_container_width=True)

                    fig_v = make_velocity_fig(res.time, res.vx, res.vy,
                                f"Velocity — {title_str}")
                    st.image(fig_to_st(fig_v), use_container_width=True)

        # ────────────────────────────────
        # TAB 3 — Parameter Sweep
        # ────────────────────────────────
        with tab_sweep:
            free = result['free_syms'] & set(user_syms.values())

            if not free:
                st.markdown(
                    '<div class="hint" style="color:#2d3748;padding:2rem;text-align:center">'
                    'No free symbols — all variables are resolved.<br>'
                    'To sweep, leave one variable symbolic (e.g. speed = 2*x + 5) '
                    'with no range constraint.'
                    '</div>',
                    unsafe_allow_html=True)
            elif len(free) > 1:
                st.markdown(
                    f'<div class="warn-box">Multiple free symbols: '
                    f'{", ".join(str(s) for s in free)}.<br>'
                    f'Sweep requires exactly one free symbol.</div>',
                    unsafe_allow_html=True)
            else:
                free_sym = next(iter(free))
                st.markdown(
                    f'<div class="section-header">// Sweep  {free_sym}</div>',
                    unsafe_allow_html=True)

                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    lo = st.number_input(f"{free_sym} min", value=0.0, key="sw_lo")
                with sc2:
                    hi = st.number_input(f"{free_sym} max", value=100.0, key="sw_hi")
                with sc3:
                    pts = st.number_input("Points", value=50, min_value=10,
                                          max_value=500, key="sw_pts")

                if lo >= hi:
                    st.markdown(
                        '<div class="err-box">min must be less than max</div>',
                        unsafe_allow_html=True)
                else:
                    vals_arr = np.linspace(lo, hi, int(pts))
                    ranges, heights, times = [], [], []

                    for v in vals_arr:
                        sub = {free_sym: v}
                        try:
                            ranges.append( float(sym['range_val'].subs(sub).evalf()))
                            heights.append(float(sym['max_height'].subs(sub).evalf()))
                            times.append(  float(sym['flight_time'].subs(sub).evalf()))
                        except Exception:
                            ranges.append(np.nan)
                            heights.append(np.nan)
                            times.append(np.nan)

                    fig_sw = make_sweep_fig(vals_arr, ranges, heights, times, free_sym)
                    st.image(fig_to_st(fig_sw), use_container_width=True)

                    # Mini table of sweep extremes
                    valid_r = [r for r in ranges  if not np.isnan(r)]
                    valid_h = [h for h in heights if not np.isnan(h)]
                    valid_t = [t for t in times   if not np.isnan(t)]
                    if valid_r:
                        st.markdown(f"""
                        <table class="results-table" style="margin-top:1rem">
                          <thead>
                            <tr><th></th><th>Range (m)</th>
                            <th>Max Height (m)</th><th>Flight Time (s)</th></tr>
                          </thead>
                          <tbody>
                            <tr>
                              <td class="var-name">Min</td>
                              <td class="num-val">{min(valid_r):.3f}</td>
                              <td class="num-val">{min(valid_h):.3f}</td>
                              <td class="num-val">{min(valid_t):.3f}</td>
                            </tr>
                            <tr>
                              <td class="var-name">Max</td>
                              <td class="num-val">{max(valid_r):.3f}</td>
                              <td class="num-val">{max(valid_h):.3f}</td>
                              <td class="num-val">{max(valid_t):.3f}</td>
                            </tr>
                          </tbody>
                        </table>
                        """, unsafe_allow_html=True)
