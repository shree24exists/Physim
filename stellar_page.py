"""
stellar_page.py
----------------
Stellar Evolution section for the Computational Physics Lab Streamlit app.
Renders inside app.py via:   from stellar_page import render_stellar_page
"""

import streamlit as st
import streamlit.components.v1 as components
from stellar_data import get_stellar_state, format_age, StellarState
import math


# ── Stage metadata ─────────────────────────────────────────────────────────
_STAGE_META = {
    'Pre-Main Sequence': {
        'color': '#a78bfa',
        'icon':  '🌱',
        'desc':  lambda s: (
            f"At {format_age(s.age)}, this {s.mass:.1f} M☉ star is still contracting "
            f"under gravity and hasn't yet ignited steady hydrogen fusion in its core. "
            f"It glows at {s.temperature:,.0f} K and will settle onto the main sequence "
            f"in roughly {format_age(s.remaining_life)}."
        ),
    },
    'Main Sequence': {
        'color': '#38bdf8',
        'icon':  '⭐',
        'desc':  lambda s: (
            f"At {format_age(s.age)}, this {s.mass:.1f} M☉ star is in the prime of its "
            f"life — steadily fusing hydrogen into helium in its core at {s.temperature:,.0f} K. "
            f"With {format_age(s.remaining_life)} left on the main sequence, "
            f"it has a radius of {s.radius:.2f} R☉."
        ),
    },
    'Subgiant': {
        'color': '#fbbf24',
        'icon':  '🌟',
        'desc':  lambda s: (
            f"At {format_age(s.age)}, this {s.mass:.1f} M☉ star has exhausted hydrogen "
            f"in its core and is beginning to expand. Its surface has cooled to {s.temperature:,.0f} K "
            f"as it swells to {s.radius:.1f} R☉. It will evolve into a red giant "
            f"in about {format_age(s.remaining_life)}."
        ),
    },
    'Red Giant': {
        'color': '#f97316',
        'icon':  '🔴',
        'desc':  lambda s: (
            f"At {format_age(s.age)}, this {s.mass:.1f} M☉ star has swelled to "
            f"{s.radius:.0f} R☉ — a red giant glowing at just {s.temperature:,.0f} K. "
            f"Its outer layers are loosely bound and will eventually be shed as a "
            f"planetary nebula, leaving a white dwarf behind in {format_age(s.remaining_life)}."
        ),
    },
    'Asymptotic Giant Branch': {
        'color': '#fb923c',
        'icon':  '🟠',
        'desc':  lambda s: (
            f"At {format_age(s.age)}, this {s.mass:.1f} M☉ star is on the asymptotic "
            f"giant branch — an enormous, cool giant of {s.radius:.0f} R☉ at {s.temperature:,.0f} K, "
            f"pulsating and shedding mass rapidly. The end is near: a white dwarf "
            f"will remain in {format_age(s.remaining_life)}."
        ),
    },
    'Red Supergiant': {
        'color': '#ef4444',
        'icon':  '💢',
        'desc':  lambda s: (
            f"At {format_age(s.age)}, this {s.mass:.1f} M☉ star is a red supergiant — "
            f"one of the largest stars known, spanning {s.radius:.0f} R☉ at {s.temperature:,.0f} K. "
            f"It is fusing progressively heavier elements in its core and will end "
            f"its life in a core collapse supernova in {format_age(s.remaining_life)}."
        ),
    },
    'Blue Supergiant': {
        'color': '#818cf8',
        'icon':  '💙',
        'desc':  lambda s: (
            f"At {format_age(s.age)}, this {s.mass:.1f} M☉ star is a luminous blue "
            f"supergiant at {s.temperature:,.0f} K, blazing with intense radiation. "
            f"Spanning {s.radius:.0f} R☉, it drives powerful stellar winds and "
            f"will explode as a supernova in {format_age(s.remaining_life)}."
        ),
    },
    'Yellow Supergiant': {
        'color': '#facc15',
        'icon':  '💛',
        'desc':  lambda s: (
            f"At {format_age(s.age)}, this {s.mass:.1f} M☉ star is passing through a "
            f"yellow supergiant phase at {s.temperature:,.0f} K across {s.radius:.0f} R☉. "
            f"This is a brief, transitional stage before a supernova "
            f"in {format_age(s.remaining_life)}."
        ),
    },
    'Luminous Blue Variable': {
        'color': '#c084fc',
        'icon':  '🔮',
        'desc':  lambda s: (
            f"At {format_age(s.age)}, this {s.mass:.1f} M☉ star is a luminous blue "
            f"variable — one of the most massive and unstable stars, prone to "
            f"violent eruptions at {s.temperature:,.0f} K. It is losing mass rapidly "
            f"and will transition to a Wolf-Rayet star in {format_age(s.remaining_life)}."
        ),
    },
    'Wolf-Rayet': {
        'color': '#e879f9',
        'icon':  '☄️',
        'desc':  lambda s: (
            f"At {format_age(s.age)}, this {s.mass:.1f} M☉ star has blown away its "
            f"outer hydrogen envelope and is now a Wolf-Rayet star — an exposed, "
            f"blazing hot core at {s.temperature:,.0f} K. It will collapse into a "
            f"black hole in {format_age(s.remaining_life)}."
        ),
    },
    'Core Collapse Supernova': {
        'color': '#fef08a',
        'icon':  '💥',
        'desc':  lambda s: (
            f"At {format_age(s.age)}, this {s.mass:.1f} M☉ star has reached the end. "
            f"Its iron core has collapsed in milliseconds, triggering a supernova "
            f"that briefly outshines an entire galaxy. The remnant will be a "
            f"neutron star or black hole."
        ),
    },
    'White Dwarf': {
        'color': '#e2e8f0',
        'icon':  '⚪',
        'desc':  lambda s: (
            f"At {format_age(s.age)}, this {s.mass:.1f} M☉ star has shed its outer "
            f"layers and left behind a white dwarf — a dense, Earth-sized remnant "
            f"at {s.temperature:,.0f} K, slowly cooling over billions of years. "
            f"No more fusion occurs; it will fade to a cold black dwarf."
        ),
    },
    'Neutron Star': {
        'color': '#7dd3fc',
        'icon':  '💠',
        'desc':  lambda s: (
            f"At {format_age(s.age)}, all that remains of this {s.mass:.1f} M☉ star "
            f"is a neutron star — a city-sized object of extraordinary density, "
            f"composed almost entirely of neutrons. It may be detectable as a "
            f"pulsar, emitting beams of radiation as it spins."
        ),
    },
    'Black Hole': {
        'color': '#6b7280',
        'icon':  '🕳️',
        'desc':  lambda s: (
            f"At {format_age(s.age)}, this {s.mass:.1f} M☉ star has collapsed into "
            f"a black hole — a region of spacetime where gravity is so extreme "
            f"that not even light can escape. Nothing remains visible; only "
            f"the gravitational imprint persists."
        ),
    },
}

def _get_stage_meta(stage: str) -> dict:
    for key in _STAGE_META:
        if key in stage:
            return _STAGE_META[key]
    return {'color': '#94a3b8', 'icon': '✨',
            'desc': lambda s: f"A {stage} phase at {format_age(s.age)}."}


# ── Star animation HTML ────────────────────────────────────────────────────

def _star_html(state: StellarState) -> str:
    """Generate animated star HTML/CSS based on stellar parameters."""
    color    = state.color_hex
    remnant  = state.is_remnant
    stage    = state.stage

    # Size: log scale, 50px (tiny WD) to 220px (supergiant)
    if remnant:
        size_px = 30
    else:
        size_px = int(np.clip(40 + 55 * math.log10(max(state.radius, 0.01) + 1), 40, 220))

    # Glow intensity scales with luminosity
    lum       = max(state.luminosity, 0.001)
    glow_px   = int(np.clip(15 + 20 * math.log10(lum + 1), 15, 80))
    glow_px2  = glow_px * 2

    # Pulse speed: faster for more luminous stars
    pulse_s   = round(max(0.8, 4.0 - 0.5 * math.log10(lum + 1)), 2)

    # Special rendering for remnants/supernova
    if 'Black Hole' in stage:
        return _black_hole_html()
    if 'Neutron Star' in stage:
        return _neutron_star_html(size_px)
    if 'Supernova' in stage:
        return _supernova_html()

    # Parse hex color for rgba
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)

    return f"""
    <div style="
      display:flex; align-items:center; justify-content:center;
      height:300px; background:#0d0f14; border-radius:8px;
      border:1px solid #1e2535;
    ">
      <div style="position:relative; display:flex;
                  align-items:center; justify-content:center;">

        <!-- Outer glow ring -->
        <div style="
          position:absolute;
          width:{size_px + glow_px2}px;
          height:{size_px + glow_px2}px;
          border-radius:50%;
          background: radial-gradient(circle,
            rgba({r},{g},{b},0.15) 0%,
            rgba({r},{g},{b},0.05) 50%,
            transparent 70%);
          animation: pulse-outer {pulse_s}s ease-in-out infinite;
        "></div>

        <!-- Mid glow -->
        <div style="
          position:absolute;
          width:{size_px + glow_px}px;
          height:{size_px + glow_px}px;
          border-radius:50%;
          background: radial-gradient(circle,
            rgba({r},{g},{b},0.3) 0%,
            rgba({r},{g},{b},0.1) 60%,
            transparent 80%);
          animation: pulse-mid {pulse_s * 0.9:.2f}s ease-in-out infinite;
        "></div>

        <!-- Star body -->
        <div style="
          position:relative;
          width:{size_px}px;
          height:{size_px}px;
          border-radius:50%;
          background: radial-gradient(circle at 35% 35%,
            #ffffff 0%,
            {color} 40%,
            rgba({r},{g},{b},0.8) 100%);
          box-shadow:
            0 0 {glow_px}px rgba({r},{g},{b},0.9),
            0 0 {glow_px * 2}px rgba({r},{g},{b},0.5),
            0 0 {glow_px * 3}px rgba({r},{g},{b},0.2);
          animation: pulse-star {pulse_s * 0.8:.2f}s ease-in-out infinite;
          z-index:2;
        "></div>
      </div>

      <style>
        @keyframes pulse-outer {{
          0%,100% {{ transform:scale(1);   opacity:0.6; }}
          50%      {{ transform:scale(1.15); opacity:1.0; }}
        }}
        @keyframes pulse-mid {{
          0%,100% {{ transform:scale(1);   opacity:0.8; }}
          50%      {{ transform:scale(1.1);  opacity:1.0; }}
        }}
        @keyframes pulse-star {{
          0%,100% {{ transform:scale(1);    }}
          50%      {{ transform:scale(1.04); }}
        }}
      </style>
    </div>
    """

def _black_hole_html() -> str:
    return """
    <div style="display:flex;align-items:center;justify-content:center;
                height:300px;background:#0d0f14;border-radius:8px;
                border:1px solid #1e2535;">
      <div style="position:relative;width:120px;height:120px;">
        <!-- Accretion disk -->
        <div style="position:absolute;top:50%;left:50%;
          transform:translate(-50%,-50%) rotateX(70deg);
          width:120px;height:120px;border-radius:50%;
          border:6px solid transparent;
          border-top:6px solid rgba(251,146,60,0.8);
          border-bottom:6px solid rgba(251,191,36,0.6);
          animation:bh-spin 2s linear infinite;"></div>
        <!-- Event horizon -->
        <div style="position:absolute;top:50%;left:50%;
          transform:translate(-50%,-50%);
          width:50px;height:50px;border-radius:50%;
          background:#000;
          box-shadow:0 0 20px rgba(251,146,60,0.6),
                     0 0 40px rgba(251,146,60,0.3);"></div>
      </div>
      <style>
        @keyframes bh-spin {
          from { transform:translate(-50%,-50%) rotateX(70deg) rotateZ(0deg); }
          to   { transform:translate(-50%,-50%) rotateX(70deg) rotateZ(360deg); }
        }
      </style>
    </div>"""

def _neutron_star_html(size_px) -> str:
    return f"""
    <div style="display:flex;align-items:center;justify-content:center;
                height:300px;background:#0d0f14;border-radius:8px;
                border:1px solid #1e2535;">
      <div style="position:relative;width:80px;height:80px;">
        <!-- Pulsar beams -->
        <div style="position:absolute;top:50%;left:50%;
          transform:translate(-50%,-50%);
          width:4px;height:120px;
          background:linear-gradient(transparent,#7dd3fc,transparent);
          animation:ns-rotate 0.5s linear infinite;transform-origin:50% 50%;"></div>
        <!-- Core -->
        <div style="position:absolute;top:50%;left:50%;
          transform:translate(-50%,-50%);
          width:20px;height:20px;border-radius:50%;
          background:radial-gradient(circle,#fff 0%,#7dd3fc 60%);
          box-shadow:0 0 15px #7dd3fc, 0 0 30px rgba(125,211,252,0.5);"></div>
      </div>
      <style>
        @keyframes ns-rotate {{
          from {{ transform:translate(-50%,-50%) rotate(0deg); }}
          to   {{ transform:translate(-50%,-50%) rotate(360deg); }}
        }}
      </style>
    </div>"""

def _supernova_html() -> str:
    return """
    <div style="display:flex;align-items:center;justify-content:center;
                height:300px;background:#0d0f14;border-radius:8px;
                border:1px solid #1e2535;overflow:hidden;">
      <div style="position:relative;width:200px;height:200px;">
        <div style="position:absolute;top:50%;left:50%;
          transform:translate(-50%,-50%);
          width:200px;height:200px;border-radius:50%;
          background:radial-gradient(circle,
            #fff 0%,#fef08a 15%,#fb923c 40%,#ef4444 65%,transparent 80%);
          animation:sn-expand 1.5s ease-out infinite;opacity:0.9;"></div>
        <div style="position:absolute;top:50%;left:50%;
          transform:translate(-50%,-50%);
          width:60px;height:60px;border-radius:50%;
          background:#fff;
          box-shadow:0 0 40px #fff,0 0 80px #fef08a;
          animation:sn-core 1.5s ease-in-out infinite;"></div>
      </div>
      <style>
        @keyframes sn-expand {
          0%   { transform:translate(-50%,-50%) scale(0.5); opacity:1; }
          100% { transform:translate(-50%,-50%) scale(1.5); opacity:0; }
        }
        @keyframes sn-core {
          0%,100% { transform:translate(-50%,-50%) scale(1); }
          50%      { transform:translate(-50%,-50%) scale(1.2); }
        }
      </style>
    </div>"""


# Fix missing numpy import
import numpy as np
def _star_html_fixed(state):
    return _star_html(state)


# ── Main render function ───────────────────────────────────────────────────

def render_stellar_page():
    """Render the full stellar evolution section."""

    st.markdown('<div class="section-header">// Stellar Evolution</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="hint" style="margin-bottom:1.5rem;color:#4a5568">
      Enter a stellar mass and an age to see the state of that star at that
      point in its life — based on stellar evolution tracks for 0.5 – 40 M☉.
    </div>
    """, unsafe_allow_html=True)

    # ── Inputs ────────────────────────────────────────────────────────────
    ic1, ic2, ic3 = st.columns([1.2, 1.5, 1])

    with ic1:
        mass = st.number_input(
            "Stellar Mass  (M☉)",
            min_value=0.5, max_value=40.0,
            value=1.0, step=0.1,
            format="%.2f",
            key="stellar_mass",
        )

    with ic2:
        age_val = st.number_input(
            "Age",
            min_value=0.0, value=4.6,
            format="%.3f",
            key="stellar_age_val",
        )

    with ic3:
        unit = st.selectbox(
            "Unit",
            ["Years", "Thousands (Kyr)", "Millions (Myr)", "Billions (Gyr)"],
            index=3,
            key="stellar_age_unit",
        )

    # Convert to years
    multipliers = {
        "Years": 1,
        "Thousands (Kyr)": 1e3,
        "Millions (Myr)":  1e6,
        "Billions (Gyr)":  1e9,
    }
    age_yr = age_val * multipliers[unit]

    simulate = st.button("⟶  SHOW STELLAR STATE", key="stellar_btn")

    if not simulate:
        st.markdown("""
        <div style="height:200px;display:flex;align-items:center;
                    justify-content:center;border:1px dashed #1e2535;
                    border-radius:8px;color:#2d3748;
                    font-family:'JetBrains Mono',monospace;font-size:0.8rem;
                    letter-spacing:0.1em;margin-top:1rem;">
          AWAITING INPUT →
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Compute ───────────────────────────────────────────────────────────
    state = get_stellar_state(mass, age_yr)
    meta  = _get_stage_meta(state.stage)

    # ── Layout: star visual | info panel ──────────────────────────────────
    vis_col, info_col = st.columns([1, 1.2], gap="large")

    with vis_col:
        components.html(_star_html(state), height=310)

        # Stage badge below star
        st.markdown(f"""
        <div style="text-align:center;margin-top:0.5rem">
          <span style="
            background:{meta['color']}22;
            border:1px solid {meta['color']}66;
            color:{meta['color']};
            font-family:'JetBrains Mono',monospace;
            font-size:0.8rem;
            font-weight:700;
            letter-spacing:0.1em;
            padding:0.3rem 1rem;
            border-radius:20px;
          ">{meta['icon']}  {state.stage.upper()}</span>
        </div>
        """, unsafe_allow_html=True)

    with info_col:
        # Stats
        def stat_row(label, value, unit_str, color='#34d399'):
            return f"""
            <div style="display:flex;justify-content:space-between;
                        align-items:baseline;padding:0.55rem 0;
                        border-bottom:1px solid #1e2535;">
              <span style="font-family:'JetBrains Mono',monospace;
                           font-size:0.7rem;color:#4a5568;
                           letter-spacing:0.1em;text-transform:uppercase">
                {label}
              </span>
              <span style="font-family:'JetBrains Mono',monospace;
                           font-size:0.95rem;font-weight:700;color:{color}">
                {value}
                <span style="font-size:0.65rem;color:#4a5568;
                             font-weight:400"> {unit_str}</span>
              </span>
            </div>"""

        stats_html = ""
        stats_html += stat_row("Mass",        f"{state.mass:.2f}",               "M☉",    '#38bdf8')
        stats_html += stat_row("Age",         format_age(state.age),              "",      '#a78bfa')
        stats_html += stat_row("Temperature", f"{state.temperature:,.0f}",        "K",     meta['color'])
        stats_html += stat_row("Radius",      f"{state.radius:.3f}",              "R☉",   '#34d399')
        stats_html += stat_row("Luminosity",  f"{state.luminosity:,.1f}",         "L☉",   '#fbbf24')

        if state.remaining_life > 0:
            stats_html += stat_row("Remaining",  format_age(state.remaining_life), "",     '#f87171')

        stats_html += stat_row("Total Life",  format_age(state.total_lifetime),   "",      '#64748b')

        st.markdown(f'<div style="margin-bottom:1rem">{stats_html}</div>',
                    unsafe_allow_html=True)

        # Description
        desc = meta['desc'](state)
        st.markdown(f"""
        <div style="
          background:#131720;
          border:1px solid #1e2535;
          border-left:3px solid {meta['color']};
          border-radius:4px;
          padding:1rem 1.25rem;
          font-family:'Space Grotesk',sans-serif;
          font-size:0.82rem;
          color:#94a3b8;
          line-height:1.7;
          margin-top:0.5rem;
        ">{desc}</div>
        """, unsafe_allow_html=True)

    # ── Lifetime progress bar ──────────────────────────────────────────────
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    pct = min(100.0, (state.age / state.total_lifetime) * 100) if state.total_lifetime > 0 else 100
    bar_color = meta['color']

    st.markdown(f"""
    <div style="margin-top:0.5rem">
      <div style="display:flex;justify-content:space-between;
                  font-family:'JetBrains Mono',monospace;
                  font-size:0.65rem;color:#4a5568;
                  letter-spacing:0.1em;margin-bottom:0.4rem">
        <span>BIRTH</span>
        <span>LIFETIME PROGRESS — {pct:.1f}%</span>
        <span>END</span>
      </div>
      <div style="background:#1e2535;border-radius:4px;height:6px;overflow:hidden">
        <div style="
          width:{pct:.1f}%;height:100%;
          background:linear-gradient(90deg,{bar_color}88,{bar_color});
          border-radius:4px;
          transition:width 0.5s ease;
        "></div>
      </div>
    </div>
    """, unsafe_allow_html=True)
