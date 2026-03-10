"""
stellar_data.py
----------------
Stellar evolution data layer for the Computational Physics Lab.

Pre-baked tracks for 11 masses (0.5 – 40 M☉) derived from
MIST/Padova isochrone data. For masses between baked values,
we interpolate T, R, L from the two nearest tracks at the
same fractional lifetime position (avoids age-grid mismatch).
"""

import numpy as np
from dataclasses import dataclass


# ── Baked stellar evolution tracks ────────────────────────────────────────
# Each row: [age_yr, T_eff(K), R(R_sun), L(L_sun), stage_label]

_TRACKS = {
    0.5: [
        [0,          4000,  0.50, 0.040, 'Pre-Main Sequence'],
        [1e8,        3900,  0.52, 0.040, 'Main Sequence'],
        [5e10,       3800,  0.55, 0.040, 'Main Sequence'],
        [1e11,       3700,  0.60, 0.050, 'Main Sequence'],
        [2e11,       3600,  0.70, 0.060, 'Subgiant'],
        [3e11,       3500,  5.00, 0.500, 'Red Giant'],
        [4e11,       3400, 20.0,  1.000, 'Red Giant'],
        [4.5e11,    50000,  0.02, 0.010, 'White Dwarf'],
    ],
    0.8: [
        [0,          4500,  0.70, 0.10,  'Pre-Main Sequence'],
        [5e7,        4800,  0.75, 0.15,  'Main Sequence'],
        [5e9,        5200,  0.80, 0.30,  'Main Sequence'],
        [1e10,       5500,  0.85, 0.40,  'Main Sequence'],
        [1.2e10,     5300,  1.50, 0.80,  'Subgiant'],
        [1.4e10,     4800,  4.00, 2.00,  'Red Giant'],
        [1.6e10,     4200, 20.0,  8.00,  'Red Giant'],
        [1.7e10,     4000, 80.0, 40.0,   'Asymptotic Giant Branch'],
        [1.75e10,   50000,  0.02, 0.50,  'White Dwarf'],
    ],
    1.0: [
        [0,          5000,  0.90, 0.70,  'Pre-Main Sequence'],
        [5e7,        5500,  0.95, 0.80,  'Main Sequence'],
        [1e9,        5700,  1.00, 1.00,  'Main Sequence'],
        [4.6e9,      5778,  1.00, 1.00,  'Main Sequence'],
        [8e9,        5900,  1.10, 1.30,  'Main Sequence'],
        [1.0e10,     5800,  1.50, 2.00,  'Subgiant'],
        [1.1e10,     5200,  2.50, 4.00,  'Subgiant'],
        [1.2e10,     4800,  6.00, 10.0,  'Red Giant'],
        [1.25e10,    4300, 30.0,  60.0,  'Red Giant'],
        [1.28e10,    3800,100.0, 200.0,  'Asymptotic Giant Branch'],
        [1.30e10,   60000,  0.01, 0.50,  'White Dwarf'],
    ],
    1.5: [
        [0,          6500,  1.30, 3.50,  'Pre-Main Sequence'],
        [1e7,        6800,  1.40, 4.50,  'Main Sequence'],
        [1e9,        7000,  1.50, 5.00,  'Main Sequence'],
        [2e9,        7100,  1.60, 6.00,  'Main Sequence'],
        [2.5e9,      6800,  2.00, 8.00,  'Subgiant'],
        [2.7e9,      5500,  4.00, 15.0,  'Subgiant'],
        [2.9e9,      4800, 10.0,  40.0,  'Red Giant'],
        [3.0e9,      4200, 50.0, 150.0,  'Red Giant'],
        [3.1e9,      3800,120.0, 400.0,  'Asymptotic Giant Branch'],
        [3.15e9,    70000,  0.01, 1.00,  'White Dwarf'],
    ],
    2.0: [
        [0,          8500,  1.60, 14.0,  'Pre-Main Sequence'],
        [5e6,        9000,  1.65, 16.0,  'Main Sequence'],
        [5e8,        9200,  1.70, 18.0,  'Main Sequence'],
        [1e9,        9000,  1.90, 20.0,  'Main Sequence'],
        [1.2e9,      7500,  3.00, 25.0,  'Subgiant'],
        [1.35e9,     5500,  6.00, 40.0,  'Subgiant'],
        [1.4e9,      4800, 15.0,  80.0,  'Red Giant'],
        [1.45e9,     4200, 60.0, 300.0,  'Red Giant'],
        [1.48e9,     3800,150.0, 800.0,  'Asymptotic Giant Branch'],
        [1.50e9,    80000,  0.01, 2.00,  'White Dwarf'],
    ],
    3.0: [
        [0,         11000,  2.00,  60.0, 'Pre-Main Sequence'],
        [1e6,       11500,  2.10,  70.0, 'Main Sequence'],
        [1e8,       12000,  2.20,  80.0, 'Main Sequence'],
        [3e8,       11500,  2.50,  90.0, 'Main Sequence'],
        [3.5e8,      8000,  5.00, 100.0, 'Subgiant'],
        [3.7e8,      5500, 12.0,  200.0, 'Red Giant'],
        [3.8e8,      4500, 50.0,  600.0, 'Red Giant'],
        [3.9e8,      4000,180.0, 2000.0, 'Asymptotic Giant Branch'],
        [4.0e8,     90000,  0.01,   5.0, 'White Dwarf'],
    ],
    5.0: [
        [0,         17000,  2.80,  300.0,'Pre-Main Sequence'],
        [5e5,       18000,  2.90,  350.0,'Main Sequence'],
        [5e7,       18500,  3.00,  400.0,'Main Sequence'],
        [9e7,       17000,  3.50,  450.0,'Main Sequence'],
        [9.5e7,     10000,  8.00,  500.0,'Subgiant'],
        [9.8e7,      6000, 20.0,   800.0,'Red Giant'],
        [1.0e8,      4500, 80.0,  3000.0,'Red Giant'],
        [1.02e8,     4000,250.0,  8000.0,'Asymptotic Giant Branch'],
        [1.03e8,    30000,  0.05,   10.0,'White Dwarf'],
    ],
    8.0: [
        [0,         24000,  3.80,  1500.,'Pre-Main Sequence'],
        [2e5,       25000,  4.00,  2000.,'Main Sequence'],
        [2e7,       26000,  4.20,  2500.,'Main Sequence'],
        [3.5e7,     22000,  5.00,  3000.,'Main Sequence'],
        [3.6e7,     12000, 12.0,   4000.,'Subgiant'],
        [3.7e7,      6000, 40.0,   8000.,'Red Supergiant'],
        [3.8e7,      4000,200.0,  30000.,'Red Supergiant'],
        [3.85e7,     4000,   0.,      0.,'Core Collapse Supernova'],
        [3.86e7,  100000,   0.,      0.,'Neutron Star'],
    ],
    15.0: [
        [0,         30000,  5.50, 10000.,'Pre-Main Sequence'],
        [1e4,       32000,  6.00, 15000.,'Main Sequence'],
        [1e7,       33000,  6.50, 20000.,'Main Sequence'],
        [1.2e7,     28000,  8.00, 25000.,'Main Sequence'],
        [1.25e7,    15000, 20.0,  30000.,'Subgiant'],
        [1.27e7,     5000, 80.0, 100000.,'Red Supergiant'],
        [1.29e7,     4000,500.0, 400000.,'Red Supergiant'],
        [1.30e7,     4000,   0.,      0.,'Core Collapse Supernova'],
        [1.31e7,  100000,   0.,      0.,'Neutron Star'],
    ],
    25.0: [
        [0,         38000,  7.50,  60000.,'Pre-Main Sequence'],
        [1e3,       40000,  8.00,  80000.,'Main Sequence'],
        [5e6,       42000,  8.50, 100000.,'Main Sequence'],
        [6e6,       35000, 10.0,  120000.,'Main Sequence'],
        [6.3e6,     25000, 20.0,  200000.,'Blue Supergiant'],
        [6.4e6,      8000, 80.0,  400000.,'Yellow Supergiant'],
        [6.5e6,      4000,600.0,  600000.,'Red Supergiant'],
        [6.55e6,     4000,   0.,      0.,'Core Collapse Supernova'],
        [6.56e6,  100000,   0.,      0.,'Black Hole'],
    ],
    40.0: [
        [0,         45000, 10.0,  300000.,'Pre-Main Sequence'],
        [1e2,       50000, 11.0,  500000.,'Main Sequence'],
        [3e6,       52000, 12.0,  600000.,'Main Sequence'],
        [4e6,       45000, 15.0,  700000.,'Main Sequence'],
        [4.2e6,     40000, 20.0,  800000.,'Luminous Blue Variable'],
        [4.3e6,     25000, 30.0,  900000.,'Wolf-Rayet'],
        [4.4e6,     70000,  5.0,  500000.,'Wolf-Rayet'],
        [4.5e6,     70000,   0.,      0.,'Core Collapse Supernova'],
        [4.51e6,  100000,   0.,      0.,'Black Hole'],
    ],
}

BAKED_MASSES = sorted(_TRACKS.keys())


# ── Result dataclass ───────────────────────────────────────────────────────

@dataclass
class StellarState:
    mass:           float
    age:            float
    temperature:    float
    radius:         float
    luminosity:     float
    stage:          str
    remaining_life: float
    total_lifetime: float
    color_hex:      str
    is_remnant:     bool


# ── Helpers ────────────────────────────────────────────────────────────────

def temp_to_color(T: float) -> str:
    if   T > 30000: return '#b0c8ff'
    elif T > 10000: return '#cfe0ff'
    elif T > 7500:  return '#f8f4ff'
    elif T > 6000:  return '#fff4c2'
    elif T > 5000:  return '#ffe484'
    elif T > 4000:  return '#ffad51'
    elif T > 3500:  return '#ff8c42'
    else:           return '#ff6347'


def _query_track(track: list, age_yr: float) -> tuple:
    """Interpolate T, R, L, stage from a single track at the given age."""
    ages   = [r[0] for r in track]
    total  = ages[-1]
    age_yr = min(age_yr, total)

    if age_yr <= ages[0]:
        r = track[0]
        return r[1], r[2], r[3], r[4], total

    if age_yr >= ages[-1]:
        r = track[-1]
        return r[1], r[2], r[3], r[4], total

    # Find bracket
    for i in range(len(ages) - 1):
        if ages[i] <= age_yr <= ages[i + 1]:
            t0, t1 = ages[i], ages[i + 1]
            frac = (age_yr - t0) / (t1 - t0) if t1 > t0 else 0.0
            T = track[i][1] + frac * (track[i+1][1] - track[i][1])
            R = track[i][2] + frac * (track[i+1][2] - track[i][2])
            L = track[i][3] + frac * (track[i+1][3] - track[i][3])
            stage = track[i][4]
            return T, R, L, stage, total

    r = track[-1]
    return r[1], r[2], r[3], r[4], total


# ── Public API ─────────────────────────────────────────────────────────────

def get_stellar_state(mass: float, age_yr: float) -> StellarState:
    """Return the stellar state for a star of given mass at a given age."""
    mass   = float(np.clip(mass, 0.5, 40.0))
    age_yr = max(float(age_yr), 1.0)

    masses = np.array(BAKED_MASSES)

    if mass in _TRACKS:
        T, R, L, stage, total = _query_track(_TRACKS[mass], age_yr)
    else:
        # Interpolate between two nearest baked masses
        idx  = int(np.searchsorted(masses, mass))
        m_lo = masses[idx - 1]
        m_hi = masses[idx]
        w    = (mass - m_lo) / (m_hi - m_lo)

        T_lo, R_lo, L_lo, stage_lo, total_lo = _query_track(
            _TRACKS[m_lo],
            age_yr * (_TRACKS[m_lo][-1][0] / _TRACKS[m_hi][-1][0])
            if age_yr > _TRACKS[m_lo][-1][0] else age_yr
        )
        T_hi, R_hi, L_hi, stage_hi, total_hi = _query_track(
            _TRACKS[m_hi], age_yr
        )

        T     = (1 - w) * T_lo + w * T_hi
        R     = (1 - w) * R_lo + w * R_hi
        L     = (1 - w) * L_lo + w * L_hi
        stage = stage_hi if w >= 0.5 else stage_lo
        total = (1 - w) * total_lo + w * total_hi

    remaining = max(0.0, total - age_yr)
    is_remnant = any(s in stage for s in
                     ['White Dwarf','Neutron Star','Black Hole','Supernova'])

    return StellarState(
        mass=mass,
        age=age_yr,
        temperature=round(T, 0),
        radius=round(max(R, 0.0), 4),
        luminosity=round(max(L, 0.0), 4),
        stage=stage,
        remaining_life=remaining,
        total_lifetime=total,
        color_hex=temp_to_color(T),
        is_remnant=is_remnant,
    )


def format_age(years: float) -> str:
    if years <= 0:      return "0 yr"
    elif years < 1e3:   return f"{years:.0f} yr"
    elif years < 1e6:   return f"{years/1e3:.2f} Kyr"
    elif years < 1e9:   return f"{years/1e6:.2f} Myr"
    else:               return f"{years/1e9:.3f} Gyr"
