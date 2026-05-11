#!/usr/bin/env python3
# process_eop_data.py
# Apply leap-second and solid-Earth tide corrections to EOP data.
#
# Usage:
#   python process_eop_data.py                           # default: C04 all-column format
#   python process_eop_data.py ../data/EOP/ori/EOP_14_C04_1962_2025.csv  # standard format

import os
import sys
import bisect
import math
import numpy as np
import pandas as pd
from pathlib import Path

# ---------- Paths ----------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent
# --------------------------

# ==================== 1. Leap-second history (IERS) ====================
# Format: (MJD, TAI-UTC cumulative seconds)
# Source: IERS Bulletin C
LEAP_SECOND_HISTORY = [
    (41317, 10), (41499, 11), (41683, 12), (42048, 13),
    (42413, 14), (42778, 15), (43144, 16), (43509, 17),
    (43874, 18), (44239, 19), (44786, 20), (45151, 21),
    (45516, 22), (46247, 23), (47161, 24), (47892, 25),
    (48257, 26), (48804, 27), (49169, 28), (49534, 29),
    (50083, 30), (50630, 31), (51179, 32), (53736, 33),
    (54832, 34), (56109, 35), (57204, 36), (57754, 37),
]


def get_leap_seconds_full(mjd):
    """Return TAI-UTC difference for a given MJD."""
    if mjd < 41317:
        return 10
    keys = [x[0] for x in LEAP_SECOND_HISTORY]
    return LEAP_SECOND_HISTORY[bisect.bisect_right(keys, mjd) - 1][1]


# ==================== 2. Astronomical functions ====================

def fundarg(t):
    """Compute fundamental lunar-solar arguments (radians)."""
    l_deg = (134.96340251 + (1717915923.2178 * t + 31.8792 * t ** 2 + 0.051635 * t ** 3 - 0.00024470 * t ** 4) / 3600)
    l = math.fmod(l_deg, 360) * math.pi / 180.0

    lp_deg = (357.52910918 + (129596581.0481 * t - 0.5532 * t ** 2 + 0.000136 * t ** 3 - 0.00001149 * t ** 4) / 3600)
    lp = math.fmod(lp_deg, 360) * math.pi / 180.0

    f_deg = (93.27209062 + (1739527262.8478 * t - 12.7512 * t ** 2 - 0.001037 * t ** 3 + 0.00000417 * t ** 4) / 3600)
    f = math.fmod(f_deg, 360) * math.pi / 180.0

    d_deg = (297.85019547 + (1602961601.2090 * t - 6.3706 * t ** 2 + 0.006593 * t ** 3 - 0.00003169 * t ** 4) / 3600)
    d = math.fmod(d_deg, 360) * math.pi / 180.0

    om_deg = (125.04455501 + (-6962890.5431 * t + 7.4722 * t ** 2 + 0.007702 * t ** 3 - 0.00005939 * t ** 4) / 3600)
    om = math.fmod(om_deg, 360) * math.pi / 180.0

    for angle in [l, lp, f, d, om]:
        if angle < 0:
            angle += 2 * math.pi
    return l, lp, f, d, om


def rg_zont2(t):
    """Compute zonal tide correction (IERS 2010, 62 terms)."""
    d2pi = 2 * math.pi
    nzont = 62

    nfund = [
        [1, 0, 2, 2, 2], [2, 0, 2, 0, 1], [2, 0, 2, 0, 2], [0, 0, 2, 2, 1], [0, 0, 2, 2, 2],
        [1, 0, 2, 0, 0], [1, 0, 2, 0, 1], [1, 0, 2, 0, 2], [3, 0, 0, 0, 0], [-1, 0, 2, 2, 1],
        [-1, 0, 2, 2, 2], [1, 0, 0, 2, 0], [2, 0, 2, -2, 2], [0, 1, 2, 0, 2], [0, 0, 2, 0, 0],
        [0, 0, 2, 0, 1], [0, 0, 2, 0, 2], [2, 0, 0, 0, -1], [2, 0, 0, 0, 0], [2, 0, 0, 0, 1],
        [0, -1, 2, 0, 2], [0, 0, 0, 2, -1], [0, 0, 0, 2, 0], [0, 0, 0, 2, 1], [0, -1, 0, 2, 0],
        [1, 0, 2, -2, 1], [1, 0, 2, -2, 2], [1, 1, 0, 0, 0], [-1, 0, 2, 0, 0], [-1, 0, 2, 0, 1],
        [-1, 0, 2, 0, 2], [1, 0, 0, 0, -1], [1, 0, 0, 0, 0], [1, 0, 0, 0, 1], [0, 0, 0, 1, 0],
        [1, -1, 0, 0, 0], [-1, 0, 0, 2, -1], [-1, 0, 0, 2, 0], [-1, 0, 0, 2, 1], [1, 0, -2, 2, -1],
        [-1, -1, 0, 2, 0], [0, 2, 2, -2, 2], [0, 1, 2, -2, 1], [0, 1, 2, -2, 2], [0, 0, 2, -2, 0],
        [0, 0, 2, -2, 1], [0, 0, 2, -2, 2], [0, 2, 0, 0, 0], [2, 0, 0, -2, -1], [2, 0, 0, -2, 0],
        [2, 0, 0, -2, 1], [0, -1, 2, -2, 1], [0, 1, 0, 0, -1], [0, -1, 2, -2, 2], [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1], [1, 0, 0, -1, 0], [2, 0, -2, 0, 0], [-2, 0, 2, 0, 1], [-1, 1, 0, 1, 0],
        [0, 0, 0, 0, 2], [0, 0, 0, 0, 1]
    ]
    tide = [
        [-0.0235, 0.0000, 0.2617, 0.0000, -0.2209, 0.0000], [-0.0404, 0.0000, 0.3706, 0.0000, -0.3128, 0.0000],
        [-0.0987, 0.0000, 0.9041, 0.0000, -0.7630, 0.0000], [-0.0508, 0.0000, 0.4499, 0.0000, -0.3797, 0.0000],
        [-0.1231, 0.0000, 1.0904, 0.0000, -0.9203, 0.0000], [-0.0385, 0.0000, 0.2659, 0.0000, -0.2244, 0.0000],
        [-0.4108, 0.0000, 2.8298, 0.0000, -2.3884, 0.0000], [-0.9926, 0.0000, 6.8291, 0.0000, -5.7637, 0.0000],
        [-0.0179, 0.0000, 0.1222, 0.0000, -0.1031, 0.0000], [-0.0818, 0.0000, 0.5384, 0.0000, -0.4544, 0.0000],
        [-0.1974, 0.0000, 1.2978, 0.0000, -1.0953, 0.0000], [-0.0761, 0.0000, 0.4976, 0.0000, -0.4200, 0.0000],
        [0.0216, 0.0000, -0.1060, 0.0000, 0.0895, 0.0000], [0.0254, 0.0000, -0.1211, 0.0000, 0.1022, 0.0000],
        [-0.2989, 0.0000, 1.3804, 0.0000, -1.1650, 0.0000], [-3.1873, 0.2010, 14.6890, 0.9266, -12.3974, -0.7820],
        [-7.8468, 0.5320, 36.0910, 2.4469, -30.4606, -2.0652], [0.0216, 0.0000, -0.0988, 0.0000, 0.0834, 0.0000],
        [-0.3384, 0.0000, 1.5433, 0.0000, -1.3025, 0.0000], [0.0179, 0.0000, -0.0813, 0.0000, 0.0686, 0.0000],
        [-0.0244, 0.0000, 0.1082, 0.0000, -0.0913, 0.0000], [0.0470, 0.0000, -0.2004, 0.0000, 0.1692, 0.0000],
        [-0.7341, 0.0000, 3.1240, 0.0000, -2.6367, 0.0000], [-0.0526, 0.0000, 0.2235, 0.0000, -0.1886, 0.0000],
        [-0.0508, 0.0000, 0.2073, 0.0000, -0.1749, 0.0000], [0.0498, 0.0000, -0.1312, 0.0000, 0.1107, 0.0000],
        [0.1006, 0.0000, -0.2640, 0.0000, 0.2228, 0.0000], [0.0395, 0.0000, -0.0968, 0.0000, 0.0817, 0.0000],
        [0.0470, 0.0000, -0.1099, 0.0000, 0.0927, 0.0000], [0.1767, 0.0000, -0.4115, 0.0000, 0.3473, 0.0000],
        [0.4352, 0.0000, -1.0093, 0.0000, 0.8519, 0.0000], [0.5339, 0.0000, -1.2224, 0.0000, 1.0317, 0.0000],
        [-8.4046, 0.2500, 19.1647, 0.5701, -16.1749, -0.4811], [0.5443, 0.0000, -1.2360, 0.0000, 1.0432, 0.0000],
        [0.0470, 0.0000, -0.1000, 0.0000, 0.0844, 0.0000], [-0.0555, 0.0000, 0.1169, 0.0000, -0.0987, 0.0000],
        [0.1175, 0.0000, -0.2332, 0.0000, 0.1968, 0.0000], [-1.8236, 0.0000, 3.6018, 0.0000, -3.0399, 0.0000],
        [0.1316, 0.0000, -0.2587, 0.0000, 0.2183, 0.0000], [0.0179, 0.0000, -0.0344, 0.0000, 0.0290, 0.0000],
        [-0.0855, 0.0000, 0.1542, 0.0000, -0.1302, 0.0000], [-0.0573, 0.0000, 0.0395, 0.0000, -0.0333, 0.0000],
        [0.0329, 0.0000, -0.0173, 0.0000, 0.0146, 0.0000], [-1.8847, 0.0000, 0.9726, 0.0000, -0.8209, 0.0000],
        [0.2510, 0.0000, -0.0910, 0.0000, 0.0768, 0.0000], [1.1703, 0.0000, -0.4135, 0.0000, 0.3490, 0.0000],
        [-49.7174, 0.4330, 17.1056, 0.1490, -14.4370, -0.1257], [-0.1936, 0.0000, 0.0666, 0.0000, -0.0562, 0.0000],
        [0.0489, 0.0000, -0.0154, 0.0000, 0.0130, 0.0000], [-0.5471, 0.0000, 0.1670, 0.0000, -0.1409, 0.0000],
        [0.0367, 0.0000, -0.0108, 0.0000, 0.0092, 0.0000], [-0.0451, 0.0000, 0.0082, 0.0000, -0.0069, 0.0000],
        [0.0921, 0.0000, -0.0167, 0.0000, 0.0141, 0.0000], [0.8281, 0.0000, -0.1425, 0.0000, 0.1202, 0.0000],
        [-15.8887, 0.1530, 2.7332, 0.0263, -2.3068, -0.0222], [-0.1382, 0.0000, 0.0225, 0.0000, -0.0190, 0.0000],
        [0.0348, 0.0000, -0.0053, 0.0000, 0.0045, 0.0000], [-0.1372, 0.0000, -0.0079, 0.0000, 0.0066, 0.0000],
        [0.4211, 0.0000, -0.0203, 0.0000, 0.0171, 0.0000], [-0.0404, 0.0000, 0.0008, 0.0000, -0.0007, 0.0000],
        [7.8998, 0.0000, 0.1460, 0.0000, -0.1232, 0.0000], [-1617.2681, 0.0000, -14.9471, 0.0000, 12.6153, 0.0000]
    ]

    l, lp, f, d, om = fundarg(t)
    dut = 0.0
    for i in range(nzont):
        arg = math.fmod(nfund[i][0] * l + nfund[i][1] * lp + nfund[i][2] * f + nfund[i][3] * d + nfund[i][4] * om, d2pi)
        if arg < 0:
            arg += d2pi
        dut += tide[i][0] * math.sin(arg) + tide[i][1] * math.cos(arg)
    return dut * 1e-4


# ==================== 3. Main processing ====================

def _detect_and_standardize_columns(eop_df):
    """
    Detect input column naming convention and standardize to:
    Year, Month, Day, MJD, UT1-UTC
    Supports both standard format (Year/Month/Day) and IERS all-column format (YR/MM/DD).
    """
    # Already standardized
    if {'Year', 'Month', 'Day', 'MJD'}.issubset(eop_df.columns):
        return eop_df

    # IERS all-column format uses YR/MM/DD
    rename_map = {}
    if 'YR' in eop_df.columns:
        rename_map['YR'] = 'Year'
    if 'MM' in eop_df.columns:
        rename_map['MM'] = 'Month'
    if 'DD' in eop_df.columns:
        rename_map['DD'] = 'Day'

    # UT1-UTC column: try multiple common names
    ut1_candidates = ['UT1-UTC(s)', 'UT1-UTC', 'UT1R-TAI', 'UT1-TAI']
    for col in ut1_candidates:
        if col in eop_df.columns:
            rename_map[col] = 'UT1-UTC'
            break

    if rename_map:
        eop_df = eop_df.rename(columns=rename_map)

    # Ensure numeric types
    for col in ['Year', 'Month', 'Day', 'MJD', 'UT1-UTC']:
        if col in eop_df.columns:
            eop_df[col] = pd.to_numeric(eop_df[col], errors='coerce')

    return eop_df


def process_eop_data(
        eop_file=None,
        start_date='1962-01-01',
        end_date='2025-08-25'
):
    """
    Apply leap-second and solid-Earth tide corrections to EOP data.

    Output: UT1R-TAI = (UT1-UTC) - (TAI-UTC) - Tidal correction

    Args:
        eop_file: Path to input CSV. If None, uses default C04 all-column file.
        start_date: Start date filter
        end_date: End date filter
    """
    if eop_file is None:
        eop_file = DATA_DIR / 'EOP' / 'ori' / 'EOP_14_C04_1962_2025_now_all.csv'
    else:
        eop_file = Path(eop_file)

    print(f"Processing EOP file: {eop_file}")

    try:
        eop_df = pd.read_csv(eop_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {eop_file}")

    print(f"Loaded {len(eop_df)} records")
    print(f"Columns: {list(eop_df.columns)}")

    # Standardize column names
    eop_df = _detect_and_standardize_columns(eop_df)

    # Verify required columns
    required = ['Year', 'Month', 'Day', 'MJD', 'UT1-UTC']
    missing = [c for c in required if c not in eop_df.columns]
    if missing:
        raise ValueError(f"Missing required columns after standardization: {missing}")

    # Interpolate missing values
    if eop_df.isnull().any().any():
        print("Warning: missing values detected, applying linear interpolation...")
        eop_df = eop_df.interpolate(method='linear')

    # Date filtering
    eop_df['date'] = pd.to_datetime(eop_df[['Year', 'Month', 'Day']], errors='coerce')
    mask = (eop_df['date'] >= start_date) & (eop_df['date'] <= end_date)
    eop_df = eop_df[mask].copy()
    print(f"Date range: {start_date} to {end_date}, {len(eop_df)} records after filtering")

    # Apply leap-second correction
    print("Applying leap-second correction...")
    eop_df['leap_seconds'] = eop_df['MJD'].apply(get_leap_seconds_full)

    # Compute tidal correction
    print("Computing tidal correction...")
    tidal_terms = []
    for mjd in eop_df['MJD']:
        jd = mjd + 2400000.5
        t = (jd - 2451545.0) / 36525.0
        tidal_terms.append(rg_zont2(t))
    eop_df['tidal_term'] = tidal_terms

    # Core formula: UT1R-TAI = (UT1-UTC) - (TAI-UTC) - Tidal
    eop_df['UT1R-TAI'] = eop_df['UT1-UTC'] - eop_df['leap_seconds'] - eop_df['tidal_term']

    # Save
    output_dir = DATA_DIR / 'EOP' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save simplified output (UT1R-TAI only, for model input)
    simple_file = output_dir / 'eop_ut1r_tai_1962_2025_full.csv'
    eop_df[['Year', 'Month', 'Day', 'MJD', 'UT1R-TAI']].to_csv(simple_file, index=False)
    print(f"Saved simplified output: {simple_file}")

    # Also save full output if extra columns exist (LOD, x, y, etc.)
    extra_cols = [c for c in ['LOD', 'x(")', 'y(")', 'dPsi(")', 'dEps(")'] if c in eop_df.columns]
    if extra_cols:
        full_file = output_dir / 'eop_full_processed_1962_2025.csv'
        save_cols = ['Year', 'Month', 'Day', 'MJD', 'UT1-UTC', 'leap_seconds', 'tidal_term', 'UT1R-TAI'] + extra_cols
        eop_df[save_cols].to_csv(full_file, index=False)
        print(f"Saved full output: {full_file}")

    print(f"Done. Valid records: {len(eop_df)}")
    return eop_df


if __name__ == "__main__":
    # Accept optional input file path as command-line argument
    if len(sys.argv) > 1:
        process_eop_data(eop_file=sys.argv[1])
    else:
        process_eop_data()