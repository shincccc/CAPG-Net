#!/usr/bin/env python3
# make_component_daily.py
# Extract a single EAM (Effective Angular Momentum) axial component from raw GFZ .asc files
# and save as a daily CSV file (selecting midnight values).
#
# Usage:
#   python make_component_daily.py AAM
#   python make_component_daily.py OAM
#   python make_component_daily.py HAM
#   python make_component_daily.py SLAM
#   python make_component_daily.py EAM  (merges all four)

import os
import sys
import pandas as pd
from glob import glob
from datetime import datetime
import numpy as np

# ---------- Parameters ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.join(DATA_DIR, 'EAM', 'ori')
OUT_DIR = os.path.join(DATA_DIR, 'EAM', 'processed')
SCALE = 86400.0  # dimensionless -> ms conversion factor


# ------------------------------


def read_eam_file(path, kind):
    """
    Read a single EAM component file, preserving full MJD (including fractional hours).

    Args:
        path: Path to .asc file
        kind: One of 'AAM', 'OAM', 'HAM', 'SLAM'

    Returns:
        DataFrame with columns [MJD, comp_z]
    """
    records = []
    skipped_lines = 0
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        header_ended = False
        for line in f:
            if line.startswith('YYYY MM DD HH'):
                header_ended = True
                continue
            if header_ended:
                parts = line.strip().split()
                if kind == 'SLAM':
                    if len(parts) < 8:
                        skipped_lines += 1
                        continue
                    try:
                        mjd = float(parts[4])
                        comp_z = float(parts[7])
                    except ValueError:
                        skipped_lines += 1
                        continue
                else:  # AAM, OAM, HAM
                    if len(parts) < 11:
                        skipped_lines += 1
                        continue
                    try:
                        mjd = float(parts[4])
                        z_mass = float(parts[7])
                        z_mot = float(parts[10])
                        comp_z = z_mass + z_mot
                    except ValueError:
                        skipped_lines += 1
                        continue
                records.append((mjd, comp_z))

    if skipped_lines > 0:
        print(f"Warning: Skipped {skipped_lines} invalid lines in {path}")

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records, columns=['MJD', 'comp_z'])
    return df


def select_midnight_values(df):
    """
    Select midnight (00:00) data points from high-frequency data.
    Falls back to linear interpolation if insufficient midnight points are found.
    """
    if df.empty:
        return df

    df = df.copy()
    df['MJD_int'] = np.floor(df['MJD']).astype(int)
    df['MJD_frac'] = df['MJD'] - df['MJD_int']

    midnight_mask = df['MJD_frac'] < 0.01
    midnight_df = df[midnight_mask].copy()

    if len(midnight_df) >= 10:
        result = midnight_df[['MJD_int', 'comp_z']].copy()
        result = result.rename(columns={'MJD_int': 'MJD'})
        result = result.sort_values('MJD').drop_duplicates(subset='MJD', keep='first')
        return result

    print(f"  Insufficient midnight data ({len(midnight_df)} days), using linear interpolation...")
    epoch = datetime(1858, 11, 17)
    df['Date'] = epoch + pd.to_timedelta(df['MJD'], unit='D')
    df = df.set_index('Date').sort_index()

    full_range = pd.date_range(df.index.min(), df.index.max(), freq='3H')
    df_resampled = df.reindex(full_range)
    df_resampled['comp_z'] = df_resampled['comp_z'].interpolate(method='time')

    daily_midnight = df_resampled[df_resampled.index.hour == 0].copy()
    daily_midnight['MJD'] = (daily_midnight.index - epoch).total_seconds() / 86400.0

    result = daily_midnight[['MJD', 'comp_z']].reset_index(drop=True)
    result['MJD'] = result['MJD'].astype(int)
    return result


def find_component_files(component_name):
    """
    Find all .asc files for a given component.
    """
    comp_dir = os.path.join(ROOT_DIR, component_name)

    if not os.path.isdir(comp_dir):
        raise RuntimeError(f'Directory not found: {comp_dir}')

    all_files = os.listdir(comp_dir)
    asc_files = sorted([f for f in all_files if f.endswith('.asc')])

    if not asc_files:
        raise RuntimeError(f'No .asc files found in {comp_dir}')

    files = [os.path.join(comp_dir, f) for f in asc_files]
    print(f"Found {len(files)} .asc file(s) for {component_name}")
    for f in files:
        print(f"  - {os.path.basename(f)}")
    return files


def build_daily_component(component_name):
    """
    Build daily CSV for a single EAM component.

    Args:
        component_name: One of 'AAM', 'OAM', 'HAM', 'SLAM'
    """
    files = find_component_files(component_name)
    print(f"Processing {len(files)} files for {component_name}...")

    comp_frames = [read_eam_file(f, component_name) for f in files]
    daily_raw = pd.concat(comp_frames, ignore_index=True)

    if daily_raw.empty:
        raise RuntimeError(f'No valid records extracted for {component_name}')

    # Select midnight values
    daily_processed = select_midnight_values(daily_raw)

    # Generate date columns
    epoch = datetime(1858, 11, 17)
    daily_processed['Date'] = epoch + pd.to_timedelta(daily_processed['MJD'], unit='D')
    daily_processed['Year'] = daily_processed['Date'].dt.year
    daily_processed['Month'] = daily_processed['Date'].dt.month
    daily_processed['Day'] = daily_processed['Date'].dt.day

    # Scale to milliseconds
    col_ms = f'{component_name}_ms'
    daily_processed[col_ms] = daily_processed['comp_z'] * SCALE * 1000

    final_cols = ['Year', 'Month', 'Day', 'MJD', col_ms]
    daily_processed = daily_processed[final_cols].sort_values(['Year', 'Month', 'Day']).reset_index(drop=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_file = os.path.join(OUT_DIR, f'{component_name}_daily_1976_present.csv')
    daily_processed.to_csv(out_file, index=False, float_format='%.8f')
    print(f'Generated {out_file}, {len(daily_processed)} days.')


def build_daily_eam():
    """
    Build Total EAM by summing all four components.
    """
    components = ['AAM', 'OAM', 'HAM', 'SLAM']
    frames = []

    for comp in components:
        files = find_component_files(comp)
        comp_frames = [read_eam_file(f, comp) for f in files]
        comp_df = pd.concat(comp_frames, ignore_index=True)
        comp_df_processed = select_midnight_values(comp_df)
        comp_df_processed['comp'] = comp
        frames.append(comp_df_processed)

    if not frames:
        raise RuntimeError('No .asc files found for any component.')

    big = pd.concat(frames)
    # Sum four components by MJD
    daily = big.groupby('MJD')['comp_z'].sum().rename('EAM_raw').reset_index()
    daily['EAM_ms'] = daily['EAM_raw'] * SCALE * 1000

    epoch = datetime(1858, 11, 17)
    daily['Date'] = epoch + pd.to_timedelta(daily['MJD'], unit='D')
    daily['Year'] = daily['Date'].dt.year
    daily['Month'] = daily['Date'].dt.month
    daily['Day'] = daily['Date'].dt.day

    daily = daily[['Year', 'Month', 'Day', 'MJD', 'EAM_ms']] \
        .sort_values(['Year', 'Month', 'Day']) \
        .reset_index(drop=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_file = os.path.join(OUT_DIR, 'EAM_daily_1976_present.csv')
    daily.to_csv(out_file, index=False, float_format='%.8f')
    print(f'Generated {out_file}, {len(daily)} days.')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python make_component_daily.py <COMPONENT>")
        print("  COMPONENT: AAM, OAM, HAM, SLAM, or EAM (to merge all four)")
        sys.exit(1)

    component = sys.argv[1].upper()
    if component not in ('AAM', 'OAM', 'HAM', 'SLAM', 'EAM'):
        print(f"Error: Unknown component '{component}'. Must be AAM, OAM, HAM, SLAM, or EAM.")
        sys.exit(1)

    if component == 'EAM':
        build_daily_eam()
    else:
        build_daily_component(component)