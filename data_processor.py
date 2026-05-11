# data_processor.py
import numpy as np
import pandas as pd


class DataProcessor:
    """
    1. Load EOP (UT1) from C04 series
    2. Load AAM, OAM, HAM, SLAM and merge into Total EAM
    3. Perform physics-based decoupling: LOD_core = LOD_obs - Total_EAM
    4. Align ENSO data
    """

    def __init__(self, config):
        self.config = config

    def load_all_data(self):

        print("📥 Loading and preprocessing multi-source data (DataProcessor V68)...")

        # 1. Load EOP (C04 only)
        eop_df = self._load_eop_data()

        # 2. Load and merge excitation functions (AAM + OAM + HAM + SLAM)
        excitation_df = self._load_and_merge_excitations()

        # 3. Load ENSO data
        nino_df = self._load_nino_data(eop_df['date'].min(), eop_df['date'].max())

        # 4. Align time ranges of the three datasets
        eop_df, nino_df, excitation_df = self._align_data_ranges(
            eop_df, nino_df, excitation_df
        )

        # 5. Perform physical decoupling
        print("⚡ Performing physical decoupling: LOD_core = LOD_obs - Total_EAM ...")
        eop_df = self._compute_physics_derived_features(eop_df, excitation_df)

        print(f"✅ Data preprocessing complete. Samples: {len(eop_df)}")

        return eop_df, nino_df, excitation_df

    def _align_data_ranges(self, eop_df, nino_df, ex_df):
        """
        Align time indices of EOP, Nino, and Excitation datasets.
        """
        # 1. Determine common time window
        start_dates = [eop_df['date'].min(), nino_df.index.min(), ex_df.index.min()]
        end_dates = [eop_df['date'].max(), nino_df.index.max(), ex_df.index.max()]

        common_start = max(d for d in start_dates if pd.notna(d))
        common_end = min(d for d in end_dates if pd.notna(d))

        # Buffer for numerical differentiation
        calc_start = common_start - pd.Timedelta(days=7)

        # 2. Slice EOP
        eop_df = eop_df[(eop_df['date'] >= calc_start) & (eop_df['date'] <= common_end)].copy()

        # 3. Align other data to EOP dates
        nino_df = nino_df.loc[nino_df.index.isin(eop_df['date'])]
        ex_df = ex_df.loc[ex_df.index.isin(eop_df['date'])]

        # 4. Filter EOP to intersection of valid dates
        valid_dates = ex_df.index.intersection(nino_df.index)
        eop_df = eop_df[eop_df['date'].isin(valid_dates)].sort_values('date').reset_index(drop=True)

        # 5. Reindex to EOP dates (fill missing with zero)
        nino_df = nino_df.reindex(eop_df['date']).fillna(0.0)
        ex_df = ex_df.reindex(eop_df['date']).fillna(0.0)

        return eop_df, nino_df, ex_df

    def _load_eop_data(self):
        """
        Load EOP (UT1) data from C04 series.
        Standardizes column names and parses dates.
        """
        try:
            eop = pd.read_csv(self.config.EOP_FILE)
        except FileNotFoundError:
            raise FileNotFoundError(f"EOP file not found: {self.config.EOP_FILE}")

        # Normalize column names to lowercase
        eop.columns = [c.lower() for c in eop.columns]
        if 'ut1r-tai' in eop.columns:
            eop.rename(columns={'ut1r-tai': 'ut1'}, inplace=True)
        elif 'ut1-tai' in eop.columns:
            eop.rename(columns={'ut1-tai': 'ut1'}, inplace=True)

        # Parse dates
        if 'date' in eop.columns:
            eop['date'] = pd.to_datetime(eop['date'])
        else:
            eop['date'] = pd.to_datetime(eop[['year', 'month', 'day']])

        eop_df = eop[['date', 'ut1']].sort_values('date').reset_index(drop=True)
        return eop_df

    def _load_and_merge_excitations(self):
        """
        Load four independent excitation function files and merge into Total EAM.
        Components: AAM, OAM, HAM, SLAM.
        """
        components = {
            'aam': self.config.AAM_FILE,
            'oam': self.config.OAM_FILE,
            'ham': self.config.HAM_FILE,
            'slam': self.config.SLAM_FILE
        }

        merged_df = None

        for name, path in components.items():
            target_col = f"{name.upper()}_ms"
            df = self._load_single_excitation(path, target_col, name)

            if merged_df is None:
                merged_df = df
            else:
                merged_df = merged_df.join(df, how='outer')

        # Fill gaps: linear interpolation for interior, zero for boundaries
        merged_df = merged_df.interpolate(method='time').fillna(0.0)

        # Compute Total EAM
        merged_df['total_eam'] = (merged_df['aam'] + merged_df['oam'] +
                                  merged_df['ham'] + merged_df['slam'])

        return merged_df

    def _load_single_excitation(self, filepath, target_col, rename_to):
        """
        Generic excitation function loader.
        Auto-detects date format, resamples to daily resolution,
        and applies 3-day rolling mean for smoothing.

        filepath: path to CSV file
        target_col: column name in CSV (e.g. 'AAM_ms')
        rename_to: renamed column in output (e.g. 'aam')
        """
        try:
            df = pd.read_csv(filepath)

            # Auto-detect date column
            if 'Date' in df.columns:
                df['date'] = pd.to_datetime(df['Date'])
            elif 'MJD' in df.columns:
                origin = pd.Timestamp('1858-11-17')
                df['date'] = origin + pd.to_timedelta(df['MJD'], unit='D')
            else:
                df['date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

            # Locate data column
            if target_col not in df.columns:
                candidates = [c for c in df.columns if rename_to.upper() in c and 'ms' in c]
                if candidates:
                    target_col = candidates[0]
                else:
                    target_col = df.columns[-1]
                    print(f"⚠️ Warning: {filepath} missing column {target_col}, using last column {target_col}.")

            df = df.set_index('date')[[target_col]].rename(columns={target_col: rename_to})

            # Daily resampling + 3-day rolling mean smoothing
            df = df.resample('D').mean().rolling(window=3, center=True, min_periods=1).mean()

            return df

        except Exception as e:
            print(f"❌ Failed to load {rename_to.upper()} ({filepath}): {e}")
            return pd.DataFrame()

    def _load_nino_data(self, min_date, max_date):
        """
        Load daily Nino-3.4 index from NOAA OISST v2.
        No monthly extrapolation needed; eliminates data leakage risk.
        """
        try:
            nino = pd.read_csv(self.config.NINO34_FILE)
            nino['date'] = pd.to_datetime(nino['date'])
            nino = nino.set_index('date').sort_index()

            target_col = 'nino34'
            if target_col not in nino.columns:
                col_candidates = [c for c in nino.columns if 'nino' in c.lower()]
                if col_candidates:
                    target_col = col_candidates[0]
                    print(f"   ℹ️ Column 'nino34' not found, using: '{target_col}'")
                else:
                    raise KeyError(f"NINO3.4 column not found. Available columns: {nino.columns.tolist()}")

            nino_daily = nino[[target_col]].astype(float)

            print(f"   📅 ENSO daily data loaded: {len(nino_daily)} days")
            return nino_daily

        except FileNotFoundError:
            print("⚠️ Warning: ENSO file not found, filling with zeros.")
            idx = pd.date_range(min_date, max_date)
            return pd.DataFrame({'nino34': 0.0}, index=idx)

    def _compute_physics_derived_features(self, eop_df, ex_df):
        """
        Compute physics-derived features.
        Core equation: LOD_core = LOD_obs - Total_EAM
        """
        # 1. Compute observed LOD from UT1 differences
        ut1_vals = eop_df['ut1'].values
        lod_obs = -1.0 * np.diff(ut1_vals, prepend=ut1_vals[0]) * 1000.0  # s -> ms
        lod_obs[0] = lod_obs[1]  # fix leading edge
        eop_df['lod_obs'] = lod_obs

        # 2. Total EAM with 1-day lag (atmosphere leads solid Earth by ~1 day)
        OPTIMAL_LAG = 1
        total_eam_series = ex_df['total_eam'].shift(OPTIMAL_LAG).fillna(0.0)

        # 3. Subtract fluid component to obtain core component
        eop_df['lod_core'] = lod_obs - total_eam_series.values

        # 4. Core UT1 integral
        eop_df['ut1_core_integral'] = np.cumsum(eop_df['lod_core']) / 1000.0

        # 5. 365-day moving average for long-term core drift
        eop_df['lod_core_trend_365'] = eop_df['lod_core'].rolling(window=365, min_periods=1).mean()

        eop_df.fillna(0.0, inplace=True)

        return eop_df