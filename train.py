#!/usr/bin/env python3
"""
CAPG-Net: Climate-Adaptive Physics-Guided Network for dUT1 Forecasting
-----------------------------------------------------------------------
Including:
  - Physics-based signal decoupling: separates LOD into core and fluid components
  - Climate-adaptive gating: encodes Niño-3.4 extremum, lag position, and seasonal phase
  - LS extrapolation for the core component
  - Training CAPG-Net with early stopping and ReduceLROnPlateau
Usage:
  python train.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import warnings
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
from configs import Config
from dl_models import create_model, MODEL_REGISTRY
from dl_models.losses import KoopmanLoss

# Default learning rates per model (unlisted models use 0.001)
LR_DEFAULTS = {
    'lstm': 0.002,
    'patchtst': 0.001,
    'dlinear': 0.001,
    'tcn': 0.001,
    'timesnet': 0.001,
}

warnings.filterwarnings('ignore')

def get_max_abs_nino_in_window(nino_series, curr_date, lag_min=200, lag_max=300, step=5):
    """
    Find the NINO3.4 value with maximum absolute value within the specified lag window

    Args:
        nino_series: NINO3.4 time series (pandas Series, index is date)
        curr_date: Current date
        lag_min: Minimum lag in days (default 200)
        lag_max: Maximum lag in days (default 300)
        step: Search step size in days (default 5)

    Returns:
        max_abs_value: Value with maximum absolute magnitude (preserving sign)
        best_lag: Corresponding lag in days
    """
    lags = range(lag_min, lag_max + 1, step)
    values = []
    valid_lags = []

    for lag in lags:
        nino_date = curr_date - pd.Timedelta(days=lag)
        if nino_date in nino_series.index:
            val = nino_series.loc[nino_date]
            values.append(val)
            valid_lags.append(lag)

    if values:
        abs_values = np.abs(values)
        abs_max_idx = np.argmax(abs_values)
        max_idx = np.argmax(values)
        min_idx = np.argmin(values)
        return values[abs_max_idx], valid_lags[abs_max_idx], values[max_idx], valid_lags[max_idx], values[min_idx], valid_lags[min_idx]
    else:
        return 0.0, None, 0.0, None, 0.0, None


class Trainer:
    def __init__(self, config, model_type_fluid='lstm', forecast_days=180,
                 fluid_roll_epochs=30, skip_pretrain=False, accumulate_weights=False):
        """
        Initialize incremental rolling trainer

        Args:
            config: Configuration object
            model_type_fluid: Fluid model type ('lstm', 'node', 'patchtst', etc.)
            forecast_days: Forecast horizon in days
            fluid_roll_epochs: Number of epochs for Fluid model rolling training (default 30)
            skip_pretrain: Whether to skip Pretrain and use cached weights directly
            accumulate_weights: Whether to enable rolling accumulated training
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type_fluid = model_type_fluid
        self.forecast_days = forecast_days
        self.fluid_roll_epochs = fluid_roll_epochs
        self.skip_pretrain = skip_pretrain
        self.accumulate_weights = accumulate_weights

        print(f"⚡ Device: {self.device}")
        print(f"🧠 Fluid model type: {model_type_fluid.upper()}")
        print(f"📅 Forecast horizon: {forecast_days} days")
        print(f"📈 Fluid rolling training epochs: {fluid_roll_epochs}")
        print(f"🔄 Skip Pretrain: {skip_pretrain}")
        print(f"🔄 Rolling accumulated training: {accumulate_weights}")

        self.OMEGA1 = 2 * np.pi / 365.2425
        self.OMEGA2 = 2 * np.pi / (365.2425 / 2)

        # Mini-Batch configuration
        self.batch_size = 1024
        self.num_workers = 0 if os.name == 'nt' else 4
        self.pin_memory = True

        # Early stopping configuration
        self.early_stopping_patience = 20
        self.early_stopping_min_delta = 1e-5
        self.min_training_epochs = 50

        # Learning rate scheduler configuration - ReduceLROnPlateau
        self.lr_scheduler_mode = 'min'
        self.lr_scheduler_factor = 0.5
        self.lr_scheduler_patience = 10
        self.lr_scheduler_min_lr = 1e-6
        self.lr_scheduler_verbose = True

        # Ensure weight save directories exist
        os.makedirs("./weight/fluid_model_weights", exist_ok=True)
        os.makedirs("./weight/core_model_weights", exist_ok=True)

        # Fluid model weight save paths
        self.model_save_path = "./weight/fluid_model_weights/trained_model_weights_" + str(forecast_days) + "_" + model_type_fluid + ".pth"
        # Pretrain initial weight save path
        self.pretrain_save_path = "./weight/fluid_model_weights/pretrained_weights_" + str(forecast_days) + "_" + model_type_fluid + ".pth"
        # Rolling accumulated weight save path
        self.roll_accumulate_path = "./weight/fluid_model_weights/roll_accumulated_" + str(forecast_days) + "_" + model_type_fluid + ".pth"

        self.model_fluid = None
        self.optimizer = None

    def _evaluate_pretrain_sufficiency(self, val_loader, criterion, eval_year, forecast_days):
        """
        Evaluate whether Pretrain training is sufficient

        Metrics:
        1. Validation loss curve (convergence)
        2. LOD space MAE (per forecast step)
        3. Predicted vs true value distribution
        4. Residual autocorrelation (check for underfitting)
        """

        self.model_fluid.eval()
        all_preds = []
        all_targets = []
        all_losses = []

        with torch.no_grad():
            for batch_X, batch_static, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_static = batch_static.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model_fluid(batch_X)
                if isinstance(outputs, tuple):
                    pred = outputs[0]
                else:
                    pred = outputs

                loss = criterion(pred, batch_y)
                all_losses.append(loss.item())
                all_preds.append(pred.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())

        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # 1. Compute MAE per forecast step
        step_mae = np.mean(np.abs(preds - targets), axis=0) * 1000  # Convert to ms

        # 2. Overall statistics
        overall_mae = np.mean(step_mae)
        max_mae = np.max(step_mae)
        min_mae = np.min(step_mae)

        # 3. Compute bias (systematic error)
        bias = np.mean(preds - targets) * 1000

        # 4. Compute residual standard deviation (random error)
        residuals = (preds - targets).flatten() * 1000
        residual_std = np.std(residuals)

        # Print evaluation results
        print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"   📈 Pretrain validation metrics:")
        print(f"   • Average MAE: {overall_mae:.3f} ms (range: {min_mae:.3f} ~ {max_mae:.3f} ms)")
        print(f"   • Systematic bias: {bias:.3f} ms")
        print(f"   • Residual std: {residual_std:.3f} ms")
        print(f"   • Validation loss: {np.mean(all_losses):.6f}")

        # Judge training sufficiency
        sufficiency_status = "✅ Sufficient" if overall_mae < 50 else ("⚠️ Moderate" if overall_mae < 100 else "❌ Insufficient")
        print(f"   • Training sufficiency: {sufficiency_status}")

        # Per-phase MAE analysis
        early_mae = np.mean(step_mae[:30])
        mid_mae = np.mean(step_mae[30:90])
        late_mae = np.mean(step_mae[90:]) if len(step_mae) > 90 else 0
        print(f"   [Phase MAE]")
        print(f"   • Early (1-30d):  {early_mae:.2f} ms")
        print(f"   • Mid (30-90d):   {mid_mae:.2f} ms")
        print(f"   • Late (90d+):    {late_mae:.2f} ms")

        # Convergence trend evaluation
        early_phase_mae = np.mean(step_mae[:10]) if len(step_mae) >= 10 else np.mean(step_mae)
        late_phase_mae = np.mean(step_mae[-10:]) if len(step_mae) >= 10 else np.mean(step_mae)
        convergence_ratio = early_phase_mae / late_phase_mae if late_phase_mae > 0 else float('inf')
        print(f"   [Convergence trend]")
        print(f"   • Early/late error ratio: {convergence_ratio:.2f}")
        if convergence_ratio > 1.5:
            print(f"   • Convergence state: 📈 Still converging rapidly, consider increasing training epochs")
        elif convergence_ratio > 1.1:
            print(f"   • Convergence state: 📉 Slowly converging, current epochs are mostly sufficient")
        else:
            print(f"   • Convergence state: ✅ Sufficiently converged")

        print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        return {
            'overall_mae': overall_mae,
            'bias': bias,
            'residual_std': residual_std,
            'status': sufficiency_status
        }

    def _init_model(self, input_dim, forecast_days, lookback):
        """Create Fluid model via dl_models model library (supports all registered models)"""
        name = self.model_type_fluid
        print(f"🔨 Initializing Fluid model: {name}")

        if name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown Fluid model '{name}'.\n"
                f"Available models: {sorted(MODEL_REGISTRY.keys())}\n"
                f"Use list_models() for details."
            )

        self.model_fluid = create_model(
            name,
            input_dim=input_dim,
            lookback=lookback,
            forecast_days=forecast_days,
            hidden_dim=64,
            dropout=0.2,
        ).to(self.device)

        self.model_fluid.summary()

        lr = LR_DEFAULTS.get(name, 0.001)
        self.optimizer = optim.AdamW(self.model_fluid.parameters(), lr=lr, weight_decay=1e-4)

    def _check_early_stopping(self, val_loss, best_loss, best_model_state, patience_counter, model, optimizer):
        """
        Check whether early stopping should be triggered

        Args:
            val_loss: Current validation loss
            best_loss: Historical best loss
            best_model_state: Best model state
            patience_counter: No-improvement counter
            model: Model object
            optimizer: Optimizer object

        Returns:
            (should_stop, new_best_loss, new_best_model_state, new_patience_counter)
        """
        if val_loss < best_loss - self.early_stopping_min_delta:
            best_loss = val_loss
            best_model_state = {
                'model_state_dict': model.state_dict().copy(),
                'optimizer_state_dict': optimizer.state_dict().copy()
            }
            patience_counter = 0
            return False, best_loss, best_model_state, 0
        else:
            patience_counter += 1
            if patience_counter >= self.early_stopping_patience:
                print(f"   ⏹️ Early stopping triggered: loss has not improved for {patience_counter} epochs (best loss: {best_loss:.6f})")
                return True, best_loss, best_model_state, patience_counter
            return False, best_loss, best_model_state, patience_counter

    def _load_model_weights(self):
        """Load previously trained model weights (to avoid re-initialization each year)"""
        # First try loading trained weights
        if os.path.exists(self.model_save_path):
            try:
                checkpoint = torch.load(self.model_save_path, map_location=self.device)
                self.model_fluid.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"✅ Successfully loaded Fluid model weights from {self.model_save_path}")
                return True
            except Exception as e:
                print(f"⚠️ Failed to load Fluid model weights: {e}")

        # Fallback: try loading pretrain initial weights
        if os.path.exists(self.pretrain_save_path):
            try:
                checkpoint = torch.load(self.pretrain_save_path, map_location=self.device)
                self.model_fluid.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"✅ Successfully loaded Fluid pretrained weights from {self.pretrain_save_path}")
                return True
            except Exception as e:
                print(f"⚠️ Failed to load Fluid pretrained weights: {e}")

        print(f"⚠️ No Fluid model weights found, will use randomly initialized model")
        return False

    def _save_model_weights(self):
        """Save current model weights"""
        try:
            torch.save({
                'model_state_dict': self.model_fluid.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, self.model_save_path)
            print(f"💾 Fluid model weights saved to {self.model_save_path}")
        except Exception as e:
            print(f"❌ Failed to save Fluid model weights: {e}")

    def _compute_baseline_aligned(self, ut1_values, idx, forecast_days, fit_window=365):
        """
        Compute UT1 LS baseline extrapolation
        """
        # 1. Local fit (using the most recent fit_window days)
        actual_window = min(fit_window, idx)
        t_fit_raw = np.arange(idx - actual_window, idx)
        t_fit_local = t_fit_raw - t_fit_raw[0]

        # Design matrix (linear + annual + semi-annual)
        X_fit = np.column_stack([
            np.ones(actual_window), t_fit_local,
            np.cos(self.OMEGA1 * t_fit_local), np.sin(self.OMEGA1 * t_fit_local),
            np.cos(self.OMEGA2 * t_fit_local), np.sin(self.OMEGA2 * t_fit_local)
        ])

        # 2. Solve coefficients
        coeffs = np.linalg.lstsq(X_fit, ut1_values[idx - actual_window: idx], rcond=1e-6)[0]

        # 3. Predict starting from idx+1
        t_future_local = np.arange(idx + 1, idx + forecast_days + 1) - t_fit_raw[0]
        X_future = np.column_stack([
            np.ones(forecast_days), t_future_local,
            np.cos(self.OMEGA1 * t_future_local), np.sin(self.OMEGA1 * t_future_local),
            np.cos(self.OMEGA2 * t_future_local), np.sin(self.OMEGA2 * t_future_local)
        ])

        return X_future @ coeffs

    def _compute_core_baseline(self, lod_core_values, idx, forecast_days, fit_window=365):
        """
        Compute LOD Core LS linear fit baseline extrapolation
        """
        # 1. Local fit (using the most recent fit_window days)
        actual_window = min(fit_window, idx)
        t_fit_raw = np.arange(idx - actual_window, idx)
        t_fit_local = t_fit_raw - t_fit_raw[0]

        # Design matrix (linear + annual + semi-annual)
        X_fit = np.column_stack([
            np.ones(actual_window), t_fit_local,
            np.cos(self.OMEGA1 * t_fit_local), np.sin(self.OMEGA1 * t_fit_local),
            np.cos(self.OMEGA2 * t_fit_local), np.sin(self.OMEGA2 * t_fit_local)
        ])

        # 2. Solve coefficients
        coeffs = np.linalg.lstsq(X_fit, lod_core_values[idx - actual_window: idx], rcond=1e-6)[0]

        # 3. Predict starting from idx (future)
        t_future_local = np.arange(idx, idx + forecast_days) - t_fit_raw[0]
        X_future = np.column_stack([
            np.ones(forecast_days), t_future_local,
            np.cos(self.OMEGA1 * t_future_local), np.sin(self.OMEGA1 * t_future_local),
            np.cos(self.OMEGA2 * t_future_local), np.sin(self.OMEGA2 * t_future_local)
        ])

        return X_future @ coeffs

    def prepare_data(self, eop_df, ex_df, nino_df, lookback=60, forecast_days=90):
        print(f"🛠️ Preparing full feature data (Forecast={forecast_days}d)...")
        X_fluid, y_fluid_seq = [], []
        X_fluid_static = []
        y_core_seq = []

        lod_fluid = (eop_df['lod_obs'] - eop_df['lod_core']).values
        lod_core = eop_df['lod_core'].values
        ut1_values = eop_df['ut1'].values

        eam_values = ex_df['total_eam'].shift(1).fillna(0.0).values
        dates = eop_df['date'].values
        valid_start = max(lookback, 365)
        valid_end = len(eop_df) - forecast_days

        date_list = []
        baseline_fluid_anchors = []
        core_ls_baseline = []
        ls_errors = []

        for i in range(valid_start, valid_end):
            # ===== Fluid model feature engineering =====
            seq_eam = eam_values[i - lookback: i]

            # Multi-scale decomposition
            eam_ma30 = np.mean(eam_values[i - 30:i]) if i >= 30 else np.mean(eam_values[:i])
            eam_ma90 = np.mean(eam_values[i - 90:i]) if i >= 90 else np.mean(eam_values[:i])

            eam_high = seq_eam - eam_ma30
            eam_season = seq_eam - eam_ma90

            # Current date
            curr_date = pd.Timestamp(dates[i])

            # ===== ENSO statistical features (static, for gating) =====
            if curr_date in nino_df.index:
                MAX_LAG = 60
                MIN_LAG = 1

                # 1. Three-period physical extremum search (with anti-leakage safety boundary)
                nino_lag_abs_max, best_lag_abs_max, nino_lag_pos_max, best_lag_pos_max, nino_lag_neg_max, best_lag_neg_max = get_max_abs_nino_in_window(
                    nino_df['nino34'], curr_date, lag_min=MIN_LAG, lag_max=MAX_LAG, step=5
                )
                [nino_lag_pos_max, best_lag_pos_max] = [0, None] if np.abs(nino_lag_abs_max) < 0.5 else [nino_lag_pos_max, best_lag_pos_max]
                [nino_lag_neg_max, best_lag_neg_max] = [0, None] if np.abs(nino_lag_abs_max) < 0.5 else [nino_lag_neg_max, best_lag_neg_max]

                # 2. Global positional encoding (reflecting order and relative distance)
                lag_pos_val = best_lag_pos_max if best_lag_pos_max is not None else MAX_LAG
                lag_neg_val = best_lag_neg_max if best_lag_neg_max is not None else MAX_LAG

                # Convert to [0, 1] interval monotonic positional coordinates
                pos_pos = (lag_pos_val - MIN_LAG) / (MAX_LAG - MIN_LAG)
                pos_neg = (lag_neg_val - MIN_LAG) / (MAX_LAG - MIN_LAG)
            else:
                print(f"⚠️ {curr_date} not in NINO index")

            # Static feature vector: [enso_pos, pos_pos, enso_neg, pos_neg, season_sin, season_cos]
            day_of_year = curr_date.dayofyear
            season_sin = np.sin(2 * np.pi * day_of_year / 365.25)
            season_cos = np.cos(2 * np.pi * day_of_year / 365.25)

            static_features = np.array([
                nino_lag_pos_max, pos_pos,
                nino_lag_neg_max, pos_neg,
                season_sin, season_cos
            ])
            # static_features = np.array([
            #     season_sin, season_cos
            # ])
            X_fluid_static.append(static_features)

            # Sequence features
            n_features = 1
            seq_features = np.zeros((lookback, n_features))
            seq_features[:, 0] = seq_eam
            X_fluid.append(seq_features)

            anchor_val = lod_fluid[i - 1]
            y_fluid_seq.append(lod_fluid[i: i + forecast_days] - anchor_val)

            # ===== Core LS baseline computation =====
            core_ls_pred = self._compute_core_baseline(lod_core, i, forecast_days, fit_window=365)

            core_ls_baseline.append(core_ls_pred)
            y_core_seq.append(lod_core[i: i + forecast_days] - core_ls_pred)

            baseline_fluid_anchors.append(anchor_val)
            date_list.append(dates[i])

            # UT1 LS Baseline
            ut1_future_true = ut1_values[i + 1: i + forecast_days + 1]
            ut1_pred_ls = self._compute_baseline_aligned(ut1_values, i, forecast_days)
            err_ms = np.abs(ut1_future_true[-1] - ut1_pred_ls[-1]) * 1000.0
            ls_errors.append(err_ms)

        core_ls_baseline_array = np.array([arr for arr in core_ls_baseline])

        return (np.array(X_fluid), np.array(y_fluid_seq), np.array(X_fluid_static)), \
            np.array(y_core_seq), \
            np.array(date_list), np.array(baseline_fluid_anchors), np.array(ls_errors), core_ls_baseline_array

    def train_rolling(self, lookback=None, forecast_days=90, epoch_pretrain=150, epoch_roll=30, start_year=2015, skip_rolling=False):

        from data_processor import DataProcessor
        processor = DataProcessor(self.config)
        eop_df, nino_df, ex_df, geo_df = processor.load_all_data()

        lookback = 365

        print(f"\n📋 Training parameters confirmed:")
        print(f"   forecast_days={forecast_days}")
        print(f"   lookback={lookback}")

        (X_f, y_f, X_f_static), y_c, dates, base_f, ls_errs, core_ls_baseline = self.prepare_data(
            eop_df, ex_df, nino_df, lookback=lookback, forecast_days=forecast_days
        )
        dates_pd = pd.DatetimeIndex(dates)

        # Print data shapes for debugging
        print(f"📊 Data preparation complete:")
        print(f"   X_fluid shape: {X_f.shape} (lookback={lookback})")
        print(f"   y_fluid shape: {y_f.shape}")

        # Check if 2025 data exists
        mask_2025 = dates_pd.year == 2025
        if mask_2025.sum() > 0:
            max_year = 2025
        else:
            print("⚠️ Warning: No data for 2025! Falling back to 2024")
            max_year = 2024
        valid_mask = dates_pd.year <= max_year

        X_f = X_f[valid_mask]
        X_f_static = X_f_static[valid_mask]
        y_f = y_f[valid_mask]
        y_c = y_c[valid_mask]
        dates_pd = dates_pd[valid_mask]
        base_f = base_f[valid_mask]
        ls_errs = ls_errs[valid_mask]
        core_ls_baseline = core_ls_baseline[valid_mask]

        print(f"📊 Valid data year range: {dates_pd.year.min()} - {dates_pd.year.max()}")
        print(f"📈 Total samples: {len(dates_pd)}")

        print(f"\n🔧 Initializing Fluid model:")
        print(f"   lookback={lookback}, forecast_days={forecast_days}")

        fluid_input_dim = 1
        self._init_model(input_dim=fluid_input_dim, forecast_days=forecast_days, lookback=lookback)

        # Try loading previously saved weights (to avoid re-initialization each year)
        weights_loaded = self._load_model_weights()

        if not weights_loaded:
            print("🆕 First training, using randomly initialized weights")

        # Ensure year ascending order
        unique_years = np.sort(np.unique(dates_pd.year))
        eval_years = unique_years[unique_years >= start_year]

        # Ensure not exceeding maximum year
        eval_years = eval_years[eval_years <= max_year]

        # If skip_rolling is set, all years use pretrained weights for evaluation directly (no Fine-tune)
        if skip_rolling and len(eval_years) > 0:
            print(f"\n⏭️ Skip rolling training mode: all years use Pretrained weights for evaluation (no Fine-tune)")

        # Check if Pretrain weight file exists
        has_pretrained_weights = os.path.exists(self.pretrain_save_path)

        print(f"\n🔄 Starting smart rolling training")
        print(f"   Evaluation years: {eval_years[0]} - {eval_years[-1]}")
        print(f"   Strategy: first year Pretrain ({epoch_pretrain} Epochs), subsequent Fine-tune ({epoch_roll} Epochs)")
        print(f"   Sampling strategy: first year 5000 samples, subsequent 2000 samples")
        print(f"   Weight reuse: {'✅ Loaded' if weights_loaded else '🆕 First training'}")
        print(f"   Pretrained cache: {'✅ Exists, will skip' if has_pretrained_weights else '🆕 Needs execution'}")
        print(f"   Core prediction: LS extrapolation (fixed)")
        print(f"   Max year limit: {max_year}")

        all_predictions = []

        criterion = nn.MSELoss()
        if model_type_fluid == 'koopa':
            criterion = KoopmanLoss()

        for i, eval_year in enumerate(eval_years):
            is_first_year = (i == 0)

            # --- 1. Dynamically determine training parameters ---
            if is_first_year:
                    if has_pretrained_weights:
                        print(f"   ⏭️ Pretrained cache detected, loading weights...")
                        try:
                            pretrain_checkpoint = torch.load(self.pretrain_save_path, map_location=self.device)
                            self.model_fluid.load_state_dict(pretrain_checkpoint['model_state_dict'])
                            self.optimizer.load_state_dict(pretrain_checkpoint['optimizer_state_dict'])
                            print(f"   ✅ Successfully loaded pretrained weights")

                            if self.skip_pretrain:
                                print(f"   ⏩ Skip pretrain enabled: skipping training phase, evaluating directly")
                                skip_training = True
                                current_epochs = 0
                            else:
                                skip_training = False
                                current_epochs = 50
                                print(f"   🔄 Skip pretrain disabled: continuing training for {current_epochs} epochs on pretrained weights")
                        except Exception as e:
                            print(f"   ⚠️ Failed to load pretrained weights: {e}, will execute normal pretrain")
                            skip_training = False
                            current_epochs = epoch_pretrain
                    else:
                        skip_training = False
                        current_epochs = epoch_pretrain

                    sample_size = 5000
                    lr_adjustment = 1.0
                    desc_str = f"Year {eval_year} [Pretrain]" if not has_pretrained_weights else f"Year {eval_year} [Fine-tune on Pretrain]"
            else:
                    if skip_rolling:
                        skip_training = True
                        print(f"   ⏭️ Skipping Fine-tune, evaluating directly with pretrained weights")
                    else:
                        skip_training = False
                        if has_pretrained_weights:
                            print(f"   🔧 Using pretrained weights as initialization, starting Fine-tune training...")
                            try:
                                pretrain_checkpoint = torch.load(self.pretrain_save_path, map_location=self.device)
                                self.model_fluid.load_state_dict(pretrain_checkpoint['model_state_dict'])
                                print(f"   ✅ Pretrained weights loaded as initialization point")
                            except Exception as e:
                                print(f"   ⚠️ Failed to load pretrained weights: {e}, continuing with current weights")
                        # If rolling accumulated training is enabled, load previous year's accumulated weights
                        if self.accumulate_weights and i > 0:
                            prev_year = eval_years[i-1]
                            prev_weight_path = f"./weight/fluid_model_weights/roll_accumulated_{self.forecast_days}_{self.model_type_fluid}_year{prev_year}.pth"
                            if os.path.exists(prev_weight_path):
                                try:
                                    prev_checkpoint = torch.load(prev_weight_path, map_location=self.device)
                                    self.model_fluid.load_state_dict(prev_checkpoint['model_state_dict'])
                                    self.optimizer.load_state_dict(prev_checkpoint['optimizer_state_dict'])
                                    print(f"   🔄 Loaded last year ({prev_year}) accumulated weights, continuing training...")
                                except Exception as e:
                                    print(f"   ⚠️ Failed to load last year's accumulated weights: {e}")

                    current_epochs = self.fluid_roll_epochs
                    sample_size = 2000
                    lr_adjustment = 0.8
                    desc_str = f"Year {eval_year} [Fine-tune]" if not skip_rolling else f"Year {eval_year} [Eval-Only]"

            # === Training window strategy: Warm-up uses full history, rolling training uses 10-year window ===
            VAL_RATIO = 0.2

            if is_first_year:
                TARGET_TRAIN_YEARS = 999
                ROLLING_WINDOW_YEARS = TARGET_TRAIN_YEARS
                desc_suffix = "[Pretrain full history]"
            else:
                ROLLING_WINDOW_YEARS = 10
                desc_suffix = "[Fine-tune 10-year window]"

            test_mask = (dates_pd.year == eval_year) & (dates_pd.day % 7 == 0)
            if is_first_year:
                train_mask = dates_pd.year < eval_year
            else:
                train_mask = (dates_pd.year < eval_year) & (dates_pd.year >= eval_year - ROLLING_WINDOW_YEARS)

            # Stratified validation set extraction
            val_mask = np.zeros(len(dates_pd), dtype=bool)
            if train_mask.any():
                available_years = np.unique(dates_pd[train_mask].year)
                for year in available_years:
                    year_indices = np.where((dates_pd.year == year) & train_mask)[0]
                    n_val = max(1, int(len(year_indices) * 0.2))
                    step = max(1, len(year_indices) // n_val)
                    val_indices = year_indices[::step][:n_val]
                    val_mask[val_indices] = True

                train_mask_final = train_mask & ~val_mask

                print(f"   📊 Data split: train={train_mask_final.sum()} | val={val_mask.sum()} | test={test_mask.sum()}")
            else:
                train_mask_final = train_mask

            if not test_mask.any():
                print(f"⚠️ Warning: No test data for {eval_year}, skipping")
                continue

            if not train_mask.any():
                print(f"⚠️ Warning: No training data for {eval_year}, skipping")
                continue

            # Data splitting
            train_X_f, train_y_f = X_f[train_mask_final], y_f[train_mask_final]
            train_X_f_static = X_f_static[train_mask_final]

            # Validation set data
            val_X_f, val_y_f = X_f[val_mask], y_f[val_mask] if val_mask.any() else (None, None)
            val_X_f_static = X_f_static[val_mask] if val_mask.any() else None

            test_X_f = X_f[test_mask]
            test_X_f_static = X_f_static[test_mask]
            test_y_f_delta = y_f[test_mask]
            test_base_f = base_f[test_mask]
            test_y_c_seq = y_c[test_mask]
            test_core_ls_baseline = core_ls_baseline[test_mask]
            test_dates_curr = dates_pd[test_mask]
            test_ls_errs_curr = ls_errs[test_mask]

            if len(test_X_f) == 0 or len(test_dates_curr) == 0:
                print(f"⚠️ Warning: Test data is empty for {eval_year}, skipping")
                continue

            # --- 2. Sample training data and create DataLoader ---
            actual_sample_size = min(len(train_X_f), sample_size)

            if len(train_X_f) > actual_sample_size:
                indices = np.random.choice(len(train_X_f), actual_sample_size, replace=False)
                train_X_sampled = train_X_f[indices]
                train_y_sampled = train_y_f[indices]
                train_X_f_static_sampled = train_X_f_static[indices]
            else:
                train_X_sampled = train_X_f
                train_y_sampled = train_y_f
                train_X_f_static_sampled = train_X_f_static

            # Create validation DataLoader
            if val_mask.any() and val_X_f is not None and val_X_f_static is not None:
                val_dataset = torch.utils.data.TensorDataset(
                    torch.FloatTensor(val_X_f),
                    torch.FloatTensor(val_X_f_static),
                    torch.FloatTensor(val_y_f)
                )
                val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=min(256, len(val_X_f)),
                    shuffle=False,
                    drop_last=False
                )
                print(f"   ✅ Validation set ready: {len(val_X_f)} samples")
            else:
                val_loader = None

            # Create DataLoader for Mini-Batch training
            train_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(train_X_sampled),
                torch.FloatTensor(train_X_f_static_sampled),
                torch.FloatTensor(train_y_sampled)
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )

            # --- 3. Training loop (Mini-Batch) ---
            if skip_training:
                print(f"   ⏭️ Skipping training phase, evaluating directly with pre-trained weights")
            else:
                self.model_fluid.train()

                # Adjust learning rate
                base_lr = LR_DEFAULTS.get(self.model_type_fluid, 0.001)
                current_lr = base_lr * lr_adjustment
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr

                # Initialize learning rate scheduler - ReduceLROnPlateau
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode=self.lr_scheduler_mode,
                    factor=self.lr_scheduler_factor,
                    patience=self.lr_scheduler_patience,
                    min_lr=self.lr_scheduler_min_lr,
                    verbose=self.lr_scheduler_verbose
                )
                print(f"   📉 LR scheduler: ReduceLROnPlateau (factor={self.lr_scheduler_factor}, patience={self.lr_scheduler_patience})")

                # Enable mixed precision training (if supported) - except FFT-based models
                use_amp = False
                if torch.cuda.is_available() and hasattr(torch.cuda.amp, 'GradScaler'):
                    fft_models = ['fedformer', 'informer', 'autoformer', 'transformer']
                    if self.model_type_fluid.lower() not in fft_models:
                        use_amp = True
                        print(f"   ⚡ Mixed precision training enabled (AMP)")
                    else:
                        print(f"   ⚠️ {self.model_type_fluid.upper()} uses FFT operators, disabling mixed precision to avoid cuFFT precision limitations")
                if use_amp:
                    scaler = torch.cuda.amp.GradScaler()

                pbar = tqdm(range(current_epochs), desc=desc_str, leave=False)

                # Early stopping initialization
                best_fluid_loss = float('inf')
                best_fluid_model_state = None
                fluid_patience_counter = 0
                best_val_loss = float('inf')

                for epoch in pbar:
                    epoch_loss = 0.0
                    num_batches = 0

                    for batch_X, batch_static, batch_y in train_loader:

                        batch_X = batch_X.to(self.device)
                        batch_static = batch_static.to(self.device)
                        batch_y = batch_y.to(self.device)

                        self.optimizer.zero_grad()

                        if use_amp:
                            with torch.cuda.amp.autocast():
                                outputs = self.model_fluid(batch_X, batch_static)
                                if isinstance(outputs, tuple):
                                    out = outputs[0]
                                    z_history = outputs[1]
                                    if hasattr(criterion, 'alpha'):
                                        loss_results = criterion(out, batch_y, z_history, self.model_fluid.K)
                                        loss = loss_results[0] if isinstance(loss_results, tuple) else loss_results
                                    else:
                                        loss = criterion(out, batch_y)
                                else:
                                    out = outputs
                                    loss = criterion(out, batch_y)

                            scaler.scale(loss).backward()
                            scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model_fluid.parameters(), 1.0)
                            scaler.step(self.optimizer)
                            scaler.update()
                        else:
                            outputs = self.model_fluid(batch_X, batch_static)

                            if isinstance(outputs, tuple):
                                out = outputs[0]
                                z_history = outputs[1]
                                if hasattr(criterion, 'alpha'):
                                    loss_results = criterion(out, batch_y, z_history, self.model_fluid.K)
                                    loss = loss_results[0] if isinstance(loss_results, tuple) else loss_results
                                else:
                                    loss = criterion(out, batch_y)
                            else:
                                out = outputs
                                loss = criterion(out, batch_y)

                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model_fluid.parameters(), 1.0)
                            self.optimizer.step()

                        epoch_loss += loss.item()
                        num_batches += 1

                    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0

                    # Compute validation loss if validation set exists
                    val_avg_loss = None
                    if val_loader is not None:
                        self.model_fluid.eval()
                        val_epoch_loss = 0.0
                        val_num_batches = 0
                        with torch.no_grad():
                            for batch_X, batch_static, batch_y in val_loader:
                                batch_X = batch_X.to(self.device)
                                batch_static = batch_static.to(self.device)
                                batch_y = batch_y.to(self.device)
                                outputs = self.model_fluid(batch_X, batch_static)

                                out = outputs[0] if isinstance(outputs, tuple) else outputs
                                if isinstance(outputs, tuple) and hasattr(criterion, 'alpha'):
                                    z_history = outputs[1]
                                    loss_results = criterion(out, batch_y, z_history, self.model_fluid.K)
                                    loss = loss_results[0] if isinstance(loss_results, tuple) else loss_results
                                else:
                                    loss = criterion(out, batch_y)
                                val_epoch_loss += loss.item()
                                val_num_batches += 1
                        val_avg_loss = val_epoch_loss / val_num_batches if val_num_batches > 0 else None
                        self.model_fluid.train()

                    # Check early stopping (use validation loss if available)
                    monitor_loss = val_avg_loss if val_avg_loss is not None else avg_loss
                    should_stop, best_fluid_loss, best_fluid_model_state, fluid_patience_counter = self._check_early_stopping(
                        monitor_loss, best_fluid_loss, best_fluid_model_state, fluid_patience_counter,
                        self.model_fluid, self.optimizer
                    )

                    # Update learning rate scheduler (based on validation loss)
                    if val_avg_loss is not None:
                        old_lr = self.optimizer.param_groups[0]['lr']
                        scheduler.step(val_avg_loss)
                        new_lr = self.optimizer.param_groups[0]['lr']
                        if new_lr < old_lr:
                            fluid_patience_counter = 0
                            print(f"   📉 LR decay: {old_lr:.2e} → {new_lr:.2e}, resetting early stopping counter")

                    if epoch % 10 == 0 or should_stop:
                        if val_avg_loss is not None:
                            pbar.set_postfix({
                                'train_loss': f"{avg_loss:.4f}",
                                'val_loss': f"{val_avg_loss:.4f}",
                                'best': f"{best_fluid_loss:.4f}",
                                'patience': fluid_patience_counter
                            })
                        else:
                            pbar.set_postfix({'loss': f"{avg_loss:.4f}", 'best': f"{best_fluid_loss:.4f}", 'patience': fluid_patience_counter})

                    if should_stop and epoch >= self.min_training_epochs:
                        break
                    elif should_stop and epoch < self.min_training_epochs:
                        should_stop = False
                        fluid_patience_counter = 0

                pbar.close()

                # Restore best model weights
                if best_fluid_model_state is not None and fluid_patience_counter >= self.early_stopping_patience:
                    print(f"   🔄 Restoring Fluid best model weights (loss: {best_fluid_loss:.6f})")
                    self.model_fluid.load_state_dict(best_fluid_model_state['model_state_dict'])
                    self.optimizer.load_state_dict(best_fluid_model_state['optimizer_state_dict'])

                # Save Pretrain weights if first year
                if is_first_year and not skip_training:
                    try:
                        torch.save({
                            'model_state_dict': self.model_fluid.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'epoch_pretrain': epoch_pretrain,
                            'eval_year': eval_year,
                        }, self.pretrain_save_path)
                        print(f"   💾 Pretrained weights saved to {self.pretrain_save_path}")
                        print(f"      Subsequent runs will automatically skip pretrain phase, each round independently fine-tuning from these weights")
                    except Exception as e:
                        print(f"   ⚠️ Failed to save pretrained weights: {e}")
                else:
                    if self.accumulate_weights and not is_first_year:
                        roll_accumulate_path = f"./weight/fluid_model_weights/roll_accumulated_{self.forecast_days}_{self.model_type_fluid}_year{eval_year}.pth"
                        try:
                            torch.save({
                                'model_state_dict': self.model_fluid.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'eval_year': eval_year,
                            }, roll_accumulate_path)
                            print(f"   💾 Rolling accumulated weights saved to {roll_accumulate_path}")
                        except Exception as e:
                            print(f"   ⚠️ Failed to save accumulated weights: {e}")
                    else:
                        print(f"   Fluid independent fine-tuning complete, accumulated weights not saved")

                # Pretrain training sufficiency evaluation (first year only after training)
                if is_first_year and val_loader is not None:
                    print(f"\n   📊 Pretrain sufficiency evaluation:")
                    self._evaluate_pretrain_sufficiency(
                        val_loader, criterion, eval_year, forecast_days
                    )

            # --- 4. Core prediction: LS extrapolation ---
            print(f"   📐 Core: using LS extrapolation")
            pred_core_abs = test_core_ls_baseline

            # --- 5. Prediction and evaluation ---
            self.model_fluid.eval()

            # Create test set DataLoader for Fluid model
            test_dataset_fluid = torch.utils.data.TensorDataset(
                torch.FloatTensor(test_X_f),
                torch.FloatTensor(test_X_f_static)
            )
            test_loader_fluid = torch.utils.data.DataLoader(
                test_dataset_fluid,
                batch_size=min(256, len(test_X_f)),
                shuffle=False,
                drop_last=False
            )

            pred_fluid_delta_list = []
            with torch.no_grad():
                for batch_X, batch_static in test_loader_fluid:
                    batch_X = batch_X.to(self.device)
                    batch_static = batch_static.to(self.device)
                    outputs = self.model_fluid(batch_X, batch_static)

                    out = outputs[0] if isinstance(outputs, tuple) else outputs
                    batch_pred = out.cpu().numpy()

                    pred_fluid_delta_list.append(batch_pred)

            pred_fluid_delta = np.concatenate(pred_fluid_delta_list, axis=0)

            # Use model predictions directly
            pred_fluid_abs = pred_fluid_delta + test_base_f[:, np.newaxis]
            true_fluid_abs = test_y_f_delta + test_base_f[:, np.newaxis]

            # ===== Core model ground truth processing =====
            # test_y_c_seq is Core residual relative to LS baseline (unit ms)
            # test_core_ls_baseline is Core LS baseline prediction (unit ms)
            # True Core LOD (absolute) = residual + LS baseline
            true_core_lod = test_y_c_seq + test_core_ls_baseline
            # Total LOD = Fluid absolute + Core absolute
            true_lod_seq = true_fluid_abs + true_core_lod

            # === LOD space: evaluate Fluid and Core prediction accuracy separately ===

            # 1. Fluid LOD prediction error (in LOD space)
            true_fluid_lod = true_fluid_abs
            pred_fluid_lod = pred_fluid_abs
            fluid_lod_error = np.abs(true_fluid_lod - pred_fluid_lod)
            fluid_lod_mae_per_step = np.mean(fluid_lod_error, axis=0)
            fluid_lod_mae_mean = np.mean(fluid_lod_mae_per_step)

            # 2. Core LOD prediction error (in LOD space)
            pred_core_lod = pred_core_abs
            core_lod_error = np.abs(true_core_lod - pred_core_lod)
            core_lod_mae_per_step = np.mean(core_lod_error, axis=0)
            core_lod_mae_mean = np.mean(core_lod_mae_per_step)

            # === UT1 space: evaluate combined model cumulative error (comparison with Bulletin A) ===
            pred_lod_seq = pred_fluid_abs + pred_core_abs

            lod_diff_seq = true_lod_seq - pred_lod_seq
            ut1_error_seq = np.cumsum(lod_diff_seq, axis=1)
            final_ut1_errors = np.abs(ut1_error_seq[:, -1])

            # Compute UT1 MAE (standard metric for Bulletin A comparison)
            mae_combined_ut1 = np.mean(final_ut1_errors)
            ls_mae = np.mean(test_ls_errs_curr)

            # Compute improvement relative to LS (based on UT1 error)
            imp_combined = (ls_mae - mae_combined_ut1) / ls_mae * 100

            # Compute LOD error proportions (for analyzing which component has larger error)
            total_lod_error = fluid_lod_mae_mean + core_lod_mae_mean
            fluid_lod_contrib = (fluid_lod_mae_mean / total_lod_error * 100) if total_lod_error > 0 else 0
            core_lod_contrib = (core_lod_mae_mean / total_lod_error * 100) if total_lod_error > 0 else 0

            # Compute UT1 statistics (for Bulletin A comparison)
            ut1_true = np.cumsum(true_lod_seq, axis=1)[:, -1]
            ut1_pred_combined = np.cumsum(pred_lod_seq, axis=1)[:, -1]
            ut1_true_mean = np.mean(ut1_true)
            ut1_pred_combined_mean = np.mean(ut1_pred_combined)

            print(f"Baseline anchor mean: {np.mean(test_base_f):.2f} ms")
            print(f"Anchor value range: {test_base_f.min():.2f} ~ {test_base_f.max():.2f}")
            print(f"True Fluid mean: {np.mean(true_fluid_abs):.2f}")
            print(f"Predicted Fluid mean: {np.mean(pred_fluid_abs):.2f}")

            # Combined model UT1 error statistics
            combined_ut1_error = np.abs(ut1_true - ut1_pred_combined)
            combined_ut1_error_mean = np.mean(combined_ut1_error)
            combined_ut1_error_std = np.std(combined_ut1_error)
            combined_ut1_error_min = np.min(combined_ut1_error)
            combined_ut1_error_max = np.max(combined_ut1_error)
            combined_ut1_bias = np.mean(ut1_pred_combined - ut1_true)

            # Compute relative error (relative to true UT1 absolute value)
            ut1_abs_mean = np.mean(np.abs(ut1_true))
            combined_ut1_rel_err = combined_ut1_error_mean / ut1_abs_mean * 100 if ut1_abs_mean > 0 else 0
            ls_rel_err = ls_mae / ut1_abs_mean * 100 if ut1_abs_mean > 0 else 0

            # LOD error statistics
            n_samples = len(test_dates_curr)
            fluid_lod_error_std = np.std(fluid_lod_error)
            core_lod_error_std = np.std(core_lod_error)

            print(f"\n   📊 {eval_year} model decomposition ({n_samples} samples)")
            print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"   [LOD space] Component prediction accuracy (length of day variation):")
            print(f"   Component   MAE(ms)    Std        Error share")
            print(f"   ─────────────────────────────────────────")
            print(f"   Fluid      {fluid_lod_mae_mean:8.3f}   {np.mean(fluid_lod_error_std):8.3f}    {fluid_lod_contrib:5.1f}%")
            print(f"   Core       {core_lod_mae_mean:8.3f}   {np.mean(core_lod_error_std):8.3f}    {core_lod_contrib:5.1f}%")
            print(f"   ─────────────────────────────────────────")
            print(f"   [UT1 space] Cumulative error (Bulletin A comparison standard):")
            print(f"   Component   MAE(ms)    Std        Bias      Rel. error")
            print(f"   ─────────────────────────────────────────────────────")
            print(f"   Combined   {combined_ut1_error_mean:8.3f}   {combined_ut1_error_std:8.3f}   {combined_ut1_bias:8.3f}    {combined_ut1_rel_err:6.2f}%")
            print(f"   LS base    {ls_mae:8.3f}      -         -       {ls_rel_err:6.2f}%")
            print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"   💡 Main LOD error source: {'Fluid' if fluid_lod_contrib > core_lod_contrib else 'Core'} ({max(fluid_lod_contrib, core_lod_contrib):.1f}%)")
            print(f"   📈 UT1 improvement over LS: {imp_combined:+.1f}% | Error range: [{combined_ut1_error_min:.1f}, {combined_ut1_error_max:.1f}] ms")
            print(f"🎯 True UT1 accumulation mean: {ut1_true_mean:.2f} ms·day | Predicted UT1 accumulation mean: {ut1_pred_combined_mean:.2f} ms·day")

            current_mae = mae_combined_ut1
            imp = imp_combined

            # Save prediction results
            for k, date in enumerate(test_dates_curr):
                all_predictions.append({
                    'date': date,
                    'year': eval_year,
                    'ut1_error': final_ut1_errors[k],
                    'ls_error': test_ls_errs_curr[k]
                })

            if eval_year >= max_year:
                print(f"🏁 Reached maximum year {max_year}, stopping training")
                break

        # Verify data integrity before final summary
        if len(all_predictions) > 0:
            self._print_final_summary(all_predictions)
            self._save_model_weights()
            print(f"   💾 Training complete: Fluid weights saved")
        else:
            print("❌ No prediction results generated, please check data")

    def _print_final_summary(self, predictions):
        pred_df = pd.DataFrame(predictions)

        print("\n" + "=" * 80)
        print(f"📅 Incremental Rolling Training Final Report ({self.model_type_fluid.upper()}, Core=LS)")
        print("-" * 80)
        print(f"{'Year':<6} | {'Samples':<8} | {'Model MAE(ms)':<14} | {'LS Base MAE(ms)':<16} | {'Improvement':<12} | {'Mark'}")
        print("-" * 80)

        total_model_mae = []
        total_ls_mae = []
        total_imp = []

        for year, group in pred_df.groupby('year'):
            count = len(group)
            mae_model = np.mean(group['ut1_error'])
            mae_ls = np.mean(group['ls_error'])
            imp = (mae_ls - mae_model) / mae_ls * 100 if mae_ls != 0 else 0

            imp_str = f"{imp:+.2f}%"
            mark = "✅" if imp > 0 else "🔻" if imp < 0 else "—"

            print(f"{year:<6} | {count:<8} | {mae_model:<14.4f} | {mae_ls:<16.4f} | {imp_str:<12} | {mark}")

            total_model_mae.extend(group['ut1_error'])
            total_ls_mae.extend(group['ls_error'])
            total_imp.append(imp * count)

        print("-" * 80)

        # Global statistics
        global_model_mae = np.mean(total_model_mae)
        global_ls_mae = np.mean(total_ls_mae)
        global_imp = (global_ls_mae - global_model_mae) / global_ls_mae * 100 if global_ls_mae != 0 else 0
        global_imp_str = f"{global_imp:+.2f}%"
        global_mark = "✅" if global_imp > 0 else "🔻" if global_imp < 0 else "—"

        print(
            f"🌟 Global summary: Model MAE = {global_model_mae:.4f} ms | LS Base MAE = {global_ls_mae:.4f} ms | Improvement = {global_imp_str} {global_mark}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    config = Config()

    # ===== Forecast horizon setting =====
    forecast_days = 1

    # ===== Model selection =====
    # Available: 'lstm', 'lstm_enso_gate', 'patchtst', 'tcn', 'dlinear', 'timesnet'
    # For BiLSTM: set bidirectional=True in ./dl_models/lstm_models.py
    model_type_fluid = 'lstm_enso_gate'

    # ===== Training settings =====
    # NOTE: For CAPG-Net results in the paper, use:
    # model_type_fluid = 'lstm_enso_gate'
    # skip_rolling = True
    # Rolling training is NOT used for CAPG-Net. 

    trainer = Trainer(
        config,
        model_type_fluid=model_type_fluid,
        forecast_days=forecast_days,
        fluid_roll_epochs=15,
        skip_pretrain=True,
        accumulate_weights=True,
    )

    trainer.train_rolling(
        forecast_days=forecast_days,
        epoch_pretrain=150,
        epoch_roll=10,
        start_year=2015,
        skip_rolling=True,
    )
