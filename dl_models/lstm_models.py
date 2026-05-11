"""
lstm_models.py - LSTM-based models for dUT1 forecasting
========================================================
Contains:
  - lstm           : Standard LSTM (baseline A1: pure data-driven)
  - lstm_enso_gate : LSTM + season-locked ENSO gating (CAPG-Net, A4)

Note for reviewers:
  A2 (BiLSTM + physical decoupling, no gating) and A3 (A2 + seasonal gating)
  are implemented by varying the static_features input to these models,
  not as separate model classes. See train.py and eval.py for details.
"""

import torch
import torch.nn as nn
from .base import FluidModelBase, register_model


# ============================================================================
# Standard LSTM (A1)
# ============================================================================

@register_model(
    'lstm',
    category='rnn',
    description='Standard LSTM: Encoder -> LayerNorm -> Linear. Baseline A1.',
    recommended_for=['short', 'medium'],
)
class LSTMFluidModel(FluidModelBase):
    """
    Standard LSTM for sequence forecasting.
    Supports both unidirectional and bidirectional modes.
    """

    def __init__(self, input_dim=1, lookback=360, forecast_days=180,
                 hidden_dim=64, num_layers=2, dropout=0.2, bidirectional=True, **kwargs):
        super().__init__(input_dim, lookback, forecast_days)
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        self.ln = nn.LayerNorm(lstm_output_dim)
        self.fc = nn.Linear(lstm_output_dim, forecast_days)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        if self.lstm.bidirectional:
            context = torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=-1)
        else:
            context = h_n[-1, :, :]
        return self.fc(self.ln(context))


# ============================================================================
# Season-locked ENSO Gate (physical gating module for CAPG-Net)
# ============================================================================

class SeasonLockedENSOGate(nn.Module):
    """
    Season-locked ENSO multi-branch gating module.

    Input features (6-dim):
      [0:2] -> El Nino (positive extremum, normalized lag position)
      [2:4] -> La Nina (negative extremum, normalized lag position)
      [4:6] -> Season (sin_DOY, cos_DOY)

    Architecture:
      1. Three independent encoders: El Nino, La Nina, Season
      2. Seasonal modulator: generates phase-specific weights for ENSO signals
      3. Gate generator: combines modulated features into final gating vector
      4. Residual connection: output = x * gate + x
    """

    def __init__(self, feat_dim, hidden_dim=16, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.elnino_encoder = nn.Linear(2, hidden_dim)
        self.lanina_encoder = nn.Linear(2, hidden_dim)
        self.season_encoder = nn.Linear(2, hidden_dim)

        self.season_modulator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Sigmoid()
        )

        self.gate_generator = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 3, feat_dim),
            nn.Sigmoid()
        )

        nn.init.constant_(self.gate_generator[-2].bias, 1.0)

    def forward(self, x, enso_features):
        elnino_feat = enso_features[:, 0:2]
        lanina_feat = enso_features[:, 2:4]
        season_feat = enso_features[:, 4:6]

        el_emb = self.elnino_encoder(elnino_feat)
        la_emb = self.lanina_encoder(lanina_feat)
        se_emb = self.season_encoder(season_feat)

        modulators = self.season_modulator(se_emb)
        el_modulator = modulators[:, :self.hidden_dim]
        la_modulator = modulators[:, self.hidden_dim:]

        el_emb_modulated = el_emb * el_modulator
        la_emb_modulated = la_emb * la_modulator

        combined = torch.cat([se_emb, el_emb_modulated, la_emb_modulated], dim=-1)
        main_gate = self.gate_generator(combined)

        return x * main_gate + x


# ============================================================================
# CAPG-Net (A4)
# ============================================================================

@register_model(
    'lstm_enso_gate',
    category='rnn',
    description='LSTM + season-locked ENSO gating. CAPG-Net (A4).',
    recommended_for=['medium', 'long'],
)
class LSTMENSOGateFluidModel(FluidModelBase):
    """
    CAPG-Net: LSTM with season-locked ENSO gating.

    The gating module dynamically modulates the backbone hidden states
    based on El Nino/La Nina phase asymmetry, lag position, and seasonal
    background. A residual connection preserves the original temporal signal.
    """

    def __init__(self, input_dim=1, lookback=360, forecast_days=180,
                 hidden_dim=64, num_layers=2, dropout=0.2,
                 bidirectional=True, gate_hidden_dim=16, **kwargs):
        super().__init__(input_dim, lookback, forecast_days)

        self.feat_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        self.enso_gate = SeasonLockedENSOGate(
            feat_dim=self.feat_dim,
            hidden_dim=gate_hidden_dim,
            dropout=dropout,
        )

        self.ln = nn.LayerNorm(self.feat_dim)
        self.fc = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, forecast_days),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def forward(self, x, enso_features=None):
        out, (h_n, c_n) = self.lstm(x)

        if self.lstm.bidirectional:
            context = torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=-1)
        else:
            context = h_n[-1, :, :]

        if enso_features is not None:
            context = self.enso_gate(context, enso_features)

        return self.fc(self.ln(context))