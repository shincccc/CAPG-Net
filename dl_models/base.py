"""
base.py - Base module for Fluid LOD prediction model library
=============================================================
Provides unified base class, model registry, and common utilities.

All models follow a unified interface:
    forward(x: Tensor[B, Lookback, Dim]) -> Tensor[B, ForecastDays]
"""

import torch
import torch.nn as nn
import math

# ======================== Global model registry ========================

MODEL_REGISTRY = {}


def register_model(name, category="", description="", recommended_for=None):
    """
    Decorator for registering models.

    Usage:
        @register_model('dlinear', category='mlp', description='...')
        class DLinearFluidModel(FluidModelBase):
            ...

    Args:
        name:             Registered name (used by create_model)
        category:         Category: 'rnn', 'transformer', 'mlp', 'cnn'
        description:      Short description
        recommended_for:  Recommended forecast scales, e.g. ['short', 'medium', 'long']
                          short: 1-10d, medium: 30-90d, long: 180-360d
    """
    def decorator(cls):
        MODEL_REGISTRY[name] = {
            'class': cls,
            'category': category,
            'description': description,
            'recommended_for': recommended_for or [],
        }
        return cls
    return decorator


# ======================== Unified base class ========================

class FluidModelBase(nn.Module):
    """
    Unified base class for all Fluid LOD prediction models.

    Standard constructor signature:
        __init__(self, input_dim=1, lookback=360, forecast_days=180, **kwargs)

    Standard forward interface:
        forward(x)  ->  predictions
        x:            (Batch, Lookback, Input_Dim)
        predictions:  (Batch, Forecast_Days)
    """

    def __init__(self, input_dim=1, lookback=360, forecast_days=180, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.lookback = lookback
        self.forecast_days = forecast_days

    def forward(self, x):
        raise NotImplementedError

    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        """Print model summary."""
        n = self.count_parameters()
        s = f"{n / 1e6:.2f}M" if n > 1e6 else (f"{n / 1e3:.1f}K" if n > 1e3 else str(n))
        print(f"[{self.__class__.__name__}] "
              f"lookback={self.lookback}, forecast={self.forecast_days}, "
              f"input_dim={self.input_dim}, params={s}")


# ======================== Common utility layers ========================

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: (B, L, D)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class RevIN(nn.Module):
    """
    Reversible Instance Normalization.
    From "Reversible Instance Normalization for Accurate Time-Series
    Forecasting against Distribution Shift" (ICLR 2022).
    """

    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode='norm'):
        """
        x: (B, L, D)
        mode: 'norm' for normalization, 'denorm' for inverse.
        """
        if mode == 'norm':
            self._mean = x.mean(dim=1, keepdim=True).detach()
            self._std = torch.sqrt(
                x.var(dim=1, keepdim=True, unbiased=False) + self.eps
            ).detach()
            x = (x - self._mean) / self._std
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        else:
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self._std + self._mean
            return x


class SeriesDecomposition(nn.Module):
    """
    Series decomposition: trend + seasonal.
    From Autoformer / FEDformer / DLinear.
    """

    def __init__(self, kernel_size=25):
        super().__init__()
        self.moving_avg = nn.AvgPool1d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )

    def forward(self, x):
        """
        x: (B, L, D)
        Returns: (trend, seasonal), each (B, L, D)
        """
        trend = self.moving_avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        trend = trend[:, :x.size(1), :]
        seasonal = x - trend
        return trend, seasonal


class TokenEmbedding(nn.Module):
    """1D convolution token embedding."""

    def __init__(self, c_in, d_model):
        super().__init__()
        self.conv = nn.Conv1d(
            c_in, d_model, kernel_size=3, padding=1, padding_mode='circular', bias=False
        )
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        """x: (B, L, C) -> (B, L, D)"""
        return self.conv(x.permute(0, 2, 1)).transpose(1, 2)


class FeedForward(nn.Module):
    """Standard FFN for Transformer-based models."""

    def __init__(self, d_model, d_ff=None, dropout=0.1, activation='gelu'):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU() if activation == 'gelu' else nn.ReLU()

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))