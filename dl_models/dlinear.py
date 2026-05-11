"""
dlinear.py - DLinear model for dUT1 forecasting
=================================================
From "Are Transformers Effective for Time Series Forecasting?" (AAAI 2023)

A minimalist yet highly competitive baseline: decomposes the series into
trend and seasonal components, then applies linear projection to each.
"""

import torch
import torch.nn as nn
from .base import FluidModelBase, SeriesDecomposition, register_model


@register_model(
    'dlinear',
    category='mlp',
    description='DLinear: trend-seasonal decomposition + dual linear projection',
    recommended_for=['short', 'medium', 'long'],
)
class DLinearFluidModel(FluidModelBase):
    """
    DLinear: Decomposition-Linear

    1. Decompose input into trend and seasonal components (moving average)
    2. Apply independent linear layers to project each to forecast horizon
    3. Sum trend and seasonal predictions to produce the final output

    """

    def __init__(self, input_dim=1, lookback=360, forecast_days=180,
                 kernel_size=25, individual=True, **kwargs):
        """
        Args:
            kernel_size: Moving average window size (controls decomposition granularity)
            individual:  If True, each channel gets its own linear layer
        """
        super().__init__(input_dim, lookback, forecast_days)
        self.decomposition = SeriesDecomposition(kernel_size)
        self.individual = individual

        if individual:
            self.linear_trend = nn.ModuleList(
                [nn.Linear(lookback, forecast_days) for _ in range(input_dim)]
            )
            self.linear_seasonal = nn.ModuleList(
                [nn.Linear(lookback, forecast_days) for _ in range(input_dim)]
            )
        else:
            self.linear_trend = nn.Linear(lookback, forecast_days)
            self.linear_seasonal = nn.Linear(lookback, forecast_days)

    def forward(self, x):
        # x: (B, L, D)
        trend, seasonal = self.decomposition(x)

        # (B, L, D) -> (B, D, L) for linear projection along time axis
        trend = trend.permute(0, 2, 1)
        seasonal = seasonal.permute(0, 2, 1)

        if self.individual:
            trend_out = torch.stack(
                [self.linear_trend[i](trend[:, i, :]) for i in range(self.input_dim)],
                dim=1,
            )
            seasonal_out = torch.stack(
                [self.linear_seasonal[i](seasonal[:, i, :]) for i in range(self.input_dim)],
                dim=1,
            )
        else:
            trend_out = self.linear_trend(trend)
            seasonal_out = self.linear_seasonal(seasonal)

        out = trend_out + seasonal_out   # (B, D, Forecast)
        out = out.permute(0, 2, 1)       # (B, Forecast, D)
        if self.input_dim == 1:
            out = out.squeeze(-1)
        return out