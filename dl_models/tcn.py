"""
tcn.py - TCN (Temporal Convolutional Network) model for dUT1 forecasting
==========================================================================
Classic causal dilated convolution architecture with exponentially
growing receptive field. Trains faster than RNNs.
"""

import torch
import torch.nn as nn
from .base import FluidModelBase, register_model


class CausalConv1d(nn.Module):
    """Causal convolution: only sees past, not future."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding,
        )

    def forward(self, x):
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalBlock(nn.Module):
    """
    TCN residual block: two causal dilated convolutions + residual connection.
    """

    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        """x: (B, C, L)"""
        residual = self.downsample(x)
        out = self.dropout(self.act(self.bn1(self.conv1(x))))
        out = self.dropout(self.act(self.bn2(self.conv2(out))))
        return out + residual


@register_model(
    'tcn',
    category='cnn',
    description='TCN: causal dilated convolutions. Fast and stable training.',
    recommended_for=['short', 'medium'],
)
class TCNFluidModel(FluidModelBase):
    """
    Temporal Convolutional Network.

    Architecture:
    1. N stacked TemporalBlocks with exponentially increasing dilation
    2. Extract last time step (causal: contains full history)
    3. Linear prediction head
    """

    def __init__(self, input_dim=1, lookback=360, forecast_days=180,
                 n_channels=64, n_layers=6, kernel_size=7, dropout=0.2,
                 **kwargs):
        super().__init__(input_dim, lookback, forecast_days)

        layers = []
        in_ch = input_dim
        for i in range(n_layers):
            dilation = 2 ** i
            out_ch = n_channels
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch

        self.network = nn.Sequential(*layers)
        self.head = nn.Linear(n_channels, forecast_days)

    def forward(self, x):
        # x: (B, L, D) -> (B, D, L) for Conv1d
        x = x.permute(0, 2, 1)
        out = self.network(x)          # (B, C, L)
        # Take last time step (causal: contains full receptive field)
        out = out[:, :, -1]            # (B, C)
        return self.head(out)          # (B, Forecast)