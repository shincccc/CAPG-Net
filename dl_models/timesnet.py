"""
timesnet.py - TimesNet model for dUT1 forecasting
====================================================
From "TimesNet: Temporal 2D-Variation Modeling for General
Time Series Analysis" (ICLR 2023).

Core idea: use FFT to detect dominant periods -> fold 1D series into
2D tensors -> apply 2D convolution (Inception) to capture intra- and
inter-period variations. Well-suited for signals with clear annual and
semi-annual cycles (e.g., LOD).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import FluidModelBase, register_model


class InceptionBlock(nn.Module):
    """
    Simplified Inception Block: multi-scale 2D convolution.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = max(out_channels // 4, 1)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid),
            nn.GELU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            nn.Conv2d(mid, mid, kernel_size=5, padding=2),
            nn.BatchNorm2d(mid),
            nn.GELU(),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, mid, kernel_size=1),
            nn.BatchNorm2d(mid),
            nn.GELU(),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1),
            nn.BatchNorm2d(mid),
            nn.GELU(),
        )
        self.proj = nn.Conv2d(mid * 4, out_channels, kernel_size=1)

    def forward(self, x):
        """x: (B, C, H, W)"""
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        return self.proj(out)


class TimesBlock(nn.Module):
    """
    TimesNet core block:
    1. FFT detects top-k dominant periods
    2. For each period: 1D -> 2D folding -> Inception 2D Conv -> unfold to 1D
    3. Adaptive aggregation across k periods
    """

    def __init__(self, d_model, d_ff=64, top_k=3):
        super().__init__()
        self.top_k = top_k
        self.conv2d = InceptionBlock(d_model, d_ff)
        self.proj_back = nn.Linear(d_ff, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """x: (B, L, D)"""
        B, L, D = x.shape
        residual = x

        # 1. FFT to detect dominant periods
        # Pad to next power of 2 for cuFFT compatibility
        L_fft = 1 << (L - 1).bit_length()
        if L_fft > L:
            x_pad_fft = F.pad(x, (0, 0, 0, L_fft - L))
        else:
            x_pad_fft = x
        x_freq = torch.fft.rfft(x_pad_fft, dim=1)
        x_freq = x_freq[:, : L // 2 + 1, :]
        freq_amp = x_freq.abs().mean(dim=-1)  # (B, F) mean amplitude
        freq_amp[:, 0] = 0  # Remove DC component

        _, top_indices = freq_amp.topk(self.top_k, dim=1)  # (B, k)

        # 2. Process each period with 2D convolution
        results = []
        weights = []

        for k_idx in range(self.top_k):
            freq_idx = top_indices[:, k_idx]  # (B,)
            # Use batch mode (mean frequency) for uniform period
            period = max(int(L / (freq_idx.float().mean().item() + 1)), 2)
            period = min(period, L)

            # Pad to integer multiple of period
            n_periods = (L + period - 1) // period
            pad_len = n_periods * period - L
            if pad_len > 0:
                x_pad = F.pad(x, (0, 0, 0, pad_len))
            else:
                x_pad = x

            # 1D -> 2D: (B, L', D) -> (B, D, n_periods, period)
            x_2d = x_pad.permute(0, 2, 1).reshape(B, D, n_periods, period)

            # 2D convolution
            out_2d = self.conv2d(x_2d)    # (B, d_ff, n_periods, period)

            # 2D -> 1D
            out_1d = out_2d.reshape(B, -1, n_periods * period).permute(0, 2, 1)
            out_1d = out_1d[:, :L, :]     # Truncate to original length

            results.append(out_1d)
            weights.append(freq_amp[:, freq_idx[0]].unsqueeze(-1).unsqueeze(-1))

        # 3. Adaptive softmax-weighted aggregation
        weights = torch.softmax(torch.stack(weights, dim=0), dim=0)  # (k, B, 1, 1)
        results = torch.stack(results, dim=0)                         # (k, B, L, d_ff)
        out = (weights * results).sum(dim=0)                         # (B, L, d_ff)

        out = self.proj_back(out)
        return self.norm(out + residual)


@register_model(
    'timesnet',
    category='cnn',
    description='TimesNet: FFT period detection + 2D convolutions for multi-period modeling.',
    recommended_for=['medium', 'long'],
)
class TimesNetFluidModel(FluidModelBase):
    """
    TimesNet (Wu et al., ICLR 2023).

    Architecture:
    1. Input embedding: input_dim -> d_model
    2. N TimesBlocks (FFT period detection + 2D Inception)
    3. Flatten -> Linear prediction head
    """

    def __init__(self, input_dim=1, lookback=360, forecast_days=180,
                 d_model=32, d_ff=32, e_layers=2, top_k=3,
                 dropout=0.2, **kwargs):
        super().__init__(input_dim, lookback, forecast_days)

        self.embed = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([
            TimesBlock(d_model, d_ff, top_k) for _ in range(e_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(lookback * d_model, forecast_days)

    def forward(self, x):
        B = x.size(0)
        x = self.embed(x)
        for block in self.blocks:
            x = self.dropout(block(x))
        x = self.norm(x)
        x = x.reshape(B, -1)
        return self.head(x)