"""
patchtst.py - PatchTST model for dUT1 forecasting
===================================================
Patch Time Series Transformer (Nie et al., ICLR 2023).
Splits time series into fixed-length patches, encodes them with a
Transformer, and maps directly to forecasts.
"""

import torch
import torch.nn as nn
from .base import FluidModelBase, register_model


@register_model(
    'patchtst',
    category='transformer',
    description='PatchTST: Patch embedding + Transformer Encoder. Strong long-term baseline.',
    recommended_for=['medium', 'long'],
)
class PatchTSTFluidModel(FluidModelBase):
    """
    Patch Time Series Transformer (Nie et al., ICLR 2023).

    Splits the time series into fixed-length patches, applies Transformer
    encoding, and maps the encoded representation to the forecast horizon.
    Uses channel-independent processing.
    """

    def __init__(self, input_dim=1, lookback=730, forecast_days=180,
                 patch_len=16, stride=8, d_model=64, n_heads=4,
                 e_layers=2, dropout=0.2, **kwargs):
        super().__init__(input_dim, lookback, forecast_days)
        self.patch_len = patch_len
        self.stride = stride

        # Learnable affine transformation
        self.affine_weight = nn.Parameter(torch.ones(input_dim))
        self.affine_bias = nn.Parameter(torch.zeros(input_dim))

        # Compute padding and number of patches
        self.pad_len = 0
        if (lookback - patch_len) % stride != 0:
            self.pad_len = stride - ((lookback - patch_len) % stride)
        self.num_patches = (lookback + self.pad_len - patch_len) // stride + 1

        self.value_embedding = nn.Linear(patch_len, d_model)
        self.position_embedding = nn.Parameter(
            torch.empty(1, self.num_patches, d_model)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=e_layers
        )
        self.head = nn.Linear(self.num_patches * d_model, forecast_days)

    def forward(self, x):
        B, L, D = x.shape

        # Instance normalization
        seq_mean = x.mean(dim=1, keepdim=True).detach()
        seq_std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = (x - seq_mean) / seq_std
        x = x * self.affine_weight + self.affine_bias

        # Padding
        if self.pad_len > 0:
            x = torch.cat([x, x[:, -1:, :].repeat(1, self.pad_len, 1)], dim=1)
            L = L + self.pad_len

        # Channel independence: (B, L, D) -> (B*D, L)
        x = x.permute(0, 2, 1).contiguous().view(B * D, L)

        # Patching: unfold into (B*D, num_patches, patch_len)
        x = x.unsqueeze(1).unfold(dimension=-1, size=self.patch_len, step=self.stride).squeeze(1)

        # Embed and encode
        x_emb = self.value_embedding(x) + self.position_embedding
        x_emb = self.dropout(x_emb)
        enc_out = self.transformer_encoder(x_emb)
        enc_out = enc_out.reshape(B * D, -1)
        out = self.head(enc_out)

        # Reshape back: (B*D, F) -> (B, F, D)
        out = out.view(B, D, -1).permute(0, 2, 1)
        if self.input_dim == 1:
            out = out.squeeze(-1)
        return out