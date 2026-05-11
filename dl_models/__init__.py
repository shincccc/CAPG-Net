"""
dl_models - Fluid LOD prediction model library
===============================================

Note for reviewers:
  Only the following models are used in the published paper:

    lstm_enso_gate  : CAPG-Net (full model, A4)
    lstm            : Standard LSTM (baseline A1)
    patchtst        : PatchTST baseline
    dlinear         : DLinear baseline
    tcn             : TCN baseline
    timesnet        : TimesNet baseline

  A2 and A3 are implemented by varying the static_features input
  to these models, not as separate model classes.
"""

from .base import MODEL_REGISTRY, FluidModelBase, register_model
from .losses import PhysicsLoss, WeightedMSELoss, HuberPhysicsLoss, LastDayLoss
from .lstm_models import LSTMFluidModel, LSTMENSOGateFluidModel
from .patchtst import PatchTSTFluidModel
from .dlinear import DLinearFluidModel
from .tcn import TCNFluidModel
from .timesnet import TimesNetFluidModel


def create_model(name, **kwargs):
    """Create a model instance by registered name."""
    if name not in MODEL_REGISTRY:
        available = sorted(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model '{name}'.\n"
            f"Available models: {available}"
        )
    return MODEL_REGISTRY[name]['class'](**kwargs)


def list_models():
    """List all registered models."""
    print(f"\n{'=' * 60}")
    print(f"  Registered models ({len(MODEL_REGISTRY)} total)")
    print(f"{'=' * 60}")
    for name, info in MODEL_REGISTRY.items():
        print(f"  {name:<20} {info['description']}")
    print(f"{'=' * 60}\n")