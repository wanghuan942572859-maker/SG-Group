"""
Autoencoder module for semantic feature compression
"""

from .train_autoencoder import AutoEncoder, train_autoencoder
from .compress_features import compress_features

__all__ = ['AutoEncoder', 'train_autoencoder', 'compress_features']
