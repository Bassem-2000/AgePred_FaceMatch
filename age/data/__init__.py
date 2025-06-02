"""
Age Estimation Data Module

This module provides data loading, preprocessing, and dataset utilities
for the age estimation model.
"""

from .dataset import UTKDataset
from .dynamic_loader import (
    UTKFaceDataset, 
    create_dynamic_data_loaders, 
    analyze_dataset_distribution,
    create_balanced_loaders
)

__all__ = [
    'UTKDataset', 
    'UTKFaceDataset',
    'create_dynamic_data_loaders',
    'analyze_dataset_distribution', 
    'create_balanced_loaders'
]