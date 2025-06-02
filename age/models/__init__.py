"""
Age Estimation Models Module

This module provides neural network models for age estimation from facial images.
"""

from .models import AgeEstimationModel, create_model

__all__ = ['AgeEstimationModel', 'create_model']