"""
Age Estimation Inference Module

This module provides inference utilities for age estimation from facial images.
"""

from .predictor import AgePredictor, predict_single_image, predict_batch_images, load_model_for_inference

__all__ = ['AgePredictor', 'predict_single_image', 'predict_batch_images', 'load_model_for_inference']