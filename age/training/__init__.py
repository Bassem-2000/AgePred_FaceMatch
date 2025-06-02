"""
Age Estimation Training Module

This module provides training utilities, loss functions, and training loops
for the age estimation model.
"""

from .trainer import AgeTrainer, train_one_epoch, validate_model
from .loss_utils import AverageMeter, create_loss_function, EarlyStopping

__all__ = [
    'AgeTrainer', 
    'train_one_epoch', 
    'validate_model',
    'AverageMeter', 
    'create_loss_function', 
    'EarlyStopping'
]