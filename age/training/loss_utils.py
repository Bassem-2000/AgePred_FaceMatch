import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config.config import config


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    Useful for tracking metrics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all tracked values"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update the meter with a new value.
        
        Args:
            val (float): New value to add
            n (int): Number of samples this value represents
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

    def __str__(self):
        return f'{self.avg:.4f}'


class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss stops improving.
    """
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True, verbose=True):
        """
        Args:
            patience (int): Number of epochs to wait before stopping
            min_delta (float): Minimum change to qualify as an improvement
            restore_best_weights (bool): Whether to restore weights of the best model
            verbose (bool): Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, val_loss, model):
        """
        Check if training should be stopped.
        
        Args:
            val_loss (float): Current validation loss
            model: The model being trained
            
        Returns:
            bool: True if training should stop
        """
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            if self.verbose:
                print(f"Validation loss improved to {val_loss:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement for {self.counter}/{self.patience} epochs")
        
        if self.counter >= self.patience:
            self.early_stop = True
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                if self.verbose:
                    print("Restored best weights")
        
        return self.early_stop


class AgeLoss(nn.Module):
    """
    Custom loss function for age estimation that combines different loss components.
    """
    
    def __init__(self, loss_type='l1', weight_l1=1.0, weight_l2=0.0, weight_huber=0.0):
        """
        Args:
            loss_type (str): Primary loss type ('l1', 'l2', 'huber', 'smooth_l1')
            weight_l1 (float): Weight for L1 loss component
            weight_l2 (float): Weight for L2 loss component  
            weight_huber (float): Weight for Huber loss component
        """
        super(AgeLoss, self).__init__()
        
        self.loss_type = loss_type
        self.weight_l1 = weight_l1
        self.weight_l2 = weight_l2
        self.weight_huber = weight_huber
        
        # Initialize loss functions
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.huber_loss = nn.SmoothL1Loss(beta=1.0)
        self.smooth_l1_loss = nn.SmoothL1Loss(beta=1.0)
    
    def forward(self, predictions, targets):
        """
        Compute the combined loss.
        
        Args:
            predictions (torch.Tensor): Model predictions
            targets (torch.Tensor): Ground truth ages
            
        Returns:
            torch.Tensor: Computed loss
        """
        
        # Ensure same shape
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        total_loss = 0.0
        
        # Primary loss
        if self.loss_type == 'l1':
            total_loss += self.l1_loss(predictions, targets)
        elif self.loss_type == 'l2':
            total_loss += self.l2_loss(predictions, targets)
        elif self.loss_type == 'huber':
            total_loss += self.huber_loss(predictions, targets)
        elif self.loss_type == 'smooth_l1':
            total_loss += self.smooth_l1_loss(predictions, targets)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Additional loss components
        if self.weight_l1 > 0:
            total_loss += self.weight_l1 * self.l1_loss(predictions, targets)
        
        if self.weight_l2 > 0:
            total_loss += self.weight_l2 * self.l2_loss(predictions, targets)
        
        if self.weight_huber > 0:
            total_loss += self.weight_huber * self.huber_loss(predictions, targets)
        
        return total_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in age prediction.
    Adapted for regression by converting to classification bins.
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, num_bins=10, age_range=(0, 100)):
        """
        Args:
            alpha (float): Weighting factor
            gamma (float): Focusing parameter
            num_bins (int): Number of age bins for classification
            age_range (tuple): Min and max age values
        """
        super(FocalLoss, self).__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.num_bins = num_bins
        self.age_range = age_range
        
        # Create age bins
        self.age_bins = torch.linspace(age_range[0], age_range[1], num_bins + 1)
    
    def forward(self, predictions, targets):
        """
        Compute focal loss by converting regression to classification.
        
        Args:
            predictions (torch.Tensor): Model predictions
            targets (torch.Tensor): Ground truth ages
            
        Returns:
            torch.Tensor: Computed focal loss
        """
        
        device = predictions.device
        self.age_bins = self.age_bins.to(device)
        
        # Convert targets to bin indices
        targets_binned = torch.bucketize(targets.view(-1), self.age_bins[1:-1])
        
        # Create logits for classification (simple approach)
        logits = predictions.repeat(1, self.num_bins) if predictions.dim() == 2 else predictions.unsqueeze(1).repeat(1, self.num_bins)
        
        # Compute cross entropy
        ce_loss = F.cross_entropy(logits, targets_binned, reduction='none')
        
        # Compute focal weight
        pt = torch.exp(-ce_loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE Loss that gives more importance to certain age ranges.
    """
    
    def __init__(self, age_weights=None, age_ranges=None):
        """
        Args:
            age_weights (list): Weights for different age ranges
            age_ranges (list): Age range boundaries
        """
        super(WeightedMSELoss, self).__init__()
        
        if age_weights is None:
            # Default: give more weight to extreme ages (young and old)
            age_weights = [2.0, 1.0, 1.0, 1.5]  # 0-18, 18-40, 40-65, 65+
            age_ranges = [0, 18, 40, 65, 100]
        
        self.age_weights = torch.tensor(age_weights, dtype=torch.float32)
        self.age_ranges = age_ranges
    
    def forward(self, predictions, targets):
        """
        Compute weighted MSE loss.
        
        Args:
            predictions (torch.Tensor): Model predictions
            targets (torch.Tensor): Ground truth ages
            
        Returns:
            torch.Tensor: Computed weighted loss
        """
        
        device = predictions.device
        self.age_weights = self.age_weights.to(device)
        
        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Compute weights based on target ages
        weights = torch.ones_like(targets)
        
        for i in range(len(self.age_ranges) - 1):
            mask = (targets >= self.age_ranges[i]) & (targets < self.age_ranges[i + 1])
            weights[mask] = self.age_weights[i]
        
        # Compute weighted MSE
        mse = (predictions - targets) ** 2
        weighted_mse = weights * mse
        
        return weighted_mse.mean()


def create_loss_function(loss_type='l1', **kwargs):
    """
    Factory function to create loss functions.
    
    Args:
        loss_type (str): Type of loss function
        **kwargs: Additional arguments for the loss function
        
    Returns:
        nn.Module: The loss function
    """
    
    if loss_type == 'l1' or loss_type == 'mae':
        return nn.L1Loss()
    
    elif loss_type == 'l2' or loss_type == 'mse':
        return nn.MSELoss()
    
    elif loss_type == 'huber' or loss_type == 'smooth_l1':
        beta = kwargs.get('beta', 1.0)
        return nn.SmoothL1Loss(beta=beta)
    
    elif loss_type == 'custom_age':
        return AgeLoss(**kwargs)
    
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    
    elif loss_type == 'weighted_mse':
        return WeightedMSELoss(**kwargs)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def compute_age_metrics(predictions, targets):
    """
    Compute various metrics for age estimation.
    
    Args:
        predictions (torch.Tensor): Model predictions
        targets (torch.Tensor): Ground truth ages
        
    Returns:
        dict: Dictionary containing various metrics
    """
    
    # Convert to numpy for easier computation
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Flatten arrays
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # Compute metrics
    mae = np.mean(np.abs(predictions - targets))
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
    
    # Accuracy within certain thresholds
    acc_1 = np.mean(np.abs(predictions - targets) <= 1) * 100
    acc_3 = np.mean(np.abs(predictions - targets) <= 3) * 100
    acc_5 = np.mean(np.abs(predictions - targets) <= 5) * 100
    acc_10 = np.mean(np.abs(predictions - targets) <= 10) * 100
    
    # R-squared
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'acc_1': acc_1,
        'acc_3': acc_3,
        'acc_5': acc_5,
        'acc_10': acc_10,
        'r2': r2
    }


class MetricTracker:
    """
    Track multiple metrics during training and validation.
    """
    
    def __init__(self, metrics=['mae', 'mse', 'rmse', 'acc_5']):
        """
        Args:
            metrics (list): List of metrics to track
        """
        self.metrics = metrics
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics"""
        self.values = {metric: AverageMeter() for metric in self.metrics}
        self.predictions = []
        self.targets = []
    
    def update(self, predictions, targets, batch_size=None):
        """
        Update metrics with new predictions and targets.
        
        Args:
            predictions (torch.Tensor): Model predictions
            targets (torch.Tensor): Ground truth values
            batch_size (int, optional): Batch size for averaging
        """
        
        if batch_size is None:
            batch_size = len(targets)
        
        # Store for final computation
        self.predictions.extend(predictions.detach().cpu().numpy().flatten())
        self.targets.extend(targets.detach().cpu().numpy().flatten())
        
        # Compute current metrics
        current_metrics = compute_age_metrics(predictions, targets)
        
        # Update meters
        for metric in self.metrics:
            if metric in current_metrics:
                self.values[metric].update(current_metrics[metric], batch_size)
    
    def compute_final_metrics(self):
        """Compute final metrics using all accumulated predictions and targets"""
        if not self.predictions or not self.targets:
            return {}
        
        return compute_age_metrics(np.array(self.predictions), np.array(self.targets))
    
    def get_current_metrics(self):
        """Get current average metrics"""
        return {metric: meter.avg for metric, meter in self.values.items()}
    
    def __str__(self):
        """String representation of current metrics"""
        metric_strs = []
        for metric, meter in self.values.items():
            metric_strs.append(f"{metric}: {meter.avg:.4f}")
        return " | ".join(metric_strs)


class AgeRangeAccuracy:
    """
    Track accuracy within different age ranges.
    Useful for understanding model performance across different demographics.
    """
    
    def __init__(self, age_ranges=None):
        """
        Args:
            age_ranges (list): List of tuples defining age ranges [(min1, max1), (min2, max2), ...]
        """
        if age_ranges is None:
            self.age_ranges = [
                (0, 18, "Children/Teens"),
                (18, 30, "Young Adults"),
                (30, 50, "Middle Age"),
                (50, 70, "Older Adults"),
                (70, 100, "Elderly")
            ]
        else:
            self.age_ranges = age_ranges
        
        self.reset()
    
    def reset(self):
        """Reset all counters"""
        self.range_stats = {}
        for age_range in self.age_ranges:
            range_key = f"{age_range[0]}-{age_range[1]}"
            self.range_stats[range_key] = {
                'name': age_range[2] if len(age_range) > 2 else range_key,
                'total': 0,
                'correct_1': 0,
                'correct_3': 0,
                'correct_5': 0,
                'mae_sum': 0.0
            }
    
    def update(self, predictions, targets):
        """
        Update accuracy statistics for each age range.
        
        Args:
            predictions (torch.Tensor or np.array): Model predictions
            targets (torch.Tensor or np.array): Ground truth ages
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        errors = np.abs(predictions - targets)
        
        for pred, target, error in zip(predictions, targets, errors):
            # Find which age range this target belongs to
            for min_age, max_age, *name in self.age_ranges:
                if min_age <= target < max_age:
                    range_key = f"{min_age}-{max_age}"
                    stats = self.range_stats[range_key]
                    
                    stats['total'] += 1
                    stats['mae_sum'] += error
                    
                    if error <= 1:
                        stats['correct_1'] += 1
                    if error <= 3:
                        stats['correct_3'] += 1
                    if error <= 5:
                        stats['correct_5'] += 1
                    
                    break
    
    def get_statistics(self):
        """Get comprehensive statistics for each age range"""
        results = {}
        
        for range_key, stats in self.range_stats.items():
            if stats['total'] > 0:
                results[range_key] = {
                    'name': stats['name'],
                    'total_samples': stats['total'],
                    'mae': stats['mae_sum'] / stats['total'],
                    'acc_1': (stats['correct_1'] / stats['total']) * 100,
                    'acc_3': (stats['correct_3'] / stats['total']) * 100,
                    'acc_5': (stats['correct_5'] / stats['total']) * 100
                }
            else:
                results[range_key] = {
                    'name': stats['name'],
                    'total_samples': 0,
                    'mae': 0.0,
                    'acc_1': 0.0,
                    'acc_3': 0.0,
                    'acc_5': 0.0
                }
        
        return results
    
    def print_statistics(self):
        """Print formatted statistics for all age ranges"""
        stats = self.get_statistics()
        
        print("\n=== Age Range Performance Analysis ===")
        print(f"{'Age Range':<15} {'Samples':<8} {'MAE':<8} {'Acc@1':<8} {'Acc@3':<8} {'Acc@5':<8}")
        print("-" * 65)
        
        for range_key, data in stats.items():
            print(f"{data['name']:<15} {data['total_samples']:<8} "
                  f"{data['mae']:<8.2f} {data['acc_1']:<8.1f} "
                  f"{data['acc_3']:<8.1f} {data['acc_5']:<8.1f}")


class LossScheduler:
    """
    Dynamic loss function scheduler that adapts the loss function during training.
    """
    
    def __init__(self, initial_loss='l1', schedule=None):
        """
        Args:
            initial_loss (str): Initial loss function type
            schedule (dict): Schedule for changing loss function {epoch: loss_type}
        """
        self.current_loss_type = initial_loss
        self.loss_function = create_loss_function(initial_loss)
        self.schedule = schedule or {}
        self.current_epoch = 0
    
    def step(self, epoch):
        """
        Update loss function based on epoch if scheduled.
        
        Args:
            epoch (int): Current epoch
        """
        self.current_epoch = epoch
        
        if epoch in self.schedule:
            new_loss_type = self.schedule[epoch]
            if new_loss_type != self.current_loss_type:
                self.current_loss_type = new_loss_type
                self.loss_function = create_loss_function(new_loss_type)
                print(f"Switched to {new_loss_type} loss at epoch {epoch}")
    
    def __call__(self, predictions, targets):
        """Forward call to current loss function"""
        return self.loss_function(predictions, targets)


class GradualUnfreezing:
    """
    Utility for gradually unfreezing model layers during training.
    """
    
    def __init__(self, model, unfreeze_schedule=None):
        """
        Args:
            model: The model to apply gradual unfreezing to
            unfreeze_schedule (dict): Schedule for unfreezing {epoch: num_layers_to_unfreeze}
        """
        self.model = model
        self.schedule = unfreeze_schedule or {}
        self.frozen_layers = self._get_layer_names()
        self.unfrozen_count = 0
    
    def _get_layer_names(self):
        """Get names of all layers that can be unfrozen"""
        layer_names = []
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:  # Don't include classifier layers
                layer_names.append(name)
        return layer_names
    
    def step(self, epoch):
        """
        Unfreeze layers based on schedule.
        
        Args:
            epoch (int): Current epoch
        """
        if epoch in self.schedule:
            layers_to_unfreeze = self.schedule[epoch]
            self._unfreeze_layers(layers_to_unfreeze)
    
    def _unfreeze_layers(self, num_layers):
        """Unfreeze specified number of layers from the top"""
        if hasattr(self.model, 'unfreeze_top_layers'):
            self.model.unfreeze_top_layers(num_layers)
            self.unfrozen_count += num_layers
            print(f"Unfroze {num_layers} layers (total unfrozen: {self.unfrozen_count})")


class AdaptiveLossWeight:
    """
    Adaptive loss weighting that adjusts loss components based on training progress.
    """
    
    def __init__(self, base_loss='l1', adaptive_components=None):
        """
        Args:
            base_loss (str): Base loss function type
            adaptive_components (dict): Components to adapt {'component': {'start_weight': w, 'end_weight': w, 'start_epoch': e, 'end_epoch': e}}
        """
        self.base_loss_fn = create_loss_function(base_loss)
        self.adaptive_components = adaptive_components or {}
        self.current_epoch = 0
    
    def _get_adaptive_weight(self, component_config, epoch):
        """Calculate adaptive weight for a component"""
        start_epoch = component_config['start_epoch']
        end_epoch = component_config['end_epoch']
        start_weight = component_config['start_weight']
        end_weight = component_config['end_weight']
        
        if epoch < start_epoch:
            return start_weight
        elif epoch > end_epoch:
            return end_weight
        else:
            # Linear interpolation
            progress = (epoch - start_epoch) / (end_epoch - start_epoch)
            return start_weight + progress * (end_weight - start_weight)
    
    def step(self, epoch):
        """Update current epoch for weight calculation"""
        self.current_epoch = epoch
    
    def __call__(self, predictions, targets):
        """Compute adaptive weighted loss"""
        # Base loss
        total_loss = self.base_loss_fn(predictions, targets)
        
        # Add adaptive components
        for component_name, config in self.adaptive_components.items():
            weight = self._get_adaptive_weight(config, self.current_epoch)
            
            if component_name == 'l2_regularization':
                # Add L2 regularization to predictions
                l2_loss = torch.mean(predictions ** 2)
                total_loss += weight * l2_loss
            
            elif component_name == 'smoothness':
                # Add smoothness penalty (minimize prediction variance in batch)
                smoothness_loss = torch.var(predictions)
                total_loss += weight * smoothness_loss
        
        return total_loss


class LossHistory:
    """
    Track and analyze loss history during training.
    """
    
    def __init__(self, window_size=10):
        """
        Args:
            window_size (int): Window size for moving average calculation
        """
        self.window_size = window_size
        self.losses = []
        self.epochs = []
    
    def update(self, loss, epoch):
        """Add new loss value"""
        self.losses.append(loss)
        self.epochs.append(epoch)
    
    def get_moving_average(self, window=None):
        """Calculate moving average of losses"""
        if window is None:
            window = self.window_size
        
        if len(self.losses) < window:
            return self.losses.copy()
        
        moving_avg = []
        for i in range(len(self.losses)):
            start_idx = max(0, i - window + 1)
            avg = np.mean(self.losses[start_idx:i+1])
            moving_avg.append(avg)
        
        return moving_avg
    
    def detect_plateau(self, patience=5, threshold=1e-4):
        """
        Detect if loss has plateaued.
        
        Args:
            patience (int): Number of epochs to check
            threshold (float): Minimum improvement threshold
            
        Returns:
            bool: True if plateau detected
        """
        if len(self.losses) < patience:
            return False
        
        recent_losses = self.losses[-patience:]
        min_loss = min(recent_losses)
        max_loss = max(recent_losses)
        
        return (max_loss - min_loss) < threshold
    
    def get_improvement_rate(self, window=10):
        """Calculate rate of improvement over recent epochs"""
        if len(self.losses) < window:
            return 0.0
        
        recent_losses = self.losses[-window:]
        if len(recent_losses) < 2:
            return 0.0
        
        # Linear regression to find slope
        x = np.arange(len(recent_losses))
        slope = np.polyfit(x, recent_losses, 1)[0]
        
        return -slope  # Negative slope means improvement


# Utility functions for loss analysis
def analyze_prediction_distribution(predictions, targets, bins=50):
    """
    Analyze the distribution of predictions vs targets.
    
    Args:
        predictions (np.array): Model predictions
        targets (np.array): Ground truth values
        bins (int): Number of bins for histogram
        
    Returns:
        dict: Analysis results
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # Basic statistics
    pred_mean, pred_std = np.mean(predictions), np.std(predictions)
    target_mean, target_std = np.mean(targets), np.std(targets)
    
    # Bias analysis
    bias = np.mean(predictions - targets)
    
    # Distribution analysis
    pred_hist, pred_bins = np.histogram(predictions, bins=bins)
    target_hist, target_bins = np.histogram(targets, bins=bins)
    
    # Correlation
    correlation = np.corrcoef(predictions, targets)[0, 1]
    
    return {
        'prediction_stats': {'mean': pred_mean, 'std': pred_std},
        'target_stats': {'mean': target_mean, 'std': target_std},
        'bias': bias,
        'correlation': correlation,
        'prediction_histogram': (pred_hist, pred_bins),
        'target_histogram': (target_hist, target_bins)
    }


def compute_confidence_intervals(predictions, targets, confidence=0.95):
    """
    Compute confidence intervals for model predictions.
    
    Args:
        predictions (np.array): Model predictions
        targets (np.array): Ground truth values
        confidence (float): Confidence level (0-1)
        
    Returns:
        dict: Confidence interval statistics
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    errors = np.abs(predictions - targets)
    
    # Calculate percentiles
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(errors, lower_percentile)
    upper_bound = np.percentile(errors, upper_percentile)
    
    # Percentage of predictions within confidence interval
    within_ci = np.mean(errors <= upper_bound) * 100
    
    return {
        'confidence_level': confidence,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'mean_error': np.mean(errors),
        'median_error': np.median(errors),
        'percentage_within_ci': within_ci
    }


# Example usage and testing
if __name__ == "__main__":
    print("Testing loss utilities...")
    
    # Create dummy data
    predictions = torch.randn(100, 1) * 10 + 30  # Around age 30
    targets = torch.randn(100, 1) * 8 + 32       # Around age 32
    
    # Test AgeRangeAccuracy
    age_tracker = AgeRangeAccuracy()
    age_tracker.update(predictions, targets)
    age_tracker.print_statistics()
    
    # Test MetricTracker
    metric_tracker = MetricTracker()
    metric_tracker.update(predictions, targets)
    final_metrics = metric_tracker.compute_final_metrics()
    print(f"\nFinal metrics: {final_metrics}")
    
    # Test loss functions
    age_loss = AgeLoss(loss_type='l1')
    loss_value = age_loss(predictions, targets)
    print(f"\nAge loss: {loss_value.item():.4f}")
    
    # Test confidence intervals
    ci_stats = compute_confidence_intervals(predictions, targets)
    print(f"\nConfidence Intervals: {ci_stats}")
    
    print("\nLoss utilities testing completed!")