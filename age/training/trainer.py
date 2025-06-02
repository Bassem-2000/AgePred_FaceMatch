import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchmetrics as tm
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from .loss_utils import AverageMeter, create_loss_function, EarlyStopping, MetricTracker
from config.config import config


class AgeTrainer:
    """
    Complete training class for age estimation models.
    
    Handles training loop, validation, checkpointing, logging, and visualization.
    """
    
    def __init__(self, model, train_loader, valid_loader, test_loader=None, 
                 loss_function=None, optimizer=None, scheduler=None, device=None):
        """
        Initialize the trainer.
        
        Args:
            model: The age estimation model
            train_loader: Training data loader
            valid_loader: Validation data loader
            test_loader: Test data loader (optional)
            loss_function: Loss function (creates default if None)
            optimizer: Optimizer (creates default if None)
            scheduler: Learning rate scheduler (optional)
            device: Device to use (uses config default if None)
        """
        
        # Set device
        self.device = device or config['device']
        
        # Model and data
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        
        # Loss function
        self.loss_function = loss_function or create_loss_function('l1')
        
        # Optimizer
        if optimizer is None:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config['lr'],
                momentum=config['momentum'],
                weight_decay=config['wd']
            )
        else:
            self.optimizer = optimizer
        
        # Scheduler
        self.scheduler = scheduler
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.valid_losses = []
        self.learning_rates = []
        
        # Checkpointing
        self.checkpoint_dir = config['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_model_path = None
        
        # Logging
        self.writer = SummaryWriter(config['tensorboard_log_dir'])
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('early_stopping_patience', 10),
            verbose=True
        )
        
        print(f"Trainer initialized with device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, epoch):
        """
        Train the model for one epoch.
        
        Args:
            epoch (int): Current epoch number
            
        Returns:
            dict: Training metrics for this epoch
        """
        
        self.model.train()
        
        # Metrics tracking
        loss_meter = AverageMeter()
        metric_tracker = MetricTracker(['mae', 'mse', 'acc_5'])
        
        # Progress bar
        pbar = tqdm(
            self.train_loader, 
            desc=f'Epoch {epoch+1}/{config["epochs"]}',
            bar_format='{desc:<16}{percentage:3.0f}%|{bar:70}{r_bar}'
        )
        
        for batch_idx, (inputs, targets, _, _) in enumerate(pbar):
            # Move to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            batch_size = inputs.size(0)
            loss_meter.update(loss.item(), batch_size)
            metric_tracker.update(outputs, targets, batch_size)
            
            # Update progress bar
            current_metrics = metric_tracker.get_current_metrics()
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'mae': f'{current_metrics.get("mae", 0):.2f}',
                'acc5': f'{current_metrics.get("acc_5", 0):.1f}%'
            })
        
        # Scheduler step (if using)
        if self.scheduler:
            self.scheduler.step()
        
        # Final metrics
        final_metrics = metric_tracker.compute_final_metrics()
        final_metrics['loss'] = loss_meter.avg
        final_metrics['lr'] = self.optimizer.param_groups[0]['lr']
        
        return final_metrics
    
    def validate_epoch(self, epoch, data_loader=None):
        """
        Validate the model.
        
        Args:
            epoch (int): Current epoch number
            data_loader: Data loader to use (uses valid_loader if None)
            
        Returns:
            dict: Validation metrics
        """
        
        if data_loader is None:
            data_loader = self.valid_loader
        
        self.model.eval()
        
        # Metrics tracking
        loss_meter = AverageMeter()
        metric_tracker = MetricTracker(['mae', 'mse', 'rmse', 'acc_1', 'acc_3', 'acc_5', 'acc_10', 'r2'])
        
        # Progress bar
        pbar = tqdm(
            data_loader,
            desc='Validating...',
            bar_format='{desc:<16}{percentage:3.0f}%|{bar:70}{r_bar}'
        )
        
        with torch.no_grad():
            for inputs, targets, _, _ in pbar:
                # Move to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, targets)
                
                # Update metrics
                batch_size = inputs.size(0)
                loss_meter.update(loss.item(), batch_size)
                metric_tracker.update(outputs, targets, batch_size)
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
        
        # Final metrics
        final_metrics = metric_tracker.compute_final_metrics()
        final_metrics['loss'] = loss_meter.avg
        
        return final_metrics
    
    def save_checkpoint(self, epoch, is_best=False, extra_info=None):
        """
        Save model checkpoint.
        
        Args:
            epoch (int): Current epoch
            is_best (bool): Whether this is the best model so far
            extra_info (dict): Additional information to save
        """
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'valid_losses': self.valid_losses,
            'config': config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if extra_info:
            checkpoint.update(extra_info)
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f'epoch-{epoch}-loss_valid-{self.valid_losses[-1]:.3f}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            # Remove previous best model
            if self.best_model_path and os.path.exists(self.best_model_path):
                os.remove(self.best_model_path)
            
            self.best_model_path = checkpoint_path
            print(f'New best model saved at epoch {epoch}')
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path (str): Path to checkpoint file
        """
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.valid_losses = checkpoint.get('valid_losses', [])
        
        print(f'Loaded checkpoint from epoch {self.current_epoch}')
    
    def train(self, num_epochs=None):
        """
        Main training loop.
        
        Args:
            num_epochs (int): Number of epochs to train (uses config default if None)
        """
        
        if num_epochs is None:
            num_epochs = config['epochs']
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.valid_loader.dataset)}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, num_epochs):
            epoch_start = time.time()
            
            # Training
            train_metrics = self.train_epoch(epoch)
            self.train_losses.append(train_metrics['loss'])
            self.learning_rates.append(train_metrics['lr'])
            
            # Validation
            valid_metrics = self.validate_epoch(epoch)
            self.valid_losses.append(valid_metrics['loss'])
            
            # Check if best model
            is_best = valid_metrics['loss'] < self.best_loss
            if is_best:
                self.best_loss = valid_metrics['loss']
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Logging
            self.log_metrics(epoch, train_metrics, valid_metrics)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f'\nEpoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s')
            print(f'Train - Loss: {train_metrics["loss"]:.4f}, MAE: {train_metrics["mae"]:.2f}')
            print(f'Valid - Loss: {valid_metrics["loss"]:.4f}, MAE: {valid_metrics["mae"]:.2f}, '
                  f'Acc@5: {valid_metrics["acc_5"]:.1f}%, R²: {valid_metrics["r2"]:.3f}')
            
            # Early stopping
            if self.early_stopping(valid_metrics['loss'], self.model):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
            
            self.current_epoch = epoch + 1
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")
        
        # Final evaluation
        self.final_evaluation()
        
        # Close tensorboard writer
        self.writer.close()
    
    def log_metrics(self, epoch, train_metrics, valid_metrics):
        """
        Log metrics to tensorboard.
        
        Args:
            epoch (int): Current epoch
            train_metrics (dict): Training metrics
            valid_metrics (dict): Validation metrics
        """
        
        # Loss curves
        self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
        self.writer.add_scalar('Loss/Valid', valid_metrics['loss'], epoch)
        
        # Learning rate
        self.writer.add_scalar('Learning_Rate', train_metrics['lr'], epoch)
        
        # Detailed metrics
        for metric in ['mae', 'mse', 'rmse', 'r2']:
            if metric in train_metrics:
                self.writer.add_scalar(f'{metric.upper()}/Train', train_metrics[metric], epoch)
            if metric in valid_metrics:
                self.writer.add_scalar(f'{metric.upper()}/Valid', valid_metrics[metric], epoch)
        
        # Accuracy metrics
        for acc_metric in ['acc_1', 'acc_3', 'acc_5', 'acc_10']:
            if acc_metric in valid_metrics:
                self.writer.add_scalar(f'Accuracy/{acc_metric}', valid_metrics[acc_metric], epoch)
    
    def final_evaluation(self):
        """Perform final evaluation on test set if available."""
        
        if self.test_loader is None:
            print("No test loader provided, skipping final evaluation")
            return
        
        print("\nPerforming final evaluation on test set...")
        
        # Load best model
        if self.best_model_path:
            checkpoint = torch.load(self.best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']}")
        
        # Evaluate on test set
        test_metrics = self.validate_epoch(-1, self.test_loader)
        
        print("Final Test Results:")
        print(f"  Loss: {test_metrics['loss']:.4f}")
        print(f"  MAE: {test_metrics['mae']:.2f} years")
        print(f"  RMSE: {test_metrics['rmse']:.2f} years")
        print(f"  R²: {test_metrics['r2']:.3f}")
        print(f"  Accuracy within 1 year: {test_metrics['acc_1']:.1f}%")
        print(f"  Accuracy within 3 years: {test_metrics['acc_3']:.1f}%")
        print(f"  Accuracy within 5 years: {test_metrics['acc_5']:.1f}%")
        print(f"  Accuracy within 10 years: {test_metrics['acc_10']:.1f}%")
        
        # Log final results
        for metric, value in test_metrics.items():
            self.writer.add_scalar(f'Final_Test/{metric}', value, 0)
    
    def plot_training_curves(self, save_path=None):
        """
        Plot training and validation curves.
        
        Args:
            save_path (str): Path to save the plot (optional)
        """
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, self.train_losses, 'r-', label='Train Loss')
        axes[0, 0].plot(epochs, self.valid_losses, 'b-', label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Learning rate
        axes[0, 1].plot(epochs, self.learning_rates, 'g-')
        axes[0, 1].set_title('Learning Rate')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].grid(True)
        
        # Loss difference
        loss_diff = [abs(t - v) for t, v in zip(self.train_losses, self.valid_losses)]
        axes[1, 0].plot(epochs, loss_diff, 'm-')
        axes[1, 0].set_title('Train-Validation Loss Difference')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('|Train Loss - Valid Loss|')
        axes[1, 0].grid(True)
        
        # Best loss line
        axes[1, 1].axhline(y=self.best_loss, color='r', linestyle='--', label=f'Best Loss: {self.best_loss:.4f}')
        axes[1, 1].plot(epochs, self.valid_losses, 'b-', label='Validation Loss')
        axes[1, 1].set_title('Best Model Performance')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Validation Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        plt.show()


# Standalone training functions for backward compatibility
def train_one_epoch(model, train_loader, loss_fn, optimizer, metric, epoch=1):
    """
    Train model for one epoch (backward compatibility function).
    
    Args:
        model: The model to train
        train_loader: Training data loader
        loss_fn: Loss function
        optimizer: Optimizer
        metric: Metric to track
        epoch: Current epoch number
        
    Returns:
        tuple: (model, average_loss, metric_value)
    """
    
    model.train()
    loss_meter = AverageMeter()
    
    if hasattr(metric, 'reset'):
        metric.reset()
    
    pbar = tqdm(
        train_loader,
        unit="batch",
        desc=f'Epoch: {epoch+1}/{config["epochs"]}',
        bar_format='{desc:<16}{percentage:3.0f}%|{bar:70}{r_bar}'
    )
    
    for inputs, targets, _, _ in pbar:
        inputs, targets = inputs.to(config['device']), targets.to(config['device'])
        
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        loss_meter.update(loss.item(), n=len(targets))
        
        if hasattr(metric, 'update'):
            metric.update(outputs, targets)
        
        metric_val = metric.compute().item() if hasattr(metric, 'compute') else 0
        pbar.set_postfix(loss=loss_meter.avg, metric=metric_val)
    
    # Clear cache
    del outputs
    torch.cuda.empty_cache()
    
    metric_val = metric.compute().item() if hasattr(metric, 'compute') else loss_meter.avg
    return model, loss_meter.avg, metric_val


def validate_model(model, valid_loader, loss_fn, metric):
    """
    Validate model (backward compatibility function).
    
    Args:
        model: The model to validate
        valid_loader: Validation data loader
        loss_fn: Loss function
        metric: Metric to track
        
    Returns:
        tuple: (average_loss, metric_value)
    """
    
    model.eval()
    loss_meter = AverageMeter()
    
    if hasattr(metric, 'reset'):
        metric.reset()
    
    pbar = tqdm(
        valid_loader,
        unit="batch",
        desc='Evaluating... ',
        bar_format='{desc:<16}{percentage:3.0f}%|{bar:70}{r_bar}'
    )
    
    with torch.no_grad():
        for inputs, targets, _, _ in pbar:
            inputs, targets = inputs.to(config['device']), targets.to(config['device'])
            
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            
            loss_meter.update(loss.item(), n=len(targets))
            
            if hasattr(metric, 'update'):
                metric.update(outputs, targets)
    
    # Clear cache
    del outputs
    torch.cuda.empty_cache()
    
    metric_val = metric.compute().item() if hasattr(metric, 'compute') else loss_meter.avg
    return loss_meter.avg, metric_val