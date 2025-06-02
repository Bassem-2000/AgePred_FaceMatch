import torch
import os

# Configuration file for Age Estimation model
config = {
    # Dimensions for input images
    'img_width': 128,  # Width of the input images
    'img_height': 128,  # Height of the input images
    'img_size': 128,  # Size of the input images

    # Normalization parameters for the images (ImageNet standards)
    'mean': [0.485, 0.456, 0.406],  # Mean values for normalization 
    'std': [0.229, 0.224, 0.225],  # Standard deviation values for normalization 
    
    # Model configuration
    'model_name': 'resnet',  # Name of the model to be used ('resnet' or 'vit')
    'pretrain_weights': 'IMAGENET1K_V2',  # Pre-trained weights to use
    'leaky_relu': False,  # Flag to indicate if Leaky ReLU should be used

    # Device configuration
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # Dataset paths - Updated for dynamic loading
    'dataset_root': os.path.join('age', 'Datasets', 'utkcropped'),
    
    # Checkpoint paths
    'checkpoint_dir': os.path.join('age','checkpoints'),
    'tensorboard_log_dir': 'runs',
    
    # File paths for testing
    'test_image_dir': os.path.join('Tested_img'),
    'output_image_dir': os.path.join('Tested_img'),
    
    # Training parameters
    'epochs': 50,  # Number of training epochs
    'batch_size': 128,  # Batch size for training
    'eval_batch_size': 256,  # Batch size for evaluation
    'seed': 42,  # Random seed value
    'num_workers': 2,  # Number of workers for data loading
    
    # Data split ratios for dynamic loading
    'train_ratio': 0.8,  # 80% for train+valid, 20% for test
    'valid_ratio': 0.85,  # 85% of train+valid goes to train (68% total), 15% to valid (12% total)
    'stratify_by': 'gender',  # Feature to stratify splits by ('gender', 'ethnicity', or None)
    
    # Dynamic dataset filtering
    'max_age': 85,  # Maximum age to include in dataset
    'min_age': 0,   # Minimum age to include in dataset
    'valid_ethnicities': None,  # List of valid ethnicities (None = all)
    'valid_genders': None,      # List of valid genders (None = all)
    'validate_files': True,     # Whether to validate file existence during loading
    
    # Learning parameters
    'lr': 0.0001,  # Learning rate for the optimizer
    'wd': 0.001,  # Weight decay (L2 regularization) for the optimizer
    'momentum': 0.9,  # Momentum for SGD optimizer
    
    # Optimizer and scheduler
    'optimizer': 'sgd',  # 'sgd' or 'adam'
    'scheduler': 'step',  # 'step', 'cosine', 'plateau', or None
    'lr_step_size': 20,   # Step size for StepLR scheduler
    'lr_gamma': 0.1,      # Gamma for step/plateau scheduler
    'lr_patience': 10,    # Patience for plateau scheduler
    'lr_min': 1e-6,       # Minimum LR for cosine scheduler
    
    # Model parameters
    'input_dim': 3,  # Number of input channels (RGB)
    'output_nodes': 1,  # Number of output nodes (age prediction)
    'dropout_rate': 0.2,  # Dropout rate for regularization
    
    # Loss function
    'loss_function': 'l1',  # 'l1', 'l2', 'huber', 'custom_age'
    
    # Training settings
    'save_best_only': True,  # Save only the best model
    'early_stopping_patience': 10,  # Early stopping patience
    'log_interval': 20,  # Logging interval for training
    'validation_interval': 1,  # Validation interval (epochs)
    
    # Data balancing options
    'balance_dataset': False,  # Whether to balance the dataset
    'balance_by': 'gender',    # Feature to balance by ('gender' or 'ethnicity')
    'samples_per_class': None, # Samples per class for balancing (None = use minimum)
}

# Validate configuration
def validate_config():
    """Validate configuration parameters"""
    
    # Check if CUDA is available when specified
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA specified but not available. Using CPU instead.")
        config['device'] = 'cpu'
    
    # Create necessary directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['tensorboard_log_dir'], exist_ok=True)
    
    # Validate model name
    valid_models = ['resnet', 'vit']
    if config['model_name'] not in valid_models:
        raise ValueError(f"Invalid model name: {config['model_name']}. Must be one of {valid_models}")
    
    # Validate ratios
    if not 0 < config['train_ratio'] <= 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 < config['valid_ratio'] <= 1:
        raise ValueError("valid_ratio must be between 0 and 1")
    
    # Validate age ranges
    if config['min_age'] >= config['max_age']:
        raise ValueError("min_age must be less than max_age")
    
    # Validate stratify_by
    valid_stratify = ['gender', 'ethnicity', None]
    if config['stratify_by'] not in valid_stratify:
        raise ValueError(f"stratify_by must be one of {valid_stratify}")
    
    # Validate balance_by
    valid_balance = ['gender', 'ethnicity']
    if config['balance_by'] not in valid_balance:
        raise ValueError(f"balance_by must be one of {valid_balance}")
    
    # Validate optimizer
    valid_optimizers = ['sgd', 'adam']
    if config['optimizer'] not in valid_optimizers:
        raise ValueError(f"optimizer must be one of {valid_optimizers}")
    
    # Validate scheduler
    valid_schedulers = ['step', 'cosine', 'plateau', None]
    if config['scheduler'] not in valid_schedulers:
        raise ValueError(f"scheduler must be one of {valid_schedulers}")
    
    # Validate loss function
    valid_losses = ['l1', 'l2', 'huber', 'custom_age', 'mae', 'mse', 'smooth_l1']
    if config['loss_function'] not in valid_losses:
        raise ValueError(f"loss_function must be one of {valid_losses}")
    
    print(f"Configuration validated. Using device: {config['device']}")
    print(f"Dataset root: {config['dataset_root']}")
    print(f"Dynamic loading enabled with {config['stratify_by']} stratification")

# Helper function to get effective split sizes
def get_split_info():
    """Get information about train/val/test split ratios."""
    
    train_ratio = config['train_ratio']
    valid_ratio = config['valid_ratio']
    
    # Calculate effective ratios
    effective_train = train_ratio * valid_ratio        # 0.8 * 0.85 = 0.68 (68%)
    effective_valid = train_ratio * (1 - valid_ratio)  # 0.8 * 0.15 = 0.12 (12%)
    effective_test = 1 - train_ratio                   # 0.2 (20%)
    
    return {
        'train_ratio': effective_train,
        'valid_ratio': effective_valid,
        'test_ratio': effective_test,
        'train_percent': effective_train * 100,
        'valid_percent': effective_valid * 100,
        'test_percent': effective_test * 100
    }

# Print configuration summary
def print_config_summary():
    """Print a summary of the current configuration."""
    
    split_info = get_split_info()
    
    print("\n" + "=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    
    print("Dataset Configuration:")
    print(f"  Root folder: {config['dataset_root']}")
    print(f"  Age range: {config['min_age']} - {config['max_age']} years")
    print(f"  File validation: {config['validate_files']}")
    
    print(f"\nData Splitting:")
    print(f"  Train: {split_info['train_percent']:.1f}%")
    print(f"  Validation: {split_info['valid_percent']:.1f}%")
    print(f"  Test: {split_info['test_percent']:.1f}%")
    print(f"  Stratify by: {config['stratify_by']}")
    
    print(f"\nModel Configuration:")
    print(f"  Architecture: {config['model_name']}")
    print(f"  Pretrained: {config['pretrain_weights']}")
    print(f"  Input size: {config['img_size']}x{config['img_size']}")
    print(f"  Dropout rate: {config['dropout_rate']}")
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['lr']}")
    print(f"  Optimizer: {config['optimizer']}")
    print(f"  Scheduler: {config['scheduler']}")
    print(f"  Loss function: {config['loss_function']}")
    print(f"  Early stopping: {config['early_stopping_patience']} epochs")
    
    if config['balance_dataset']:
        print(f"\nData Balancing:")
        print(f"  Balance by: {config['balance_by']}")
        print(f"  Samples per class: {config['samples_per_class'] or 'minimum'}")

# Validate configuration on import
validate_config()