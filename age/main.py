#!/usr/bin/env python3
"""
Main Training Script for Age Estimation Model (Enhanced with Image Prediction)

This script provides the main entry point for training age estimation models
with the restructured codebase and dynamic data loading. Now includes image prediction functionality.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
import torchmetrics as tm
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# Import our modules
from config.config import config, validate_config, print_config_summary
from models.models import create_model, save_model_checkpoint
from data.dynamic_loader import create_dynamic_data_loaders, analyze_dataset_distribution
from training.trainer import AgeTrainer, train_one_epoch, validate_model
from training.loss_utils import create_loss_function, EarlyStopping, MetricTracker
from inference.predictor import AgePredictor


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Age Estimation Model with Image Prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to custom configuration file (JSON format)'
    )
    
    # Dataset path
    parser.add_argument(
        '--dataset_path', type=str, default=None,
        help='Path to UTKFace dataset folder (overrides config)'
    )
    
    # Training control
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Number of training epochs (overrides config)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=None,
        help='Batch size for training (overrides config)'
    )
    parser.add_argument(
        '--lr', type=float, default=None,
        help='Learning rate (overrides config)'
    )
    
    # Model selection
    parser.add_argument(
        '--model', type=str, choices=['resnet', 'vit'], default=None,
        help='Model architecture to use (overrides config)'
    )
    parser.add_argument(
        '--pretrained', action='store_true',
        help='Use pretrained weights'
    )
    parser.add_argument(
        '--no_pretrained', action='store_true',
        help='Do not use pretrained weights'
    )
    
    # Data filtering
    parser.add_argument(
        '--min_age', type=int, default=None,
        help='Minimum age to include in dataset'
    )
    parser.add_argument(
        '--max_age', type=int, default=None,
        help='Maximum age to include in dataset'
    )
    
    # Resume training
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint to resume training from'
    )
    
    # Evaluation mode
    parser.add_argument(
        '--eval_only', action='store_true',
        help='Only run evaluation, do not train'
    )
    
    # Image prediction mode
    parser.add_argument(
        '--predict', type=str, default=None,
        help='Path to image file or folder for age prediction'
    )
    parser.add_argument(
        '--model_checkpoint', type=str, default=None,
        help='Path to model checkpoint for prediction (required with --predict)'
    )
    parser.add_argument(
        '--save_predictions', action='store_true',
        help='Save prediction results with annotated images'
    )
    parser.add_argument(
        '--show_predictions', action='store_true',
        help='Display prediction results (requires GUI)'
    )
    
    # Data preparation
    parser.add_argument(
        '--analyze_data', action='store_true',
        help='Analyze dataset before training'
    )
    
    # Output control
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Output directory for checkpoints and logs'
    )
    
    # Debug options
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable debug mode (smaller dataset, more verbose output)'
    )
    parser.add_argument(
        '--dry_run', action='store_true',
        help='Dry run - check configuration and data loading without training'
    )
    
    return parser.parse_args()


def load_custom_config(config_path):
    """Load custom configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            custom_config = json.load(f)
        
        # Update global config
        config.update(custom_config)
        print(f"Loaded custom configuration from: {config_path}")
        
    except Exception as e:
        print(f"Error loading custom config: {e}")
        sys.exit(1)


def update_config_from_args(args):
    """Update configuration based on command line arguments."""
    
    # Dataset path
    if args.dataset_path is not None:
        config['dataset_root'] = args.dataset_path
        print(f"Using dataset path: {args.dataset_path}")
    
    # Training parameters
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.lr is not None:
        config['lr'] = args.lr
    
    # Model parameters
    if args.model is not None:
        config['model_name'] = args.model
    
    # Handle pretrained weights
    if args.pretrained:
        config['pretrain_weights'] = 'IMAGENET1K_V2'
    elif args.no_pretrained:
        config['pretrain_weights'] = False
    
    # Age filtering
    if args.min_age is not None:
        config['min_age'] = args.min_age
    if args.max_age is not None:
        config['max_age'] = args.max_age
    
    # Output directory
    if args.output_dir is not None:
        config['checkpoint_dir'] = os.path.join(args.output_dir, 'checkpoints')
        config['tensorboard_log_dir'] = os.path.join(args.output_dir, 'logs')
    
    # Debug mode adjustments
    if args.debug:
        config['epochs'] = min(config['epochs'], 5)
        config['batch_size'] = min(config['batch_size'], 32)
        config['num_workers'] = 0
        print("Debug mode enabled - reduced epochs and batch size")


def get_image_transform():
    """Get the image transformation pipeline for prediction."""
    
    # Get the same transforms used during training
    img_size = config.get('img_size', 224)
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.get('mean', [0.485, 0.456, 0.406]),
            std=config.get('std', [0.229, 0.224, 0.225])
        )
    ])
    
    return transform


def load_and_preprocess_image(image_path):
    """Load and preprocess a single image for prediction."""
    
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_image = image.copy()
        
        # Apply transforms
        transform = get_image_transform()
        processed_image = transform(image).unsqueeze(0)  # Add batch dimension
        
        return processed_image, original_image
        
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None


def predict_single_image(model, image_path, device='cpu'):
    """Predict age for a single image."""
    
    # Load and preprocess image
    processed_image, original_image = load_and_preprocess_image(image_path)
    
    if processed_image is None:
        return None, None, None
    
    # Move to device
    processed_image = processed_image.to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        prediction = model(processed_image)
        predicted_age = prediction.cpu().numpy()[0][0]  # Assuming single output
    
    return predicted_age, original_image, os.path.basename(image_path)


def annotate_image_with_prediction(image, predicted_age, filename):
    """Annotate image with prediction result."""
    
    # Convert PIL image to matplotlib format
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image)
    ax.axis('off')
    
    # Add prediction text
    ax.text(
        10, 30, 
        f'Predicted Age: {predicted_age:.1f}',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        fontsize=14,
        color='black',
        weight='bold'
    )
    
    # Add filename
    ax.text(
        10, image.size[1] - 20,
        f'File: {filename}',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        fontsize=10,
        color='black'
    )
    
    plt.tight_layout()
    return fig


def get_supported_image_extensions():
    """Get list of supported image file extensions."""
    return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']


def find_image_files(path):
    """Find all image files in a given path."""
    
    image_extensions = get_supported_image_extensions()
    image_files = []
    
    if os.path.isfile(path):
        # Single file
        if any(path.lower().endswith(ext) for ext in image_extensions):
            image_files.append(path)
        else:
            print(f"Warning: {path} is not a supported image format")
    
    elif os.path.isdir(path):
        # Directory - find all images
        for root, dirs, files in os.walk(path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
    
    else:
        print(f"Error: Path {path} does not exist")
    
    return sorted(image_files)


def run_image_prediction(model_checkpoint_path, predict_path, save_predictions=False, show_predictions=False):
    """Run age prediction on image(s)."""
    
    print("\n" + "=" * 60)
    print("IMAGE PREDICTION MODE")
    print("=" * 60)
    
    # Load model
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Create model and load weights
        model = create_model()
        checkpoint = torch.load(model_checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        print(f"✅ Model loaded from: {model_checkpoint_path}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Find image files
    image_files = find_image_files(predict_path)
    
    if not image_files:
        print(f"❌ No supported image files found in: {predict_path}")
        return
    
    print(f"Found {len(image_files)} image(s) to process")
    
    # Process each image
    results = []
    predictions_dir = None
    
    if save_predictions:
        predictions_dir = os.path.join(config.get('checkpoint_dir', './'), 'predictions')
        os.makedirs(predictions_dir, exist_ok=True)
        print(f"Predictions will be saved to: {predictions_dir}")
    
    print("\nProcessing images...")
    for i, image_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Processing: {os.path.basename(image_path)}")
        
        # Make prediction
        predicted_age, original_image, filename = predict_single_image(model, image_path, device)
        
        if predicted_age is not None:
            results.append({
                'filename': filename,
                'path': image_path,
                'predicted_age': predicted_age,
                'image': original_image
            })
            
            print(f"  Predicted age: {predicted_age:.1f} years")
            
            # Save annotated image
            if save_predictions and original_image is not None:
                try:
                    fig = annotate_image_with_prediction(original_image, predicted_age, filename)
                    save_path = os.path.join(predictions_dir, f"prediction_{filename}")
                    fig.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    print(f"  Saved to: {save_path}")
                except Exception as e:
                    print(f"  Warning: Could not save prediction: {e}")
            
            # Show prediction
            if show_predictions and original_image is not None:
                try:
                    fig = annotate_image_with_prediction(original_image, predicted_age, filename)
                    plt.show()
                    plt.close(fig)
                except Exception as e:
                    print(f"  Warning: Could not display prediction: {e}")
        
        else:
            print(f"  Failed to process image")
    
    # Print summary
    print("\n" + "=" * 40)
    print("PREDICTION SUMMARY")
    print("=" * 40)
    
    if results:
        ages = [r['predicted_age'] for r in results]
        print(f"Successfully processed: {len(results)} images")
        print(f"Age predictions range: {min(ages):.1f} - {max(ages):.1f} years")
        print(f"Average predicted age: {np.mean(ages):.1f} ± {np.std(ages):.1f} years")
        
        # Show individual results
        print("\nIndividual Results:")
        for result in results:
            print(f"  {result['filename']}: {result['predicted_age']:.1f} years")
    
    else:
        print("❌ No images were successfully processed")


def check_dataset_path():
    """Check and validate dataset path."""
    
    dataset_path = config['dataset_root']
    
    print(f"Checking dataset path: {dataset_path}")
    
    # Convert to absolute path for clarity
    abs_path = os.path.abspath(dataset_path)
    print(f"Absolute path: {abs_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset folder not found: {dataset_path}")
        print("\nPlease ensure the UTKFace dataset is available.")
        print("Expected structure:")
        print("  Datasets/")
        print("    └── utkcropped/")
        print("        ├── 1_0_0_20161219203650636.jpg")
        print("        ├── 2_1_3_20161219203709615.jpg")
        print("        └── ...")
        
        # Suggest some common locations
        common_paths = [
            os.path.join('..', 'Datasets', 'utkcropped'),
            os.path.join('Datasets', 'utkcropped'),
            os.path.join('..', '..', 'Datasets', 'utkcropped'),
            r'D:\Datasets\utkcropped',
            r'C:\Datasets\utkcropped'
        ]
        
        print("\nTrying common dataset locations:")
        found_alternative = None
        for path in common_paths:
            abs_alt = os.path.abspath(path)
            print(f"  Checking: {abs_alt}")
            if os.path.exists(path):
                print(f"  ✅ Found dataset at: {abs_alt}")
                found_alternative = path
                break
            else:
                print(f"  ❌ Not found")
        
        if found_alternative:
            user_input = input(f"\nUse found dataset at {found_alternative}? (y/n): ")
            if user_input.lower() in ['y', 'yes']:
                config['dataset_root'] = found_alternative
                return found_alternative
        
        sys.exit(1)
    
    print(f"✅ Dataset folder found: {dataset_path}")
    return dataset_path


def prepare_dataset(analyze_data=False):
    """Prepare the dataset by validating the folder and optionally analyzing."""
    
    print("=" * 60)
    print("PREPARING DATASET")
    print("=" * 60)
    
    # Check and validate dataset path
    dataset_folder = check_dataset_path()
    
    # Analyze dataset if requested
    if analyze_data:
        print("\nAnalyzing dataset distribution...")
        try:
            stats = analyze_dataset_distribution(dataset_folder)
            print(f"✅ Dataset analysis completed")
            print(f"   Total samples: {stats['total_samples']}")
            print(f"   Age range: {stats['age_stats']['min']} - {stats['age_stats']['max']}")
            print(f"   Mean age: {stats['age_stats']['mean']:.1f} ± {stats['age_stats']['std']:.1f}")
        except Exception as e:
            print(f"⚠️  Warning: Could not analyze dataset: {e}")
    
    return dataset_folder


def setup_data_loaders(dataset_folder):
    """Setup data loaders for training, validation, and testing."""
    
    print("\n" + "=" * 60)
    print("SETTING UP DATA LOADERS")
    print("=" * 60)
    
    # Create data loaders dynamically
    print("Creating dynamic data loaders...")
    try:
        train_loader, valid_loader, test_loader = create_dynamic_data_loaders(
            dataset_folder=dataset_folder,
            min_age=config['min_age'],
            max_age=config['max_age'],
            validate_files=True
        )
        print("✅ Data loaders created successfully")
        return train_loader, valid_loader, test_loader
    except Exception as e:
        print(f"❌ Error creating data loaders: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def create_optimizer_and_scheduler(model):
    """Create optimizer and learning rate scheduler."""
    
    # Create optimizer
    if config.get('optimizer', 'sgd').lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['wd']
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['lr'],
            momentum=config.get('momentum', 0.9),
            weight_decay=config['wd']
        )
    
    # Create learning rate scheduler
    scheduler_type = config.get('scheduler', 'step')
    
    if scheduler_type == 'step':
        scheduler = StepLR(
            optimizer, 
            step_size=config.get('lr_step_size', 20),
            gamma=config.get('lr_gamma', 0.1)
        )
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config['epochs'],
            eta_min=config.get('lr_min', 1e-6)
        )
    elif scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.get('lr_gamma', 0.1),
            patience=config.get('lr_patience', 10),
            verbose=True
        )
    else:
        scheduler = None
    
    print(f"✅ Created {config.get('optimizer', 'sgd').upper()} optimizer")
    if scheduler:
        print(f"✅ Created {scheduler_type} learning rate scheduler")
    
    return optimizer, scheduler


def setup_training_components(train_loader, valid_loader, test_loader):
    """Setup all training components: model, optimizer, loss, etc."""
    
    print("\n" + "=" * 60)
    print("SETTING UP TRAINING COMPONENTS")
    print("=" * 60)
    
    # Create model
    print("Creating model...")
    model = create_model()
    print("✅ Model created successfully")
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model)
    
    # Create loss function
    loss_type = config.get('loss_function', 'l1')
    loss_function = create_loss_function(loss_type)
    print(f"✅ Created {loss_type.upper()} loss function")
    
    # Create trainer
    trainer = AgeTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        scheduler=scheduler
    )
    print("✅ Trainer initialized successfully")
    
    return trainer


def run_evaluation_only(checkpoint_path, test_loader):
    """Run evaluation only mode."""
    
    print("\n" + "=" * 60)
    print("EVALUATION ONLY MODE")
    print("=" * 60)
    
    try:
        # Load model for evaluation
        predictor = AgePredictor(model_path=checkpoint_path)
        
        # If test loader is available, run comprehensive evaluation
        if test_loader:
            print("Running evaluation on test set...")
            
            # Create a simple evaluation loop
            predictor.model.eval()
            metric_tracker = MetricTracker(['mae', 'mse', 'rmse', 'acc_1', 'acc_3', 'acc_5', 'acc_10', 'r2'])
            
            with torch.no_grad():
                for inputs, targets, _, _ in test_loader:
                    inputs = inputs.to(predictor.device)
                    targets = targets.to(predictor.device)
                    
                    outputs = predictor.model(inputs)
                    metric_tracker.update(outputs, targets)
            
            # Print results
            final_metrics = metric_tracker.compute_final_metrics()
            print("\n" + "=" * 40)
            print("FINAL EVALUATION RESULTS")
            print("=" * 40)
            
            for metric, value in final_metrics.items():
                if metric.startswith('acc_'):
                    print(f"{metric.upper()}: {value:.1f}%")
                else:
                    print(f"{metric.upper()}: {value:.4f}")
        
        else:
            print("No test loader available for evaluation")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        sys.exit(1)


def run_training(trainer, resume_checkpoint=None):
    """Run the main training loop."""
    
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    # Resume from checkpoint if specified
    if resume_checkpoint:
        print(f"Resuming training from: {resume_checkpoint}")
        try:
            trainer.load_checkpoint(resume_checkpoint)
            print("✅ Successfully resumed from checkpoint")
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            sys.exit(1)
    
    # Start training
    try:
        start_time = time.time()
        trainer.train()
        total_time = time.time() - start_time
        
        print(f"\n🎉 Training completed successfully!")
        print(f"Total training time: {total_time/3600:.2f} hours")
        
        # Plot training curves
        try:
            plot_path = os.path.join(config['checkpoint_dir'], 'training_curves.png')
            trainer.plot_training_curves(plot_path)
            print(f"📊 Training curves saved to: {plot_path}")
        except Exception as e:
            print(f"⚠️  Could not save training curves: {e}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Main execution function."""
    
    print("🚀 Age Estimation Model Training & Prediction")
    print("=" * 60)
    
    # Parse arguments
    args = parse_arguments()
    
    # Check if this is prediction mode
    if args.predict:
        if not args.model_checkpoint:
            print("❌ Error: --predict requires --model_checkpoint")
            print("Example: python main.py --predict /path/to/image.jpg --model_checkpoint /path/to/model.pth")
            sys.exit(1)
        
        # Load minimal config for prediction
        if args.config:
            load_custom_config(args.config)
        
        # Run prediction
        run_image_prediction(
            args.model_checkpoint, 
            args.predict,
            save_predictions=args.save_predictions,
            show_predictions=args.show_predictions
        )
        return
    
    # Load custom configuration if provided
    if args.config:
        load_custom_config(args.config)
    
    # Update configuration from command line arguments
    update_config_from_args(args)
    
    # Validate and setup configuration
    validate_config()
    
    # Print configuration summary
    print_config_summary()
    
    # Early exit for dry run
    if args.dry_run:
        print("\n✅ Dry run completed - configuration and setup look good!")
        return
    
    # Prepare dataset
    dataset_folder = prepare_dataset(analyze_data=args.analyze_data)
    
    # Setup data loaders
    train_loader, valid_loader, test_loader = setup_data_loaders(dataset_folder)
    
    # Evaluation only mode
    if args.eval_only:
        if not args.resume:
            print("❌ Error: --eval_only requires --resume checkpoint")
            sys.exit(1)
        run_evaluation_only(args.resume, test_loader)
        return
    
    # Setup training components
    trainer = setup_training_components(train_loader, valid_loader, test_loader)
    
    # Run training
    success = run_training(trainer, args.resume)
    
    if success:
        print("\n🎉 All tasks completed successfully!")
    else:
        print("\n❌ Training was not completed successfully")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Program interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)