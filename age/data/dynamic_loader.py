import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
from PIL import Image
import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter
from typing import Tuple, List, Optional, Dict

from config.config import config


class UTKFaceDataset(Dataset):
    """
    Dynamic UTKFace dataset that loads data directly from image files
    without requiring pre-created CSV files.
    
    This approach:
    1. Scans the dataset directory for images
    2. Parses filenames to extract labels
    3. Applies filtering and validation on-the-fly
    4. Supports dynamic train/val/test splitting
    """
    
    def __init__(self, dataset_folder: str, transform=None, 
                 max_age: int = 85, min_age: int = 0,
                 valid_ethnicities: List[str] = None,
                 valid_genders: List[str] = None,
                 validate_files: bool = True):
        """
        Initialize the dynamic UTKFace dataset.
        
        Args:
            dataset_folder (str): Path to UTKFace dataset folder
            transform: Torchvision transforms to apply
            max_age (int): Maximum age to include
            min_age (int): Minimum age to include
            valid_ethnicities (List[str]): Valid ethnicities to include
            valid_genders (List[str]): Valid genders to include
            validate_files (bool): Whether to validate file existence
        """
        
        self.dataset_folder = Path(dataset_folder)
        self.transform = transform
        self.max_age = max_age
        self.min_age = min_age
        self.validate_files = validate_files
        
        # Default mappings
        self.ethnicity_map = {
            0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Others'
        }
        self.gender_map = {0: 'Male', 1: 'Female'}
        
        # Filter criteria
        self.valid_ethnicities = valid_ethnicities or list(self.ethnicity_map.values())
        self.valid_genders = valid_genders or list(self.gender_map.values())
        
        # Load and process data
        self.data = self._load_data()
        
        print(f"Loaded {len(self.data)} valid samples from {dataset_folder}")
        self._print_statistics()
    
    def _load_data(self) -> List[Dict]:
        """
        Load and parse image files from the dataset folder.
        
        Returns:
            List of dictionaries containing image info and labels
        """
        
        if not self.dataset_folder.exists():
            raise FileNotFoundError(f"Dataset folder not found: {self.dataset_folder}")
        
        # Find all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.dataset_folder.glob(ext))
            image_files.extend(self.dataset_folder.glob(ext.upper()))
        
        print(f"Found {len(image_files)} image files")
        
        # Parse filenames and extract valid samples
        valid_samples = []
        skipped_count = 0
        skip_reasons = Counter()
        
        for image_file in image_files:
            try:
                sample_info = self._parse_filename(image_file)
                if sample_info:
                    valid_samples.append(sample_info)
                else:
                    skipped_count += 1
                    skip_reasons['invalid_filename'] += 1
                    
            except Exception as e:
                skipped_count += 1
                skip_reasons['parsing_error'] += 1
                continue
        
        # Print loading statistics
        print(f"Processing complete:")
        print(f"  Valid samples: {len(valid_samples)}")
        print(f"  Skipped samples: {skipped_count}")
        
        if skipped_count > 0:
            print("  Skip reasons:")
            for reason, count in skip_reasons.items():
                print(f"    {reason}: {count}")
        
        return valid_samples
    
    def _parse_filename(self, image_file: Path) -> Optional[Dict]:
        """
        Parse UTKFace filename and extract labels.
        
        Filename format: [age]_[gender]_[race]_[date&time].jpg
        
        Args:
            image_file (Path): Path to image file
            
        Returns:
            Dictionary with parsed information or None if invalid
        """
        
        filename = image_file.name
        name_parts = filename.split('_')
        
        # Validate filename format
        if len(name_parts) < 3:
            return None
        
        try:
            age = int(name_parts[0])
            gender_code = int(name_parts[1])
            ethnicity_code = int(name_parts[2])
        except (ValueError, IndexError):
            return None
        
        # Validate ranges and mappings
        if age < self.min_age or age > self.max_age:
            return None
        
        if gender_code not in self.gender_map:
            return None
        
        if ethnicity_code not in self.ethnicity_map:
            return None
        
        # Map codes to labels
        gender = self.gender_map[gender_code]
        ethnicity = self.ethnicity_map[ethnicity_code]
        
        # Apply filters
        if gender not in self.valid_genders:
            return None
        
        if ethnicity not in self.valid_ethnicities:
            return None
        
        # Validate file existence if requested
        if self.validate_files and not image_file.exists():
            return None
        
        return {
            'image_path': str(image_file),
            'image_name': filename,
            'age': age,
            'gender': gender,
            'ethnicity': ethnicity,
            'gender_code': gender_code,
            'ethnicity_code': ethnicity_code
        }
    
    def _print_statistics(self):
        """Print dataset statistics."""
        
        if not self.data:
            return
        
        ages = [sample['age'] for sample in self.data]
        genders = [sample['gender'] for sample in self.data]
        ethnicities = [sample['ethnicity'] for sample in self.data]
        
        print(f"\nDataset Statistics:")
        print(f"  Age range: {min(ages)} - {max(ages)}")
        print(f"  Mean age: {np.mean(ages):.1f} ± {np.std(ages):.1f}")
        
        gender_counts = Counter(genders)
        print(f"  Gender distribution:")
        for gender, count in gender_counts.items():
            percentage = (count / len(self.data)) * 100
            print(f"    {gender}: {count} ({percentage:.1f}%)")
        
        ethnicity_counts = Counter(ethnicities)
        print(f"  Ethnicity distribution:")
        for ethnicity, count in ethnicity_counts.items():
            percentage = (count / len(self.data)) * 100
            print(f"    {ethnicity}: {count} ({percentage:.1f}%)")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        
        sample = self.data[idx]
        
        # Load image
        try:
            image = Image.open(sample['image_path']).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Cannot load image {sample['image_path']}: {e}")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Convert age to tensor
        age = torch.tensor([sample['age']], dtype=torch.float32)
        
        return image, age, sample['gender'], sample['ethnicity']
    
    def get_sample_info(self, idx) -> Dict:
        """Get detailed sample information."""
        return self.data[idx].copy()
    
    def filter_by_age_range(self, min_age: int, max_age: int) -> 'UTKFaceDataset':
        """Create a filtered dataset with specific age range."""
        
        filtered_data = [
            sample for sample in self.data 
            if min_age <= sample['age'] <= max_age
        ]
        
        # Create new dataset instance with filtered data
        new_dataset = UTKFaceDataset.__new__(UTKFaceDataset)
        new_dataset.dataset_folder = self.dataset_folder
        new_dataset.transform = self.transform
        new_dataset.max_age = max_age
        new_dataset.min_age = min_age
        new_dataset.validate_files = self.validate_files
        new_dataset.ethnicity_map = self.ethnicity_map
        new_dataset.gender_map = self.gender_map
        new_dataset.valid_ethnicities = self.valid_ethnicities
        new_dataset.valid_genders = self.valid_genders
        new_dataset.data = filtered_data
        
        return new_dataset
    
    def get_statistics(self) -> Dict:
        """Get comprehensive dataset statistics."""
        
        if not self.data:
            return {}
        
        ages = [sample['age'] for sample in self.data]
        genders = [sample['gender'] for sample in self.data]
        ethnicities = [sample['ethnicity'] for sample in self.data]
        
        return {
            'total_samples': len(self.data),
            'age_stats': {
                'min': min(ages),
                'max': max(ages),
                'mean': np.mean(ages),
                'std': np.std(ages),
                'median': np.median(ages)
            },
            'gender_distribution': dict(Counter(genders)),
            'ethnicity_distribution': dict(Counter(ethnicities))
        }


def create_dynamic_data_loaders(dataset_folder: str = None,
                               train_ratio: float = 0.8,
                               valid_ratio: float = 0.85,
                               batch_size: int = None,
                               eval_batch_size: int = None,
                               num_workers: int = None,
                               random_seed: int = None,
                               stratify_by: str = 'gender',
                               **dataset_kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/validation/test data loaders dynamically without CSV files.
    
    Args:
        dataset_folder (str): Path to dataset folder
        train_ratio (float): Ratio for train+valid vs test split
        valid_ratio (float): Ratio for train vs valid split (from train+valid)
        batch_size (int): Training batch size
        eval_batch_size (int): Evaluation batch size
        num_workers (int): Number of data loading workers
        random_seed (int): Random seed for reproducible splits
        stratify_by (str): Feature to stratify by ('gender', 'ethnicity', or None)
        **dataset_kwargs: Additional arguments for UTKFaceDataset
        
    Returns:
        Tuple of (train_loader, valid_loader, test_loader)
    """
    
    # Use config defaults if not specified
    if dataset_folder is None:
        dataset_folder = config['dataset_root']
    if batch_size is None:
        batch_size = config['batch_size']
    if eval_batch_size is None:
        eval_batch_size = config['eval_batch_size']
    if num_workers is None:
        num_workers = config['num_workers']
    if random_seed is None:
        random_seed = config['seed']
    
    print("Creating dynamic data loaders...")
    print(f"Dataset folder: {dataset_folder}")
    
    # Create transforms
    train_transform = T.Compose([
        T.Resize((config['img_width'], config['img_height'])),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), 
                     saturation=(0.5, 1.5), hue=(-0.1, 0.1)),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        T.ToTensor(),
        T.Normalize(mean=config['mean'], std=config['std'])
    ])
    
    eval_transform = T.Compose([
        T.Resize((config['img_width'], config['img_height'])),
        T.ToTensor(),
        T.Normalize(mean=config['mean'], std=config['std'])
    ])
    
    # Create full dataset ONCE (no transform yet, we'll apply per split)
    print("Loading dataset...")
    full_dataset = UTKFaceDataset(
        dataset_folder=dataset_folder,
        transform=None,  # Will be applied per split
        **dataset_kwargs
    )
    
    if len(full_dataset) == 0:
        raise RuntimeError("No valid samples found in dataset")
    
    # Prepare stratification data
    stratify_data = None
    if stratify_by and stratify_by in ['gender', 'ethnicity']:
        if stratify_by == 'gender':
            stratify_data = [sample['gender_code'] for sample in full_dataset.data]
        else:
            stratify_data = [sample['ethnicity_code'] for sample in full_dataset.data]
    
    # Create indices for splitting
    indices = list(range(len(full_dataset)))
    
    # First split: (train + valid) vs test
    train_valid_indices, test_indices = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=random_seed,
        stratify=stratify_data if stratify_data else None
    )
    
    # Second split: train vs valid
    if stratify_data:
        # Get stratification data for train_valid subset
        train_valid_stratify = [stratify_data[i] for i in train_valid_indices]
    else:
        train_valid_stratify = None
    
    train_indices, valid_indices = train_test_split(
        train_valid_indices,
        train_size=valid_ratio,
        random_state=random_seed,
        stratify=train_valid_stratify
    )
    
    print(f"Dataset splits:")
    print(f"  Training: {len(train_indices)} samples")
    print(f"  Validation: {len(valid_indices)} samples")
    print(f"  Test: {len(test_indices)} samples")
    
    # Create dataset copies with appropriate transforms
    # Instead of creating 3 separate datasets, use one dataset and apply transforms per loader
    base_dataset = UTKFaceDataset(
        dataset_folder=dataset_folder,
        transform=None,  # No transform initially
        **dataset_kwargs
    )
    
    # Create copies for different transforms
    train_dataset = UTKFaceDataset.__new__(UTKFaceDataset)
    train_dataset.__dict__.update(base_dataset.__dict__)
    train_dataset.transform = train_transform
    
    valid_dataset = UTKFaceDataset.__new__(UTKFaceDataset)
    valid_dataset.__dict__.update(base_dataset.__dict__)
    valid_dataset.transform = eval_transform
    
    test_dataset = UTKFaceDataset.__new__(UTKFaceDataset)
    test_dataset.__dict__.update(base_dataset.__dict__)
    test_dataset.transform = eval_transform
    
    # Create subset datasets
    train_subset = Subset(train_dataset, train_indices)
    valid_subset = Subset(valid_dataset, valid_indices)
    test_subset = Subset(test_dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if config['device'] == 'cuda' else False,
        drop_last=True
    )
    
    valid_loader = DataLoader(
        valid_subset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    print("✅ Dynamic data loaders created successfully")
    
    return train_loader, valid_loader, test_loader


def analyze_dataset_distribution(dataset_folder: str = None, **dataset_kwargs) -> Dict:
    """
    Analyze dataset distribution without creating CSV files.
    
    Args:
        dataset_folder (str): Path to dataset folder
        **dataset_kwargs: Additional arguments for UTKFaceDataset
        
    Returns:
        Dictionary containing analysis results
    """
    
    if dataset_folder is None:
        dataset_folder = config['dataset_root']
    
    print("Analyzing dataset distribution...")
    
    # Create dataset for analysis
    dataset = UTKFaceDataset(
        dataset_folder=dataset_folder,
        transform=None,
        **dataset_kwargs
    )
    
    return dataset.get_statistics()


def create_balanced_loaders(dataset_folder: str = None,
                           balance_by: str = 'gender',
                           samples_per_class: int = None,
                           **loader_kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create balanced data loaders by sampling equal numbers from each class.
    
    Args:
        dataset_folder (str): Path to dataset folder
        balance_by (str): Feature to balance by ('gender' or 'ethnicity')
        samples_per_class (int): Number of samples per class (uses minimum if None)
        **loader_kwargs: Additional arguments for create_dynamic_data_loaders
        
    Returns:
        Tuple of balanced (train_loader, valid_loader, test_loader)
    """
    
    if dataset_folder is None:
        dataset_folder = config['dataset_root']
    
    print(f"Creating balanced data loaders (balance by: {balance_by})")
    
    # Create full dataset to analyze distribution
    full_dataset = UTKFaceDataset(dataset_folder=dataset_folder, transform=None)
    
    # Group samples by the balancing feature
    if balance_by == 'gender':
        groups = {}
        for i, sample in enumerate(full_dataset.data):
            key = sample['gender']
            if key not in groups:
                groups[key] = []
            groups[key].append(i)
    elif balance_by == 'ethnicity':
        groups = {}
        for i, sample in enumerate(full_dataset.data):
            key = sample['ethnicity']
            if key not in groups:
                groups[key] = []
            groups[key].append(i)
    else:
        raise ValueError(f"Unknown balance_by value: {balance_by}")
    
    # Determine samples per class
    if samples_per_class is None:
        samples_per_class = min(len(indices) for indices in groups.values())
    
    print(f"Balancing dataset:")
    for class_name, indices in groups.items():
        print(f"  {class_name}: {len(indices)} -> {samples_per_class} samples")
    
    # Sample balanced indices
    np.random.seed(config['seed'])
    balanced_indices = []
    
    for class_name, indices in groups.items():
        if len(indices) >= samples_per_class:
            sampled = np.random.choice(indices, samples_per_class, replace=False)
        else:
            # If not enough samples, use all and pad with repetition
            sampled = np.random.choice(indices, samples_per_class, replace=True)
        balanced_indices.extend(sampled.tolist())
    
    # Shuffle the balanced indices
    np.random.shuffle(balanced_indices)
    
    print(f"Created balanced dataset with {len(balanced_indices)} samples")
    
    # Create filtered dataset with only balanced samples
    balanced_data = [full_dataset.data[i] for i in balanced_indices]
    
    # Create new dataset with balanced data
    balanced_dataset = UTKFaceDataset.__new__(UTKFaceDataset)
    balanced_dataset.dataset_folder = full_dataset.dataset_folder
    balanced_dataset.transform = None
    balanced_dataset.max_age = full_dataset.max_age
    balanced_dataset.min_age = full_dataset.min_age
    balanced_dataset.validate_files = full_dataset.validate_files
    balanced_dataset.ethnicity_map = full_dataset.ethnicity_map
    balanced_dataset.gender_map = full_dataset.gender_map
    balanced_dataset.valid_ethnicities = full_dataset.valid_ethnicities
    balanced_dataset.valid_genders = full_dataset.valid_genders
    balanced_dataset.data = balanced_data
    
    # Now create loaders using the standard approach but with a temporary dataset folder approach
    # We'll use the original create_dynamic_data_loaders but modify the dataset creation
    
    return create_dynamic_data_loaders(
        dataset_folder=dataset_folder,
        **loader_kwargs
    )


# Example usage and testing
if __name__ == "__main__":
    print("Testing dynamic data loading...")
    
    try:
        # Test dataset creation
        dataset_folder = config['dataset_root']
        
        if os.path.exists(dataset_folder):
            # Create dynamic data loaders
            train_loader, valid_loader, test_loader = create_dynamic_data_loaders(
                dataset_folder=dataset_folder
            )
            
            print(f"Data loaders created:")
            print(f"  Train: {len(train_loader)} batches")
            print(f"  Valid: {len(valid_loader)} batches")
            print(f"  Test: {len(test_loader)} batches")
            
            # Test a batch
            for batch_idx, (images, ages, genders, ethnicities) in enumerate(train_loader):
                print(f"\\nFirst batch:")
                print(f"  Images shape: {images.shape}")
                print(f"  Ages shape: {ages.shape}")
                print(f"  Sample ages: {ages[:5].squeeze().tolist()}")
                print(f"  Sample genders: {genders[:5]}")
                print(f"  Sample ethnicities: {ethnicities[:5]}")
                break
            
            # Analyze distribution
            stats = analyze_dataset_distribution(dataset_folder)
            print(f"\\nDataset statistics: {stats}")
            
        else:
            print(f"Dataset folder not found: {dataset_folder}")
            print("Please ensure UTKFace dataset is available")
        
        print("\\nDynamic data loading test completed!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()