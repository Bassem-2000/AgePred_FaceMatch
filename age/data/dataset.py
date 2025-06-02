import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import pandas as pd
import os
from config.config import config


class UTKDataset(Dataset):
    """
    Custom dataset class for UTKFace dataset.
    
    The UTKFace dataset contains face images with age, gender, and ethnicity labels
    encoded in the filename format: [age]_[gender]_[race]_[date&time].jpg
    
    Args:
        root_dir (str): Root directory containing the images
        csv_file (str): Path to CSV file containing the dataset splits
        transform (callable, optional): Optional transform to be applied on images
        mode (str): Dataset mode - 'train', 'valid', or 'test'
    """
    
    def __init__(self, root_dir, csv_file, transform=None, mode='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.data = pd.read_csv(csv_file)
        
        # Validate that the CSV contains required columns
        required_columns = ['image_name', 'age', 'ethnicity', 'gender']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"CSV file missing required columns: {missing_columns}")
        
        print(f"Loaded {mode} dataset with {len(self.data)} samples")
    
    def __len__(self):
        """Return the size of the dataset"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image, age, gender, ethnicity) where:
                - image: Transformed PIL image
                - age: Age as float tensor
                - gender: Gender as string
                - ethnicity: Ethnicity as string
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get the row data
        row_data = self.data.iloc[idx]
        
        # Construct image path
        img_path = os.path.join(self.root_dir, row_data['image_name'])
        
        # Load and process image
        try:
            image = Image.open(img_path).convert('RGB')
        except (IOError, OSError) as e:
            raise RuntimeError(f"Cannot load image {img_path}: {e}")
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        # Get labels
        age = torch.tensor([row_data['age']], dtype=torch.float32)
        gender = row_data['gender']
        ethnicity = row_data['ethnicity']
        
        return image, age, gender, ethnicity
    
    def get_age_statistics(self):
        """Get age statistics for the dataset"""
        ages = self.data['age'].values
        return {
            'mean': float(ages.mean()),
            'std': float(ages.std()),
            'min': int(ages.min()),
            'max': int(ages.max()),
            'median': float(pd.Series(ages).median())
        }
    
    def get_gender_distribution(self):
        """Get gender distribution in the dataset"""
        return self.data['gender'].value_counts().to_dict()
    
    def get_ethnicity_distribution(self):
        """Get ethnicity distribution in the dataset"""
        return self.data['ethnicity'].value_counts().to_dict()
    
    def print_dataset_info(self):
        """Print comprehensive dataset information"""
        print(f"\n=== {self.mode.upper()} Dataset Information ===")
        print(f"Total samples: {len(self.data)}")
        
        # Age statistics
        age_stats = self.get_age_statistics()
        print(f"\nAge Statistics:")
        print(f"  Mean: {age_stats['mean']:.2f}")
        print(f"  Std: {age_stats['std']:.2f}")
        print(f"  Range: {age_stats['min']} - {age_stats['max']}")
        print(f"  Median: {age_stats['median']:.2f}")
        
        # Gender distribution
        gender_dist = self.get_gender_distribution()
        print(f"\nGender Distribution:")
        for gender, count in gender_dist.items():
            percentage = (count / len(self.data)) * 100
            print(f"  {gender}: {count} ({percentage:.1f}%)")
        
        # Ethnicity distribution
        ethnicity_dist = self.get_ethnicity_distribution()
        print(f"\nEthnicity Distribution:")
        for ethnicity, count in ethnicity_dist.items():
            percentage = (count / len(self.data)) * 100
            print(f"  {ethnicity}: {count} ({percentage:.1f}%)")


def get_transforms(mode='train'):
    """
    Get appropriate transforms for training/validation/testing.
    
    Args:
        mode (str): 'train', 'valid', or 'test'
        
    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    
    if mode == 'train':
        # Training transforms with data augmentation
        return T.Compose([
            T.Resize((config['img_width'], config['img_height'])),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(
                brightness=(0.5, 1.5), 
                contrast=(0.5, 1.5), 
                saturation=(0.5, 1.5), 
                hue=(-0.1, 0.1)
            ),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            T.ToTensor(),
            T.Normalize(mean=config['mean'], std=config['std'])
        ])
    
    else:
        # Validation/Test transforms without augmentation
        return T.Compose([
            T.Resize((config['img_width'], config['img_height'])),
            T.ToTensor(),
            T.Normalize(mean=config['mean'], std=config['std'])
        ])


def validate_image_files(root_dir, csv_file):
    """
    Validate that all images listed in CSV exist in the directory.
    
    Args:
        root_dir (str): Root directory containing images
        csv_file (str): Path to CSV file
        
    Returns:
        tuple: (valid_files, missing_files)
    """
    df = pd.read_csv(csv_file)
    image_names = df['image_name'].tolist()
    
    valid_files = []
    missing_files = []
    
    for img_name in image_names:
        img_path = os.path.join(root_dir, img_name)
        if os.path.exists(img_path):
            valid_files.append(img_name)
        else:
            missing_files.append(img_name)
    
    if missing_files:
        print(f"Warning: {len(missing_files)} images are missing:")
        for missing in missing_files[:10]:  # Show first 10
            print(f"  - {missing}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
    
    return valid_files, missing_files