import os
import pandas as pd
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class FacePairDataset(Dataset):
    """Dataset for face pairs with same/different person labels"""
    
    def __init__(self, identities_df, images_dir, transform=None, pairs_per_identity=5):
        self.identities_df = identities_df
        self.images_dir = images_dir
        self.transform = transform
        self.pairs = []
        self.labels = []
        
        self._create_pairs(pairs_per_identity)
    
    def _create_pairs(self, pairs_per_identity):
        """Create positive and negative pairs"""
        
        # Group images by identity
        identity_groups = self.identities_df.groupby('identity')['image'].apply(list).to_dict()
        
        # Create positive pairs (same person)
        for identity, images in identity_groups.items():
            if len(images) > 1:
                pairs_created = 0
                for i in range(len(images)):
                    for j in range(i + 1, len(images)):
                        if pairs_created < pairs_per_identity:
                            self.pairs.append((images[i], images[j]))
                            self.labels.append(1)  # Same person
                            pairs_created += 1
        
        # Create negative pairs (different persons)
        identities = list(identity_groups.keys())
        num_positive = len([l for l in self.labels if l == 1])
        
        for _ in range(num_positive):
            # Random two different identities
            id1, id2 = random.sample(identities, 2)
            img1 = random.choice(identity_groups[id1])
            img2 = random.choice(identity_groups[id2])
            
            self.pairs.append((img1, img2))
            self.labels.append(0)  # Different person
        
        print(f"Dataset created: {len(self.pairs)} pairs")
        print(f"Positive pairs: {sum(self.labels)}")
        print(f"Negative pairs: {len(self.labels) - sum(self.labels)}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img1_name, img2_name = self.pairs[idx]
        label = self.labels[idx]
        
        # Load images
        img1_path = os.path.join(self.images_dir, img1_name)
        img2_path = os.path.join(self.images_dir, img2_name)
        
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, torch.tensor(label, dtype=torch.float32)

class DataPreprocessor:
    """Preprocess and prepare data for training"""
    
    def __init__(self, logger=None):
        self.logger = logger
    
    def log(self, message):
        """Log message if logger is available"""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
    
    def create_train_val_test_splits(self, identities_file, train_ratio=0.7, val_ratio=0.15):
        """Create train/validation/test splits"""
        self.log("[*] Creating data splits...")
        
        df = pd.read_csv(identities_file)
        
        # Split by identities to avoid data leakage
        unique_identities = df['identity'].unique()
        
        # Ensure we have enough identities for meaningful splits
        min_identities_needed = 10
        if len(unique_identities) < min_identities_needed:
            self.log(f"Only {len(unique_identities)} identities found. Using simple split.")
            # For small datasets, just split randomly
            train_df = df.sample(frac=train_ratio, random_state=42)
            remaining_df = df.drop(train_df.index)
            val_df = remaining_df.sample(frac=val_ratio/(1-train_ratio), random_state=42)
            test_df = remaining_df.drop(val_df.index)
        else:
            # Normal split by identities
            train_ids, temp_ids = train_test_split(
                unique_identities, test_size=(1-train_ratio), random_state=42
            )
            
            # Calculate validation split size correctly
            val_split_size = val_ratio / (1 - train_ratio)
            val_split_size = max(0.1, min(0.9, val_split_size))
            
            val_ids, test_ids = train_test_split(
                temp_ids, test_size=(1-val_split_size), random_state=42
            )
            
            # Create splits
            train_df = df[df['identity'].isin(train_ids)]
            val_df = df[df['identity'].isin(val_ids)]
            test_df = df[df['identity'].isin(test_ids)]
        
        # Save splits
        splits_dir = 'data/splits'
        os.makedirs(splits_dir, exist_ok=True)
        train_df.to_csv(f'{splits_dir}/train_identities.csv', index=False)
        val_df.to_csv(f'{splits_dir}/val_identities.csv', index=False)
        test_df.to_csv(f'{splits_dir}/test_identities.csv', index=False)
        
        self.log(f"[+] Data splits created:")
        self.log(f"   Train: {len(train_df['identity'].unique())} identities, {len(train_df)} images")
        self.log(f"   Val: {len(val_df['identity'].unique())} identities, {len(val_df)} images")
        self.log(f"   Test: {len(test_df['identity'].unique())} identities, {len(test_df)} images")
        
        return train_df, val_df, test_df