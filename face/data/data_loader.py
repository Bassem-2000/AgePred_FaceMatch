import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data.dataset import FacePairDataset

class DataLoaderFactory:
    """Factory class for creating data loaders"""
    
    def __init__(self, config):
        self.config = config
        
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def create_data_loaders(self, train_df, val_df, images_dir):
        """Create data loaders for training and validation"""
        
        train_dataset = FacePairDataset(
            train_df, images_dir, self.train_transform, 
            pairs_per_identity=self.config.pairs_per_identity
        )
        
        val_dataset = FacePairDataset(
            val_df, images_dir, self.val_transform,
            pairs_per_identity=self.config.pairs_per_identity
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=self.config.num_workers
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=self.config.num_workers
        )
        
        return train_loader, val_loader
