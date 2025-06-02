import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import logging
from models.model import SiameseNetwork
from training.loss_utils import ContrastiveLoss
from data.data_loader import DataLoaderFactory

class FaceMatchingTrainer:
    """Complete training system for face matching"""
    
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = SiameseNetwork(embedding_dim=config.embedding_dim).to(self.device)
        
        # Initialize training components
        self.criterion = ContrastiveLoss(margin=config.margin)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=config.scheduler_step, 
            gamma=config.scheduler_gamma
        )
        
        # Initialize data loader factory
        self.data_loader_factory = DataLoaderFactory(config)
        
        # Metrics tracking - FIXED: Initialize empty lists
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc='Training')
        
        for batch_idx, (img1, img2, labels) in enumerate(progress_bar):
            img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            emb1, emb2 = self.model(img1, img2)
            loss = self.criterion(emb1, emb2, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        return total_loss / len(train_loader)
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for img1, img2, labels in tqdm(val_loader, desc='Validation'):
                img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)
                
                emb1, emb2 = self.model(img1, img2)
                loss = self.criterion(emb1, emb2, labels)
                total_loss += loss.item()
                
                # Calculate accuracy based on distance threshold
                distances = F.pairwise_distance(emb1, emb2, p=2)
                predictions = (distances < self.config.distance_threshold).float()
                
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def train(self, train_df, val_df, images_dir):
        """Complete training loop"""
        self.logger.info(f"[*] Starting training on {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Create data loaders
        train_loader, val_loader = self.data_loader_factory.create_data_loaders(
            train_df, val_df, images_dir
        )
        
        for epoch in range(self.config.num_epochs):
            self.logger.info(f"\n[*] Epoch {epoch+1}/{self.config.num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_accuracy = self.validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            
            # Log results
            self.logger.info(f"Train Loss: {train_loss:.4f}")
            self.logger.info(f"Val Loss: {val_loss:.4f}")
            self.logger.info(f"Val Accuracy: {val_accuracy:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, 'best_model.pth')
                self.logger.info(f"[+] New best model saved! Val Loss: {val_loss:.4f}")
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch+1}.pth')
        
        self.logger.info("[+] Training completed!")
        return self.train_losses, self.val_losses, self.val_accuracies
    
    def save_checkpoint(self, epoch, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict()
        }
        
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        filepath = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        self.logger.info(f"[+] Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_accuracies = checkpoint['val_accuracies']
        self.best_val_loss = checkpoint['best_val_loss']
        
        return checkpoint['epoch']