# config.py - Face Matching Project
import json
import os
from datetime import datetime

class Config:
    """Configuration class for face matching system"""
    
    def __init__(self):
        # Model parameters
        self.embedding_dim = 512
        self.margin = 1.0
        
        # Training parameters
        self.learning_rate = 0.0001
        self.weight_decay = 1e-4
        self.batch_size = 32
        self.num_epochs = 30
        self.pairs_per_identity = 5
        
        # Scheduler parameters
        self.scheduler_step = 10
        self.scheduler_gamma = 0.5
        
        # Evaluation parameters
        self.distance_threshold = 0.5
        
        # Data parameters
        self.num_workers = 4
        self.save_every = 5
        
        # Dataset creation (for synthetic fallback)
        self.num_identities = 100
        self.images_per_identity = 10
        
        # Paths
        self.data_dir = 'face/data'
        self.checkpoint_dir = 'face/checkpoints'
        self.results_dir = 'face/results'
        self.logs_dir = 'face/logs'
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save(self, filepath='config/config.json'):
        """Save configuration to JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
        print(f"✅ Config saved to {filepath}")
    
    @classmethod
    def load(cls, filepath='face/config/config.json'):
        """Load configuration from JSON file"""
        config = cls()
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
                for key, value in config_dict.items():
                    setattr(config, key, value)
            print(f"✅ Config loaded from {filepath}")
        else:
            print(f"⚠️ Config file not found, using defaults")
            config.save(filepath)
        return config