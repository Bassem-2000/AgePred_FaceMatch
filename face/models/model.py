import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class FaceEmbeddingNetwork(nn.Module):
    """Deep learning model for extracting face embeddings"""
    
    def __init__(self, embedding_dim=512, pretrained=True):
        super(FaceEmbeddingNetwork, self).__init__()
        
        # Use ResNet50 as backbone
        self.backbone = resnet50(pretrained=pretrained)
        
        # Remove the final classification layer
        self.backbone.fc = nn.Identity()
        
        # Add custom embedding layers
        self.embedding_layers = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        
        # Generate embeddings
        embeddings = self.embedding_layers(features)
        
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings

class SiameseNetwork(nn.Module):
    """Siamese network for face matching"""
    
    def __init__(self, embedding_dim=512):
        super(SiameseNetwork, self).__init__()
        
        self.face_embedding = FaceEmbeddingNetwork(embedding_dim)
        
    def forward(self, img1, img2):
        # Get embeddings for both images
        emb1 = self.face_embedding(img1)
        emb2 = self.face_embedding(img2)
        
        return emb1, emb2
    
    def get_embedding(self, img):
        """Get embedding for a single image"""
        return self.face_embedding(img)