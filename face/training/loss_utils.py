import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """Contrastive loss for siamese networks"""
    
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, emb1, emb2, label):
        distance = F.pairwise_distance(emb1, emb2, p=2)
        
        loss_positive = label * torch.pow(distance, 2)
        loss_negative = (1 - label) * torch.pow(
            torch.clamp(self.margin - distance, min=0.0), 2
        )
        
        return (loss_positive + loss_negative).mean()

class TripletLoss(nn.Module):
    """Triplet loss for learning discriminative embeddings"""
    
    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()