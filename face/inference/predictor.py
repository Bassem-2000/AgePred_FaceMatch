# ============================================================================
# QUICK FIX - Let me provide you with the exact content for each file
# Copy each section EXACTLY as shown below
# ============================================================================

# FILE: inference/predictor.py
# Copy this ENTIRE content to inference/predictor.py

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import logging
import os
from models.model import SiameseNetwork
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.dataset import FacePairDataset

class FaceMatchingPredictor:
    """Face matching predictor for inference and evaluation"""
    
    def __init__(self, model_path, config, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = SiameseNetwork(embedding_dim=config.embedding_dim).to(self.device)
        self._load_model(model_path)
        
        # Transform for evaluation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, model_path):
        """Load model from checkpoint"""
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.logger.info(f"[+] Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    def predict_single_pair(self, img1_path, img2_path):
        """Predict if two images contain the same person"""
        
        # Load and preprocess images
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        img1_tensor = self.transform(img1).unsqueeze(0).to(self.device)
        img2_tensor = self.transform(img2).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            emb1, emb2 = self.model(img1_tensor, img2_tensor)
            
            distance = F.pairwise_distance(emb1, emb2, p=2).item()
            similarity = F.cosine_similarity(emb1, emb2).item()
            
            is_same_distance = distance < self.config.distance_threshold
            is_same_similarity = similarity > 0.5
            
            return {
                'is_same_person_distance': is_same_distance,
                'is_same_person_similarity': is_same_similarity,
                'distance': distance,
                'similarity': similarity,
                'confidence_distance': max(0, 1 - distance),
                'confidence_similarity': similarity
            }
    
    def get_embedding(self, img_path):
        """Get face embedding for a single image"""
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model.get_embedding(img_tensor)
            return embedding.cpu().numpy()
    
    def evaluate_on_dataset(self, test_df, images_dir):
        """Evaluate model on test dataset"""
        self.logger.info("[*] Evaluating model on test set...")
        
        test_dataset = FacePairDataset(
            test_df, images_dir, self.transform,
            pairs_per_identity=self.config.pairs_per_identity
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=self.config.num_workers
        )
        
        all_distances = []
        all_labels = []
        all_similarities = []
        
        with torch.no_grad():
            for img1, img2, labels in tqdm(test_loader, desc='Evaluating'):
                img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)
                
                emb1, emb2 = self.model(img1, img2)
                
                # Calculate distances and similarities
                distances = F.pairwise_distance(emb1, emb2, p=2)
                similarities = F.cosine_similarity(emb1, emb2)
                
                all_distances.extend(distances.cpu().numpy())
                all_similarities.extend(similarities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        distance_threshold = self.config.distance_threshold
        distance_predictions = [1 if d < distance_threshold else 0 for d in all_distances]
        
        similarity_threshold = 0.5
        similarity_predictions = [1 if s > similarity_threshold else 0 for s in all_similarities]
        
        # Calculate accuracies
        distance_accuracy = accuracy_score(all_labels, distance_predictions)
        similarity_accuracy = accuracy_score(all_labels, similarity_predictions)
        
        # Calculate AUC scores
        distance_auc = roc_auc_score(all_labels, [1-d for d in all_distances])
        similarity_auc = roc_auc_score(all_labels, all_similarities)
        
        # Generate detailed metrics
        metrics = {
            'distance_accuracy': distance_accuracy,
            'similarity_accuracy': similarity_accuracy,
            'distance_auc': distance_auc,
            'similarity_auc': similarity_auc,
            'distance_threshold': distance_threshold,
            'similarity_threshold': similarity_threshold
        }
        
        self.logger.info("[*] Test Results:")
        self.logger.info(f"Distance-based Accuracy: {distance_accuracy:.4f}")
        self.logger.info(f"Similarity-based Accuracy: {similarity_accuracy:.4f}")
        self.logger.info(f"Distance-based AUC: {distance_auc:.4f}")
        self.logger.info(f"Similarity-based AUC: {similarity_auc:.4f}")
        
        # Save detailed results
        results = {
            'metrics': metrics,
            'distances': all_distances,
            'similarities': all_similarities,
            'labels': all_labels,
            'distance_predictions': distance_predictions,
            'similarity_predictions': similarity_predictions
        }
        
        return results