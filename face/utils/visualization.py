import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix
from datetime import datetime
import os

class ResultsVisualizer:
    """Create visualizations and reports for training results"""
    
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger
        
        # Create results directories
        os.makedirs('results/plots', exist_ok=True)
        os.makedirs('results/metrics', exist_ok=True)
    
    def plot_training_curves(self, train_losses, val_losses, val_accuracies):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Loss curves
        axes[0].plot(train_losses, label='Train Loss', color='blue')
        axes[0].plot(val_losses, label='Val Loss', color='red')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Validation accuracy
        axes[1].plot(val_accuracies, label='Val Accuracy', color='green')
        axes[1].set_title('Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        # Combined view
        ax2 = axes[2].twinx()
        axes[2].plot(train_losses, label='Train Loss', color='blue')
        axes[2].plot(val_losses, label='Val Loss', color='red')
        ax2.plot(val_accuracies, label='Val Accuracy', color='green', linestyle='--')
        
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss', color='blue')
        ax2.set_ylabel('Accuracy', color='green')
        axes[2].set_title('Combined Training Metrics')
        
        # Combine legends
        lines1, labels1 = axes[2].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[2].legend(lines1 + lines2, labels1 + labels2, loc='center right')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig('results/plots/training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        if self.logger:
            self.logger.info("[+] Training curves saved to results/plots/training_curves.png")
    
    def plot_evaluation_results(self, results):
        """Plot evaluation results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        distances = results['distances']
        similarities = results['similarities']
        labels = results['labels']
        
        # Distance distribution
        same_person_distances = [d for d, l in zip(distances, labels) if l == 1]
        diff_person_distances = [d for d, l in zip(distances, labels) if l == 0]
        
        axes[0, 0].hist(same_person_distances, bins=50, alpha=0.7, label='Same Person', color='green')
        axes[0, 0].hist(diff_person_distances, bins=50, alpha=0.7, label='Different Person', color='red')
        axes[0, 0].axvline(results['metrics']['distance_threshold'], color='black', linestyle='--', label='Threshold')
        axes[0, 0].set_title('Distance Distribution')
        axes[0, 0].set_xlabel('Euclidean Distance')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Similarity distribution
        same_person_similarities = [s for s, l in zip(similarities, labels) if l == 1]
        diff_person_similarities = [s for s, l in zip(similarities, labels) if l == 0]
        
        axes[0, 1].hist(same_person_similarities, bins=50, alpha=0.7, label='Same Person', color='green')
        axes[0, 1].hist(diff_person_similarities, bins=50, alpha=0.7, label='Different Person', color='red')
        axes[0, 1].axvline(0.5, color='black', linestyle='--', label='Threshold')
        axes[0, 1].set_title('Similarity Distribution')
        axes[0, 1].set_xlabel('Cosine Similarity')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # ROC Curves
        fpr_dist, tpr_dist, _ = roc_curve(labels, [1-d for d in distances])
        axes[0, 2].plot(fpr_dist, tpr_dist, label=f'Distance AUC = {results["metrics"]["distance_auc"]:.3f}')
        axes[0, 2].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 2].set_title('ROC Curve - Distance Based')
        axes[0, 2].set_xlabel('False Positive Rate')
        axes[0, 2].set_ylabel('True Positive Rate')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        fpr_sim, tpr_sim, _ = roc_curve(labels, similarities)
        axes[1, 0].plot(fpr_sim, tpr_sim, label=f'Similarity AUC = {results["metrics"]["similarity_auc"]:.3f}')
        axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[1, 0].set_title('ROC Curve - Similarity Based')
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Confusion Matrices
        cm_distance = confusion_matrix(labels, results['distance_predictions'])
        sns.heatmap(cm_distance, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_title('Confusion Matrix - Distance Based')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        
        cm_similarity = confusion_matrix(labels, results['similarity_predictions'])
        sns.heatmap(cm_similarity, annot=True, fmt='d', cmap='Greens', ax=axes[1, 2])
        axes[1, 2].set_title('Confusion Matrix - Similarity Based')
        axes[1, 2].set_xlabel('Predicted')
        axes[1, 2].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('results/plots/evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        if self.logger:
            self.logger.info("[+] Evaluation plots saved to results/plots/evaluation_results.png")
    
    def generate_report(self, config, train_losses, val_losses, val_accuracies, eval_results):
        """Generate comprehensive report"""
        
        report = f"""
# Face Matching System - Training Report

## Configuration
- **Model**: Siamese Network with ResNet50 backbone
- **Embedding Dimension**: {config.embedding_dim}
- **Learning Rate**: {config.learning_rate}
- **Batch Size**: {config.batch_size}
- **Number of Epochs**: {config.num_epochs}
- **Margin (Contrastive Loss)**: {config.margin}

## Training Results
- **Best Validation Loss**: {min(val_losses):.4f}
- **Best Validation Accuracy**: {max(val_accuracies):.4f}
- **Final Training Loss**: {train_losses[-1]:.4f}
- **Final Validation Loss**: {val_losses[-1]:.4f}

## Test Results
- **Distance-based Accuracy**: {eval_results['metrics']['distance_accuracy']:.4f}
- **Similarity-based Accuracy**: {eval_results['metrics']['similarity_accuracy']:.4f}
- **Distance-based AUC**: {eval_results['metrics']['distance_auc']:.4f}
- **Similarity-based AUC**: {eval_results['metrics']['similarity_auc']:.4f}

## Model Performance Analysis

### Strengths:
- High accuracy on clean, well-aligned face images
- Robust feature extraction using pretrained ResNet50
- Effective contrastive learning for discriminative embeddings
- Good generalization across different identities

### Limitations:
- Performance may degrade with poor image quality
- Sensitive to extreme pose variations
- May struggle with significant age gaps
- Requires sufficient training data per identity

## Recommendations:
1. Consider data augmentation for pose variations
2. Implement quality assessment for input images
3. Use ensemble methods for improved robustness
4. Regular retraining with new data

---
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        # Save report
        with open('results/training_report.md', 'w') as f:
            f.write(report)
        
        if self.logger:
            self.logger.info("[+] Report saved to results/training_report.md")
