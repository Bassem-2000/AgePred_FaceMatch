#!/usr/bin/env python3
"""
Fixed model loading for the integrated pipeline.

This fixes the checkpoint loading issues by properly handling different checkpoint formats
and model architectures.
"""

import torch
import os
import sys
from pathlib import Path

# Add paths for imports
sys.path.append('age')
sys.path.append('face')

def load_age_model_checkpoint(checkpoint_path, device='cuda'):
    """
    Load age estimation model from checkpoint with proper error handling.
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        device (str): Device to load model on
        
    Returns:
        Loaded model
    """
    
    print(f"Loading age model from: {checkpoint_path}")
    
    # Import age model components
    from age.models.models import AgeEstimationModel, create_model
    from age.config.config import config
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            # Training checkpoint format
            state_dict = checkpoint['model_state_dict']
            print("Loading from training checkpoint format")
        elif 'state_dict' in checkpoint:
            # Model-only checkpoint format
            state_dict = checkpoint['state_dict']
            print("Loading from model checkpoint format")
        else:
            # Direct state dict
            state_dict = checkpoint
            print("Loading from direct state dict")
    else:
        state_dict = checkpoint
    
    # Create model with proper architecture
    model = create_model()
    
    # Try to load state dict with error handling
    try:
        model.load_state_dict(state_dict, strict=True)
        print("✅ Model loaded successfully with strict matching")
    except RuntimeError as e:
        print(f"⚠️ Strict loading failed: {e}")
        
        # Try flexible loading
        try:
            model.load_state_dict(state_dict, strict=False)
            print("✅ Model loaded with flexible matching")
        except RuntimeError as e2:
            print(f"❌ Flexible loading also failed: {e2}")
            
            # Try to manually fix state dict keys
            print("🔧 Attempting to fix state dict keys...")
            fixed_state_dict = fix_state_dict_keys(state_dict, model.state_dict())
            model.load_state_dict(fixed_state_dict, strict=False)
            print("✅ Model loaded with key fixing")
    
    model.to(device)
    model.eval()
    return model


def fix_state_dict_keys(loaded_state_dict, model_state_dict):
    """
    Attempt to fix mismatched state dict keys.
    
    Args:
        loaded_state_dict (dict): State dict from checkpoint
        model_state_dict (dict): Current model's state dict
        
    Returns:
        dict: Fixed state dict
    """
    
    fixed_state_dict = {}
    model_keys = set(model_state_dict.keys())
    loaded_keys = set(loaded_state_dict.keys())
    
    print(f"Model expects {len(model_keys)} keys")
    print(f"Checkpoint has {len(loaded_keys)} keys")
    
    # Direct matches
    for key in model_keys.intersection(loaded_keys):
        fixed_state_dict[key] = loaded_state_dict[key]
    
    # Try to find mappings for missing keys
    missing_keys = model_keys - loaded_keys
    extra_keys = loaded_keys - model_keys
    
    print(f"Missing keys: {len(missing_keys)}")
    print(f"Extra keys: {len(extra_keys)}")
    
    # Common key transformations
    key_mappings = {
        'backbone.': 'model.0.',  # backbone -> model.0
        'classifier.': 'model.1.',  # classifier -> model.1
        'model.0.': 'backbone.',   # reverse mapping
        'model.1.': 'classifier.'  # reverse mapping
    }
    
    for missing_key in missing_keys:
        found = False
        for old_prefix, new_prefix in key_mappings.items():
            if missing_key.startswith(old_prefix.split('.')[0]):
                # Try to find corresponding key with different prefix
                candidate_key = missing_key.replace(old_prefix.split('.')[0], new_prefix.split('.')[0])
                if candidate_key in loaded_keys:
                    fixed_state_dict[missing_key] = loaded_state_dict[candidate_key]
                    found = True
                    break
        
        if not found:
            # Initialize with current model weights (keep pretrained)
            fixed_state_dict[missing_key] = model_state_dict[missing_key]
    
    print(f"Fixed state dict has {len(fixed_state_dict)} keys")
    return fixed_state_dict


class ImprovedAgePredictor:
    """
    Improved age predictor with better model loading.
    """
    
    def __init__(self, model_path=None, device=None):
        """Initialize with robust model loading."""
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_path and os.path.exists(model_path):
            self.model = load_age_model_checkpoint(model_path, self.device)
        else:
            # Create fresh model if no checkpoint
            print("Creating fresh age estimation model...")
            from age.models.models import create_model
            self.model = create_model()
            self.model.to(self.device)
            self.model.eval()
            print("⚠️ Using untrained model - predictions may be inaccurate")
        
        # Setup transform
        from age.config.config import config
        import torchvision.transforms as T
        
        self.transform = T.Compose([
            T.Resize((config['img_size'], config['img_size'])),
            T.ToTensor(),
            T.Normalize(mean=config['mean'], std=config['std'])
        ])
    
    def predict_single(self, image_path, return_confidence=False):
        """Predict age for single image."""
        
        from PIL import Image
        import numpy as np
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            predicted_age = float(output.item())
        
        # Simple confidence estimation
        confidence = min(1.0, max(0.0, 1.0 - abs(predicted_age - 30) / 50))
        
        if return_confidence:
            return predicted_age, confidence
        return predicted_age


def load_face_model_checkpoint(checkpoint_path, device='cuda'):
    """
    Load face matching model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        device (str): Device to load model on
        
    Returns:
        Loaded model
    """
    
    print(f"Loading face model from: {checkpoint_path}")
    
    # Import face model components
    from face.models.model import SiameseNetwork
    from face.config.config import Config
    
    # Load config
    config = Config.load()
    
    # Create model
    model = SiameseNetwork(embedding_dim=config.embedding_dim).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Load state dict
    model.load_state_dict(state_dict)
    model.eval()
    
    print("✅ Face model loaded successfully")
    return model


class ImprovedFaceMatchingPredictor:
    """
    Improved face matching predictor with better model loading.
    """
    
    def __init__(self, model_path, config):
        """Initialize with robust model loading."""
        
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = load_face_model_checkpoint(model_path, self.device)
        
        # Setup transform
        import torchvision.transforms as transforms
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict_single_pair(self, img1_path, img2_path):
        """Predict if two images contain the same person."""
        
        from PIL import Image
        import torch.nn.functional as F
        
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


# Create the improved integrated pipeline
class ImprovedAgeVariantFaceMatcher:
    """
    Improved integrated system with robust model loading.
    """
    
    def __init__(self, age_model_path=None, face_model_path=None):
        """Initialize with improved model loading."""
        
        print("🚀 Initializing Improved Age-Variant Face Matching System...")
        
        # Initialize age predictor
        print("📊 Loading age estimation model...")
        try:
            self.age_predictor = ImprovedAgePredictor(model_path=age_model_path)
            print("✅ Age estimation model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load age model: {e}")
            raise
        
        # Initialize face matcher
        print("👥 Loading face matching model...")
        try:
            from face.config.config import Config
            face_config = Config.load()
            self.face_matcher = ImprovedFaceMatchingPredictor(
                face_model_path or 'face/checkpoints/best_model.pth', 
                face_config
            )
            print("✅ Face matching model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load face matching model: {e}")
            raise
        
        # Age difference thresholds for scoring adjustment
        self.age_diff_thresholds = {
            'low': 5,      # 0-5 years: minimal aging effects
            'medium': 15,  # 5-15 years: moderate aging effects
            'high': 30     # 15-30+ years: significant aging effects
        }
        
        print("🎉 System initialized successfully!")
    
    def predict_age_variant_match(self, image1_path, image2_path, 
                                detailed_output=True, save_visualization=None):
        """
        Main pipeline: predict if two images show the same person at different ages.
        """
        
        print(f"\n🔍 Analyzing image pair:")
        print(f"  Image 1: {Path(image1_path).name}")
        print(f"  Image 2: {Path(image2_path).name}")
        
        from datetime import datetime
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'image1_path': image1_path,
            'image2_path': image2_path,
            'age_predictions': {},
            'face_matching': {},
            'age_variant_analysis': {},
            'final_decision': {},
            'confidence_scores': {}
        }
        
        try:
            # Step 1: Age Prediction
            print("\n📊 Step 1: Predicting ages...")
            age1, conf1 = self.age_predictor.predict_single(image1_path, return_confidence=True)
            age2, conf2 = self.age_predictor.predict_single(image2_path, return_confidence=True)
            
            results['age_predictions'] = {
                'image1_age': float(age1),
                'image1_confidence': float(conf1),
                'image2_age': float(age2),
                'image2_confidence': float(conf2),
                'age_difference': abs(float(age1) - float(age2))
            }
            
            print(f"  Image 1 age: {age1:.1f} years (confidence: {conf1:.2f})")
            print(f"  Image 2 age: {age2:.1f} years (confidence: {conf2:.2f})")
            print(f"  Age difference: {results['age_predictions']['age_difference']:.1f} years")
            
            # Step 2: Face Matching
            print("\n👥 Step 2: Performing face matching...")
            face_match_result = self.face_matcher.predict_single_pair(image1_path, image2_path)
            
            results['face_matching'] = {
                'distance': float(face_match_result['distance']),
                'similarity': float(face_match_result['similarity']),
                'same_person_distance': bool(face_match_result['is_same_person_distance']),
                'same_person_similarity': bool(face_match_result['is_same_person_similarity']),
                'confidence_distance': float(face_match_result['confidence_distance']),
                'confidence_similarity': float(face_match_result['confidence_similarity'])
            }
            
            print(f"  Face distance: {face_match_result['distance']:.4f}")
            print(f"  Face similarity: {face_match_result['similarity']:.4f}")
            print(f"  Basic face match (distance): {face_match_result['is_same_person_distance']}")
            print(f"  Basic face match (similarity): {face_match_result['is_same_person_similarity']}")
            
            # Step 3: Age-Variant Analysis
            print("\n🧠 Step 3: Age-variant analysis...")
            age_variant_analysis = self._analyze_age_variant_matching(
                results['age_predictions'], 
                results['face_matching']
            )
            results['age_variant_analysis'] = age_variant_analysis
            
            # Step 4: Final Decision
            print("\n🎯 Step 4: Making final decision...")
            final_decision = self._make_final_decision(
                results['age_predictions'],
                results['face_matching'],
                results['age_variant_analysis']
            )
            results['final_decision'] = final_decision
            
            # Step 5: Confidence Scoring
            confidence_scores = self._calculate_confidence_scores(results)
            results['confidence_scores'] = confidence_scores
            
            # Print final results
            self._print_final_results(results)
            
            # Create visualization if requested
            if save_visualization:
                self._create_visualization(image1_path, image2_path, results, save_visualization)
                print(f"\n📊 Visualization saved to: {save_visualization}")
            
            return results
            
        except Exception as e:
            print(f"\n❌ Error during analysis: {e}")
            results['error'] = str(e)
            return results
    
    def _analyze_age_variant_matching(self, age_pred, face_match):
        """Analyze age-variant matching considering aging effects."""
        
        import numpy as np
        
        age_diff = age_pred['age_difference']
        distance = face_match['distance']
        similarity = face_match['similarity']
        
        # Determine aging category
        if age_diff <= self.age_diff_thresholds['low']:
            aging_category = 'minimal'
            aging_factor = 1.0  # No adjustment needed
        elif age_diff <= self.age_diff_thresholds['medium']:
            aging_category = 'moderate'
            aging_factor = 1.15  # Slightly more lenient
        elif age_diff <= self.age_diff_thresholds['high']:
            aging_category = 'significant'
            aging_factor = 1.3   # More lenient
        else:
            aging_category = 'extreme'
            aging_factor = 1.5   # Very lenient
        
        # Adjust thresholds based on aging
        base_distance_threshold = 0.5
        base_similarity_threshold = 0.5
        
        adjusted_distance_threshold = base_distance_threshold * aging_factor
        adjusted_similarity_threshold = base_similarity_threshold / aging_factor
        
        # Age-aware matching decisions
        age_aware_distance_match = distance < adjusted_distance_threshold
        age_aware_similarity_match = similarity > adjusted_similarity_threshold
        
        # Calculate aging probability (how likely is this age difference)
        aging_probability = 1 / (1 + np.exp((age_diff - 20) / 10))
        
        # Calculate age compatibility score
        max_realistic_diff = 50
        age_compatibility_score = max(0, 1 - (age_diff / max_realistic_diff))
        
        return {
            'aging_category': aging_category,
            'aging_factor': aging_factor,
            'adjusted_distance_threshold': adjusted_distance_threshold,
            'adjusted_similarity_threshold': adjusted_similarity_threshold,
            'age_aware_distance_match': age_aware_distance_match,
            'age_aware_similarity_match': age_aware_similarity_match,
            'aging_probability': aging_probability,
            'age_compatibility_score': age_compatibility_score
        }
    
    def _make_final_decision(self, age_pred, face_match, age_analysis):
        """Make final decision combining all evidence."""
        
        # Collect evidence
        evidence = {
            'basic_distance_match': face_match['same_person_distance'],
            'basic_similarity_match': face_match['same_person_similarity'], 
            'age_aware_distance_match': age_analysis['age_aware_distance_match'],
            'age_aware_similarity_match': age_analysis['age_aware_similarity_match'],
            'age_compatibility_high': age_analysis['age_compatibility_score'] > 0.7,
            'aging_probability_high': age_analysis['aging_probability'] > 0.5
        }
        
        # Count positive evidence
        positive_evidence = sum(evidence.values())
        total_evidence = len(evidence)
        
        # Decision logic
        if positive_evidence >= 4:  # Strong evidence
            decision = 'same_person'
            confidence_level = 'high'
        elif positive_evidence >= 3:  # Moderate evidence  
            decision = 'same_person'
            confidence_level = 'medium'
        elif positive_evidence >= 2:  # Weak evidence
            decision = 'possibly_same_person'
            confidence_level = 'low'
        else:  # Insufficient evidence
            decision = 'different_person'
            confidence_level = 'high' if positive_evidence <= 1 else 'medium'
        
        # Special case: very high age difference with low compatibility
        if (age_pred['age_difference'] > 40 and 
            age_analysis['age_compatibility_score'] < 0.3):
            decision = 'different_person'
            confidence_level = 'high'
            reasoning = 'Age difference too large to be realistic'
        else:
            reasoning = f'Based on {positive_evidence}/{total_evidence} positive indicators'
        
        return {
            'decision': decision,
            'confidence_level': confidence_level,
            'evidence_score': f"{positive_evidence}/{total_evidence}",
            'reasoning': reasoning,
            'evidence_breakdown': evidence
        }
    
    def _calculate_confidence_scores(self, results):
        """Calculate overall confidence scores."""
        
        age_conf = (results['age_predictions']['image1_confidence'] + 
                   results['age_predictions']['image2_confidence']) / 2
        
        face_conf = (results['face_matching']['confidence_distance'] + 
                    results['face_matching']['confidence_similarity']) / 2
        
        age_compatibility = results['age_variant_analysis']['age_compatibility_score']
        
        # Overall confidence is the minimum of all components
        overall_confidence = min(age_conf, face_conf, age_compatibility)
        
        return {
            'age_prediction_confidence': float(age_conf),
            'face_matching_confidence': float(face_conf), 
            'age_compatibility_confidence': float(age_compatibility),
            'overall_confidence': float(overall_confidence)
        }
    
    def _print_final_results(self, results):
        """Print formatted final results."""
        
        print("\n" + "="*60)
        print("🎯 FINAL RESULTS")
        print("="*60)
        
        decision = results['final_decision']['decision']
        confidence = results['final_decision']['confidence_level']
        
        # Use emojis based on decision
        if decision == 'same_person':
            emoji = "✅"
        elif decision == 'possibly_same_person':
            emoji = "🤔"
        else:
            emoji = "❌"
        
        print(f"{emoji} Decision: {decision.replace('_', ' ').title()}")
        print(f"🎯 Confidence: {confidence.title()}")
        print(f"📊 Evidence: {results['final_decision']['evidence_score']}")
        print(f"💭 Reasoning: {results['final_decision']['reasoning']}")
        
        print(f"\n📈 Key Metrics:")
        print(f"  Age difference: {results['age_predictions']['age_difference']:.1f} years")
        print(f"  Face distance: {results['face_matching']['distance']:.4f}")
        print(f"  Face similarity: {results['face_matching']['similarity']:.4f}")
        print(f"  Age compatibility: {results['age_variant_analysis']['age_compatibility_score']:.3f}")
        print(f"  Overall confidence: {results['confidence_scores']['overall_confidence']:.3f}")
    
    def _create_visualization(self, image1_path, image2_path, results, save_path):
        """Create basic visualization of results."""
        
        try:
            import matplotlib.pyplot as plt
            from PIL import Image
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Load and display images
            img1 = Image.open(image1_path).convert('RGB')
            img2 = Image.open(image2_path).convert('RGB')
            
            axes[0].imshow(img1)
            axes[0].set_title(f"Image 1\nAge: {results['age_predictions']['image1_age']:.1f} years")
            axes[0].axis('off')
            
            axes[1].imshow(img2)
            axes[1].set_title(f"Image 2\nAge: {results['age_predictions']['image2_age']:.1f} years")
            axes[1].axis('off')
            
            # Results summary
            decision = results['final_decision']['decision']
            confidence = results['final_decision']['confidence_level']
            
            axes[2].text(0.5, 0.5, f"DECISION:\n{decision.replace('_', ' ').title()}\n\nConfidence: {confidence.title()}", 
                        ha='center', va='center', fontsize=14, fontweight='bold')
            axes[2].set_xlim(0, 1)
            axes[2].set_ylim(0, 1)
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not create visualization: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Improved Age-Variant Face Matching Pipeline')
    parser.add_argument('--image1', type=str, required=True, help='Path to first image')
    parser.add_argument('--image2', type=str, required=True, help='Path to second image')
    parser.add_argument('--age_model', type=str, help='Path to age estimation model')
    parser.add_argument('--face_model', type=str, help='Path to face matching model')
    parser.add_argument('--save_viz', type=str, help='Path to save visualization')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.image1):
        print(f"❌ Image 1 not found: {args.image1}")
        exit(1)
    
    if not os.path.exists(args.image2):
        print(f"❌ Image 2 not found: {args.image2}")
        exit(1)
    
    # Initialize pipeline
    try:
        matcher = ImprovedAgeVariantFaceMatcher(
            age_model_path=args.age_model,
            face_model_path=args.face_model
        )
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        exit(1)
    
    # Process images
    results = matcher.predict_age_variant_match(
        args.image1, 
        args.image2,
        save_visualization=args.save_viz
    )
    
    print("\n🎉 Analysis complete!")