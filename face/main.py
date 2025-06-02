import os
import sys
import argparse
import logging
import pandas as pd

# Import project modules
from config.config import Config
from utils.util import (
    setup_logging, setup_sklearn_lfw, setup_olivetti_faces, 
    create_synthetic_dataset, get_device
)
from data.dataset import DataPreprocessor
from training.trainer import FaceMatchingTrainer
from inference.predictor import FaceMatchingPredictor
from utils.visualization import ResultsVisualizer

def setup_real_data(config, logger):
    """Setup real face datasets with fallback chain"""
    
    # Priority 1: Check for existing sklearn LFW dataset
    lfw_sklearn_path = os.path.join('Dataset/raw/lfw_sklearn/identities.csv')
    if os.path.exists(lfw_sklearn_path):
        processed_dir = "Dataset/raw/lfw_sklearn"
        identities_file = lfw_sklearn_path
        images_dir = os.path.join(processed_dir, "images")
        
        df = pd.read_csv(identities_file)
        logger.info("[+] Using existing sklearn LFW dataset (REAL FACES)")
        logger.info(f"    Total images: {len(df)}")
        logger.info(f"    Total people: {df['identity'].nunique()}")
        logger.info(f"    Images per person: {len(df) / df['identity'].nunique():.1f} average")
        
        return processed_dir, identities_file, images_dir
    
    # Priority 2: Check for existing Olivetti dataset
    olivetti_path = os.path.join('Dataset/raw/olivetti/identities.csv')
    if os.path.exists(olivetti_path):
        processed_dir = "Dataset/raw/olivetti"
        identities_file = olivetti_path
        images_dir = os.path.join(processed_dir, "images")
        
        df = pd.read_csv(identities_file)
        logger.info("[+] Using existing Olivetti dataset (REAL FACES)")
        logger.info(f"    Total images: {len(df)}")
        logger.info(f"    Total people: {df['identity'].nunique()}")
        
        return processed_dir, identities_file, images_dir
    
    # Priority 3: Try to download sklearn LFW automatically
    logger.info("[*] No existing real dataset found. Setting up sklearn LFW...")
    result = setup_sklearn_lfw()
    
    if result:
        processed_dir, identities_file, images_dir = result
        logger.info("[+] Successfully setup sklearn LFW dataset (REAL FACES)")
        return processed_dir, identities_file, images_dir
    
    # Priority 4: Try Olivetti as backup
    logger.info("[*] LFW setup failed. Trying Olivetti faces...")
    result = setup_olivetti_faces()
    
    if result:
        processed_dir, identities_file, images_dir = result
        logger.info("[+] Successfully setup Olivetti dataset (REAL FACES)")
        return processed_dir, identities_file, images_dir
    
    # Priority 5: Fallback to synthetic
    logger.info("[!] Real dataset setup failed. Creating synthetic dataset...")
    processed_dir, identities_file, images_dir = create_synthetic_dataset(
        num_identities=config.num_identities,
        images_per_identity=config.images_per_identity
    )
    logger.info("[+] Using synthetic dataset as fallback")
    
    return processed_dir, identities_file, images_dir

def main():
    """Main execution pipeline"""
    
    parser = argparse.ArgumentParser(description='Face Matching System')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'predict', 'full'], 
                       default='full', help='Operation mode')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--img1', type=str, help='First image for prediction')
    parser.add_argument('--img2', type=str, help='Second image for prediction')
    parser.add_argument('--config', type=str, default='config/config.json',
                       help='Configuration file path')
    parser.add_argument('--setup-real-data', action='store_true',
                       help='Setup real face datasets')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Load configuration
    config = Config.load(args.config)
    
    print("Face Matching System - Modular Architecture")
    print("=" * 50)
    
    if args.setup_real_data:
        # Setup real datasets
        logger.info("[*] Setting up real face datasets...")
        result1 = setup_sklearn_lfw()
        if not result1:
            result2 = setup_olivetti_faces()
            if not result2:
                logger.error("[!] Failed to setup any real datasets")
            else:
                logger.info("[+] Olivetti dataset setup complete")
        else:
            logger.info("[+] LFW dataset setup complete")
        return
    
    if args.mode == 'full':
        try:
            # Step 1: Data Preparation
            logger.info("[*] Step 1: Data Preparation")
            processed_dir, identities_file, images_dir = setup_real_data(config, logger)
            
            # Create data splits
            preprocessor = DataPreprocessor(logger)
            train_df, val_df, test_df = preprocessor.create_train_val_test_splits(identities_file)
            
            # Step 2: Model Training
            logger.info("[*] Step 2: Model Training")
            trainer = FaceMatchingTrainer(config, logger)
            train_losses, val_losses, val_accuracies = trainer.train(train_df, val_df, images_dir)
            
            # Step 3: Model Evaluation
            logger.info("[*] Step 3: Model Evaluation")
            predictor = FaceMatchingPredictor(args.model_path, config, logger)
            eval_results = predictor.evaluate_on_dataset(test_df, images_dir)
            
            # Step 4: Visualization
            logger.info("[*] Step 4: Results Visualization")
            visualizer = ResultsVisualizer(config, logger)
            visualizer.plot_training_curves(train_losses, val_losses, val_accuracies)
            visualizer.plot_evaluation_results(eval_results)
            visualizer.generate_report(config, train_losses, val_losses, val_accuracies, eval_results)
            
            # Step 5: Demo Prediction
            logger.info("[*] Step 5: Demo Predictions")
            test_images = test_df['image'].head(4).tolist()
            if len(test_images) >= 2:
                img1_path = os.path.join(images_dir, test_images[0])
                img2_path = os.path.join(images_dir, test_images[1])
                
                result = predictor.predict_single_pair(img1_path, img2_path)
                
                logger.info("Demo Prediction Results:")
                logger.info(f"Image 1: {test_images[0]}")
                logger.info(f"Image 2: {test_images[1]}")
                logger.info(f"Same Person (Distance): {result['is_same_person_distance']}")
                logger.info(f"Same Person (Similarity): {result['is_same_person_similarity']}")
                logger.info(f"Distance: {result['distance']:.4f}")
                logger.info(f"Similarity: {result['similarity']:.4f}")
            
            logger.info("[+] Pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"[!] Pipeline failed with error: {e}")
            raise
    
    elif args.mode == 'predict' and args.img1 and args.img2:
        # Single prediction mode
        predictor = FaceMatchingPredictor(args.model_path, config, logger)
        result = predictor.predict_single_pair(args.img1, args.img2)
        
        print("[*] Face Matching Prediction")
        print(f"Image 1: {args.img1}")
        print(f"Image 2: {args.img2}")
        print(f"Same Person (Distance): {result['is_same_person_distance']}")
        print(f"Same Person (Similarity): {result['is_same_person_similarity']}")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Similarity: {result['similarity']:.4f}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()