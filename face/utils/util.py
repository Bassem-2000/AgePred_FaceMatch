import os
import sys
import logging
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.datasets import fetch_lfw_people, fetch_olivetti_faces
from PIL import Image
from tqdm import tqdm
import cv2
import random

def setup_logging(log_dir='logs'):
    """Setup logging configuration - Windows compatible"""
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"{log_dir}/face_matching_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Create file handler with UTF-8 encoding
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    
    # Create console handler that handles encoding issues
    console_handler = logging.StreamHandler()
    
    # Set formatters
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Prevent encoding issues on Windows
    if hasattr(console_handler.stream, 'reconfigure'):
        try:
            console_handler.stream.reconfigure(encoding='utf-8')
        except:
            pass
    
    return logger

def setup_sklearn_lfw(output_dir='Dataset/raw/lfw_sklearn'):
    """Download and setup LFW dataset using scikit-learn"""
    try:
        print("[*] Downloading LFW dataset via scikit-learn...")
        print("    This may take a few minutes (downloading ~200MB)...")
        
        # Fetch LFW people dataset
        lfw_people = fetch_lfw_people(
            min_faces_per_person=2,
            resize=0.4,
            color=True,
            slice_=None,
            download_if_missing=True
        )
        
        print(f"[+] Downloaded {len(lfw_people.data)} images from {len(lfw_people.target_names)} people")
        
        # Create directory structure
        images_dir = os.path.join(output_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        # Save images and create identities file
        identities_data = []
        
        print("[*] Processing and saving images...")
        for i, (image, target) in enumerate(tqdm(zip(lfw_people.images, lfw_people.target), desc="Processing LFW")):
            # Get person name
            person_name = lfw_people.target_names[target]
            
            # Create filename
            img_name = f"{person_name}_{i:04d}.jpg"
            img_path = os.path.join(images_dir, img_name)
            
            # Convert and save image
            if len(image.shape) == 3:  # RGB
                img_pil = Image.fromarray((image * 255).astype('uint8'))
            else:  # Grayscale
                img_pil = Image.fromarray((image * 255).astype('uint8'))
                img_pil = img_pil.convert('RGB')
            
            # Resize to our standard size
            img_pil = img_pil.resize((224, 224))
            img_pil.save(img_path)
            
            identities_data.append({
                'image': img_name,
                'identity': person_name
            })
        
        # Save identities file
        identities_df = pd.DataFrame(identities_data)
        identities_file = os.path.join(output_dir, 'identities.csv')
        identities_df.to_csv(identities_file, index=False)
        
        print(f"[+] LFW Dataset Ready!")
        print(f"    Total images: {len(identities_data)}")
        print(f"    Total people: {identities_df['identity'].nunique()}")
        print(f"    Images per person: {len(identities_data) / identities_df['identity'].nunique():.1f} average")
        
        return output_dir, identities_file, images_dir
        
    except ImportError:
        print("[!] scikit-learn not available. Install with: pip install scikit-learn")
        return None
    except Exception as e:
        print(f"[!] Error setting up LFW: {e}")
        return None

def setup_olivetti_faces(output_dir='data/raw/olivetti'):
    """Setup Olivetti faces dataset as backup real data"""
    try:
        print("[*] Setting up Olivetti Faces dataset...")
        
        # Fetch dataset
        faces = fetch_olivetti_faces(shuffle=True, random_state=42)
        
        # Create directories
        images_dir = os.path.join(output_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        # Save images and create identities file
        identities_data = []
        
        for i, (image, target) in enumerate(tqdm(zip(faces.data, faces.target), desc="Processing Olivetti")):
            # Reshape image (64x64)
            img_array = image.reshape(64, 64)
            
            # Convert to PIL Image and resize
            img = Image.fromarray((img_array * 255).astype('uint8'))
            img = img.resize((224, 224))
            
            # Save image
            img_name = f"person_{target:02d}_img_{i:03d}.jpg"
            img_path = os.path.join(images_dir, img_name)
            img.save(img_path)
            
            identities_data.append({
                'image': img_name,
                'identity': f"person_{target:02d}"
            })
        
        # Save identities file
        identities_df = pd.DataFrame(identities_data)
        identities_file = os.path.join(output_dir, 'identities.csv')
        identities_df.to_csv(identities_file, index=False)
        
        print(f"[+] Olivetti dataset created: {len(identities_data)} images from 40 people")
        return output_dir, identities_file, images_dir
        
    except ImportError:
        print("[!] scikit-learn not available for Olivetti faces")
        return None
    except Exception as e:
        print(f"[!] Error setting up Olivetti: {e}")
        return None

def create_synthetic_dataset(output_dir='data/raw/synthetic', num_identities=100, images_per_identity=10):
    """Create synthetic dataset for testing"""
    print(f"[*] Creating synthetic dataset with {num_identities} identities...")
    
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Create synthetic face images
    identities_data = []
    
    for identity_id in tqdm(range(num_identities), desc="Creating identities"):
        # Random base color for this identity
        base_color = np.random.randint(0, 255, 3)
        
        for img_idx in range(images_per_identity):
            # Create image with variations
            img = np.ones((224, 224, 3), dtype=np.uint8)
            
            # Add identity-specific pattern
            color_variation = base_color + np.random.randint(-30, 30, 3)
            color_variation = np.clip(color_variation, 0, 255)
            
            img[:, :] = color_variation
            
            # Add some noise to simulate real variations
            noise = np.random.randint(-20, 20, (224, 224, 3))
            img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
            
            # Save image
            img_name = f"person_{identity_id:03d}_img_{img_idx:02d}.jpg"
            img_path = os.path.join(images_dir, img_name)
            
            cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            identities_data.append({
                'image': img_name,
                'identity': f"person_{identity_id:03d}"
            })
    
    # Save identities file
    identities_df = pd.DataFrame(identities_data)
    identities_file = os.path.join(output_dir, 'identities.csv')
    identities_df.to_csv(identities_file, index=False)
    
    print(f"[+] Synthetic dataset created: {len(identities_data)} images")
    return output_dir, identities_file, images_dir

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[+] Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("[*] Using CPU")
    return device