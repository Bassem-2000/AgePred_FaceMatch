import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os
import glob
from pathlib import Path
import time
from typing import Union, List, Tuple, Optional, Dict

from config.config import config
from models.models import AgeEstimationModel, load_model_checkpoint


class AgePredictor:
    """
    High-level age prediction class for inference on facial images.
    
    This class handles model loading, image preprocessing, prediction,
    and result visualization for age estimation tasks.
    """
    
    def __init__(self, model_path: Optional[str] = None, model: Optional[AgeEstimationModel] = None, 
                 device: Optional[str] = None, confidence_threshold: float = 0.8):
        """
        Initialize the age predictor.
        
        Args:
            model_path (str, optional): Path to trained model checkpoint
            model (AgeEstimationModel, optional): Pre-loaded model instance
            device (str, optional): Device to run inference on
            confidence_threshold (float): Confidence threshold for predictions
        """
        
        self.device = device or config['device']
        self.confidence_threshold = confidence_threshold
        
        # Load model
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = self._load_model(model_path)
        else:
            # Try to load latest checkpoint from default directory
            self.model = self._load_latest_checkpoint()
        
        self.model.to(self.device)
        self.model.eval()
        
        # Setup transforms
        self.transform = self._create_transform()
        
        # Prediction statistics
        self.prediction_count = 0
        self.total_inference_time = 0.0
        
        print(f"AgePredictor initialized on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _load_model(self, model_path: str) -> AgeEstimationModel:
        """Load model from checkpoint file"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        model = load_model_checkpoint(model_path, device=self.device)
        print(f"Loaded model from: {model_path}")
        return model
    
    def _load_latest_checkpoint(self) -> AgeEstimationModel:
        """Load the latest checkpoint from the default checkpoint directory"""
        checkpoint_dir = config['checkpoint_dir']
        
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        # Find all checkpoint files
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'epoch-*-loss_valid-*.pt'))
        
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
        
        # Get the latest checkpoint (by modification time)
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        
        return self._load_model(latest_checkpoint)
    
    def _create_transform(self) -> T.Compose:
        """Create image preprocessing transform"""
        return T.Compose([
            T.Resize((config['img_width'], config['img_height'])),
            T.ToTensor(),
            T.Normalize(mean=config['mean'], std=config['std'])
        ])
    
    def predict_single(self, image: Union[str, Path, Image.Image, np.ndarray], 
                      return_confidence: bool = False) -> Union[float, Tuple[float, float]]:
        """
        Predict age for a single image.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            return_confidence: Whether to return confidence score
            
        Returns:
            Predicted age (and confidence if requested)
        """
        
        start_time = time.time()
        
        # Load and preprocess image
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            pil_image = image.convert('RGB')
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Transform image
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            predicted_age = output.item()
        
        # Calculate confidence (simple approach using prediction consistency)
        confidence = self._calculate_confidence(input_tensor)
        
        # Update statistics
        self.prediction_count += 1
        self.total_inference_time += time.time() - start_time
        
        if return_confidence:
            return predicted_age, confidence
        return predicted_age
    
    def predict_batch(self, images: List[Union[str, Path, Image.Image, np.ndarray]], 
                     batch_size: int = 32) -> List[Tuple[float, float]]:
        """
        Predict ages for multiple images in batches.
        
        Args:
            images: List of input images
            batch_size: Batch size for processing
            
        Returns:
            List of (predicted_age, confidence) tuples
        """
        
        results = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_results = self._predict_batch_internal(batch_images)
            results.extend(batch_results)
        
        return results
    
    def _predict_batch_internal(self, images: List) -> List[Tuple[float, float]]:
        """Internal batch prediction method"""
        
        # Preprocess images
        batch_tensors = []
        for image in images:
            if isinstance(image, (str, Path)):
                pil_image = Image.open(image).convert('RGB')
            elif isinstance(image, Image.Image):
                pil_image = image.convert('RGB')
            elif isinstance(image, np.ndarray):
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                continue
            
            tensor = self.transform(pil_image)
            batch_tensors.append(tensor)
        
        if not batch_tensors:
            return []
        
        # Stack into batch
        batch_tensor = torch.stack(batch_tensors).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            predicted_ages = outputs.squeeze().cpu().numpy()
        
        # Calculate confidences
        confidences = [self._calculate_confidence(tensor.unsqueeze(0)) for tensor in batch_tensors]
        
        # Handle single prediction case
        if predicted_ages.ndim == 0:
            predicted_ages = [predicted_ages.item()]
        
        results = list(zip(predicted_ages.tolist(), confidences))
        self.prediction_count += len(results)
        
        return results
    
    def _calculate_confidence(self, input_tensor: torch.Tensor) -> float:
        """
        Calculate prediction confidence using model uncertainty estimation.
        
        This is a simplified approach - for production, consider using
        Monte Carlo Dropout or ensemble methods.
        """
        
        with torch.no_grad():
            # Method 1: Use prediction consistency with slight augmentations
            original_pred = self.model(input_tensor).item()
            
            # Apply small augmentations and check consistency
            augmented_preds = []
            for _ in range(5):
                # Add small noise
                noise = torch.randn_like(input_tensor) * 0.01
                noisy_input = input_tensor + noise
                pred = self.model(noisy_input).item()
                augmented_preds.append(pred)
            
            # Calculate confidence based on prediction variance
            pred_std = np.std(augmented_preds + [original_pred])
            
            # Convert to confidence score (lower variance = higher confidence)
            confidence = max(0.0, min(1.0, 1.0 - (pred_std / 10.0)))  # Normalize by expected std
            
            return confidence
    
    def predict_with_visualization(self, image_path: Union[str, Path], 
                                 output_path: Optional[Union[str, Path]] = None,
                                 show_confidence: bool = True,
                                 font_size: int = 20,
                                 font_color: Tuple[int, int, int] = (255, 0, 0)) -> Tuple[float, float]:
        """
        Predict age and create visualization with annotation.
        
        Args:
            image_path: Path to input image
            output_path: Path to save annotated image (optional)
            show_confidence: Whether to show confidence in annotation
            font_size: Font size for annotation
            font_color: RGB color for annotation text
            
        Returns:
            Tuple of (predicted_age, confidence)
        """
        
        # Predict age
        predicted_age, confidence = self.predict_single(image_path, return_confidence=True)
        
        # Load original image
        image = Image.open(image_path).convert('RGB')
        
        # Create annotated image
        annotated_image = self._annotate_image(
            image, predicted_age, confidence, 
            show_confidence, font_size, font_color
        )
        
        # Save if output path provided
        if output_path:
            annotated_image.save(output_path)
            print(f"Annotated image saved to: {output_path}")
        
        return predicted_age, confidence
    
    def _annotate_image(self, image: Image.Image, age: float, confidence: float,
                       show_confidence: bool, font_size: int, 
                       font_color: Tuple[int, int, int]) -> Image.Image:
        """Add age prediction annotation to image"""
        
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        
        # Create annotation text
        if show_confidence:
            text = f"Age: {age:.1f} (Conf: {confidence:.2f})"
        else:
            text = f"Age: {age:.1f}"
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except (OSError, IOError):
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        # Get text size for positioning
        if font:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            # Estimate text size for default font
            text_width = len(text) * 6
            text_height = 11
        
        # Position text (top-left with some padding)
        x, y = 10, 10
        
        # Draw text background for better visibility
        padding = 5
        draw.rectangle([x-padding, y-padding, x+text_width+padding, y+text_height+padding], 
                      fill=(0, 0, 0, 128))
        
        # Draw text
        draw.text((x, y), text, fill=font_color, font=font)
        
        return annotated_image
    
    def predict_from_directory(self, input_dir: Union[str, Path], 
                             output_dir: Optional[Union[str, Path]] = None,
                             image_extensions: List[str] = None) -> Dict[str, Tuple[float, float]]:
        """
        Predict ages for all images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save annotated images (optional)
            image_extensions: List of image file extensions to process
            
        Returns:
            Dictionary mapping filename to (age, confidence)
        """
        
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_dir.glob(f"*{ext}"))
            image_files.extend(input_dir.glob(f"*{ext.upper()}"))
        
        print(f"Found {len(image_files)} images to process")
        
        results = {}
        
        for image_file in image_files:
            try:
                # Predict age
                age, confidence = self.predict_single(image_file, return_confidence=True)
                results[image_file.name] = (age, confidence)
                
                # Create annotated image if output directory specified
                if output_dir:
                    output_path = output_dir / f"{image_file.stem}_predicted{image_file.suffix}"
                    self.predict_with_visualization(image_file, output_path)
                
                print(f"Processed {image_file.name}: Age {age:.1f} (Confidence: {confidence:.2f})")
                
            except Exception as e:
                print(f"Error processing {image_file.name}: {e}")
                results[image_file.name] = (None, None)
        
        return results
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get inference performance statistics"""
        
        if self.prediction_count == 0:
            return {"predictions": 0, "avg_time_ms": 0.0, "fps": 0.0}
        
        avg_time = self.total_inference_time / self.prediction_count
        fps = 1.0 / avg_time if avg_time > 0 else 0.0
        
        return {
            "predictions": self.prediction_count,
            "total_time_s": self.total_inference_time,
            "avg_time_ms": avg_time * 1000,
            "fps": fps
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.prediction_count = 0
        self.total_inference_time = 0.0


def predict_single_image(image_path: Union[str, Path], model_path: Optional[str] = None,
                        output_path: Optional[Union[str, Path]] = None) -> float:
    """
    Standalone function to predict age for a single image.
    
    Args:
        image_path: Path to input image
        model_path: Path to model checkpoint (uses latest if None)
        output_path: Path to save annotated image (optional)
        
    Returns:
        Predicted age
    """
    
    predictor = AgePredictor(model_path=model_path)
    
    if output_path:
        age, _ = predictor.predict_with_visualization(image_path, output_path)
    else:
        age = predictor.predict_single(image_path)
    
    return age


def predict_batch_images(image_paths: List[Union[str, Path]], 
                        model_path: Optional[str] = None,
                        batch_size: int = 32) -> List[Tuple[float, float]]:
    """
    Standalone function to predict ages for multiple images.
    
    Args:
        image_paths: List of paths to input images
        model_path: Path to model checkpoint (uses latest if None)
        batch_size: Batch size for processing
        
    Returns:
        List of (predicted_age, confidence) tuples
    """
    
    predictor = AgePredictor(model_path=model_path)
    return predictor.predict_batch(image_paths, batch_size=batch_size)


def load_model_for_inference(model_path: Optional[str] = None, device: Optional[str] = None) -> AgePredictor:
    """
    Load and return an AgePredictor instance ready for inference.
    
    Args:
        model_path: Path to model checkpoint (uses latest if None)
        device: Device to run inference on (uses config default if None)
        
    Returns:
        AgePredictor instance
    """
    
    return AgePredictor(model_path=model_path, device=device)


# Utility functions for inference
def validate_image_format(image_path: Union[str, Path]) -> bool:
    """
    Validate if image format is supported.
    
    Args:
        image_path: Path to image file
        
    Returns:
        True if format is supported
    """
    
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    try:
        path = Path(image_path)
        return path.suffix.lower() in supported_formats
    except:
        return False


def preprocess_image_for_inference(image: Union[str, Path, Image.Image, np.ndarray],
                                 target_size: Tuple[int, int] = None) -> torch.Tensor:
    """
    Preprocess image for age estimation inference.
    
    Args:
        image: Input image in various formats
        target_size: Target size (width, height) for resizing
        
    Returns:
        Preprocessed tensor ready for model input
    """
    
    if target_size is None:
        target_size = (config['img_width'], config['img_height'])
    
    # Convert to PIL Image
    if isinstance(image, (str, Path)):
        pil_image = Image.open(image).convert('RGB')
    elif isinstance(image, Image.Image):
        pil_image = image.convert('RGB')
    elif isinstance(image, np.ndarray):
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    # Create transform
    transform = T.Compose([
        T.Resize(target_size),
        T.ToTensor(),
        T.Normalize(mean=config['mean'], std=config['std'])
    ])
    
    # Apply transform and add batch dimension
    tensor = transform(pil_image).unsqueeze(0)
    
    return tensor


# Example usage and testing
if __name__ == "__main__":
    print("Testing age prediction inference...")
    
    try:
        # Test predictor initialization
        predictor = AgePredictor()
        
        # Test with dummy image (if you have a test image)
        test_image_path = config.get('test_image_dir', 'test_image.jpg')
        
        if os.path.exists(test_image_path):
            # Single prediction
            age = predictor.predict_single(test_image_path)
            print(f"Predicted age: {age:.1f}")
            
            # Prediction with visualization
            output_path = "test_output_with_age.jpg"
            age, confidence = predictor.predict_with_visualization(
                test_image_path, output_path
            )
            print(f"Predicted age: {age:.1f}, Confidence: {confidence:.2f}")
            
            # Performance stats
            stats = predictor.get_performance_stats()
            print(f"Performance: {stats}")
        
        else:
            print(f"Test image not found: {test_image_path}")
            print("Skipping inference tests")
        
        print("Inference testing completed!")
        
    except Exception as e:
        print(f"Error during inference testing: {e}")