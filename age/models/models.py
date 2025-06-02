import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import timm
from config.config import config


class AgeEstimationModel(nn.Module):
    """
    Age Estimation Model supporting multiple architectures.
    
    Supports:
    - ResNet50 (pretrained on ImageNet)
    - Vision Transformer (ViT) variants
    
    Args:
        input_dim (int): Number of input channels (default: 3 for RGB)
        output_nodes (int): Number of output nodes (default: 1 for age regression)
        model_name (str): Model architecture ('resnet' or 'vit')
        pretrain_weights (str or bool): Pretrained weights to use
        dropout_rate (float): Dropout rate for regularization
    """
    
    def __init__(self, input_dim=3, output_nodes=1, model_name='resnet', 
                 pretrain_weights='IMAGENET1K_V2', dropout_rate=0.2):
        super(AgeEstimationModel, self).__init__()
        
        self.input_dim = input_dim
        self.output_nodes = output_nodes
        self.model_name = model_name.lower()
        self.pretrain_weights = pretrain_weights
        self.dropout_rate = dropout_rate
        
        # Build the model based on the specified architecture
        self._build_model()
        
        # Initialize weights if needed
        self._initialize_weights()
    
    def _build_model(self):
        """Build the model architecture based on model_name"""
        
        if self.model_name == 'resnet':
            self._build_resnet()
        elif self.model_name == 'vit':
            self._build_vit()
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}. "
                           f"Supported models: ['resnet', 'vit']")
    
    def _build_resnet(self):
        """Build ResNet50 architecture"""
        
        # Load pretrained ResNet50
        if self.pretrain_weights == 'IMAGENET1K_V2':
            weights = ResNet50_Weights.IMAGENET1K_V2
        elif self.pretrain_weights:
            weights = ResNet50_Weights.DEFAULT
        else:
            weights = None
            
        self.backbone = resnet50(weights=weights)
        
        # Get the number of features from the last layer
        num_features = self.backbone.fc.in_features
        
        # Replace the classifier with our custom head
        self.backbone.fc = nn.Identity()  # Remove original classifier
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_rate, inplace=True),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=self.dropout_rate, inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=self.dropout_rate, inplace=True),
            nn.Linear(256, self.output_nodes)
        )
        
        self.model = nn.Sequential(self.backbone, self.classifier)
    
    def _build_vit(self):
        """Build Vision Transformer architecture"""
        
        # Create ViT model using timm
        self.backbone = timm.create_model(
            'vit_small_patch14_dinov2.lvd142m',
            img_size=config['img_size'],
            pretrained=bool(self.pretrain_weights),
            num_classes=0  # Remove classification head
        )
        
        # Get number of features
        num_features = self.backbone.num_features
        
        # Custom classification head for ViT
        self.classifier = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(num_features, 384),
            nn.GELU(),
            nn.LayerNorm(384),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(384, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(256, self.output_nodes)
        )
        
        self.model = nn.Sequential(self.backbone, self.classifier)
    
    def _initialize_weights(self):
        """Initialize weights for custom layers"""
        
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Age predictions of shape (batch_size, output_nodes)
        """
        
        # Extract features using backbone
        features = self.backbone(x)
        
        # Apply classifier
        output = self.classifier(features)
        
        return output
    
    def get_num_parameters(self):
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self):
        """Get detailed model information"""
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'model_name': self.model_name,
            'input_dim': self.input_dim,
            'output_nodes': self.output_nodes,
            'pretrain_weights': self.pretrain_weights,
            'dropout_rate': self.dropout_rate,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }
        
        return info
    
    def freeze_backbone(self, freeze=True):
        """
        Freeze or unfreeze the backbone parameters.
        
        Args:
            freeze (bool): Whether to freeze the backbone
        """
        
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        
        print(f"Backbone {'frozen' if freeze else 'unfrozen'}")
    
    def unfreeze_top_layers(self, num_layers=2):
        """
        Unfreeze the top N layers of the backbone for fine-tuning.
        
        Args:
            num_layers (int): Number of top layers to unfreeze
        """
        
        if self.model_name == 'resnet':
            # Unfreeze last few layers of ResNet
            layers = [self.backbone.layer4, self.backbone.avgpool]
            if num_layers > 1:
                layers.append(self.backbone.layer3)
            
            for layer in layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
                    
        elif self.model_name == 'vit':
            # Unfreeze last few transformer blocks
            blocks = list(self.backbone.blocks)
            for block in blocks[-num_layers:]:
                for param in block.parameters():
                    param.requires_grad = True
        
        print(f"Unfroze top {num_layers} layers of {self.model_name}")


class AgeRegressionHead(nn.Module):
    """
    Flexible regression head for age estimation.
    
    Args:
        input_features (int): Number of input features
        hidden_dim (int): Hidden dimension size
        output_dim (int): Output dimension (default: 1)
        dropout_rate (float): Dropout rate
        use_batch_norm (bool): Whether to use batch normalization
    """
    
    def __init__(self, input_features, hidden_dim=512, output_dim=1, 
                 dropout_rate=0.2, use_batch_norm=True):
        super(AgeRegressionHead, self).__init__()
        
        layers = []
        
        # First layer
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(input_features, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        
        # Second layer
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_dim, hidden_dim // 2))
        layers.append(nn.ReLU(inplace=True))
        
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim // 2))
        
        # Output layer
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_dim // 2, output_dim))
        
        self.head = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.head(x)


def create_model(model_name=None, pretrain_weights=None, **kwargs):
    """
    Factory function to create an age estimation model.
    
    Args:
        model_name (str, optional): Model architecture. Uses config default if None.
        pretrain_weights (str or bool, optional): Pretrained weights. Uses config default if None.
        **kwargs: Additional arguments for the model
        
    Returns:
        AgeEstimationModel: The created model
    """
    
    if model_name is None:
        model_name = config['model_name']
    
    if pretrain_weights is None:
        pretrain_weights = config['pretrain_weights']
    
    # Default parameters from config
    model_params = {
        'input_dim': config['input_dim'],
        'output_nodes': config['output_nodes'],
        'model_name': model_name,
        'pretrain_weights': pretrain_weights,
        'dropout_rate': config['dropout_rate']
    }
    
    # Update with any provided kwargs
    model_params.update(kwargs)
    
    # Create and return the model
    model = AgeEstimationModel(**model_params)
    
    # Print model information
    info = model.get_model_info()
    print(f"\nCreated {info['model_name']} model:")
    print(f"  Total parameters: {info['total_parameters']:,}")
    print(f"  Trainable parameters: {info['trainable_parameters']:,}")
    print(f"  Model size: {info['model_size_mb']:.2f} MB")
    print(f"  Pretrained weights: {info['pretrain_weights']}")
    
    return model


def load_model_checkpoint(checkpoint_path, model=None, device=None):
    """
    Load a model from a checkpoint file.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        model (AgeEstimationModel, optional): Model instance. Creates new if None.
        device (str, optional): Device to load the model on. Uses config default if None.
        
    Returns:
        AgeEstimationModel: The loaded model
    """
    
    if device is None:
        device = config['device']
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model if not provided
    if model is None:
        model = create_model()
    
    # Load state dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    return model


def save_model_checkpoint(model, checkpoint_path, epoch=None, loss=None, optimizer=None, **kwargs):
    """
    Save a model checkpoint.
    
    Args:
        model (AgeEstimationModel): The model to save
        checkpoint_path (str): Path to save the checkpoint
        epoch (int, optional): Current epoch
        loss (float, optional): Current loss
        optimizer (torch.optim.Optimizer, optional): Optimizer state
        **kwargs: Additional information to save
    """
    
    checkpoint = {
        'state_dict': model.state_dict(),
        'model_info': model.get_model_info()
    }
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if loss is not None:
        checkpoint['loss'] = loss
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    
    # Add any additional kwargs
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")


# Example usage and testing
if __name__ == "__main__":
    print("Testing age estimation model...")
    
    # Test model creation
    model = create_model()
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        print(f"Sample predictions: {output.squeeze().tolist()}")
    
    # Test different architectures
    print("\nTesting ViT model...")
    vit_model = create_model(model_name='vit')
    
    with torch.no_grad():
        vit_output = vit_model(dummy_input)
        print(f"ViT output shape: {vit_output.shape}")
    
    print("\nModel testing completed!")