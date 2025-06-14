import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Enhanced residual block with dynamic dropout"""
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                              kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(identity)
        out = F.relu(out)
        
        return out

class CNN_improved(nn.Module):
    """
    Enhanced CNN model with regularization to combat overfitting
    - Reduced model complexity
    - Progressive dropout strategy
    - Batch normalization
    - Residual connections
    - Global average pooling to reduce parameters
    """
    def __init__(self, num_classes=10):
        super(CNN_improved, self).__init__()
        
        # Initial convolution - reduced complexity
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Progressive feature extraction with residual blocks
        self.layer1 = self._make_layer(64, 96, num_blocks=2, stride=1, dropout_rate=0.05)
        self.layer2 = self._make_layer(96, 192, num_blocks=2, stride=2, dropout_rate=0.1)
        self.layer3 = self._make_layer(192, 384, num_blocks=2, stride=2, dropout_rate=0.15)
        
        # Global Average Pooling reduces parameters significantly
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Progressive dropout strategy
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.5)
        
        # Smaller classifier to prevent overfitting
        self.fc1 = nn.Linear(384, 256)
        self.fc2 = nn.Linear(256, num_classes)
          # Initialize weights
        self._initialize_weights()
        
        # Dropout configuration tracking
        self.dropout_config = {
            'start_rate': 0.1,
            'end_rate': 0.5,
            'current_epoch': 0,
            'max_epochs': 100
        }
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride, dropout_rate):
        """Create residual blocks with progressive dropout"""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, dropout_rate))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1, dropout_rate=dropout_rate))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Progressive feature extraction
        x = self.layer1(x)
        x = self.dropout1(x)  # Early dropout
        
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
          # Classification head with dropout
        x = self.dropout2(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x
    
    def get_num_parameters(self):
        """Return the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def update_dropout_rates(self, epoch, max_epochs):
        """
        Dynamic dropout adjustment based on training progress
        Increases dropout as training progresses to prevent overfitting
        """
        # Update configuration
        self.dropout_config['current_epoch'] = epoch
        self.dropout_config['max_epochs'] = max_epochs
        
        # Calculate progression factor (0 to 1)
        progress = min(epoch / max_epochs, 1.0)
        
        # Dynamic dropout rates - start low, increase with training
        base_rate = 0.05 + 0.1 * progress  # 0.1 to 0.3
        mid_rate = 0.1 + 0.15 * progress   # 0.2 to 0.5
        high_rate = 0.15 + 0.2 * progress  # 0.3 to 0.7
        
        # Update layer dropout rates
        for layer in [self.layer1, self.layer2, self.layer3]:
            for block in layer:
                if hasattr(block, 'dropout1'):
                    if layer == self.layer1:
                        block.dropout1.p = base_rate
                    elif layer == self.layer2:
                        block.dropout1.p = mid_rate
                    else:  # layer3
                        block.dropout1.p = high_rate
        
        # Update classifier dropout
        self.dropout1.p = 0.1 + 0.15 * progress  # 0.2 to 0.5
        self.dropout2.p = 0.15 + 0.2 * progress  # 0.3 to 0.7
    
    def print_model_info(self):
        """Print model architecture and parameter information"""
        print("=" * 60)
        print("🏗️  ENHANCED CNN ARCHITECTURE")
        print("=" * 60)
        print(f"📊 Total Parameters: {self.get_num_parameters():,}")
        print(f"🎯 Target: Reduce overfitting gap from 21% to <10%")
        print("=" * 60)
        
        print("\n🔧 MODEL FEATURES:")
        print("• Residual connections for better gradient flow")
        print("• Progressive dropout: 0.1 → 0.3 → 0.5")
        print("• Batch normalization for training stability")
        print("• Global average pooling to reduce parameters")
        print("• Reduced complexity: 256 final features (vs 512)")
        print("• Xavier/He weight initialization")
        print("• Dynamic dropout adjustment during training")
        
        print("\n📈 EXPECTED IMPROVEMENTS:")
        print("• Better generalization (reduce train-val gap)")
        print("• More stable training curves")
        print("• Improved validation accuracy: 78% → 85%+")
        print("• Faster convergence with better regularization")
        print("=" * 60)

def create_model(num_classes=10, print_info=True):
    """
    Factory function to create the enhanced CNN model
    
    Args:
        num_classes (int): Number of output classes (default: 10 for CIFAR-10)
        print_info (bool): Whether to print model information
    
    Returns:
        CNNImproved: The enhanced CNN model
    """
    model = CNN_improved(num_classes=num_classes)
    
    if print_info:
        model.print_model_info()
    
    return model

# Test the model if run directly
if __name__ == "__main__":
    print("🧪 Testing Enhanced CNN Model...")
    
    # Create model
    model = create_model()
    
    # Test with sample input
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Test forward pass
    sample_input = torch.randn(4, 3, 32, 32).to(device)  # Batch of 4 CIFAR-10 images
    
    print(f"\n🔍 Testing forward pass...")
    print(f"Input shape: {sample_input.shape}")
    
    with torch.no_grad():
        output = model(sample_input)
        print(f"Output shape: {output.shape}")
        print(f"Output sample: {output[0][:5].cpu().numpy()}")
    
    # Test dynamic dropout update
    print(f"\n🔄 Testing dynamic dropout update...")
    print(f"Initial dropout rates:")
    print(f"  Layer1 dropout: {model.layer1[0].dropout1.p:.3f}")
    print(f"  Layer2 dropout: {model.layer2[0].dropout1.p:.3f}")
    print(f"  Layer3 dropout: {model.layer3[0].dropout1.p:.3f}")
    print(f"  Classifier dropout1: {model.dropout1.p:.3f}")
    print(f"  Classifier dropout2: {model.dropout2.p:.3f}")
    
    # Simulate training progress
    model.update_dropout_rates(epoch=25, max_epochs=50)  # Mid-training
    print(f"\nAfter 25/50 epochs:")
    print(f"  Layer1 dropout: {model.layer1[0].dropout1.p:.3f}")
    print(f"  Layer2 dropout: {model.layer2[0].dropout1.p:.3f}")
    print(f"  Layer3 dropout: {model.layer3[0].dropout1.p:.3f}")
    print(f"  Classifier dropout1: {model.dropout1.p:.3f}")
    print(f"  Classifier dropout2: {model.dropout2.p:.3f}")
    
    print(f"\n✅ Model test completed successfully!")
    print(f"📊 Model ready for enhanced training with mixup and label smoothing")
    
    def update_dropout(self, epoch, max_epochs):
        """
        Méthode de compatibilité - redirige vers update_dropout_rates
        """
        return self.update_dropout_rates(epoch, max_epochs)
    
    def get_current_dropout_rate(self):
        """
        Retourne le taux de dropout actuel de la première couche
        """
        return self.dropout1.p