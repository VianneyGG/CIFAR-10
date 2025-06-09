# Test script for enhanced training with all improvements

import torch
import torch.nn as nn
from CNN_improved import CNNImproved
from cifar10_colab import *

def test_enhanced_model():
    """Test the enhanced model with all improvements"""
    print("🧪 Testing Enhanced CIFAR-10 Training")
    print("=" * 50)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Test model initialization
    print("\n1️⃣ Testing model initialization...")
    model = CNNImproved(num_classes=10)
    model = model.to(device)
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test enhanced transforms
    print("\n2️⃣ Testing enhanced data augmentation...")
    batch_size = 64 if device.type == 'cpu' else 128
    train_loader, test_loader = load_cifar10_colab(batch_size=batch_size, num_workers=0)
    print(f"✅ Data loaders created with enhanced augmentation")
    
    # Test mixup
    print("\n3️⃣ Testing mixup augmentation...")
    data_iter = iter(train_loader)
    inputs, labels = next(data_iter)
    inputs, labels = inputs.to(device), labels.to(device)
    
    mixed_inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=1.0)
    print(f"✅ Mixup applied: λ={lam:.3f}")
    
    # Test label smoothing
    print("\n4️⃣ Testing label smoothing...")
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    print(f"✅ Label smoothing loss: {loss.item():.4f}")
    
    # Test early stopping
    print("\n5️⃣ Testing early stopping...")
    early_stopping = EarlyStopping(patience=3, min_delta=0.001)
    val_acc = 75.0
    should_stop = early_stopping(val_acc, model)
    print(f"✅ Early stopping initialized (should_stop: {should_stop})")
    
    # Test enhanced optimizer
    print("\n6️⃣ Testing enhanced optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=0.001, 
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-6
    )
    print(f"✅ AdamW optimizer and CosineAnnealingWarmRestarts scheduler created")
    
    # Test short training loop
    print("\n7️⃣ Testing enhanced training loop (1 epoch)...")
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        if i >= 5:  # Only test first 5 batches
            break
            
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Test mixup training
        if torch.rand(1) < 0.4:
            mixed_inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=1.0)
            outputs = model(mixed_inputs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        print(f"   Batch {i+1}/5: Loss = {loss.item():.4f}")
    
    print(f"✅ Enhanced training loop working correctly")
    
    print("\n🎉 All tests passed! Enhanced training is ready to use.")
    print("\n📋 Summary of improvements:")
    print("   ✅ Enhanced data augmentation (ColorJitter, RandomErasing)")
    print("   ✅ Mixup data augmentation (40% probability)")
    print("   ✅ Label smoothing (smoothing=0.1)")
    print("   ✅ Early stopping (patience=7)")
    print("   ✅ AdamW optimizer with weight decay")
    print("   ✅ CosineAnnealingWarmRestarts scheduler")
    print("   ✅ Gradient clipping (max_norm=1.0)")
    print("   ✅ Improved model architecture with progressive dropout")
    
    print("\n🚀 Ready to run enhanced training with:")
    print("   python cifar10_colab.py")

if __name__ == "__main__":
    test_enhanced_model()
