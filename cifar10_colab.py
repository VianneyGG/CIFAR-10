# Google Colab optimized version of CIFAR-10 CNN Training
# Adapted for Google Colab with GPU acceleration and optimized settings

# === COLAB SETUP CELL ===
# Run this first to install dependencies and check GPU
!pip install -r requirements.txt

import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# Check GPU availability and setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    # Enable GPU optimizations
    torch.backends.cudnn.benchmark = True
    batch_size = 512
    num_workers = 2  # Colab works better with 2 workers
else:
    print("Using CPU - consider enabling GPU in Runtime > Change runtime type")
    batch_size = 256
    num_workers = 2

print(f"Batch size: {batch_size}")

# === UPLOAD CNN MODELS CELL ===
# Upload your CNN model files
print("Please upload your CNN_simple.py and CNN_improved.py files")
from google.colab import files
uploaded = files.upload()

# Import your models after upload
try:
    from CNN_simple import CNN_simple 
    from CNN_improved import CNN_improved
    print("✅ CNN models imported successfully")
except ImportError as e:
    print(f"❌ Error importing CNN models: {e}")
    print("Make sure you uploaded CNN_simple.py and CNN_improved.py")

# === CONSTANTS AND HELPER FUNCTIONS ===
# Les 10 classes de CIFAR-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def load_cifar10_colab(batch_size=512, num_workers=2):
    """Colab-optimized CIFAR-10 data loading with GPU acceleration"""
    # Transformations pour les données
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Chargement du dataset CIFAR-10 (auto-download enabled)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,  # Enable for GPU acceleration
        drop_last=True    # Ensures consistent batch sizes
    )
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=False
    )
    
    print(f"✅ Dataset loaded - Train: {len(trainset)}, Test: {len(testset)}")
    print(f"   Batches - Train: {len(trainloader)}, Test: {len(testloader)}")
    
    return trainloader, testloader

def save_model_colab(model, model_type='improved'):
    """Save model and auto-download in Colab"""
    model_path = f'{model_type}_model_cifar10.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Auto-download the model file in Colab
    try:
        files.download(model_path)
        print(f"✅ Model downloaded to your computer")
    except:
        print("💡 Model saved but download failed - you can download manually")

def display_sample_images(dataloader, num_samples=8):
    """Display sample images from the dataset"""
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    
    # Denormalize for display
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2023, 0.1994, 0.2010])
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        img = images[i] * std.view(3, 1, 1) + mean.view(3, 1, 1)
        img = torch.clamp(img, 0, 1)
        
        axes[i].imshow(img.permute(1, 2, 0))
        axes[i].set_title(f'{classes[labels[i]]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def training_colab(trainloader, testloader, model, model_type, criterion, optimizer, scheduler, device, num_epochs):
    """Colab-optimized training function with enhanced monitoring"""
    
    # Lists to store training statistics
    train_losses = []
    test_losses = []
    test_acc = []
    train_acc = []
    
    # Display training info
    print(f"🚀 Starting training on {device}")
    print(f"   Model: {model_type}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {trainloader.batch_size}")
    print(f"   Total batches per epoch: {len(trainloader)}")
    print("-" * 60)

    for epoch in tqdm(range(num_epochs), desc='Training Epochs', unit='epoch', colour='blue'):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Update dropout rate if using improved model
        if hasattr(model, 'update_dropout'):
            model.update_dropout(epoch, num_epochs)
        
        # Training phase with progress bar
        batch_pbar = tqdm(enumerate(trainloader, 0), 
                         total=len(trainloader), 
                         desc=f'Epoch {epoch+1}/{num_epochs}',
                         colour='green',
                         leave=False)        
        
        for i, (inputs, labels) in batch_pbar:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
            # Update progress bar every 20 batches
            if i % 20 == 0:
                batch_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * correct / total:.2f}%'
                })
        
        batch_pbar.close()
        
        # Calculate training accuracy and loss
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_acc.append(epoch_acc)
        
        # Validation phase
        model.eval()
        test_running_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(testloader, desc='Validation', leave=False, colour='red'):
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        # Calculate validation accuracy and loss
        test_epoch_loss = test_running_loss / len(testloader)
        test_epoch_acc = 100 * test_correct / test_total
        test_losses.append(test_epoch_loss)
        test_acc.append(test_epoch_acc)
        
        # Update learning rate
        scheduler.step()
        
        # Print statistics
        print(f'Epoch [{epoch + 1:2d}/{num_epochs}] | '
              f'Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | '
              f'Val Loss: {test_epoch_loss:.4f} | Val Acc: {test_epoch_acc:.2f}%')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'{model_type}_checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_loss,
                'val_acc': test_epoch_acc,
            }, checkpoint_path)
            print(f"   💾 Checkpoint saved: {checkpoint_path}")

    print("\n🎉 Training completed!")
    
    # Save the final trained model
    save_model_colab(model, model_type)
    
    # Plot training curves
    plot_training_curves(train_losses, test_losses, train_acc, test_acc)
    
    return train_losses, test_losses, train_acc, test_acc

def plot_training_curves(train_losses, test_losses, train_acc, test_acc):
    """Plot training and validation curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(test_losses, label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(train_acc, label='Training Accuracy', color='blue')
    ax2.plot(test_acc, label='Validation Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def evaluate_model_colab(model, testloader, device):
    """Evaluate model and show confusion matrix"""
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    
    print("🔍 Evaluating model...")
    
    with torch.no_grad():
        for images, labels in tqdm(testloader, desc='Evaluation'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Collect all labels and predictions for confusion matrix
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    accuracy = 100 * correct / total
    print(f'🎯 Final Test Accuracy: {accuracy:.2f}%')
    
    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Test Accuracy: {accuracy:.2f}%')
    plt.show()
    
    return accuracy

# === MAIN TRAINING EXECUTION ===
if __name__ == "__main__":
    print("🚀 CIFAR-10 CNN Training - Google Colab Version")
    print("=" * 60)
    
    # Load CIFAR-10 dataset
    print("📥 Loading CIFAR-10 dataset...")
    train_loader, testloader = load_cifar10_colab(batch_size, num_workers)
    
    # Display sample images
    print("📸 Sample images from dataset:")
    display_sample_images(train_loader)
    
    # Initialize model
    print("🧠 Initializing CNN model...")
    model = CNN_improved(num_classes=10)
    
    # Configuration personnalisée du dropout
    if hasattr(model, 'set_dropout_config'):
        model.set_dropout_config(start_rate=0.6, end_rate=0.1)
        print(f"   Dropout: {model.get_current_dropout_rate():.3f} → {model.dropout_config['end_rate']:.3f}")
    
    model = model.to(device)
    model_type = 'improved'
    
    # Training parameters
    num_epochs = 30
    print(f"   Epochs: {num_epochs}")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1e4
    )
    
    print("⚙️ Training configuration:")
    print(f"   Optimizer: Adam (lr=0.001)")
    print(f"   Scheduler: OneCycleLR (max_lr=0.01)")
    print(f"   Loss: CrossEntropyLoss")
    
    # Start training
    print("\n🎯 Starting training...")
    train_losses, test_losses, train_acc, test_acc = training_colab(
        train_loader, testloader, model, model_type, 
        criterion, optimizer, scheduler, device, num_epochs
    )
    
    # Final evaluation
    print("\n📊 Final evaluation:")
    final_accuracy = evaluate_model_colab(model, testloader, device)
    
    print(f"\n🏆 Training Summary:")
    print(f"   Final Validation Accuracy: {final_accuracy:.2f}%")
    print(f"   Best Validation Accuracy: {max(test_acc):.2f}%")
    print(f"   Model saved and downloaded successfully!")
