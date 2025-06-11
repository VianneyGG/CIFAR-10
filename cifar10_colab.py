# Google Colab optimized version of CIFAR-10 CNN Training
# Adapted for Google Colab with GPU acceleration and optimized settings

# === COLAB SETUP CELL ===
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torch.optim as optim
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

# Import your models after upload
try:
    from CNN_simple import CNN_simple 
    from CNN_improved import CNN_improved, create_model
    print("✅ CNN models imported successfully")
except ImportError as e:
    print(f"❌ Error importing CNN models: {e}")
    print("Make sure you uploaded CNN_simple.py and CNN_improved.py")

# === CONSTANTS AND HELPER FUNCTIONS ===
# Les 10 classes de CIFAR-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label smoothing prevents the model from being overconfident
    This helps with your overfitting issue
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - self.smoothing) + (1 - one_hot) * self.smoothing / (n_class - 1)
        log_prob = F.log_softmax(pred, dim=1)
        return F.kl_div(log_prob, one_hot, reduction='batchmean')

def mixup_data(x, y, alpha=1.0):
    """
    Mixup augmentation that blends images and labels
    Proven to reduce overfitting significantly
    """
    if alpha > 0:
        lam = torch.distributions.beta.Beta(alpha, alpha).sample()
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def load_cifar10_colab(batch_size=512, num_workers=2):
    """Colab-optimized CIFAR-10 data loading with enhanced data augmentation"""    # Enhanced transformations for training (combat overfitting)
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),  # Increased from 10
        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # NEW
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        transforms.RandomErasing(p=0.05),  # Applied after ToTensor
    ])
    
    # Standard transformations for testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # Chargement du dataset CIFAR-10 (auto-download enabled)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,  # Enable for GPU acceleration
        drop_last=True    # Ensures consistent batch sizes
    )
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
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
    """Save model to repository (Git-friendly)"""
    model_path = f'{model_type}_model_cifar10.pth'
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")
    
    # Also save a timestamped version for tracking
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_path = f'{model_type}_model_cifar10_{timestamp}.pth'
    torch.save(model.state_dict(), timestamped_path)
    print(f"📁 Timestamped backup: {timestamped_path}")
    
    # Create a simple training info file
    info_file = f'{model_type}_training_info.txt'
    with open(info_file, 'w') as f:
        f.write(f"CIFAR-10 Training Information\n")
        f.write(f"={'='*40}\n")
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
        f.write(f"PyTorch Version: {torch.__version__}\n")
        f.write(f"Model File: {model_path}\n")
        f.write(f"Backup File: {timestamped_path}\n")
    
    print(f"📄 Training info saved to {info_file}")
    
    # Show Git status
    try:
        import subprocess
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True, cwd='.')
        if result.returncode == 0:
            print(f"\n📊 Git Status:")
            if result.stdout.strip():
                for line in result.stdout.strip().split('\n'):
                    if '.pth' in line or '.txt' in line:
                        print(f"   {line}")
            else:
                print("   No changes to commit")
        else:
            print("💡 Not in a Git repository or Git not available")
    except:
        print("💡 Git status check failed")

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
    """Enhanced training function with Mixup, Early Stopping and advanced strategies"""
    
    # Lists to store training statistics
    train_losses = []
    test_losses = []
    test_acc = []
    train_acc = []
    
    # Initialize Early Stopping
    early_stopping = EarlyStopping(patience=7, min_delta=0.001)
    
    # Display training info
    print(f"🚀 Starting Enhanced Training on {device}")
    print(f"   Model: {model_type}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {trainloader.batch_size}")
    print(f"   Total batches per epoch: {len(trainloader)}")
    print(f"   🎯 Targeting overfitting: Enhanced regularization strategies")
    print("-" * 60)

    for epoch in tqdm(range(num_epochs), desc='Training Epochs', unit='epoch', colour='blue'):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
          # Update dropout rate if using improved model
        if hasattr(model, 'update_dropout_rates'):
            model.update_dropout_rates(epoch, num_epochs)
        
        # Training phase with progress bar and MIXUP
        batch_pbar = tqdm(enumerate(trainloader, 0), 
                         total=len(trainloader), 
                         desc=f'Epoch {epoch+1}/{num_epochs}',
                         colour='green',
                         leave=False)        
        
        for i, (inputs, labels) in batch_pbar:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Apply Mixup augmentation 40% of the time
            if torch.rand(1) < 0.2:
                mixed_inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=1.0)
                outputs = model(mixed_inputs)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                
                # For accuracy calculation with mixup (use original labels)
                _, predicted = torch.max(outputs.data, 1)
                correct += (lam * predicted.eq(labels_a).sum().item() + 
                           (1 - lam) * predicted.eq(labels_b).sum().item())
            else:
                # Standard training
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics for non-mixup batches
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient Clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            total += labels.size(0)
        
            # Update progress bar every 20 batches
            if i % 20 == 0:
                current_acc = 100 * correct / total
                batch_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.2f}%'
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
        
        # Print statistics with overfitting gap monitoring
        overfitting_gap = abs(epoch_acc - test_epoch_acc)
        print(f'Epoch [{epoch + 1:2d}/{num_epochs}] | '
              f'Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | '
              f'Val Loss: {test_epoch_loss:.4f} | Val Acc: {test_epoch_acc:.2f}% | '
              f'Gap: {overfitting_gap:.2f}%')
        
        # Check for early stopping
        if early_stopping(test_epoch_acc, model):
            print(f"✅ Early stopping triggered at epoch {epoch + 1}")
            print(f"🎯 Best validation accuracy: {early_stopping.best_val_acc:.2f}%")
            break
        
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

    print("\n🎉 Enhanced Training completed!")
    print(f"🏆 Best validation accuracy achieved: {early_stopping.best_val_acc:.2f}%")
      # Save the final trained model
    save_model_colab(model, model_type)
    
    # Plot training curves and save to repository
    plot_filename = plot_training_curves(train_losses, test_losses, train_acc, test_acc)
    
    return train_losses, test_losses, train_acc, test_acc, plot_filename

def plot_training_curves(train_losses, test_losses, train_acc, test_acc):
    """Plot training and validation curves and save to repository"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
    ax1.plot(test_losses, label='Validation Loss', color='red', linewidth=2)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(train_acc, label='Training Accuracy', color='blue', linewidth=2)
    ax2.plot(test_acc, label='Validation Accuracy', color='red', linewidth=2)
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot to repository
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f'training_curves_{timestamp}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📈 Training curves saved to {plot_filename}")
    
    plt.show()
    return plot_filename

def evaluate_model_colab(model, testloader, device):
    """Evaluate model and show confusion matrix, save results to repository"""
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
    
    # Save confusion matrix to repository
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cm_filename = f'confusion_matrix_{timestamp}.png'
    plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Confusion matrix saved to {cm_filename}")
    
    plt.show()
    
    # Save detailed results
    results_filename = f'evaluation_results_{timestamp}.txt'
    with open(results_filename, 'w') as f:
        f.write(f"CIFAR-10 Model Evaluation Results\n")
        f.write(f"={'='*40}\n")
        f.write(f"Test Accuracy: {accuracy:.2f}%\n")
        f.write(f"Total Test Samples: {total}\n")
        f.write(f"Correct Predictions: {correct}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n\n")
        
        # Per-class accuracy
        f.write("Per-Class Accuracy:\n")
        f.write("-" * 20 + "\n")
        for i, class_name in enumerate(classes):
            class_correct = sum((torch.tensor(all_preds) == i) & (torch.tensor(all_labels) == i))
            class_total = sum(torch.tensor(all_labels) == i)
            class_acc = 100.0 * class_correct / class_total if class_total > 0 else 0
            f.write(f"{class_name:>10}: {class_acc:>6.2f}%\n")
    
    print(f"📄 Detailed results saved to {results_filename}")
    
    return accuracy, cm_filename, results_filename

# =============================================================================
# ADVANCED TRAINING STRATEGIES - EARLY STOPPING
# =============================================================================

class EarlyStopping:
    """
    Stop training when validation accuracy stops improving
    This addresses your issue of training for too many epochs without improvement
    """
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_val_acc = 0
        self.wait = 0
        self.best_weights = None
        
    def __call__(self, val_acc, model):
        if val_acc > self.best_val_acc + self.min_delta:
            self.best_val_acc = val_acc
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

# === MAIN TRAINING EXECUTION ===
if __name__ == "__main__":
    print("🚀 CIFAR-10 CNN Training - Google Colab Version")
    print("=" * 60)
    
    # Load CIFAR-10 dataset
    print("📥 Loading CIFAR-10 dataset...")
    train_loader, testloader = load_cifar10_colab(batch_size, num_workers)
    
    # Display sample images
    print("📸 Sample images from dataset:")
    display_sample_images(train_loader)    # Initialize model
    print("🧠 Initializing CNN model...")
    model = CNN_improved(num_classes=10)
    
    model = model.to(device)
    model_type = 'improved'
    
    # Training parameters
    num_epochs = 30
    print(f"   Epochs: {num_epochs}")
    
    # Define loss function and optimizer
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = torch.optim.AdamW(  # AdamW includes better weight decay
        model.parameters(), 
        lr=0.003, 
        weight_decay=5e-5,  # L2 regularization
        betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # First restart after 10 epochs
        T_mult=1,  # Period multiplier
        eta_min=1e-6  # Minimum learning rate
    )
    
    print("⚙️ Training configuration:")
    print(f"   Optimizer: Adam (lr=0.001)")
    print(f"   Scheduler: OneCycleLR (max_lr=0.01)")
    print(f"   Loss: CrossEntropyLoss")
      # Start training
    print("\n🎯 Starting training...")
    train_losses, test_losses, train_acc, test_acc, plot_file = training_colab(
        train_loader, testloader, model, model_type, 
        criterion, optimizer, scheduler, device, num_epochs
    )
    
    # Final evaluation
    print("\n📊 Final evaluation:")
    final_accuracy, cm_file, results_file = evaluate_model_colab(model, testloader, device)
    
    # Git integration - show what files were created
    print(f"\n📁 Files created in repository:")
    print(f"   🤖 Model: {model_type}_model_cifar10.pth")
    print(f"   📈 Training curves: {plot_file}")
    print(f"   📊 Confusion matrix: {cm_file}")
    print(f"   📄 Results: {results_file}")
    print(f"   ℹ️  Training info: {model_type}_training_info.txt")
    
    # Provide Git commands for saving to repository
    print(f"\n🔧 To save results to Git repository, run:")
    print(f"   !git add *.pth *.png *.txt")
    print(f"   !git commit -m 'Training results: {final_accuracy:.2f}% accuracy'")
    print(f"   !git push")
    
    print(f"\n🏆 Training Summary:")
    print(f"   Final Validation Accuracy: {final_accuracy:.2f}%")
    print(f"   Best Validation Accuracy: {max(test_acc):.2f}%")
    print(f"   All files saved to repository!")
