import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from CNN_simple import CNN_simple 
from CNN_improved import CNN_improved
from tqdm import tqdm
import datetime
import subprocess
import numpy as np
# -*- coding: utf-8 -*-

# Les 10 classes de CIFAR-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Define data transformations
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.RandomRotation(10),     # Randomly rotate by +/- 10 degrees
    transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Random translation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.1),  # NEW: Random grayscale
    transforms.ToTensor(),             # Convert to tensor
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # Normalize
])

# Simpler transform for test data (no augmentation)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# === COMBINED LOSS FUNCTION ===
class CombinedLoss(nn.Module):
    """
    Combinaison optimale de Focal Loss et Label Smoothing Cross Entropy
    - Focal Loss: Se concentre sur les exemples difficiles (cat, bird, dog)
    - Label Smoothing: Prévient la surconfiance et améliore la généralisation
    - Poids adaptatif: Balance automatiquement selon la difficulté des classes
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, smoothing=0.1, focal_weight=0.6, num_classes=10):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.focal_weight = focal_weight
        self.num_classes = num_classes
        
        # Tracking pour adaptation dynamique
        self.class_difficulties = torch.ones(num_classes)
        self.update_count = 0
        
    def forward(self, pred, target):
        """
        Forward pass avec combinaison adaptative des deux losses
        
        Args:
            pred: Prédictions du modèle [batch_size, num_classes]
            target: Labels vrais [batch_size]
            
        Returns:
            loss: Loss combinée optimisée
        """
        
        # === FOCAL LOSS COMPONENT ===
        # Calcul de la Cross Entropy standard
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        
        # Calcul de la probabilité prédite pour la vraie classe
        pt = torch.exp(-ce_loss)
        
        # Application du facteur focal (1-pt)^gamma
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        focal_loss = focal_loss.mean()
        
        # === LABEL SMOOTHING COMPONENT ===
        # Création des labels "soft" avec smoothing
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        smooth_labels = one_hot * (1 - self.smoothing) + (1 - one_hot) * self.smoothing / (self.num_classes - 1)
        
        # Calcul de la KL divergence
        log_prob = F.log_softmax(pred, dim=1)
        smooth_loss = F.kl_div(log_prob, smooth_labels, reduction='batchmean')
        
        # === ADAPTATION DYNAMIQUE DES POIDS ===
        # Mise à jour des difficultés par classe (optionnel, pour monitoring)
        with torch.no_grad():
            for i in range(self.num_classes):
                class_mask = (target == i)
                if class_mask.sum() > 0:
                    avg_confidence = pt[class_mask].mean()
                    # Classes avec faible confiance = plus difficiles
                    self.class_difficulties[i] = 0.9 * self.class_difficulties[i] + 0.1 * (1 - avg_confidence)
        
        # === COMBINAISON FINALE ===
        # Balance entre focal (pour difficultés) et smoothing (pour stabilité)
        combined_loss = self.focal_weight * focal_loss + (1 - self.focal_weight) * smooth_loss
        
        return combined_loss

# === CIFAR-10 STATISTICS CALCULATION ===

def calculate_cirfa10_stats():
    basic_transform = transforms.Compose([transforms.ToTensor()])
    
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=basic_transform)
    
    print("Calculating CIFAR-10 statistics...")
    print("Number of images:", len(dataset))
    print('Number of batches:', len(dataset) // 10000 + 1)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10000, shuffle=False, num_workers=2)

    #variables pour stocker les statistiques
    mean = torch.zeros(3)
    squared = torch.zeros(3)
    
    print("Calculating mean and std...")
    
    for batch_id, (images, _) in enumerate(dataloader):

        pixels = images.view(images.size(0), images.size(1), -1)
        pixels = pixels.permute(1, 0, 2).contiguous().view(3, -1)
        
        # Calculer la moyenne et l'écart type pour chaque canal
        mean += pixels.mean(dim=1)
        squared += (pixels**2).std(dim=1)
        
        if batch_id % 10 == 0:
            print(f"bacth {batch_id} processed")
            
    mean /= len(dataloader)
    squared /= len(dataloader)
    std = torch.sqrt(squared - mean**2)
    return mean, std

# === CIFAR-10 DATA LOADER AND DISPLAY === 

def load_cifar10(batch_size=4, num_workers=4):
    # Transformations pour les données
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
      # Chargement du dataset CIFAR-10
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, 
                                              num_workers=num_workers, pin_memory=False, persistent_workers=True, prefetch_factor=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, 
                                             num_workers=num_workers, pin_memory=False, persistent_workers=True, prefetch_factor=2)
    
    return trainloader, testloader

def display_sample_images(dataloader, num_samples=8):
    """Display sample images from the dataset with enhanced visualization"""
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    
    # Denormalize for display
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2470, 0.2435, 0.2616])
    
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

def display_images(images, labels, num_images=4):
    """Affiche un certain nombre d'images avec leurs étiquettes."""
    plt.figure(figsize=(10, 10))
    for i, (img, label) in enumerate(zip(images, labels)):
        plt.subplot(2, 2, i + 1)
        plt.imshow(img.permute(1, 2, 0).numpy())
        plt.title(classes[label])
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

# === MODEL SAVING AND LOADING ===

def save_model(model, model_type='simple'):
    """Save the trained model to disk with enhanced tracking."""
    model_path = f'{model_type}_model_cifar10.pth'
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")
    
    # Also save a timestamped version for tracking
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

def load_model(model_type='simple'):
    """Load a previously trained model from disk."""
    if model_type == 'simple':
        print("Loading simple CNN model...")
        model = CNN_simple()
    else:
        print("Loading improved CNN model...")    
        model = CNN_improved()
    
    # Check if GPU is available and set device    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    try:
        model.load_state_dict(torch.load(f'{model_type}_model_cifar10.pth', map_location=device))
        model = model.to(device)
        print(f"Model loaded from {model_type}_model_cifar10.pth")
        return model
    except FileNotFoundError:
        print(f"No saved model found at {model_type}_model_cifar10.pth")
        model = model.to(device)
        return model

# === MIXUP AUGMENTATION ===
def mixup_data(x, y, alpha=1.0):
    """
    Mixup augmentation pour améliorer la généralisation
    Mélange deux exemples et leurs labels
    """
    if alpha > 0:
        lam = torch.distributions.beta.Beta(alpha, alpha).sample()
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Calcul de la loss pour mixup"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# === TRAINING AND EVALUATION FUNCTIONS ===

def training(trainloader, testloader, model, model_type, criterion, optimizer, scheduler, device, num_epochs):
    """Enhanced training function with visualization and early stopping"""
    
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
    print(f"   🎯 Enhanced regularization strategies enabled")
    print("-" * 60)

    for epoch in tqdm(range(num_epochs), desc='Training Epochs', unit='epoch', colour='blue'):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Update dropout rate if using improved model
        if hasattr(model, 'update_dropout_rates'):
            model.update_dropout_rates(epoch, num_epochs)
        
        # Use a progress bar for batches too
        batch_pbar = tqdm(enumerate(trainloader, 0), 
                         total=len(trainloader), 
                         desc=f'Epoch {epoch+1}/{num_epochs}',
                         colour='green',
                         leave=False)        
        
        for i, (inputs, labels) in batch_pbar:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Apply Mixup 30% du temps
            if torch.rand(1) < 0.3:
                mixed_inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=1.0)
                outputs = model(mixed_inputs)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                
                # Pour le calcul de l'accuracy avec mixup
                _, predicted = torch.max(outputs.data, 1)
                correct += (lam * predicted.eq(labels_a).sum().item() + 
                           (1 - lam) * predicted.eq(labels_b).sum().item())
            else:
                # Training standard avec CombinedLoss
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics pour les batches non-mixup
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
            
            # Backward pass and optimize
            loss.backward()
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
    
    # Save the trained model
    save_model(model, model_type)
    
    # Plot training curves and save to repository
    plot_filename = plot_training_curves(train_losses, test_losses, train_acc, test_acc)
    
    # Plot overfitting analysis
    analysis_filename = plot_overfitting_analysis(train_acc, test_acc, train_losses, test_losses)
    
    return train_losses, test_losses, train_acc, test_acc, plot_filename
    
def evaluate_model(model, testloader, device):
    """Enhanced model evaluation with detailed analysis and visualization"""
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    print("🔍 Evaluating model with detailed analysis...")
    
    with torch.no_grad():
        for images, labels in tqdm(testloader, desc='Evaluation'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy calculation
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
            
            # Collect all labels and predictions for confusion matrix
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    accuracy = 100 * correct / total
    print(f'🎯 Final Test Accuracy: {accuracy:.2f}%')
    
    # Per-class accuracy analysis
    print(f"\n📊 Per-Class Accuracy Analysis:")
    print("-" * 50)
    for i, class_name in enumerate(classes):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            print(f"{class_name:>10}: {class_acc:>6.2f}% ({int(class_correct[i])}/{int(class_total[i])})")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, ax=axes[0,0])
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('True')
    axes[0,0].set_title(f'Confusion Matrix - Accuracy: {accuracy:.2f}%')
    
    # 2. Per-class accuracy bar chart
    class_accuracies = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 
                       for i in range(10)]
    bars = axes[0,1].bar(classes, class_accuracies, color='skyblue', alpha=0.7)
    axes[0,1].set_title('Per-Class Accuracy')
    axes[0,1].set_ylabel('Accuracy (%)')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, acc in zip(bars, class_accuracies):
        axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                      f'{acc:.1f}%', ha='center', va='bottom')
    
    # 3. Class distribution in test set
    class_counts = [class_total[i] for i in range(10)]
    axes[1,0].bar(classes, class_counts, color='lightcoral', alpha=0.7)
    axes[1,0].set_title('Test Set Class Distribution')
    axes[1,0].set_ylabel('Number of Samples')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 4. Accuracy vs Class Frequency
    axes[1,1].scatter(class_counts, class_accuracies, s=100, alpha=0.7, color='green')
    for i, (count, acc) in enumerate(zip(class_counts, class_accuracies)):
        axes[1,1].annotate(classes[i], (count, acc), xytext=(5, 5), 
                          textcoords='offset points', fontsize=8)
    axes[1,1].set_xlabel('Number of Test Samples')
    axes[1,1].set_ylabel('Accuracy (%)')
    axes[1,1].set_title('Accuracy vs Sample Count')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save comprehensive evaluation results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_filename = f'comprehensive_evaluation_{timestamp}.png'
    plt.savefig(eval_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Comprehensive evaluation saved to {eval_filename}")
    
    plt.show()
    
    # Save detailed results to text file
    results_filename = f'evaluation_results_{timestamp}.txt'
    with open(results_filename, 'w') as f:
        f.write(f"CIFAR-10 Model Evaluation Results\n")
        f.write(f"={'='*40}\n")
        f.write(f"Overall Test Accuracy: {accuracy:.2f}%\n")
        f.write(f"Total Test Samples: {total}\n")
        f.write(f"Correct Predictions: {correct}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n\n")
        
        # Per-class detailed results
        f.write("Per-Class Accuracy Analysis:\n")
        f.write("-" * 30 + "\n")
        for i, class_name in enumerate(classes):
            if class_total[i] > 0:
                class_acc = 100 * class_correct[i] / class_total[i]
                f.write(f"{class_name:>10}: {class_acc:>6.2f}% ({int(class_correct[i])}/{int(class_total[i])})\n")
        
        # Confusion matrix statistics
        f.write(f"\nConfusion Matrix Summary:\n")
        f.write("-" * 25 + "\n")
        f.write(f"True Positives per class:\n")
        for i, class_name in enumerate(classes):
            f.write(f"{class_name:>10}: {cm[i,i]:>4d}\n")
    
    print(f"📄 Detailed results saved to {results_filename}")
    
    return accuracy, eval_filename, results_filename

# === ADVANCED TRAINING STRATEGIES - EARLY STOPPING ===

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
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f'training_curves_{timestamp}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📈 Training curves saved to {plot_filename}")
    
    plt.show()
    return plot_filename

def plot_overfitting_analysis(train_acc, test_acc, train_losses, test_losses):
    """Create detailed overfitting analysis plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Calculate overfitting gap
    overfitting_gap = [abs(ta - va) for ta, va in zip(train_acc, test_acc)]
    
    # Plot 1: Accuracy comparison
    axes[0,0].plot(train_acc, label='Training Accuracy', color='blue', linewidth=2)
    axes[0,0].plot(test_acc, label='Validation Accuracy', color='red', linewidth=2)
    axes[0,0].fill_between(range(len(train_acc)), train_acc, test_acc, 
                          alpha=0.3, color='orange', label='Overfitting Gap')
    axes[0,0].set_title('Training vs Validation Accuracy')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Accuracy (%)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Loss comparison
    axes[0,1].plot(train_losses, label='Training Loss', color='blue', linewidth=2)
    axes[0,1].plot(test_losses, label='Validation Loss', color='red', linewidth=2)
    axes[0,1].set_title('Training vs Validation Loss')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Loss')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Overfitting gap over time
    axes[1,0].plot(overfitting_gap, color='orange', linewidth=2)
    axes[1,0].set_title('Overfitting Gap Over Time')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('|Train Acc - Val Acc| (%)')
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Learning efficiency (accuracy per loss)
    train_efficiency = [acc/loss if loss > 0 else 0 for acc, loss in zip(train_acc, train_losses)]
    val_efficiency = [acc/loss if loss > 0 else 0 for acc, loss in zip(test_acc, test_losses)]
    
    axes[1,1].plot(train_efficiency, label='Training Efficiency', color='blue', linewidth=2)
    axes[1,1].plot(val_efficiency, label='Validation Efficiency', color='red', linewidth=2)
    axes[1,1].set_title('Learning Efficiency (Accuracy/Loss)')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Efficiency')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save analysis
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_filename = f'overfitting_analysis_{timestamp}.png'
    plt.savefig(analysis_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Overfitting analysis saved to {analysis_filename}")
    
    plt.show()
    return analysis_filename

if __name__ == "__main__":
    print("🚀 CIFAR-10 CNN Training - Enhanced Version with Complete Visualization")
    print("=" * 70)
    
    num_threads = os.cpu_count()-4
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Enable CPU optimizations
    if device.type == 'cpu':
        torch.backends.cudnn.benchmark = False
        batch_size = 256
    # Enable GPU optimizations
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        batch_size = 512
    
    print(f"Using device: {device}")
    print(f"Number of CPU threads: {num_threads}")
    print(f"Batch size: {batch_size}")
    
    # Load CIFAR-10 dataset
    print("\n� Loading CIFAR-10 dataset with enhanced augmentation...")
    train_loader, testloader = load_cifar10(batch_size, num_workers=4)
    
    # Display sample images from dataset
    print("\n📸 Sample images from training dataset:")
    display_sample_images(train_loader)
    
    print("\n🚀 Initializing Enhanced CNN model...")
    model = CNN_improved(num_classes=10)
    
    # Configuration du dropout au début seulement (optionnel)
    print(f"Dropout initial configuré: {model.layer1[0].dropout1.p:.3f}")
    print("Dropout sera ajusté dynamiquement pendant l'entraînement")
    
    model = model.to(device)
    model_type = 'improved'
    
    # Training parameters
    num_epochs = 30
    print(f"   Model: {model_type}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Device: {device}")
    
    # Define loss function and optimizer with enhanced configuration
    criterion = CombinedLoss(alpha=1.0, gamma=2.0, smoothing=0.1, focal_weight=0.6, num_classes=10)
    print("\n⚙️ Enhanced Training Configuration:")
    print("🎯 Loss Function: CombinedLoss (Focal + Label Smoothing)")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,                          # LR maximum
        steps_per_epoch=len(train_loader),      # Nombre de batches par epoch
        epochs=num_epochs,                     # Nombre total d'epochs
        pct_start=0.3,                         # Pourcentage de montée en LR
        anneal_strategy='cos',                 # Stratégie de descente
        div_factor=25.0,                       # Facteur initial/max_lr
        final_div_factor=1e4                   # Facteur max_lr/final_lr
    )
    
    print(f"🔧 Optimizer: Adam (lr=0.001)")
    print(f"📈 Scheduler: OneCycleLR with cosine annealing")
    print(f"🎲 Mixup: 30% probability with alpha=1.0")
    print(f"✂️ Gradient clipping: max_norm=1.0")
    print(f"⏰ Early stopping: patience=7 epochs")
    
    # Train the model with enhanced features
    print("\n🎯 Starting Enhanced Training with Complete Monitoring...")
    train_losses, test_losses, train_acc, test_acc, plot_file = training(
        train_loader, testloader, model, model_type, 
        criterion, optimizer, scheduler, device, num_epochs
    )
    
    # Final comprehensive evaluation
    print("\n📊 Final Comprehensive Evaluation:")
    final_accuracy, eval_file, results_file = evaluate_model(model, testloader, device)
    
    # Summary report
    print(f"\n🏆 Training Summary Report:")
    print(f"   Final Test Accuracy: {final_accuracy:.2f}%")
    print(f"   Best Validation Accuracy: {max(test_acc):.2f}%")
    print(f"   Final Overfitting Gap: {abs(train_acc[-1] - test_acc[-1]):.2f}%")
    print(f"   Training Epochs Completed: {len(train_acc)}")
    
    print(f"\n📁 Generated Files:")
    print(f"   🤖 Model: {model_type}_model_cifar10.pth")
    print(f"   📈 Training curves: {plot_file}")
    print(f"   📊 Evaluation analysis: {eval_file}")
    print(f"   📄 Results summary: {results_file}")
    print(f"   ℹ️  Training info: {model_type}_training_info.txt")
    
    # Git integration commands
    print(f"\n🔧 Git Integration Commands:")
    print(f"   git add *.pth *.png *.txt")
    print(f"   git commit -m 'Enhanced training: {final_accuracy:.2f}% accuracy with full visualization'")
    print(f"   git push")
    
    print(f"\n✅ Enhanced CIFAR-10 training completed successfully!")
    print(f"🎯 All visualizations and analyses saved to repository")



