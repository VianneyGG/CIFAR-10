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
# -*- coding: utf-8 -*-

from tqdm import tqdm

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
    """Save the trained model to disk."""
    torch.save(model.state_dict(), f'{model_type}_model_cifar10.pth')
    print(f"Model saved to {model_type}_model_cifar10.pth")

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
    """Train the CNN model on CIFAR-10 dataset."""
    
    # Lists to store training statistics
    train_losses = []
    test_losses = []
    test_acc = []
    train_acc = []

    for epoch in tqdm(range(num_epochs), desc='Training Epochs', unit='epoch', colour='blue'):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        model.update_dropout_rates(epoch, num_epochs)  # Update dropout rate if using improved model
        
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
        
            # Update progress bar
            if i % 10 == 0:
                batch_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * correct / total:.2f}%'
                })
        
        # Update the progress bar description
        batch_pbar.set_description(f'Epoch {epoch+1}/{num_epochs} - Loss: {running_loss / len(trainloader):.4f} - Acc: {100 * correct / total:.2f}%')
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
            for images, labels in tqdm(testloader, desc='Validation Batches', leave=False, colour='red'):
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
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
            f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, '
            f'Test Loss: {test_epoch_loss:.4f}, Test Acc: {test_epoch_acc:.2f}%')
        

    print('Finished Training')
    
    # Save the trained model
    save_model(model, model_type)
    
def evaluate_model(model, testloader):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Collect all labels and predictions for confusion matrix
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
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
    
    print("🚀 Utilisation du modèle amélioré avec Cosine Scheduled Dropout")
    model = CNN_improved(num_classes=10)
      # Configuration du dropout au début seulement (optionnel)
    print(f"Dropout initial configuré: {model.layer1[0].dropout1.p:.3f}")
    print("Dropout sera ajusté dynamiquement pendant l'entraînement")
    
    model = model.to(device)
    model_type = 'improved'
    
    #Number of epochs
    num_epochs = 30
    
    # Load CIFAR-10 dataset
    train_loader, testloader = load_cifar10(batch_size, num_workers=4)
      # Define loss function and optimizer
    criterion = CombinedLoss(alpha=1.0, gamma=2.0, smoothing=0.1, focal_weight=0.6, num_classes=10)
    print("🎯 Utilisation de CombinedLoss (Focal + Label Smoothing)")
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
    
    # Train the model
    training(train_loader, testloader, model, model_type, criterion, optimizer, scheduler, device, num_epochs)



