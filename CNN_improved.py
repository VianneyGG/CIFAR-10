import torch
import torch.nn as nn
import torch.nn.functional as F
import math  # Nécessaire pour les calculs cosinus

class ResidualBlock(nn.Module):
    """
    Bloc résiduel de base pour ResNet
    Implémente : output = F(x) + x
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # Première couche convolutionnelle du bloc
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Deuxième couche convolutionnelle du bloc
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                              kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Connexion skip - si les dimensions changent, on adapte x
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        # Sauvegarder l'entrée pour la connexion skip
        identity = x
        
        # Passer par les couches du bloc
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Ajouter la connexion skip (adapte x si nécessaire)
        out += self.shortcut(identity)
        
        # Activation finale
        out = F.relu(out)
        
        return out

class CNN_improved(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_improved, self).__init__()
        
        # ═══════════════════════════════════════
        # BLOC INITIAL - Extraction des features de base
        # ═══════════════════════════════════════
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ═══════════════════════════════════════
        # BLOCS RÉSIDUELS - Apprentissage profond stable
        # ═══════════════════════════════════════
        
        # Niveau 1: 64 filtres, même taille (32x32)
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        
        # Niveau 2: 128 filtres, réduction taille (16x16)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        
        # Niveau 3: 256 filtres, réduction taille (8x8)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        
        # Niveau 4: 512 filtres, réduction taille (4x4)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)
        
        
        # ═══════════════════════════════════════
        # CLASSIFICATEUR FINAL avec COSINE SCHEDULED DROPOUT
        # ═══════════════════════════════════════
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        
        # Configuration du Cosine Scheduled Dropout
        self.dropout_config = {
            'start_rate': 0.6,      # Taux de départ (60%)
            'end_rate': 0.1,        # Taux final (10%)
            'current_rate': 0.6     # Taux actuel (initialisé au départ)
        }
        
        # Couche dropout principale (sera mise à jour dynamiquement)
        self.dropout = nn.Dropout(self.dropout_config['current_rate'])
        
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """
        Crée une séquence de blocs résiduels
        """
        layers = []
        
        # Premier bloc (peut changer les dimensions)
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        # Blocs suivants (mêmes dimensions)
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Bloc initial
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 64, 32, 32]
        
        # Passage par les blocs résiduels
        x = self.layer1(x)  # [B, 64, 32, 32]
        x = self.layer2(x)  # [B, 128, 16, 16]
        x = self.layer3(x)  # [B, 256, 8, 8]
        x = self.layer4(x)  # [B, 512, 4, 4]
        
        # Classification finale
        x = self.avgpool(x)  # [B, 512, 1, 1]
        x = torch.flatten(x, 1)  # [B, 512]
        x = self.dropout(x)
        x = self.fc(x)  # [B, 10]
        
        return x

    def update_dropout(self, epoch, total_epochs):
        """
        Met à jour le taux de dropout selon une fonction de type cosinus
        """
        # Calculer le taux de dropout actuel basé sur la progression de l'époque
        progress = epoch / total_epochs  # Valeur entre 0 et 1
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        
        # Interpoler le taux de dropout entre start_rate et end_rate
        new_rate = self.dropout_config['end_rate'] + \
                   (self.dropout_config['start_rate'] - self.dropout_config['end_rate']) * cosine_decay
        
        # Mettre à jour le taux de la couche dropout
        self.dropout.p = new_rate  # Met à jour le taux de la couche dropout
        
        # Sauvegarder le taux actuel
        self.dropout_config['current_rate'] = new_rate
    
    def get_current_dropout_rate(self):
        """Retourne le taux de dropout actuel"""
        return self.dropout_config['current_rate']
    
    def set_dropout_config(self, start_rate=0.6, end_rate=0.1):
        """
        Configure les paramètres du dropout scheduling
        
        Args:
            start_rate (float): Taux de dropout initial (0.0 à 1.0)
            end_rate (float): Taux de dropout final (0.0 à 1.0)
        """
        self.dropout_config['start_rate'] = start_rate
        self.dropout_config['end_rate'] = end_rate
        self.dropout_config['current_rate'] = start_rate
        self.dropout.p = start_rate

