import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class CNN_simple(nn.Module):
    
    def __init__(self):
        super (CNN_simple, self).__init__()

        # Convolutional layer 1
        # Input: 3 channels (RGB), Output: 32 channels, Kernel size: 3x3, Padding: 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization after conv1
        
        # Convolutional layer 2
        # Input: 32 channels, Output: 64 channels, Kernel size: 3x3, Padding: 1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Convolutional layer 3
        # Input: 64 channels, Output: 128 channels, Kernel size: 3x3, Padding: 1
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # 128 channels, 4x4 feature map after pooling
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 10)  # 10 classes for CIFAR-10
    
    def forward(self, x):
        
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.fc2(x)
        
        return x



        
        
        
    