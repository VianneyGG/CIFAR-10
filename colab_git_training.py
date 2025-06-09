# CIFAR-10 Colab with Git Integration - Single Cell Script
# Copy this entire script into a single Colab cell and run

# 1. Setup and Git clone
print("🚀 CIFAR-10 Training with Git Integration")
print("=" * 50)

# Clone repository (modify URL as needed)
import os

# ⚠️ IMPORTANT: Replace with your actual GitHub repository URL
REPO_URL = "https://github.com/YOUR_USERNAME/CIFAR-10.git"

if not os.path.exists('CIFAR-10'):
    print("📥 Cloning repository...")
    print(f"🔗 Repository URL: {REPO_URL}")
    print("⚠️  Make sure to replace YOUR_USERNAME with your actual GitHub username!")
    !git clone {REPO_URL}
    %cd CIFAR-10
else:
    print("📁 Repository already exists, pulling latest changes...")
    %cd CIFAR-10
    !git pull

# 2. Install dependencies
print("\n📦 Installing dependencies...")
!pip install -q torch torchvision matplotlib seaborn scikit-learn tqdm

# 3. Import libraries and setup
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import datetime
import subprocess

# Import your models
from CNN_simple import CNN_simple 
from CNN_improved import CNN_improved

# Check GPU and setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️  Using device: {device}")

if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    torch.backends.cudnn.benchmark = True
    batch_size = 512
else:
    batch_size = 256

print(f"   Batch size: {batch_size}")

# 4. Define classes and functions
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def auto_commit_results(accuracy, files_created):
    """Automatically commit training results to Git repository"""
    try:
        # Configure git if needed
        subprocess.run(['git', 'config', '--global', 'user.email', 'colab@google.com'], cwd='.')
        subprocess.run(['git', 'config', '--global', 'user.name', 'Google Colab'], cwd='.')
        
        # Add files
        for file_pattern in ['*.pth', '*.png', '*.txt']:
            subprocess.run(['git', 'add', file_pattern], cwd='.')
        
        # Create commit message
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        commit_msg = f"🚀 CIFAR-10 training - {accuracy:.2f}% accuracy ({timestamp})"
        
        # Commit
        result = subprocess.run(['git', 'commit', '-m', commit_msg], 
                              cwd='.', capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"\n✅ Results committed to Git!")
            print(f"📝 Commit: {commit_msg}")
            print(f"📁 Files: {', '.join(files_created)}")
            
            # Show push command (user needs to authenticate)
            print(f"\n🔧 To push to remote repository, run:")
            print(f"   !git push")
            
            return True
        else:
            print(f"\n⚠️  Commit failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"\n❌ Git operation failed: {e}")
        return False

# 5. Run the actual training
print(f"\n🎯 Starting CIFAR-10 training...")

# Execute the main training script
exec(open('cifar10_colab.py').read())

print(f"\n🎉 Training completed! Check your repository for saved models and results.")
