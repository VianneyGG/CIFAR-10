# CIFAR-10 Google Colab Setup Instructions (Git Repository)

## Quick Start Guide

### 1. Create a new Google Colab notebook
- Go to https://colab.research.google.com/
- Create a new notebook
- Enable GPU: Runtime → Change runtime type → Hardware accelerator → GPU

### 2. Clone your Git repository
```python
# Clone your repository
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
%cd YOUR_REPO_NAME
```

### 3. Setup authentication (if private repo)
```python
# For private repositories, you'll need to authenticate
# Option 1: Using personal access token
!git config --global user.name "Your Name"
!git config --global user.email "your.email@example.com"

# Option 2: Mount Google Drive (if you have the repo there)
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/path/to/your/repo
```

### 4. Install dependencies and run training:
```python
# Install dependencies
!pip install -r requirements.txt

# Run the training
exec(open('cifar10_colab.py').read())
```

### 5. Save results back to repository:
```python
# After training, commit and push the trained model
!git add *.pth
!git add *.png  # Training plots if any
!git commit -m "Add trained CIFAR-10 model from Colab"
!git push
```

## Alternative: Copy-paste approach

If you prefer, you can copy the content of `cifar10_colab.py` directly into Colab cells:

### Cell 1: Setup and imports
```python
!pip install torch torchvision matplotlib seaborn scikit-learn tqdm

import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

### Cell 3-N: Copy the rest from cifar10_colab.py

## Complete Git Workflow

Here's a complete workflow for using your Git repository with Colab:

```python
# 1. Clone repository
!git clone https://github.com/YOUR_USERNAME/CIFAR-10.git
%cd CIFAR-10

# 2. Check current status
!git status
!git log --oneline -5  # Show last 5 commits

# 3. Install and run
!pip install -r requirements.txt
exec(open('cifar10_colab.py').read())

# 4. Check what was created
!ls -la *.pth *.png *.txt  # List all generated files
!git status   # Check git status

# 5. Commit and push results
!git add *.pth *.png *.txt
!git commit -m "Training results from Google Colab GPU - $(date)"
!git push

# 6. Verify upload
!git log --oneline -1  # Show latest commit
```

## Advanced Git Integration

### Automatic commit after training

```python
# Add this to your Colab notebook after training
import subprocess
import datetime

def commit_training_results(accuracy, model_type="improved"):
    """Automatically commit training results to Git"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Add all training artifacts
    subprocess.run(['git', 'add', '*.pth', '*.png', '*.txt'], cwd='.')
    
    # Create detailed commit message
    commit_msg = f"🚀 CIFAR-10 training results - {accuracy:.2f}% accuracy ({timestamp})"
    
    # Commit and push
    subprocess.run(['git', 'commit', '-m', commit_msg], cwd='.')
    result = subprocess.run(['git', 'push'], cwd='.')
    
    if result.returncode == 0:
        print(f"✅ Results successfully pushed to repository!")
        print(f"📝 Commit message: {commit_msg}")
    else:
        print("❌ Failed to push to repository")

# Use after training completes
# commit_training_results(final_accuracy)
```

### Managing large model files

```python
# Setup Git LFS for large model files (>100MB)
!git lfs track "*.pth"
!git add .gitattributes
!git commit -m "Setup Git LFS for model files"
!git push
```

## Key Optimizations for Colab:

1. **GPU Detection**: Automatic GPU/CPU detection and optimization
2. **Batch Size**: Optimized for Colab GPU memory (512 vs 256)
3. **Workers**: Reduced to 2 workers (Colab limitation)
4. **Pin Memory**: Enabled for faster GPU transfers
5. **Auto Download**: Models automatically download after training
6. **Progress Bars**: Enhanced monitoring with tqdm
7. **Checkpoints**: Automatic checkpoints every 10 epochs
8. **Visualization**: Training curves and confusion matrix
9. **Memory Management**: Optimized for Colab's memory constraints

## Expected Performance:
- **With GPU**: ~30-60 seconds per epoch
- **Total time**: ~15-30 minutes for 30 epochs
- **Speedup**: 10-20x faster than your current CPU setup

## Tips:
- Use smaller batch sizes if you run out of GPU memory
- Enable GPU in Runtime settings for best performance
- Save checkpoints frequently in case of disconnection
- Download your trained model immediately after training
