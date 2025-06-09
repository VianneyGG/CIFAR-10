# CIFAR-10 Google Colab Setup Instructions

## Quick Start Guide

### 1. Create a new Google Colab notebook
- Go to https://colab.research.google.com/
- Create a new notebook
- Enable GPU: Runtime → Change runtime type → Hardware accelerator → GPU

### 2. Upload files to Colab
Upload these files to your Colab session:
- `requirements.txt`
- `cifar10_colab.py`
- `CNN_simple.py`
- `CNN_improved.py`

### 3. Run the setup cell first:
```python
# Install dependencies
!pip install -r requirements.txt

# Import and run the main script
exec(open('cifar10_colab.py').read())
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

### Cell 2: Upload CNN models
```python
from google.colab import files
uploaded = files.upload()  # Upload CNN_simple.py and CNN_improved.py
```

### Cell 3-N: Copy the rest from cifar10_colab.py

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
