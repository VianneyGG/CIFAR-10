# CIFAR-10 CNN Training in Google Colab

## Quick Start Guide

### 1. Open Google Colab
Go to [Google Colab](https://colab.research.google.com/) and create a new notebook.

### 2. Clone the Repository
```python
# Clone the repository
!git clone https://github.com/VianneyGG/CIFAR-10.git
%cd CIFAR-10
```

### 3. Install Dependencies
```python
# Install required packages
!pip install -r requirements.txt
```

### 4. Run Training (Option A: Full Script)
```python
# Execute the complete training script
%run cifar10_colab.py
```

### 5. Run Training (Option B: Single Cell)
Copy and paste the entire content of `colab_git_training.py` into a single Colab cell and run it.

## Expected Performance

### CPU Training (Original):
- **Time**: 4+ hours for 10 epochs
- **Speed**: ~15-20 seconds per epoch

### GPU Training (Colab):
- **Time**: 15-30 minutes for 10 epochs  
- **Speed**: ~1-2 seconds per epoch
- **Speedup**: 10-20x faster

## Automatic Features

✅ **GPU Detection**: Automatically uses GPU if available  
✅ **Progress Tracking**: Enhanced progress bars with ETA  
✅ **Model Saving**: Timestamped model checkpoints  
✅ **Visualization**: Training curves and confusion matrix  
✅ **Git Integration**: Auto-commit results to repository  

## Output Files

After training, the following files will be generated:
- `cifar10_model_YYYYMMDD_HHMMSS.pth` - Trained model
- `training_curves_YYYYMMDD_HHMMSS.png` - Loss/accuracy plots
- `confusion_matrix_YYYYMMDD_HHMMSS.png` - Classification results
- `training_results_YYYYMMDD_HHMMSS.txt` - Detailed metrics

## Git Workflow

The script automatically:
1. Configures Git with Colab user info
2. Stages all result files
3. Commits with descriptive message
4. Shows Git status and files

### Manual Git Push (if needed)
```python
# Configure Git credentials (replace with your info)
!git config --global user.email "your.email@example.com"
!git config --global user.name "Your Name"

# Push results back to repository
!git push origin main
```

## Troubleshooting

### Issue: CUDA Out of Memory
```python
# Reduce batch size
batch_size = 256  # Instead of 512
```

### Issue: Git Push Fails
```python
# Check Git status
!git status

# Check remote
!git remote -v

# Force push if needed (use carefully)
!git push --force origin main
```

### Issue: Package Installation Fails
```python
# Update pip first
!pip install --upgrade pip
!pip install -r requirements.txt
```

## Performance Monitoring

The script provides real-time monitoring:
- **Training Progress**: Live loss/accuracy updates
- **GPU Utilization**: Memory usage tracking
- **Time Estimates**: ETA for completion
- **Validation Metrics**: Per-epoch evaluation

## Model Architecture

- **CNN**: 3 convolutional layers + 2 fully connected
- **Features**: Batch normalization, dropout, ReLU activation
- **Optimizer**: Adam with learning rate 0.001
- **Criterion**: CrossEntropyLoss
- **Classes**: 10 CIFAR-10 categories

## Next Steps

1. Run the training in Colab
2. Download trained models
3. Experiment with hyperparameters
4. Try transfer learning
5. Deploy the model

---

**Repository**: https://github.com/VianneyGG/CIFAR-10  
**Documentation**: Complete setup guides in repository
