# Quick GitHub Setup for CIFAR-10 Project

## Current Issue
Your local Git repository is trying to push to `https://github.com/VianneyGG/CIFAR-10.git` which doesn't exist.

## Quick Fix (Choose One Option)

### Option A: Create New Repository on GitHub (Recommended)

1. **Go to GitHub and create a new repository:**
   - Visit: https://github.com/new
   - Repository name: `CIFAR-10`
   - Description: `CIFAR-10 CNN Classification with PyTorch`
   - Choose Public or Private
   - **Don't** check "Initialize with README"
   - Click "Create repository"

2. **Connect your local repository:**
   ```powershell
   cd "c:\Users\gauth\Projets\ML\CIFAR-10"
   
   # Add your new repository (replace YOUR_USERNAME with your GitHub username)
   git remote add origin https://github.com/YOUR_USERNAME/CIFAR-10.git
   
   # Push your code
   git branch -M main
   git push -u origin main
   ```

3. **For Google Colab, use this code:**
   ```python
   # Replace YOUR_USERNAME with your actual GitHub username
   !git clone https://github.com/YOUR_USERNAME/CIFAR-10.git
   %cd CIFAR-10
   !pip install -r requirements.txt
   exec(open('cifar10_colab.py').read())
   ```

### Option B: Use Automated Setup Script

Run the PowerShell script I created:
```powershell
cd "c:\Users\gauth\Projets\ML\CIFAR-10"
.\setup_github.ps1
```

This script will:
- Ask for your GitHub username
- Set up the remote repository
- Push your code to GitHub
- Update all Colab files with correct URLs

## After Setup

Your repository URL will be:
`https://github.com/YOUR_USERNAME/CIFAR-10.git`

### Test the Setup
```powershell
# Check remote URL
git remote -v

# Test connection
git ls-remote origin

# Should show your repository without errors
```

### For Google Colab Training
Once your repository is set up, you can use Google Colab with these simple steps:

1. **Open Google Colab** (https://colab.research.google.com/)
2. **Enable GPU** (Runtime → Change runtime type → GPU)
3. **Run these commands:**
   ```python
   !git clone https://github.com/YOUR_USERNAME/CIFAR-10.git
   %cd CIFAR-10
   !pip install -r requirements.txt
   exec(open('cifar10_colab.py').read())
   ```

This will:
- Clone your repository
- Install all dependencies
- Train your CNN model on GPU (15-30 minutes vs 4+ hours on CPU)
- Save results back to your repository

## Troubleshooting

**If you get authentication errors:**
```powershell
git config --global user.name "Your Name"
git config --global user.email "your.email@github.com"
```

**If model files are too large (>100MB):**
```powershell
git lfs install
git lfs track "*.pth"
git add .gitattributes
git commit -m "Track model files with Git LFS"
git push
```

## Ready to Go!
Once you complete Option A or B above, your CIFAR-10 project will be ready for fast GPU training in Google Colab! 🚀
