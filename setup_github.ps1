# Setup Your CIFAR-10 Repository for GitHub and Colab
# Run this script in PowerShell after creating your GitHub repository

Write-Host "🚀 CIFAR-10 GitHub Repository Setup" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green

# Get user input for GitHub username
$username = Read-Host "VianneyGG"
$repoName = Read-Host "CIFAR-10"


$repoUrl = "https://github.com/$username/$repoName.git"

Write-Host ""
Write-Host "📋 Repository Details:" -ForegroundColor Yellow
Write-Host "   Username: $username"
Write-Host "   Repository: $repoName" 
Write-Host "   URL: $repoUrl"
Write-Host ""

# Confirm the repository exists
$confirm = Read-Host "Have you created the repository '$repoName' on GitHub? (y/n)"
if ($confirm -ne "y" -and $confirm -ne "Y") {
    Write-Host "❌ Please create the repository on GitHub first:" -ForegroundColor Red
    Write-Host "   1. Go to https://github.com/new"
    Write-Host "   2. Repository name: $repoName"
    Write-Host "   3. Don't initialize with README"
    Write-Host "   4. Click 'Create repository'"
    Write-Host "   5. Run this script again"
    exit
}

# Navigate to project directory
Set-Location "c:\Users\gauth\Projets\ML\CIFAR-10"

Write-Host "📁 Setting up Git repository..." -ForegroundColor Blue

# Add remote origin
git remote add origin $repoUrl

# Set main branch
git branch -M main

# Push to GitHub
Write-Host "📤 Pushing to GitHub..." -ForegroundColor Blue
git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Successfully pushed to GitHub!" -ForegroundColor Green
    
    # Update Colab files with correct repository URL
    Write-Host "🔧 Updating Colab configuration files..." -ForegroundColor Blue
    
    # Update colab_git_training.py
    $colabFile = "colab_git_training.py"
    if (Test-Path $colabFile) {
        $content = Get-Content $colabFile -Raw
        $content = $content -replace "https://github.com/YOUR_USERNAME/CIFAR-10.git", $repoUrl
        $content = $content -replace "YOUR_USERNAME", $username
        Set-Content $colabFile -Value $content
        Write-Host "   ✓ Updated $colabFile"
    }
    
    # Update COLAB_SETUP.md
    $setupFile = "COLAB_SETUP.md"
    if (Test-Path $setupFile) {
        $content = Get-Content $setupFile -Raw
        $content = $content -replace "YOUR_USERNAME", $username
        $content = $content -replace "YOUR_REPO_NAME", $repoName
        Set-Content $setupFile -Value $content
        Write-Host "   ✓ Updated $setupFile"
    }
    
    Write-Host ""
    Write-Host "🎉 Setup Complete!" -ForegroundColor Green
    Write-Host "Your repository is now ready for Google Colab!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📋 Next Steps:" -ForegroundColor Yellow
    Write-Host "1. Open Google Colab: https://colab.research.google.com/"
    Write-Host "2. Create a new notebook"
    Write-Host "3. Enable GPU: Runtime → Change runtime type → GPU"
    Write-Host "4. Run this code in Colab:"
    Write-Host ""
    Write-Host "!git clone $repoUrl" -ForegroundColor Cyan
    Write-Host "%cd $repoName" -ForegroundColor Cyan
    Write-Host "!pip install -r requirements.txt" -ForegroundColor Cyan  
    Write-Host "exec(open('cifar10_colab.py').read())" -ForegroundColor Cyan
    Write-Host ""
    
} else {
    Write-Host "❌ Failed to push to GitHub!" -ForegroundColor Red
    Write-Host "Please check:" -ForegroundColor Yellow
    Write-Host "1. Repository exists on GitHub"
    Write-Host "2. You have push access to the repository"
    Write-Host "3. Your Git credentials are configured"
    Write-Host ""
    Write-Host "To configure Git credentials:" -ForegroundColor Yellow
    Write-Host "git config --global user.name `"Your Name`""
    Write-Host "git config --global user.email `"your.email@example.com`""
}
