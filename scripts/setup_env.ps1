# PowerShell script for setting up development environment

# Function to check if a command exists
function Test-Command($cmdname) {
    return [bool](Get-Command -Name $cmdname -ErrorAction SilentlyContinue)
}

# Error handling
$ErrorActionPreference = "Stop"

Write-Host "🚀 Setting up Sector-H development environment..." -ForegroundColor Cyan

# Check if Conda is installed
if (-not (Test-Command "conda")) {
    Write-Host "❌ Conda is not installed. Please install Miniconda or Anaconda first." -ForegroundColor Red
    Write-Host "Download from: https://docs.conda.io/en/latest/miniconda.html" -ForegroundColor Yellow
    exit 1
}

# Create and activate conda environment
Write-Host "📦 Creating Conda environment..." -ForegroundColor Green
conda env create -f environment.yml
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to create Conda environment" -ForegroundColor Red
    exit 1
}

# Activate environment
Write-Host "🔄 Activating Conda environment..." -ForegroundColor Green
conda activate sector-h

# Install Python packages
Write-Host "📚 Installing additional Python packages..." -ForegroundColor Green
pip install -r api/requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install Python packages" -ForegroundColor Red
    exit 1
}

# Install pre-commit hooks
if (Test-Command "pre-commit") {
    Write-Host "🔧 Installing pre-commit hooks..." -ForegroundColor Green
    pre-commit install
}

# Download NLTK data
Write-Host "📥 Downloading NLTK data..." -ForegroundColor Green
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet')"

# Download spaCy models
Write-Host "📥 Downloading spaCy models..." -ForegroundColor Green
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md

# Setup Jupyter kernel
Write-Host "🔧 Setting up Jupyter kernel..." -ForegroundColor Green
python -m ipykernel install --user --name sector-h --display-name "Python (Sector-H)"

Write-Host "✅ Environment setup complete!" -ForegroundColor Green
Write-Host @"

🎉 Setup Complete! Next steps:
1. Create a copy of api/.env.example as api/.env and fill in your credentials
2. Run 'conda activate sector-h' to activate the environment
3. Start developing with 'docker-compose up'

For development:
- API runs on: http://localhost:8000
- Website runs on: http://localhost:3000
- Jupyter runs on: http://localhost:8888

"@ -ForegroundColor Cyan
