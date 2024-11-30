#!/bin/bash

# Exit on error
set -e

# Colors for pretty output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Setting up Sector-H development environment...${NC}"

# Function to check system requirements
check_requirements() {
    echo -e "\n${BLUE}📋 Checking system requirements...${NC}"
    
    # Check Python version
    if ! command -v python &> /dev/null; then
        echo -e "${RED}❌ Python is not installed. Please install Python 3.8 or higher.${NC}"
        exit 1
    fi
    
    # Check Conda
    if ! command -v conda &> /dev/null; then
        echo -e "${RED}❌ Conda is not installed. Please install Miniconda or Anaconda first.${NC}"
        echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    
    # Check Git
    if ! command -v git &> /dev/null; then
        echo -e "${RED}❌ Git is not installed. Please install Git first.${NC}"
        echo "Download from: https://git-scm.com/downloads"
        exit 1
    fi
    
    echo -e "${GREEN}✅ All system requirements met!${NC}"
}

# Function to setup development environment
setup_env() {
    # Create and activate conda environment
    echo -e "\n${BLUE}📦 Creating Conda environment...${NC}"
    conda env create -f environment.yml
    
    # Activate environment
    echo -e "\n${BLUE}🔄 Activating Conda environment...${NC}"
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate sector-h
    
    # Install Python packages
    echo -e "\n${BLUE}📚 Installing additional Python packages...${NC}"
    pip install -r api/requirements.txt
    
    # Install development tools
    echo -e "\n${BLUE}🛠️ Installing development tools...${NC}"
    
    # Install pre-commit hooks if available
    if command -v pre-commit &> /dev/null; then
        echo "🔧 Installing pre-commit hooks..."
        pre-commit install
    fi
    
    # Setup Jupyter kernel
    echo -e "\n${BLUE}🔧 Setting up Jupyter kernel...${NC}"
    python -m ipykernel install --user --name sector-h --display-name "Python (Sector-H)"
}

# Function to download required ML models and data
setup_ml_dependencies() {
    echo -e "\n${BLUE}📥 Downloading ML dependencies...${NC}"
    
    # Download NLTK data
    echo "📥 Downloading NLTK data..."
    python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet')"
    
    # Download spaCy models
    echo "📥 Downloading spaCy models..."
    python -m spacy download en_core_web_sm
    python -m spacy download en_core_web_md
}

# Function to verify installation
verify_installation() {
    echo -e "\n${BLUE}🔍 Verifying installation...${NC}"
    
    # Check if conda environment is active
    if [[ "$CONDA_DEFAULT_ENV" != "sector-h" ]]; then
        echo -e "${RED}❌ Conda environment 'sector-h' is not active${NC}"
        return 1
    fi
    
    # Run basic imports to verify installation
    python -c "import torch; import transformers; import spacy; import nltk" &> /dev/null
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ All dependencies successfully installed!${NC}"
    else
        echo -e "${RED}❌ Some dependencies failed to install correctly${NC}"
        return 1
    fi
}

# Main installation process
main() {
    check_requirements
    setup_env
    setup_ml_dependencies
    verify_installation
    
    echo -e "\n${GREEN}🎉 Setup completed successfully!${NC}"
    echo -e "\n${BLUE}📘 Quick Start:${NC}"
    echo "1. Activate the environment: conda activate sector-h"
    echo "2. Start the API: python api/main.py"
    echo "3. Visit the docs at: http://localhost:8000/docs"
    echo -e "\n${BLUE}📚 Documentation:${NC} See README.md for detailed usage instructions"
    echo """
    🎉 Setup Complete! Next steps:
    1. Create a copy of api/.env.example as api/.env and fill in your credentials
    2. Run 'conda activate sector-h' to activate the environment
    3. Start developing with 'docker-compose up'

    For development:
    - API runs on: http://localhost:8000
    - Website runs on: http://localhost:3000
    - Jupyter runs on: http://localhost:8888
    """
}

# Run main function
main
