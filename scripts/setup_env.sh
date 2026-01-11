#!/bin/bash
# setup_env.sh
# Script to set up the Python environment for the Capstone Part B project

echo "Setting up environment for Capstone Part B project..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found! Please install Python 3.10+."
    exit 1
fi

# Check Python version
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [[ "$(printf '%s\n' "3.10" "$PY_VERSION" | sort -V | head -n1)" != "3.10" ]]; then
    echo "Python version is $PY_VERSION, but Python 3.10+ is required!"
    exit 1
else
    echo "Python version $PY_VERSION is OK"
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt

# Verify installations
echo -e "\nVerifying installations..."
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"

# Check key packages
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available());" 2>/dev/null || echo "PyTorch not available"
python -c "import transformers; print('Transformers version:', transformers.__version__)" 2>/dev/null || echo "Transformers not available"
python -c "import datasets; print('Datasets version:', datasets.__version__)" 2>/dev/null || echo "Datasets not available"
python -c "import sklearn; print('Scikit-learn version:', sklearn.__version__)" 2>/dev/null || echo "Scikit-learn not available"
python -c "import captum; print('Captum version:', captum.__version__)" 2>/dev/null || echo "Captum not available"
python -c "import lime; print('Lime version:', lime.__version__)" 2>/dev/null || echo "Lime not available"

echo -e "\nEnvironment setup complete!"
echo "Virtual environment created and activated in .venv/"
echo "All required packages installed."