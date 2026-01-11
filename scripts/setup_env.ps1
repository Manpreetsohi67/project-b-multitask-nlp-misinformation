# setup_env.ps1
# Script to set up the Python environment for the Capstone Part B project

Write-Host "Setting up environment for Capstone Part B project..." -ForegroundColor Green

# Check if Python is installed
try {
    $python_version = python --version 2>&1
    Write-Host "Python found: $python_version" -ForegroundColor Green
} catch {
    Write-Host "Python not found! Please install Python 3.10+." -ForegroundColor Red
    exit 1
}

# Check Python version
$py_version = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
if ($py_version -lt "3.10") {
    Write-Host "Python version is $py_version, but Python 3.10+ is required!" -ForegroundColor Red
    exit 1
} else {
    Write-Host "Python version $py_version is OK" -ForegroundColor Green
}

# Create virtual environment if it doesn't exist
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& "$PWD\.venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install requirements
Write-Host "Installing requirements from requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt

# Verify installations
Write-Host "`nVerifying installations..." -ForegroundColor Green
Write-Host "Python version: $(python --version)" -ForegroundColor Cyan
Write-Host "Pip version: $(pip --version)" -ForegroundColor Cyan

# Check key packages
try {
    python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available());"
} catch { Write-Host "PyTorch not available" -ForegroundColor Red }

try {
    python -c "import transformers; print('Transformers version:', transformers.__version__)"
} catch { Write-Host "Transformers not available" -ForegroundColor Red }

try {
    python -c "import datasets; print('Datasets version:', datasets.__version__)"
} catch { Write-Host "Datasets not available" -ForegroundColor Red }

try {
    python -c "import sklearn; print('Scikit-learn version:', sklearn.__version__)"
} catch { Write-Host "Scikit-learn not available" -ForegroundColor Red }

try {
    python -c "import captum; print('Captum version:', captum.__version__)"
} catch { Write-Host "Captum not available" -ForegroundColor Red }

try {
    python -c "import lime; print('Lime version:', lime.__version__)"
} catch { Write-Host "Lime not available" -ForegroundColor Red }

Write-Host "`nEnvironment setup complete!" -ForegroundColor Green
Write-Host "Virtual environment created and activated in .venv/" -ForegroundColor Green
Write-Host "All required packages installed." -ForegroundColor Green