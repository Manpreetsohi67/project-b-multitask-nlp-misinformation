# Detecting and Mitigating Online Misinformation via Multi-Task NLP

## Project Overview
This project implements a multi-task NLP system for detecting and mitigating online misinformation through joint sentiment, stance, and veracity detection with explainability (IG/LIME) and event-held-out evaluation.

## Repository Structure
```
├── data/
│   ├── raw/           # Raw data files (not tracked by Git)
│   └── processed/     # Processed data files (not tracked by Git)
├── notebooks/         # Jupyter notebooks for exploration
├── src/              # Source code
│   ├── data/         # Data loading and preprocessing
│   ├── models/       # Model definitions
│   ├── eval/         # Evaluation metrics
│   ├── explain/      # Explanation methods (IG/LIME)
│   └── utils/        # Utility functions
├── tests/            # Unit and integration tests
├── docs/             # Documentation
├── outputs/          # Model outputs and results (not tracked)
├── scripts/          # Utility scripts
└── requirements.txt  # Python dependencies
```

## Setup Instructions

### Prerequisites
- Python 3.10 or higher
- Git
- Windows 10/11 (PowerShell), macOS, or Linux

### Environment Setup

#### Windows (PowerShell)
```powershell
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Run setup script (creates virtual environment and installs dependencies)
.\scripts\setup_env.ps1
```

#### macOS/Linux (Bash)
```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Run setup script (creates virtual environment and installs dependencies)
chmod +x ./scripts/setup_env.sh
./scripts/setup_env.sh
```

### Manual Setup (Alternative)
If the automated scripts don't work, you can manually set up the environment:

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

2. Activate the virtual environment:
   - Windows: `.venv\Scripts\Activate.ps1` (PowerShell) or `.venv\Scripts\activate.bat` (CMD)
   - macOS/Linux: `source .venv/bin/activate`

3. Upgrade pip:
   ```bash
   python -m pip install --upgrade pip
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running Tests and Smoke Tests

### Individual Components

#### Run Smoke Test (End-to-End Pipeline)
```bash
# Windows
python scripts/run_smoke_test.py

# macOS/Linux
python scripts/run_smoke_test.py
```

#### Run Specific Tests
```bash
# Test event splits
python -m pytest tests/test_splits.py -v

# Test model shapes
python -m pytest tests/test_model_shapes.py -v
```

### Complete Pipeline Run

#### Windows (PowerShell)
```powershell
.\scripts\run_all.ps1
```

#### macOS/Linux (Bash)
```bash
./scripts/run_all.sh
```

## Week 1 Evidence Screenshots Checklist

Take screenshots of the following to document Week 1 progress:

1. **Git Setup Evidence**
   - Screenshot of cloned repository in your file system
   - Screenshot of `git log` showing initial commit

2. **Environment Verification Output**
   - Run `.\scripts\setup_env.ps1` (Windows) or `./scripts/setup_env.sh` (macOS/Linux)
   - Capture the output showing successful installation and verification

3. **Toy Dataset Loading Output**
   - Run `python -c "from src.data.load_toy import load_toy_data; load_toy_data()"`
   - Capture the output showing dataset loading and class distributions

4. **Event-Held-Out Split Output**
   - Run `python -c "from src.data.split_event_heldout import split_by_events, print_split_summary; from src.data.load_toy import load_toy_data; df = load_toy_data(); train_df, test_df = split_by_events(df); print_split_summary(train_df, test_df)"`
   - Capture the output showing event separation

5. **Model Initialization Output**
   - Run `python -c "from src.models.multitask_model import MultiTaskModel, print_model_summary; model = MultiTaskModel(); print_model_summary(model)"`
   - Capture the output showing model architecture

6. **Integrated Gradients Output**
   - Run `python -c "from src.explain.ig_demo import run_ig_demo; from src.models.multitask_model import MultiTaskModel; from transformers import AutoTokenizer; model = MultiTaskModel(); tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased'); run_ig_demo(model, tokenizer, 'This is a sample text for explanation.')"`
   - Capture the output showing IG explanations

7. **Complete Smoke Test Output**
   - Run `python scripts/run_smoke_test.py`
   - Capture the complete output showing all components working together

## Key Features

### Multi-Task Learning
- Joint sentiment analysis (positive, neutral, negative)
- Stance detection (support, neutral, oppose) 
- Veracity assessment (true, false, uncertain)

### Explainability
- Integrated Gradients (IG) for feature importance
- Event-held-out evaluation to prevent data leakage

### Data Processing
- Label harmonization across different sources
- Event-based train/test splits to ensure generalizability

## Project Status
- [x] Environment setup automation
- [x] Data loading and preprocessing modules
- [x] Multi-task model architecture
- [x] Evaluation metrics implementation
- [x] Explanation methods (IG)
- [x] Event-held-out splitting
- [x] Smoke testing pipeline
- [x] Unit tests
- [ ] Full training pipeline
- [ ] Advanced evaluation
- [ ] LIME integration

## Dependencies
See `requirements.txt` for a complete list of pinned dependencies.

## Contributing
1. Create a feature branch: `git checkout -b feature/amazing-feature`
2. Make your changes
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a pull request

## License
This project is licensed under the MIT License - see the LICENSE file for details.