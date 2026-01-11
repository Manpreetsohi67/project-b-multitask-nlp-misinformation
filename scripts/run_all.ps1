# run_all.ps1
# Runner script for Windows to execute all setup, tests, and smoke test

Write-Host "Running complete pipeline for Capstone Part B project..." -ForegroundColor Green

# Verify environment
Write-Host "`n=== VERIFYING ENVIRONMENT ===" -ForegroundColor Yellow
try {
    $python_version = python --version
    Write-Host "Python: $python_version" -ForegroundColor Cyan
} catch {
    Write-Host "Python not found!" -ForegroundColor Red
    exit 1
}

# Activate virtual environment if it exists
if (Test-Path ".venv") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & "$PWD\.venv\Scripts\Activate.ps1"
} else {
    Write-Host "Virtual environment (.venv) not found. Please run setup first." -ForegroundColor Red
    exit 1
}

# Run tests
Write-Host "`n=== RUNNING TESTS ===" -ForegroundColor Yellow

# Test event splits
Write-Host "Running event split tests..." -ForegroundColor Cyan
python -m pytest tests/test_splits.py -v

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Event split tests passed!" -ForegroundColor Green
} else {
    Write-Host "✗ Event split tests failed!" -ForegroundColor Red
    exit 1
}

# Test model shapes
Write-Host "`nRunning model shape tests..." -ForegroundColor Cyan
python -m pytest tests/test_model_shapes.py -v

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Model shape tests passed!" -ForegroundColor Green
} else {
    Write-Host "✗ Model shape tests failed!" -ForegroundColor Red
    exit 1
}

# Run smoke test
Write-Host "`n=== RUNNING SMOKE TEST ===" -ForegroundColor Yellow
Write-Host "Running end-to-end smoke test pipeline..." -ForegroundColor Cyan
python scripts/run_smoke_test.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Smoke test passed!" -ForegroundColor Green
} else {
    Write-Host "✗ Smoke test failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n=== ALL TESTS COMPLETED SUCCESSFULLY ===" -ForegroundColor Green
Write-Host "Pipeline execution completed without errors." -ForegroundColor Green
Write-Host "Ready for Week 1 evidence submission!" -ForegroundColor Green