#!/bin/bash
# run_all.sh
# Runner script for bash to execute all setup, tests, and smoke test

echo "Running complete pipeline for Capstone Part B project..."

# Verify environment
echo -e "\n=== VERIFYING ENVIRONMENT ==="
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found!"
    exit 1
fi

python_version=$(python3 --version)
echo "Python: $python_version"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Virtual environment (.venv) not found. Please run setup first."
    exit 1
fi

# Run tests
echo -e "\n=== RUNNING TESTS ==="

# Test event splits
echo "Running event split tests..."
python -m pytest tests/test_splits.py -v
SPLIT_TEST_RESULT=$?

if [ $SPLIT_TEST_RESULT -eq 0 ]; then
    echo "✓ Event split tests passed!"
else
    echo "✗ Event split tests failed!"
    exit 1
fi

# Test model shapes
echo -e "\nRunning model shape tests..."
python -m pytest tests/test_model_shapes.py -v
SHAPE_TEST_RESULT=$?

if [ $SHAPE_TEST_RESULT -eq 0 ]; then
    echo "✓ Model shape tests passed!"
else
    echo "✗ Model shape tests failed!"
    exit 1
fi

# Run smoke test
echo -e "\n=== RUNNING SMOKE TEST ==="
echo "Running end-to-end smoke test pipeline..."
python scripts/run_smoke_test.py
SMOKE_TEST_RESULT=$?

if [ $SMOKE_TEST_RESULT -eq 0 ]; then
    echo "✓ Smoke test passed!"
else
    echo "✗ Smoke test failed!"
    exit 1
fi

echo -e "\n=== ALL TESTS COMPLETED SUCCESSFULLY ==="
echo "Pipeline execution completed without errors."
echo "Ready for Week 1 evidence submission!"