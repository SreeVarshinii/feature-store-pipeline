#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "=========================================================="
echo "🚀 Initializing Feature Store Pipeline Setup & Execution"
echo "=========================================================="

# Check Python version
python3 --version

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip and install requirements
echo "Installing/updating dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Run the pipeline end-to-end
echo "Executing Feature Store & Data Drift Pipeline Orchestrator..."
python3 src/orchestrator.py

echo "=========================================================="
echo "✅ Pipeline Execution Complete!"
echo "=========================================================="
echo "To launch the Streamlit monitoring dashboard, run:"
echo "  source venv/bin/activate && streamlit run src/dashboard.py"
echo "=========================================================="
