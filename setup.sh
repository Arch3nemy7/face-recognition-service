#!/bin/bash

# Shelia Face Recognition Service - Setup Script
# This script sets up the Python virtual environment and installs dependencies

set -e  # Exit on error

echo "=========================================="
echo "Shelia Face Recognition Service Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Check if Python 3.11+ is available
if ! python3 -c 'import sys; assert sys.version_info >= (3, 11)' 2>/dev/null; then
    echo "Error: Python 3.11 or higher is required"
    echo "Current version: $python_version"
    exit 1
fi

echo ""
echo "Creating virtual environment..."
python3 -m venv .venv

echo ""
echo "Activating virtual environment..."
source .venv/bin/activate

echo ""
echo "Upgrading pip..."
pip install --upgrade pip

echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the virtual environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To start the service, run:"
echo "  uvicorn shelia_face_recognition_service.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "Or simply:"
echo "  python -m shelia_face_recognition_service.main"
echo ""
echo "API documentation will be available at:"
echo "  http://localhost:8000/docs"
echo ""
