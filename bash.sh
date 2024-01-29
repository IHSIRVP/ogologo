#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the Python version and the virtual environment name.
PYTHON_VERSION=3.8
VENV_NAME=venv_dummy_dl

# Create a Python virtual environment.
echo "Creating virtual environment..."
python$PYTHON_VERSION -m venv $VENV_NAME

# Activate the virtual environment.
source $VENV_NAME/bin/activate

# Upgrade pip to the latest version.
echo "Upgrading pip..."
pip install --upgrade pip

# Install TensorFlow (you can specify the version as needed).
echo "Installing TensorFlow..."
pip install tensorflow

# Install PyTorch (select the version and CUDA version as needed, check the official PyTorch website for the correct command).
echo "Installing PyTorch..."
pip install torch torchvision torchaudio

# Install other common deep learning libraries.
echo "Installing other common libraries..."
pip install keras numpy pandas matplotlib scikit-learn

# Deactivate the virtual environment.
deactivate

echo "Dependencies installed successfully."
