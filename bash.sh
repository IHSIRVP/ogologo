# Exit immediately if a command exits with a non-zero status.
set -e
PYTHON_VERSION=3.8
VENV_NAME=venv_dummy_dl
echo "Creating virtual environment..."
python${PYTHON_VERSION} -m venv $VENV_NAME
source $VENV_NAME/bin/activate
pip install --upgrade pip
pip install pytorch=1.13.1
pip install torchvision 
pip install torchaudio
pip install numpy pandas matplotlib seaborn scikit-learn
pip install scipy pillow h5py
echo "Dependencies installed successfully."
