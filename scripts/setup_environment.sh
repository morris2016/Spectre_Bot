#!/usr/bin/env bash
# Environment setup script
# Installs required system packages, sets up a RAPIDS-enabled conda env,
# installs Python dependencies, downloads NLTK data, and installs UI packages.
set -e

# Install base system packages
apt-get update
apt-get install -y wget curl git ca-certificates bzip2 --no-install-recommends

# Install Miniforge for Python 3.11
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
INSTALL_DIR="$HOME/miniforge"
if [ ! -d "$INSTALL_DIR" ]; then
  wget -O /tmp/miniforge.sh "$MINIFORGE_URL"
  bash /tmp/miniforge.sh -b -p "$INSTALL_DIR"
fi

export PATH="$INSTALL_DIR/bin:$PATH"
source "$INSTALL_DIR/etc/profile.d/conda.sh"

# Create RAPIDS environment if it doesn't exist
if ! conda info --envs | grep -q '^rapids-25\.04'; then
  conda create -y -n rapids-25.04 -c rapidsai -c conda-forge -c nvidia \
    rapids=25.04 python=3.11 'cuda-version>=12.0,<=12.8' \
    'pytorch=*=*cuda*' tensorflow dash networkx nx-cugraph=25.04 \
    jupyterlab graphistry xarray-spatial
fi

conda activate rapids-25.04

# Upgrade pip and install Python requirements excluding RAPIDS packages
pip install --upgrade pip
sed '/^cudf/d;/^cuml/d;/^dash/d;/^networkx/d;/^tensorflow/d;/^torch/d' \
  requirements.txt > /tmp/requirements_no_rapids.txt
pip install -r /tmp/requirements_no_rapids.txt

# Download required NLTK data
python scripts/download_nltk_data.py

# Install UI dependencies
cd ui && npm install && cd ..

echo "Environment setup complete. Activate with 'conda activate rapids-25.04'"
