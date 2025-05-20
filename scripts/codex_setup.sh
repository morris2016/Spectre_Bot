#!/usr/bin/env bash
# Setup script for Codex agent environment
# Installs Miniforge and creates a RAPIDS-enabled conda environment.
# GPU libraries are installed with conda, while remaining requirements are
# installed via pip.
set -e

# Install base system packages
apt-get update
apt-get install -y wget bzip2 curl git ca-certificates --no-install-recommends

# Install Miniforge for Python 3.11
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
INSTALL_DIR="$HOME/miniforge"

if [ ! -d "$INSTALL_DIR" ]; then
  wget -O /tmp/miniforge.sh "$MINIFORGE_URL"
  bash /tmp/miniforge.sh -b -p "$INSTALL_DIR"
fi

export PATH="$INSTALL_DIR/bin:$PATH"
source "$INSTALL_DIR/etc/profile.d/conda.sh"

# Create rapids-25.04 environment if it doesn't exist
if ! conda info --envs | grep -q '^rapids-25\.04'; then
  conda create -y -n rapids-25.04 -c rapidsai -c conda-forge -c nvidia \
    rapids=25.04 python=3.11 'cuda-version>=12.0,<=12.8' \
    'pytorch=*=*cuda*' tensorflow dash networkx nx-cugraph=25.04 \
    jupyterlab graphistry xarray-spatial
fi

conda activate rapids-25.04

# Upgrade pip
pip install --upgrade pip

# Install requirements excluding RAPIDS packages handled by conda
sed '/^cudf/d;/^cuml/d;/^dash/d;/^networkx/d;/^tensorflow/d;/^torch/d' \
  requirements.txt > /tmp/requirements_no_rapids.txt
pip install -r /tmp/requirements_no_rapids.txt

