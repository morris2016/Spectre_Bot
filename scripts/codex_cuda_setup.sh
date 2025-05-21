#!/usr/bin/env bash
# Setup script for Codex agent on Ubuntu 24.04 with CUDA support
set -e

# Install common build tools and utilities
apt-get update
apt-get install -y --no-install-recommends \
    wget curl git ca-certificates build-essential

# Install NVIDIA CUDA repository for Ubuntu 24.04
CUDA_KEYRING=/tmp/cuda-keyring.deb
wget -O $CUDA_KEYRING https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i $CUDA_KEYRING
apt-get update
apt-get install -y cuda-toolkit-12-4

# Install Miniforge to manage Python environments
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
INSTALL_DIR="$HOME/miniforge"
if [ ! -d "$INSTALL_DIR" ]; then
  wget -O /tmp/miniforge.sh "$MINIFORGE_URL"
  bash /tmp/miniforge.sh -b -p "$INSTALL_DIR"
fi
export PATH="$INSTALL_DIR/bin:$PATH"
source "$INSTALL_DIR/etc/profile.d/conda.sh"

# Create and activate a Conda environment
conda create -y -n spectre-cuda python=3.11
conda activate spectre-cuda

# Upgrade pip and install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install UI dependencies
cd ui && npm install && cd ..

echo "Setup complete. Activate the environment with 'conda activate spectre-cuda'"
