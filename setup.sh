#!/usr/bin/env bash
# Setup script for Codex agent environment
# Installs Miniforge (Conda) with Python 3.11 and project dependencies
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

# Create conda environment if it doesn't exist
if ! conda info --envs | grep -q '^spectre'; then
  conda create -y -n spectre python=3.11
fi

conda activate spectre

# Upgrade pip and install Python requirements
pip install --upgrade pip
pip install -r requirements.txt
