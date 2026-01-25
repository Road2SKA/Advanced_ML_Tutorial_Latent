#!/bin/bash
# ============================================================
# Setup script for Clay Foundation Model environment
# Session 2B: Generate Your Own Clay Embeddings
# ============================================================

set -e  # Exit on error

echo "=============================================="
echo "Setting up Clay environment (r2ska-clay)"
echo "=============================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Anaconda or Miniconda first."
    echo "https://www.anaconda.com/docs/getting-started/miniconda/main"
    exit 1
fi

# Create the environment
echo ""
echo "Step 1: Creating conda environment from environment-clay.yml..."
echo "This may take several minutes..."
echo ""

conda env create -f environment-clay.yml --yes || {
    echo ""
    echo "Environment may already exist. Updating instead..."
    conda env update -f environment-clay.yml --prune
}

echo ""
echo "=============================================="
echo "Step 2: Downloading Clay v1.5 checkpoint..."
echo "=============================================="

# Activate the environment for downloading
eval "$(conda shell.bash hook)"
conda activate r2ska-clay

# Download checkpoint if not present
CHECKPOINT="clay-v1.5.ckpt"
CHECKPOINT_URL="https://huggingface.co/made-with-clay/Clay/resolve/main/v1.5/clay-v1.5.ckpt"

if [ -f "$CHECKPOINT" ]; then
    echo "Checkpoint already exists: $CHECKPOINT"
else
    echo "Downloading $CHECKPOINT (~1.2 GB)..."
    echo "URL: $CHECKPOINT_URL"

    # Try wget first, then curl
    if command -v wget &> /dev/null; then
        wget -O "$CHECKPOINT" "$CHECKPOINT_URL"
    elif command -v curl &> /dev/null; then
        curl -L -o "$CHECKPOINT" "$CHECKPOINT_URL"
    else
        echo "ERROR: Neither wget nor curl found. Please install one and download manually:"
        echo "  $CHECKPOINT_URL"
        exit 1
    fi
fi

# Verify checkpoint
if [ -f "$CHECKPOINT" ]; then
    SIZE=$(ls -lh "$CHECKPOINT" | awk '{print $5}')
    echo "Checkpoint downloaded: $CHECKPOINT ($SIZE)"
else
    echo "ERROR: Checkpoint download failed"
    exit 1
fi

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "To use the Clay environment:"
echo "  conda activate r2ska-clay"
echo ""
echo "To start Jupyter Lab:"
echo "  conda activate r2ska-clay"
echo "  jupyter lab"
echo ""
echo "Then open Session2B_Generate_Own_Embeddings.ipynb"
echo ""
