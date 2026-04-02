#!/bin/bash
# GNN-ProtoNet Setup Script
set -e

echo "============================================"
echo "  GNN-ProtoNet Setup: EEG Parkinson's Detection"
echo "============================================"

# Install dependencies
echo ""
echo "[1/3] Installing Python dependencies..."
pip install -r requirements.txt

# Create directories
echo ""
echo "[2/3] Creating directories..."
mkdir -p data/raw/UC data/raw/UNM data/raw/Iowa data/processed results

# Download datasets
echo ""
echo "[3/3] Downloading EEG datasets from OpenNeuro..."
cd src && python download_data.py --dataset all
cd ..

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Quick test (synthetic data):"
echo "    cd src && python main.py --n_subjects 6 --k_shot 5"
echo ""
echo "  Real data:"
echo "    cd src && python main.py --real --k_shot 5"
echo ""
echo "  Cross-dataset evaluation:"
echo "    cd src && python main.py --real --cross-dataset --k_shot 5"
echo ""
echo "  Full ablation:"
echo "    cd src && python main.py --real --ablation --k_shot 5"
echo "============================================"
