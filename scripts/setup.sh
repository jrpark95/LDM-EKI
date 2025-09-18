#!/bin/bash
# LDM-EKI Environment Setup Script

set -e

echo "Setting up LDM-EKI integrated environment..."

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"

# Create necessary directories
echo "Creating directory structure..."
mkdir -p data/output/{ldm_results,eki_results,visualization}
mkdir -p logs/{ldm_logs,eki_logs,integration_logs}

# Set executable permissions
echo "Setting permissions..."
chmod +x ldm/ldm 2>/dev/null || echo "LDM executable not found (will be created during build)"
chmod +x integration/workflows/coupled_simulation.py
chmod +x scripts/*.sh

# Check dependencies
echo "Checking dependencies..."

# Check CUDA
if command -v nvcc &> /dev/null; then
    echo "✓ CUDA found: $(nvcc --version | grep "release" | awk '{print $6}')"
else
    echo "⚠ CUDA not found - GPU acceleration may not be available"
fi

# Check Python
if command -v python3 &> /dev/null; then
    echo "✓ Python3 found: $(python3 --version)"
else
    echo "✗ Python3 not found"
    exit 1
fi

# Check conda environment
if [ -f "/opt/miniconda3/bin/python" ]; then
    echo "✓ Miniconda found"
else
    echo "⚠ Miniconda not found at expected location"
fi

# Check required Python packages
echo "Checking Python packages..."
REQUIRED_PACKAGES=("numpy" "matplotlib" "json")
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if /opt/miniconda3/bin/python -c "import $pkg" 2>/dev/null; then
        echo "✓ $pkg found"
    else
        echo "⚠ $pkg not found"
    fi
done

# Build LDM
echo "Building LDM..."
cd ldm
if [ -f "Makefile" ]; then
    make clean || true
    make
    echo "✓ LDM build completed"
else
    echo "⚠ LDM Makefile not found"
fi

cd "$ROOT_DIR"

# Test EKI imports
echo "Testing EKI imports..."
cd eki
if /opt/miniconda3/bin/python -c "
import sys
sys.path.append('src')
try:
    import Optimizer_EKI_np
    import Model_Connection_np_Ensemble
    print('✓ EKI imports successful')
except ImportError as e:
    print(f'⚠ EKI import error: {e}')
" 2>/dev/null; then
    echo "✓ EKI ready"
else
    echo "⚠ EKI imports failed"
fi

cd "$ROOT_DIR"

echo ""
echo "Setup completed!"
echo ""
echo "Quick start commands:"
echo "  make run-coupled    # Run integrated simulation"
echo "  make run-ldm        # Run LDM server only"
echo "  make run-eki        # Run EKI client only"
echo "  make help           # Show all available commands"