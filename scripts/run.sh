#!/bin/bash

# LDM-CRAM4 Execution Script
# Sets up the correct directory and runs the simulation

echo "Starting LDM-CRAM4 Lagrangian Dispersion Model..."
echo "=============================================="

# Check build
if [ ! -f "ldm" ]; then
    echo "Executable not found. Starting build..."
    make
fi

# Create output directories
mkdir -p output validation logs

# Run simulation
echo "Running simulation..."
./ldm > logs/simulation_output.log 2>&1

echo "Simulation completed!"

# Check results
if [ -d "output_1" ]; then
    echo "Output files generated: $(ls output_1/*.vtk | wc -l) VTK files"
fi

if [ -f "validation/nuclide_totals.csv" ]; then
    echo "Validation data generated: validation/nuclide_totals.csv"
fi

echo ""
echo "Generating visualization graphs..."
echo "================================"

# Run integrated visualization
if [ -f "scripts/visualization/visualize_cram4_60_nuclides.py" ]; then
    echo "Creating comprehensive 60-nuclide visualizations..."
    python3 scripts/visualization/visualize_cram4_60_nuclides.py
elif [ -f "scripts/visualization/visualize_all_60_nuclides.py" ]; then
    echo "Creating comprehensive 60-nuclide visualizations..."
    python3 scripts/visualization/visualize_all_60_nuclides.py
elif [ -f "scripts/visualization/create_nuclide_time_3d_english_log.py" ]; then
    echo "Creating 3D nuclide visualization..."
    python3 scripts/visualization/create_nuclide_time_3d_english_log.py
fi

# Check generated graph files
echo ""
echo "Generated visualization files:"
if [ -f "cram_result/ldm_heatmap.png" ]; then
    echo "✓ Heatmap: cram_result/ldm_heatmap.png"
fi
if [ -f "cram_result/ldm_stacked.png" ]; then
    echo "✓ Stacked plot: cram_result/ldm_stacked.png"
fi
if [ -f "cram_result/simulation_summary.csv" ]; then
    echo "✓ Simulation summary: cram_result/simulation_summary.csv"
fi

echo ""
echo "All tasks completed!"