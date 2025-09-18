#!/usr/bin/env python3
"""
LDM 3D Heatmap Visualization
Creates 3D visualization of nuclear concentration data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import os

def create_3d_heatmap(csv_file, output_file=None, title_suffix=""):
    """Create 3D heatmap from concentration grid data"""
    
    # Read the data
    df = pd.read_csv(csv_file)
    
    # Extract timestep from filename
    timestep = os.path.basename(csv_file).split('_')[-1].split('.')[0]
    
    # Create 3D figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract coordinates and concentrations
    x = df['lon'].values
    y = df['lat'].values  
    z = df['alt'].values
    c = df['concentration'].values
    
    # Create scatter plot with color mapping
    scatter = ax.scatter(x, y, z, c=c, cmap='plasma', s=60, alpha=0.8)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('Concentration', rotation=270, labelpad=20, fontsize=12)
    
    # Set labels and title
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_zlabel('Altitude (m)', fontsize=12)
    
    title = f'LDM Nuclear Concentration Distribution - Timestep {timestep}'
    if title_suffix:
        title += f' {title_suffix}'
    ax.set_title(title, fontsize=14, pad=20)
    
    # Add statistics text
    stats_text = f"""
    Particles: {len(df)}
    Max Conc: {c.max():.4f}
    Mean Conc: {c.mean():.4f}
    Min Conc: {c.min():.4f}
    """
    ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, 
              verticalalignment='top', bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8),
              fontsize=10)
    
    # Improve layout
    plt.tight_layout()
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"3D heatmap saved to {output_file}")
    else:
        plt.show()
    
    return fig, ax

def create_temporal_3d_animation_frames(timesteps=None, output_dir="3d_frames"):
    """Create multiple 3D heatmap frames for different timesteps"""
    
    # Find available concentration grid files
    grid_files = sorted(glob.glob("validation/concentration_grid_*.csv"))
    
    if not grid_files:
        print("No concentration grid files found!")
        return
    
    # Filter by timesteps if specified
    if timesteps:
        filtered_files = []
        for ts in timesteps:
            pattern = f"validation/concentration_grid_{ts:05d}.csv"
            if os.path.exists(pattern):
                filtered_files.append(pattern)
        grid_files = filtered_files
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create frames
    created_files = []
    for i, csv_file in enumerate(grid_files):
        timestep = os.path.basename(csv_file).split('_')[-1].split('.')[0]
        output_file = f"{output_dir}/ldm_3d_heatmap_t{timestep}.png"
        
        try:
            fig, ax = create_3d_heatmap(csv_file, output_file, f"(Frame {i+1}/{len(grid_files)})")
            created_files.append(output_file)
            plt.close(fig)  # Free memory
            
            print(f"Created frame {i+1}/{len(grid_files)}: timestep {timestep}")
            
        except Exception as e:
            print(f"Error creating frame for {csv_file}: {e}")
    
    print(f"\nCreated {len(created_files)} 3D heatmap frames in {output_dir}/")
    return created_files

def create_multi_timestep_comparison():
    """Create a comparison of multiple timesteps in subplots"""
    
    # Select interesting timesteps
    timesteps = [100, 500, 1000, 2000, 3000, 4000]
    available_files = []
    
    for ts in timesteps:
        csv_file = f"validation/concentration_grid_{ts:05d}.csv"
        if os.path.exists(csv_file):
            available_files.append((ts, csv_file))
    
    if not available_files:
        print("No concentration grid files found for comparison!")
        return
    
    # Create subplot figure
    n_plots = len(available_files)
    cols = 3
    rows = (n_plots + cols - 1) // cols
    
    fig = plt.figure(figsize=(18, 6*rows))
    
    for i, (timestep, csv_file) in enumerate(available_files):
        df = pd.read_csv(csv_file)
        
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        
        # Extract coordinates and concentrations
        x = df['lon'].values
        y = df['lat'].values  
        z = df['alt'].values
        c = df['concentration'].values
        
        # Create scatter plot
        scatter = ax.scatter(x, y, z, c=c, cmap='plasma', s=40, alpha=0.8)
        
        # Labels and title
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Altitude (m)')
        ax.set_title(f'Timestep {timestep}\n({len(df)} particles, max: {c.max():.3f})')
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=15)
    
    plt.suptitle('LDM Nuclear Concentration Evolution - 3D Heatmap Comparison', fontsize=16, y=0.98)
    plt.tight_layout()
    
    output_file = "cram_result/ldm_3d_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Multi-timestep 3D comparison saved to {output_file}")
    
    return fig

def main():
    """Main function to create various 3D visualizations"""
    
    print("=== LDM 3D Heatmap Visualization ===\n")
    
    # Create output directory
    os.makedirs("cram_result", exist_ok=True)
    
    # 1. Create single 3D heatmap for a specific timestep
    print("1. Creating single 3D heatmap...")
    single_file = "validation/concentration_grid_01000.csv"
    if os.path.exists(single_file):
        create_3d_heatmap(single_file, "cram_result/ldm_3d_heatmap_single.png")
    else:
        print(f"File {single_file} not found!")
    
    # 2. Create multi-timestep comparison
    print("\n2. Creating multi-timestep comparison...")
    create_multi_timestep_comparison()
    
    # 3. Create animation frames for selected timesteps
    print("\n3. Creating animation frames...")
    selected_timesteps = [100, 200, 500, 1000, 1500, 2000, 3000, 4000]
    create_temporal_3d_animation_frames(selected_timesteps, "cram_result/3d_frames")
    
    print("\n=== 3D Visualization Complete! ===")
    print("Generated files:")
    print("- cram_result/ldm_3d_heatmap_single.png")
    print("- cram_result/ldm_3d_comparison.png") 
    print("- cram_result/3d_frames/ldm_3d_heatmap_t*.png")

if __name__ == "__main__":
    main()