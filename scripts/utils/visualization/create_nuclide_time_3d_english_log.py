#!/usr/bin/env python3
"""
Nuclide Concentration over Time - 3D Visualization (English + Log Scale)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def create_nuclide_time_3d_surface_log(csv_file, output_file=None):
    """Create 3D surface plot of nuclide concentrations over time with log scale"""
    
    # Read the nuclide ratios data
    df = pd.read_csv(csv_file)
    
    # Extract time and nuclide data
    time = df['time(s)'].values
    timesteps = df['timestep'].values
    
    # Get nuclide columns (ratio_Q_0 to ratio_Q_59)
    nuclide_cols = [col for col in df.columns if col.startswith('ratio_Q_')]
    nuclide_data = df[nuclide_cols].values
    
    # Apply log scale (add small value to avoid log(0))
    nuclide_data_log = np.log10(nuclide_data + 1e-10)
    
    # Create nuclide indices
    nuclide_indices = np.arange(len(nuclide_cols))
    
    # Create meshgrid for 3D surface
    T, N = np.meshgrid(time, nuclide_indices)
    Z = nuclide_data_log.T  # Transpose so nuclides are rows, time is columns
    
    # Create 3D figure
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create surface plot
    surf = ax.plot_surface(T, N, Z, cmap='plasma', alpha=0.8, 
                          linewidth=0, antialiased=True, shade=True)
    
    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.8, aspect=20, pad=0.1)
    cbar.set_label('Log10(Concentration Ratio)', rotation=270, labelpad=25, fontsize=12)
    
    # Labels and title
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Nuclide Index', fontsize=12)
    ax.set_zlabel('Log10(Concentration Ratio)', fontsize=12)
    ax.set_title('Nuclide Concentration Evolution Over Time - 3D Surface (Log Scale)', 
                 fontsize=14, pad=20)
    
    # Set better viewing angle
    ax.view_init(elev=30, azim=45)
    
    # Add statistics
    max_conc = np.max(nuclide_data)
    min_conc = np.min(nuclide_data[nuclide_data > 0])  # Exclude zeros
    mean_conc = np.mean(nuclide_data)
    
    stats_text = f"""
    Simulation Statistics:
    Max Concentration: {max_conc:.6f}
    Min Concentration: {min_conc:.6f}
    Mean Concentration: {mean_conc:.6f}
    Total Nuclides: {len(nuclide_cols)}
    Time Range: {time.min():.0f}-{time.max():.0f} sec
    Log Scale Range: {np.min(Z):.2f} to {np.max(Z):.2f}
    """
    ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, 
              verticalalignment='top', 
              bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8),
              fontsize=10)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"3D log-scale surface plot saved: {output_file}")
    
    return fig, ax

def create_nuclide_time_3d_wireframe_log(csv_file, output_file=None):
    """Create 3D wireframe plot with log scale"""
    
    df = pd.read_csv(csv_file)
    time = df['time(s)'].values
    
    # Get nuclide columns
    nuclide_cols = [col for col in df.columns if col.startswith('ratio_Q_')]
    nuclide_data = df[nuclide_cols].values
    
    # Apply log scale
    nuclide_data_log = np.log10(nuclide_data + 1e-10)
    nuclide_indices = np.arange(len(nuclide_cols))
    
    # Create meshgrid
    T, N = np.meshgrid(time, nuclide_indices)
    Z = nuclide_data_log.T
    
    # Create 3D wireframe
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create wireframe
    wire = ax.plot_wireframe(T, N, Z, cmap='viridis', alpha=0.7, linewidth=0.8)
    
    # Add some scatter points for key nuclides
    key_nuclides = [0, 10, 20, 30, 40, 50]  # Sample key nuclides
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    for i, (nuc_idx, color) in enumerate(zip(key_nuclides, colors)):
        if nuc_idx < len(nuclide_cols):
            ax.plot(time, [nuc_idx]*len(time), nuclide_data_log[:, nuc_idx], 
                   color=color, linewidth=2, alpha=0.9, 
                   label=f'Nuclide {nuc_idx}')
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Nuclide Index', fontsize=12)
    ax.set_zlabel('Log10(Concentration Ratio)', fontsize=12)
    ax.set_title('Nuclide Concentration Wireframe - Log Scale', 
                 fontsize=14, pad=20)
    
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
    ax.view_init(elev=25, azim=60)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"3D log-scale wireframe saved: {output_file}")
    
    return fig, ax

def create_nuclide_time_3d_scatter_log(csv_file, output_file=None):
    """Create 3D scatter plot with log scale"""
    
    df = pd.read_csv(csv_file)
    time = df['time(s)'].values
    
    # Get nuclide columns
    nuclide_cols = [col for col in df.columns if col.startswith('ratio_Q_')]
    nuclide_data = df[nuclide_cols].values
    
    # Apply log scale
    nuclide_data_log = np.log10(nuclide_data + 1e-10)
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create scatter plot with varying sizes and colors
    for t_idx, t in enumerate(time):
        concentrations = nuclide_data[t_idx, :]
        concentrations_log = nuclide_data_log[t_idx, :]
        nuclide_indices = np.arange(len(concentrations))
        
        # Size proportional to original concentration
        sizes = 20 + 200 * (concentrations / np.max(nuclide_data))
        
        scatter = ax.scatter(
            [t] * len(concentrations),  # Time coordinate
            nuclide_indices,            # Nuclide index
            concentrations_log,         # Log concentration (height)
            c=concentrations,           # Color by original concentration
            s=sizes,                    # Size by original concentration
            cmap='plasma',
            alpha=0.6
        )
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('Original Concentration Ratio', rotation=270, labelpad=20, fontsize=12)
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Nuclide Index', fontsize=12)
    ax.set_zlabel('Log10(Concentration Ratio)', fontsize=12)
    ax.set_title('Nuclide Concentration Scatter Plot - Log Scale', 
                 fontsize=14, pad=20)
    
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"3D log-scale scatter plot saved: {output_file}")
    
    return fig, ax

def create_top_nuclides_3d_evolution_log(csv_file, output_file=None, top_n=10):
    """Create 3D plot focusing on top N nuclides with log scale"""
    
    df = pd.read_csv(csv_file)
    time = df['time(s)'].values
    
    # Get nuclide columns
    nuclide_cols = [col for col in df.columns if col.startswith('ratio_Q_')]
    nuclide_data = df[nuclide_cols].values
    
    # Apply log scale
    nuclide_data_log = np.log10(nuclide_data + 1e-10)
    
    # Find top N nuclides by maximum concentration
    max_concentrations = np.max(nuclide_data, axis=0)
    top_indices = np.argsort(max_concentrations)[-top_n:]
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color map for different nuclides
    colors = plt.cm.tab10(np.linspace(0, 1, top_n))
    
    # Plot each top nuclide as a line in 3D space
    for i, nuc_idx in enumerate(top_indices):
        concentrations_log = nuclide_data_log[:, nuc_idx]
        concentrations_orig = nuclide_data[:, nuc_idx]
        
        ax.plot(time, [nuc_idx]*len(time), concentrations_log, 
               color=colors[i], linewidth=3, alpha=0.8,
               label=f'Nuclide {nuc_idx} (max: {max_concentrations[nuc_idx]:.4f})')
        
        # Add scatter points at key time intervals
        key_times = np.arange(0, len(time), max(1, len(time)//10))
        ax.scatter(time[key_times], [nuc_idx]*len(key_times), 
                  concentrations_log[key_times], 
                  color=colors[i], s=50, alpha=0.9)
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Nuclide Index', fontsize=12)
    ax.set_zlabel('Log10(Concentration Ratio)', fontsize=12)
    ax.set_title(f'Top {top_n} Nuclides Concentration Evolution - Log Scale', 
                 fontsize=14, pad=20)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Top nuclides 3D log-scale analysis saved: {output_file}")
    
    return fig, ax

def create_linear_vs_log_comparison(csv_file, output_file=None):
    """Create side-by-side comparison of linear vs log scale"""
    
    df = pd.read_csv(csv_file)
    time = df['time(s)'].values
    
    # Get nuclide columns
    nuclide_cols = [col for col in df.columns if col.startswith('ratio_Q_')]
    nuclide_data = df[nuclide_cols].values
    nuclide_data_log = np.log10(nuclide_data + 1e-10)
    nuclide_indices = np.arange(len(nuclide_cols))
    
    # Create meshgrid
    T, N = np.meshgrid(time, nuclide_indices)
    Z_linear = nuclide_data.T
    Z_log = nuclide_data_log.T
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 10))
    
    # Linear scale subplot
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(T, N, Z_linear, cmap='plasma', alpha=0.8, 
                            linewidth=0, antialiased=True)
    ax1.set_xlabel('Time (seconds)', fontsize=11)
    ax1.set_ylabel('Nuclide Index', fontsize=11)
    ax1.set_zlabel('Concentration Ratio', fontsize=11)
    ax1.set_title('Linear Scale', fontsize=13, pad=15)
    ax1.view_init(elev=30, azim=45)
    
    # Log scale subplot
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(T, N, Z_log, cmap='plasma', alpha=0.8, 
                            linewidth=0, antialiased=True)
    ax2.set_xlabel('Time (seconds)', fontsize=11)
    ax2.set_ylabel('Nuclide Index', fontsize=11)
    ax2.set_zlabel('Log10(Concentration Ratio)', fontsize=11)
    ax2.set_title('Log Scale', fontsize=13, pad=15)
    ax2.view_init(elev=30, azim=45)
    
    # Add colorbars
    fig.colorbar(surf1, ax=ax1, shrink=0.6, aspect=15)
    fig.colorbar(surf2, ax=ax2, shrink=0.6, aspect=15)
    
    plt.suptitle('Nuclide Concentration Evolution - Linear vs Log Scale Comparison', 
                 fontsize=16, y=0.95)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Linear vs log comparison saved: {output_file}")
    
    return fig, (ax1, ax2)

def main():
    """Main function for English log-scale 3D visualizations"""
    
    print("=== Nuclide Concentration over Time - 3D Visualization (English + Log Scale) ===\n")
    
    # Create output directory
    os.makedirs("cram_result/nuclide_time_3d_log", exist_ok=True)
    
    # Use the nuclide ratios data
    data_file = "all_particles_nuclide_ratios.csv"
    
    if not os.path.exists(data_file):
        print(f"File not found: {data_file}")
        print("Nuclide concentration data required.")
        return
    
    print(f"Using data file: {data_file}")
    
    # 1. Surface plot with log scale
    print("1. Creating 3D log-scale surface plot...")
    create_nuclide_time_3d_surface_log(data_file, "cram_result/nuclide_time_3d_log/nuclide_surface_log.png")
    
    # 2. Wireframe plot with log scale
    print("2. Creating 3D log-scale wireframe...")
    create_nuclide_time_3d_wireframe_log(data_file, "cram_result/nuclide_time_3d_log/nuclide_wireframe_log.png")
    
    # 3. Scatter plot with log scale
    print("3. Creating 3D log-scale scatter plot...")
    create_nuclide_time_3d_scatter_log(data_file, "cram_result/nuclide_time_3d_log/nuclide_scatter_log.png")
    
    # 4. Top nuclides focus with log scale
    print("4. Creating top nuclides log-scale analysis...")
    create_top_nuclides_3d_evolution_log(data_file, "cram_result/nuclide_time_3d_log/top_nuclides_log.png", top_n=15)
    
    # 5. Linear vs log comparison
    print("5. Creating linear vs log scale comparison...")
    create_linear_vs_log_comparison(data_file, "cram_result/nuclide_time_3d_log/linear_vs_log_comparison.png")
    
    print("\n=== 3D Log-Scale Nuclide Visualization Complete! ===")
    print("Generated files:")
    print("- cram_result/nuclide_time_3d_log/nuclide_surface_log.png")
    print("- cram_result/nuclide_time_3d_log/nuclide_wireframe_log.png") 
    print("- cram_result/nuclide_time_3d_log/nuclide_scatter_log.png")
    print("- cram_result/nuclide_time_3d_log/top_nuclides_log.png")
    print("- cram_result/nuclide_time_3d_log/linear_vs_log_comparison.png")

if __name__ == "__main__":
    main()