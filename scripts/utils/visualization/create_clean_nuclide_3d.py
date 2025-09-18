#!/usr/bin/env python3
"""
Clean 3D Nuclide Visualization with Color-coded Nuclides
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from scipy.interpolate import interp1d

def smooth_data(time, data, smooth_factor=5):
    """Apply smoothing to reduce jagged appearance"""
    if len(time) < 4:
        return time, data
    
    # Create interpolation function
    f = interp1d(time, data, kind='cubic', fill_value='extrapolate')
    
    # Create smoother time array
    time_smooth = np.linspace(time.min(), time.max(), len(time) * smooth_factor)
    data_smooth = f(time_smooth)
    
    return time_smooth, data_smooth

def get_nuclide_categories():
    """Define nuclide categories with distinct colors"""
    categories = {
        'Short-lived (high decay)': {
            'indices': [3, 6, 7, 10, 12, 13, 14, 15, 18, 28, 29, 34, 37, 39, 41, 46, 48],  # Fast decay nuclides
            'color': 'red',
            'alpha': 0.8
        },
        'Medium-lived': {
            'indices': [0, 4, 8, 11, 16, 17, 19, 20, 21, 22, 23, 25, 27, 31, 33, 35, 44, 45, 47, 49, 50, 52],  # Medium decay
            'color': 'blue', 
            'alpha': 0.7
        },
        'Long-lived (low decay)': {
            'indices': [1, 2, 9, 24, 26, 30, 36, 38, 40, 42, 43, 51, 53, 55, 56, 57, 58, 59],  # Slow decay nuclides
            'color': 'green',
            'alpha': 0.6
        },
        'I-131 (main isotope)': {
            'indices': [32],  # I-131 is special
            'color': 'orange',
            'alpha': 1.0
        }
    }
    return categories

def create_clean_3d_lines(csv_file, output_file=None, use_log=True, smooth=True):
    """Create clean 3D line plot with nuclide categories"""
    
    df = pd.read_csv(csv_file)
    time = df['time(s)'].values
    
    # Get nuclide columns
    nuclide_cols = [col for col in df.columns if col.startswith('ratio_Q_')]
    nuclide_data = df[nuclide_cols].values
    
    # Apply log scale if requested
    if use_log:
        nuclide_data_plot = np.log10(nuclide_data + 1e-10)
        z_label = 'Log10(Concentration Ratio)'
        title_suffix = '(Log Scale)'
    else:
        nuclide_data_plot = nuclide_data
        z_label = 'Concentration Ratio'
        title_suffix = '(Linear Scale)'
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get nuclide categories
    categories = get_nuclide_categories()
    
    # Plot each category
    for cat_name, cat_info in categories.items():
        for nuc_idx in cat_info['indices']:
            if nuc_idx < len(nuclide_cols):
                # Get data for this nuclide
                nuc_data = nuclide_data_plot[:, nuc_idx]
                
                # Apply smoothing if requested
                if smooth and len(time) > 3:
                    time_smooth, nuc_data_smooth = smooth_data(time, nuc_data, smooth_factor=3)
                else:
                    time_smooth, nuc_data_smooth = time, nuc_data
                
                # Plot as line
                ax.plot(time_smooth, [nuc_idx] * len(time_smooth), nuc_data_smooth,
                       color=cat_info['color'], alpha=cat_info['alpha'], linewidth=2,
                       label=cat_name if nuc_idx == cat_info['indices'][0] else "")
    
    # Set labels and title
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Nuclide Index', fontsize=12)
    ax.set_zlabel(z_label, fontsize=12)
    ax.set_title(f'Nuclide Concentration Evolution by Category {title_suffix}', fontsize=14, pad=20)
    
    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
    
    # Set viewing angle
    ax.view_init(elev=25, azim=45)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Clean 3D lines plot saved: {output_file}")
    
    return fig, ax

def create_top_nuclides_clean(csv_file, output_file=None, top_n=15, use_log=True):
    """Create clean visualization focusing on top nuclides"""
    
    df = pd.read_csv(csv_file)
    time = df['time(s)'].values
    
    # Get nuclide columns
    nuclide_cols = [col for col in df.columns if col.startswith('ratio_Q_')]
    nuclide_data = df[nuclide_cols].values
    
    # Find top N nuclides by maximum concentration
    max_concentrations = np.max(nuclide_data, axis=0)
    top_indices = np.argsort(max_concentrations)[-top_n:]
    
    # Apply log scale if requested
    if use_log:
        nuclide_data_plot = np.log10(nuclide_data + 1e-10)
        z_label = 'Log10(Concentration Ratio)'
        title_suffix = '(Log Scale)'
    else:
        nuclide_data_plot = nuclide_data
        z_label = 'Concentration Ratio'
        title_suffix = '(Linear Scale)'
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate distinct colors for each nuclide
    colors = plt.cm.Set3(np.linspace(0, 1, top_n))
    
    # Plot each top nuclide
    for i, nuc_idx in enumerate(top_indices):
        nuc_data = nuclide_data_plot[:, nuc_idx]
        
        # Apply smoothing
        time_smooth, nuc_data_smooth = smooth_data(time, nuc_data, smooth_factor=3)
        
        # Plot as thick line
        ax.plot(time_smooth, [nuc_idx] * len(time_smooth), nuc_data_smooth,
               color=colors[i], linewidth=3, alpha=0.8,
               label=f'Nuclide {nuc_idx} (max: {max_concentrations[nuc_idx]:.4f})')
        
        # Add scatter points at key intervals
        key_times = np.arange(0, len(time), max(1, len(time)//8))
        ax.scatter(time[key_times], [nuc_idx]*len(key_times), 
                  nuc_data[key_times],
                  color=colors[i], s=60, alpha=0.9, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Nuclide Index', fontsize=12)
    ax.set_zlabel(z_label, fontsize=12)
    ax.set_title(f'Top {top_n} Nuclides - Clean Visualization {title_suffix}', fontsize=14, pad=20)
    
    # Add legend outside plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Top nuclides clean plot saved: {output_file}")
    
    return fig, ax

def create_surface_by_category(csv_file, output_file=None, use_log=True):
    """Create surface plot but color by nuclide category"""
    
    df = pd.read_csv(csv_file)
    time = df['time(s)'].values
    
    # Get nuclide columns
    nuclide_cols = [col for col in df.columns if col.startswith('ratio_Q_')]
    nuclide_data = df[nuclide_cols].values
    
    # Apply log scale if requested
    if use_log:
        nuclide_data_plot = np.log10(nuclide_data + 1e-10)
        z_label = 'Log10(Concentration Ratio)'
        title_suffix = '(Log Scale)'
    else:
        nuclide_data_plot = nuclide_data
        z_label = 'Concentration Ratio'
        title_suffix = '(Linear Scale)'
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    
    # Get categories
    categories = get_nuclide_categories()
    
    # Create subplots for each category
    subplot_positions = [(2, 2, 1), (2, 2, 2), (2, 2, 3), (2, 2, 4)]
    
    for idx, (cat_name, cat_info) in enumerate(categories.items()):
        if idx >= len(subplot_positions):
            break
            
        ax = fig.add_subplot(*subplot_positions[idx], projection='3d')
        
        # Get data for this category
        cat_indices = cat_info['indices']
        cat_data = nuclide_data_plot[:, cat_indices]
        
        # Create meshgrid
        nuclide_indices = np.array(cat_indices)
        T, N = np.meshgrid(time, nuclide_indices)
        Z = cat_data.T
        
        # Create surface with single color
        surf = ax.plot_surface(T, N, Z, color=cat_info['color'], alpha=cat_info['alpha'],
                              linewidth=0, antialiased=True, shade=True)
        
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Nuclide Index', fontsize=10)
        ax.set_zlabel(z_label, fontsize=10)
        ax.set_title(f'{cat_name}', fontsize=12, pad=15)
        ax.view_init(elev=25, azim=45)
    
    plt.suptitle(f'Nuclide Categories - Surface View {title_suffix}', fontsize=16, y=0.95)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Category surface plot saved: {output_file}")
    
    return fig

def create_concentration_height_plot(csv_file, output_file=None):
    """Create plot where concentration is only shown as height, nuclides as colors"""
    
    df = pd.read_csv(csv_file)
    time = df['time(s)'].values
    
    # Get nuclide columns
    nuclide_cols = [col for col in df.columns if col.startswith('ratio_Q_')]
    nuclide_data = df[nuclide_cols].values
    
    # Use log scale for better visualization
    nuclide_data_log = np.log10(nuclide_data + 1e-10)
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get categories
    categories = get_nuclide_categories()
    
    # Plot each nuclide with its category color
    for cat_name, cat_info in categories.items():
        for nuc_idx in cat_info['indices']:
            if nuc_idx < len(nuclide_cols):
                # Get smoothed data
                nuc_data = nuclide_data_log[:, nuc_idx]
                time_smooth, nuc_data_smooth = smooth_data(time, nuc_data, smooth_factor=3)
                
                # Plot as filled area under curve
                ax.plot(time_smooth, [nuc_idx] * len(time_smooth), nuc_data_smooth,
                       color=cat_info['color'], linewidth=2, alpha=cat_info['alpha'],
                       label=cat_name if nuc_idx == cat_info['indices'][0] else "")
                
                # Add vertical lines to show concentration as "buildings"
                for i, (t, conc) in enumerate(zip(time[::5], nuc_data[::5])):  # Every 5th point
                    ax.plot([t, t], [nuc_idx, nuc_idx], [np.min(nuclide_data_log), conc],
                           color=cat_info['color'], alpha=0.3, linewidth=1)
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Nuclide Index', fontsize=12)
    ax.set_zlabel('Log10(Concentration Ratio)', fontsize=12)
    ax.set_title('Nuclide Concentrations - Height-based Visualization', fontsize=14, pad=20)
    
    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
    ax.view_init(elev=20, azim=50)
    
    # Add statistics
    stats_text = f"""
    Visualization Notes:
    • Height = Concentration (log scale)
    • Colors = Nuclide categories
    • Smooth lines reduce jaggedness
    • {len(nuclide_cols)} total nuclides
    • Time: {time.min():.0f}-{time.max():.0f} seconds
    """
    ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, 
              verticalalignment='top', 
              bbox=dict(boxstyle="round", facecolor='lightgray', alpha=0.8),
              fontsize=10)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Height-based concentration plot saved: {output_file}")
    
    return fig, ax

def main():
    """Main function for clean 3D visualizations"""
    
    print("=== Clean 3D Nuclide Visualization (Color-coded by Nuclide Type) ===\n")
    
    # Create output directory
    os.makedirs("cram_result", exist_ok=True)
    
    # Use the nuclide ratios data
    data_file = "all_particles_nuclide_ratios.csv"
    
    if not os.path.exists(data_file):
        print(f"File not found: {data_file}")
        return
    
    print(f"Using data file: {data_file}")
    print("Creating smooth, color-coded visualizations...\n")
    
    # 1. Clean line plot by category (Log scale)
    print("1. Creating clean 3D lines by category (Log scale)...")
    create_clean_3d_lines(data_file, "cram_result/clean_lines_log.png", use_log=True, smooth=True)
    
    # 2. Clean line plot by category (Linear scale)
    print("2. Creating clean 3D lines by category (Linear scale)...")
    create_clean_3d_lines(data_file, "cram_result/clean_lines_linear.png", use_log=False, smooth=True)
    
    # 3. Top nuclides clean visualization
    print("3. Creating top nuclides clean visualization...")
    create_top_nuclides_clean(data_file, "cram_result/top_nuclides_clean.png", top_n=12, use_log=True)
    
    # 4. Surface plots by category
    print("4. Creating surface plots by category...")
    create_surface_by_category(data_file, "cram_result/surface_by_category.png", use_log=True)
    
    # 5. Height-based concentration plot
    print("5. Creating height-based concentration visualization...")
    create_concentration_height_plot(data_file, "cram_result/height_based_plot.png")
    
    print("\n=== Clean 3D Visualization Complete! ===")
    print("Generated files:")
    print("- cram_result/clean_lines_log.png          (Smooth lines, log scale)")
    print("- cram_result/clean_lines_linear.png       (Smooth lines, linear scale)")
    print("- cram_result/top_nuclides_clean.png       (Top 12 nuclides, distinct colors)")
    print("- cram_result/surface_by_category.png      (Surfaces by nuclide type)")
    print("- cram_result/height_based_plot.png        (Height = concentration only)")
    print("\nKey improvements:")
    print("• Reduced jaggedness with data smoothing")
    print("• Color-coded by nuclide categories, not concentration")
    print("• Height represents concentration level")
    print("• Cleaner, more readable visualization")

if __name__ == "__main__":
    main()