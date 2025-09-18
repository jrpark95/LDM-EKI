#!/usr/bin/env python3
"""
Advanced LDM 3D Visualization
Creates enhanced 3D visualizations with volume rendering and interactive features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata
import os

def create_volume_rendered_3d(csv_file, output_file=None):
    """Create volume-rendered 3D visualization"""
    
    df = pd.read_csv(csv_file)
    timestep = os.path.basename(csv_file).split('_')[-1].split('.')[0]
    
    # Create figure with custom styling
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract coordinates
    x = df['lon'].values
    y = df['lat'].values  
    z = df['alt'].values
    c = df['concentration'].values
    
    # Normalize concentrations for better visualization
    c_norm = (c - c.min()) / (c.max() - c.min()) if c.max() > c.min() else c
    
    # Create multi-layer visualization with different sizes based on concentration
    sizes = 20 + 200 * c_norm  # Size proportional to concentration
    alpha_values = 0.3 + 0.7 * c_norm  # Transparency proportional to concentration
    
    # Create scatter plot with varying sizes and transparency
    for i in range(len(x)):
        ax.scatter(x[i], y[i], z[i], c=c[i], cmap='plasma', 
                  s=sizes[i], alpha=alpha_values[i], edgecolors='white', linewidth=0.5)
    
    # Add convex hull for particle cloud boundary
    if len(df) >= 4:  # Need at least 4 points for 3D hull
        try:
            points = np.column_stack((x, y, z))
            hull = ConvexHull(points)
            
            # Draw hull edges
            for simplex in hull.simplices:
                # Draw triangular faces with low alpha
                triangle = points[simplex]
                ax.plot_trisurf(*triangle.T, alpha=0.1, color='cyan', linewidth=0.1)
        except:
            print("Could not create convex hull (degenerate points)")
    
    # Enhanced colorbar
    scatter_ref = ax.scatter([], [], [], c=[], cmap='plasma')  # Reference for colorbar
    cbar = plt.colorbar(scatter_ref, ax=ax, shrink=0.8, aspect=20, pad=0.1)
    cbar.set_label('Nuclear Concentration', rotation=270, labelpad=25, fontsize=14, color='white')
    cbar.ax.tick_params(colors='white')
    
    # Styling
    ax.set_xlabel('Longitude', fontsize=14, color='white')
    ax.set_ylabel('Latitude', fontsize=14, color='white')
    ax.set_zlabel('Altitude (m)', fontsize=14, color='white')
    ax.tick_params(colors='white')
    
    # Enhanced title
    title = f'LDM Nuclear Plume - Volume Visualization\nTimestep {timestep} ({len(df)} Active Particles)'
    ax.set_title(title, fontsize=16, pad=20, color='white', weight='bold')
    
    # Add detailed statistics box
    stats_text = f"""
    ╔══ Simulation Statistics ══╗
    ║ Active Particles: {len(df):,}     ║
    ║ Max Concentration: {c.max():.4f} ║
    ║ Mean Concentration: {c.mean():.4f}║
    ║ Std Deviation: {c.std():.4f}    ║
    ║ Altitude Range: {z.min():.0f}-{z.max():.0f}m  ║
    ║ Lat Range: {y.min():.3f}-{y.max():.3f}°    ║
    ║ Lon Range: {x.min():.3f}-{x.max():.3f}°    ║
    ╚═══════════════════════════╝
    """
    ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, 
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.8, edgecolor='cyan'),
              fontsize=10, color='cyan')
    
    # Set viewing angle for better perspective
    ax.view_init(elev=20, azim=45)
    
    # Enhanced grid
    ax.grid(True, alpha=0.3)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"Volume-rendered 3D visualization saved to {output_file}")
    
    plt.style.use('default')  # Reset style
    return fig, ax

def create_concentration_isosurfaces(csv_file, output_file=None):
    """Create isosurface visualization of concentration levels"""
    
    df = pd.read_csv(csv_file)
    timestep = os.path.basename(csv_file).split('_')[-1].split('.')[0]
    
    if len(df) < 10:
        print("Not enough data points for isosurface visualization")
        return None, None
    
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract coordinates
    x = df['lon'].values
    y = df['lat'].values  
    z = df['alt'].values
    c = df['concentration'].values
    
    # Create regular grid for interpolation
    xi = np.linspace(x.min(), x.max(), 30)
    yi = np.linspace(y.min(), y.max(), 30)
    zi = np.linspace(z.min(), z.max(), 20)
    
    # Create mesh grid
    X, Y, Z = np.meshgrid(xi, yi, zi)
    
    # Interpolate concentration values to grid
    points = np.column_stack((x, y, z))
    grid_points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    try:
        # Interpolate concentrations
        C_interp = griddata(points, c, grid_points, method='linear', fill_value=0)
        C_grid = C_interp.reshape(X.shape)
        
        # Define isosurface levels
        c_max = c.max()
        c_min = c.min()
        levels = [c_min + (c_max - c_min) * f for f in [0.2, 0.5, 0.8]]
        colors = ['blue', 'green', 'red']
        alphas = [0.2, 0.3, 0.4]
        
        # Plot isosurfaces
        for level, color, alpha in zip(levels, colors, alphas):
            # Find contour surfaces
            mask = C_grid >= level
            if np.any(mask):
                ax.scatter(X[mask], Y[mask], Z[mask], 
                          c=color, alpha=alpha, s=20, label=f'Level {level:.3f}')
        
        # Plot original data points
        scatter = ax.scatter(x, y, z, c=c, cmap='plasma', s=60, alpha=0.8, 
                           edgecolors='black', linewidth=0.5, label='Data Points')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label('Concentration', rotation=270, labelpad=20, fontsize=12)
        
        # Labels and title
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_zlabel('Altitude (m)', fontsize=12)
        ax.set_title(f'Concentration Isosurfaces - Timestep {timestep}', fontsize=14, pad=20)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Isosurface visualization saved to {output_file}")
        
    except Exception as e:
        print(f"Error creating isosurfaces: {e}")
        return None, None
    
    return fig, ax

def create_cross_section_views(csv_file, output_file=None):
    """Create cross-sectional views of the 3D data"""
    
    df = pd.read_csv(csv_file)
    timestep = os.path.basename(csv_file).split('_')[-1].split('.')[0]
    
    # Create subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'LDM Cross-Sectional Views - Timestep {timestep}', fontsize=16, weight='bold')
    
    x = df['lon'].values
    y = df['lat'].values  
    z = df['alt'].values
    c = df['concentration'].values
    
    # XY view (top-down)
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(x, y, c=c, cmap='plasma', s=60, alpha=0.7)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('Top View (XY Plane)')
    plt.colorbar(scatter1, ax=ax1)
    
    # XZ view (side view)
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(x, z, c=c, cmap='plasma', s=60, alpha=0.7)
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Altitude (m)')
    ax2.set_title('Side View (XZ Plane)')
    plt.colorbar(scatter2, ax=ax2)
    
    # YZ view (front view)
    ax3 = axes[1, 0]
    scatter3 = ax3.scatter(y, z, c=c, cmap='plasma', s=60, alpha=0.7)
    ax3.set_xlabel('Latitude')
    ax3.set_ylabel('Altitude (m)')
    ax3.set_title('Front View (YZ Plane)')
    plt.colorbar(scatter3, ax=ax3)
    
    # Concentration histogram
    ax4 = axes[1, 1]
    ax4.hist(c, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.set_xlabel('Concentration')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Concentration Distribution')
    ax4.axvline(c.mean(), color='red', linestyle='--', label=f'Mean: {c.mean():.4f}')
    ax4.axvline(np.median(c), color='green', linestyle='--', label=f'Median: {np.median(c):.4f}')
    ax4.legend()
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Cross-sectional views saved to {output_file}")
    
    return fig, axes

def main():
    """Main function for advanced 3D visualizations"""
    
    print("=== Advanced LDM 3D Visualization ===\n")
    
    # Create output directory
    os.makedirs("cram_result/advanced_3d", exist_ok=True)
    
    # Select a representative timestep
    test_file = "validation/concentration_grid_01000.csv"
    
    if not os.path.exists(test_file):
        print(f"File {test_file} not found! Looking for alternatives...")
        import glob
        grid_files = glob.glob("validation/concentration_grid_*.csv")
        if grid_files:
            test_file = grid_files[len(grid_files)//2]  # Take middle file
            print(f"Using {test_file} instead")
        else:
            print("No concentration grid files found!")
            return
    
    # 1. Volume-rendered visualization
    print("1. Creating volume-rendered 3D visualization...")
    create_volume_rendered_3d(test_file, "cram_result/advanced_3d/ldm_volume_render.png")
    
    # 2. Isosurface visualization
    print("2. Creating isosurface visualization...")
    create_concentration_isosurfaces(test_file, "cram_result/advanced_3d/ldm_isosurfaces.png")
    
    # 3. Cross-sectional views
    print("3. Creating cross-sectional views...")
    create_cross_section_views(test_file, "cram_result/advanced_3d/ldm_cross_sections.png")
    
    print("\n=== Advanced 3D Visualization Complete! ===")
    print("Generated files:")
    print("- cram_result/advanced_3d/ldm_volume_render.png")
    print("- cram_result/advanced_3d/ldm_isosurfaces.png")
    print("- cram_result/advanced_3d/ldm_cross_sections.png")

if __name__ == "__main__":
    main()