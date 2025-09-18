#!/usr/bin/env python3
"""
Particle Distribution Visualization Script
Visualizes LDM particle distributions with concentration-based coloring and geographic maps
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle

def load_particle_data(logs_dir="../logs/ldm_logs"):
    """Load particle position data from CSV files"""
    particle_files = glob.glob(os.path.join(logs_dir, "particles_hour_*.csv"))
    particle_data = {}
    
    for file in sorted(particle_files):
        # Extract hour from filename
        filename = os.path.basename(file)
        hour = int(filename.split('_')[2].split('.')[0])
        
        try:
            df = pd.read_csv(file)
            if not df.empty:
                particle_data[hour] = df
                print(f"Loaded {len(df)} particles for hour {hour}")
            else:
                print(f"No particles found for hour {hour}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return particle_data

def load_particle_count_data(logs_dir="../logs/ldm_logs"):
    """Load particle count data"""
    count_file = os.path.join(logs_dir, "particle_count.csv")
    
    if not os.path.exists(count_file):
        print(f"Particle count file not found: {count_file}")
        return None
    
    try:
        df = pd.read_csv(count_file)
        print(f"Loaded particle count data with {len(df)} time points")
        return df
    except Exception as e:
        print(f"Error loading particle count data: {e}")
        return None

def create_geographic_map_plot(particle_data, output_dir="../logs/ldm_logs"):
    """Create geographic map plots for each hour"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the geographic bounds (focused around NYC area)
    lon_min, lon_max = -75.0, -73.0
    lat_min, lat_max = 40.0, 42.0
    
    for hour, df in particle_data.items():
        if df.empty:
            continue
            
        # Filter particles within our region of interest
        mask = ((df['longitude'] >= lon_min) & (df['longitude'] <= lon_max) & 
                (df['latitude'] >= lat_min) & (df['latitude'] <= lat_max))
        df_filtered = df[mask]
        
        if df_filtered.empty:
            print(f"No particles in region for hour {hour}")
            continue
        
        # Create figure with cartopy projection
        fig = plt.figure(figsize=(12, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Set map extent
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.3)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.5)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.top_labels = False
        gl.right_labels = False
        
        # Plot particles with concentration-based coloring
        if len(df_filtered) > 0:
            concentrations = df_filtered['concentration']
            
            # Use log scale for concentration coloring
            scatter = ax.scatter(df_filtered['longitude'], df_filtered['latitude'], 
                               c=concentrations, s=20, alpha=0.7, 
                               cmap='plasma', norm=LogNorm(vmin=concentrations.min(), 
                                                          vmax=concentrations.max()),
                               transform=ccrs.PlateCarree())
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', 
                              pad=0.05, shrink=0.8)
            cbar.set_label('Concentration (Bq)', fontsize=12)
            cbar.ax.tick_params(labelsize=10)
        
        # Mark source location
        source_lon, source_lat = -73.9650, 40.7490
        ax.plot(source_lon, source_lat, 'r*', markersize=15, 
                label='Source Location', transform=ccrs.PlateCarree())
        
        # Mark receptor locations
        receptor_coords = [
            (-73.98, 40.7490),
            (-74.00, 40.7490), 
            (-74.02, 40.7490)
        ]
        
        for i, (rec_lon, rec_lat) in enumerate(receptor_coords):
            ax.plot(rec_lon, rec_lat, 'bs', markersize=8, 
                    label=f'Receptor {i}' if i == 0 else '', 
                    transform=ccrs.PlateCarree())
        
        plt.title(f'Particle Distribution - Hour {hour}\n'
                 f'Active Particles: {len(df_filtered)}', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right')
        
        # Save plot
        filename = os.path.join(output_dir, f'particle_map_hour_{hour:02d}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved geographic map for hour {hour}: {filename}")

def create_simple_scatter_plot(particle_data, output_dir="../logs/ldm_logs"):
    """Create simple scatter plots without map background"""
    os.makedirs(output_dir, exist_ok=True)
    
    for hour, df in particle_data.items():
        if df.empty:
            continue
            
        plt.figure(figsize=(10, 8))
        
        # Plot particles with concentration-based coloring
        concentrations = df['concentration']
        
        scatter = plt.scatter(df['longitude'], df['latitude'], 
                           c=concentrations, s=20, alpha=0.7, 
                           cmap='plasma', norm=LogNorm(vmin=concentrations.min(), 
                                                      vmax=concentrations.max()))
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Concentration (Bq)', fontsize=12)
        
        # Mark source location
        source_lon, source_lat = -73.9650, 40.7490
        plt.plot(source_lon, source_lat, 'r*', markersize=15, label='Source Location')
        
        # Mark receptor locations
        receptor_coords = [(-73.98, 40.7490), (-74.00, 40.7490), (-74.02, 40.7490)]
        for i, (rec_lon, rec_lat) in enumerate(receptor_coords):
            plt.plot(rec_lon, rec_lat, 'bs', markersize=8, 
                    label=f'Receptor {i}' if i == 0 else '')
        
        plt.xlabel('Longitude (degrees)', fontsize=12)
        plt.ylabel('Latitude (degrees)', fontsize=12)
        plt.title(f'Particle Distribution - Hour {hour}\nActive Particles: {len(df)}', 
                 fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        filename = os.path.join(output_dir, f'particle_scatter_hour_{hour:02d}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved scatter plot for hour {hour}: {filename}")

def create_particle_count_plot(count_data, output_dir="../logs/ldm_logs"):
    """Create particle count vs time plot"""
    if count_data is None:
        print("No particle count data available")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    # Plot total particles
    plt.subplot(2, 1, 1)
    plt.plot(count_data['time_hours'], count_data['total_particles'], 
             'b-', linewidth=2, label='Total Particles')
    plt.plot(count_data['time_hours'], count_data['active_particles'], 
             'r-', linewidth=2, label='Active Particles')
    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('Particle Count', fontsize=12)
    plt.title('Particle Count vs Time', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot particle emission rate (difference between consecutive total counts)
    plt.subplot(2, 1, 2)
    if len(count_data) > 1:
        emission_rate = np.diff(count_data['total_particles']) / np.diff(count_data['time_hours'])
        time_midpoints = (count_data['time_hours'][1:].values + count_data['time_hours'][:-1].values) / 2
        plt.plot(time_midpoints, emission_rate, 'g-', linewidth=2, label='Emission Rate')
    
    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('Particles/hour', fontsize=12)
    plt.title('Particle Emission Rate', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    filename = os.path.join(output_dir, 'particle_count_vs_time.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved particle count plot: {filename}")

def create_concentration_statistics_plot(particle_data, output_dir="../logs/ldm_logs"):
    """Create concentration statistics plot"""
    os.makedirs(output_dir, exist_ok=True)
    
    hours = []
    mean_conc = []
    min_conc = []
    max_conc = []
    particle_counts = []
    
    for hour in sorted(particle_data.keys()):
        df = particle_data[hour]
        if not df.empty:
            hours.append(hour)
            mean_conc.append(df['concentration'].mean())
            min_conc.append(df['concentration'].min())
            max_conc.append(df['concentration'].max())
            particle_counts.append(len(df))
    
    if not hours:
        print("No data available for concentration statistics")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Concentration statistics
    ax1.plot(hours, mean_conc, 'b-o', linewidth=2, markersize=6, label='Mean Concentration')
    ax1.fill_between(hours, min_conc, max_conc, alpha=0.3, color='blue', label='Min-Max Range')
    ax1.set_xlabel('Hour', fontsize=12)
    ax1.set_ylabel('Concentration (Bq)', fontsize=12)
    ax1.set_title('Particle Concentration Statistics Over Time', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Particle count per hour
    ax2.bar(hours, particle_counts, alpha=0.7, color='green', label='Active Particles')
    ax2.set_xlabel('Hour', fontsize=12)
    ax2.set_ylabel('Number of Particles', fontsize=12)
    ax2.set_title('Active Particles Per Hour', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    filename = os.path.join(output_dir, 'concentration_statistics.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved concentration statistics plot: {filename}")

def main():
    """Main visualization function"""
    print("Starting particle visualization...")
    
    # Load data
    logs_dir = "../logs/ldm_logs"
    particle_data = load_particle_data(logs_dir)
    count_data = load_particle_count_data(logs_dir)
    
    if not particle_data:
        print("No particle data found. Make sure LDM simulation has been run.")
        return
    
    print(f"Found particle data for {len(particle_data)} hours")
    
    # Create visualizations
    try:
        print("Creating geographic map plots...")
        create_geographic_map_plot(particle_data, logs_dir)
    except Exception as e:
        print(f"Error creating geographic maps (cartopy may not be available): {e}")
        print("Creating simple scatter plots instead...")
        create_simple_scatter_plot(particle_data, logs_dir)
    
    print("Creating particle count plots...")
    create_particle_count_plot(count_data, logs_dir)
    
    print("Creating concentration statistics plots...")
    create_concentration_statistics_plot(particle_data, logs_dir)
    
    print("Visualization complete! Check ../logs/ldm_logs/ for generated plots.")

if __name__ == "__main__":
    main()