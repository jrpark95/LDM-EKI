#!/usr/bin/env python3
"""
Map-based particle visualization with both contextily and cartopy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def load_particle_data(logs_dir="../logs/ldm_logs"):
    """Load particle data from CSV files"""
    particle_data = {}
    
    particle_files = glob.glob(os.path.join(logs_dir, "particles_hour_*.csv"))
    
    for file in sorted(particle_files):
        try:
            df = pd.read_csv(file)
            hour = int(file.split('_hour_')[1].split('.')[0])
            particle_data[hour] = df
            print(f"Loaded {len(df)} particles for hour {hour}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return particle_data

def create_contextily_animation_frames(particle_data, output_dir="../logs/ldm_logs"):
    """Create animation frames with OpenStreetMap background using contextily"""
    import contextily as ctx
    from matplotlib.patches import Rectangle
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Zoomed out coordinates
    lon_min, lon_max = -75.0, -73.0
    lat_min, lat_max = 40.0, 41.5
    
    # Find global concentration range
    all_concentrations = []
    for df in particle_data.values():
        if not df.empty:
            mask = ((df['longitude'] >= lon_min) & (df['longitude'] <= lon_max) & 
                    (df['latitude'] >= lat_min) & (df['latitude'] <= lat_max))
            df_region = df[mask]
            if not df_region.empty:
                all_concentrations.extend(df_region['concentration'].values)
    
    if not all_concentrations:
        return
        
    global_vmin, global_vmax = min(all_concentrations), max(all_concentrations)
    
    for hour, df in particle_data.items():
        if df.empty:
            continue
        
        mask = ((df['longitude'] >= lon_min) & (df['longitude'] <= lon_max) & 
                (df['latitude'] >= lat_min) & (df['latitude'] <= lat_max))
        df_region = df[mask]
        
        if df_region.empty:
            continue
        
        fig, ax = plt.subplots(figsize=(12, 9))
        
        # Create scatter plot
        scatter = ax.scatter(df_region['longitude'], df_region['latitude'], 
                           c=df_region['concentration'], s=30, alpha=0.7, 
                           cmap='turbo', vmin=global_vmin, vmax=global_vmax, zorder=5)
        
        # Add source and receptors
        source_lon, source_lat = -74.1, 40.7490
        ax.plot(source_lon, source_lat, 'r*', markersize=25, 
                label='Source Location', markeredgecolor='black', markeredgewidth=2, zorder=6)
        
        receptor_coords = [(-73.98, 40.7490), (-74.00, 40.7490), (-74.02, 40.7490)]
        for i, (rec_lon, rec_lat) in enumerate(receptor_coords):
            ax.plot(rec_lon, rec_lat, 'bs', markersize=12, 
                    label='Receptors' if i == 0 else '', 
                    markeredgecolor='black', markeredgewidth=1, zorder=6)
        
        # Set extent
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        
        # Add map background
        try:
            ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.7)
        except Exception as e:
            print(f"Could not add basemap: {e}")
            ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Concentration (Bq)', fontsize=14)
        
        # Labels and title
        ax.set_xlabel('Longitude', fontsize=14)
        ax.set_ylabel('Latitude', fontsize=14)
        ax.set_title(f'NYC Particle Dispersion (OpenStreetMap) - Hour {hour}\n'
                     f'{len(df_region)} active particles | Time: {hour}:00', 
                     fontsize=16, fontweight='bold')
        
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
        
        # Save
        filename = os.path.join(output_dir, f'map_contextily_frame_{hour:02d}.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved contextily frame for hour {hour}: {filename}")

def create_cartopy_animation_frames(particle_data, output_dir="../logs/ldm_logs"):
    """Create animation frames with natural features using cartopy"""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Zoomed out coordinates
    lon_min, lon_max = -75.0, -73.0
    lat_min, lat_max = 40.0, 41.5
    
    # Find global concentration range
    all_concentrations = []
    for df in particle_data.values():
        if not df.empty:
            mask = ((df['longitude'] >= lon_min) & (df['longitude'] <= lon_max) & 
                    (df['latitude'] >= lat_min) & (df['latitude'] <= lat_max))
            df_region = df[mask]
            if not df_region.empty:
                all_concentrations.extend(df_region['concentration'].values)
    
    if not all_concentrations:
        return
        
    global_vmin, global_vmax = min(all_concentrations), max(all_concentrations)
    
    for hour, df in particle_data.items():
        if df.empty:
            continue
        
        mask = ((df['longitude'] >= lon_min) & (df['longitude'] <= lon_max) & 
                (df['latitude'] >= lat_min) & (df['latitude'] <= lat_max))
        df_region = df[mask]
        
        if df_region.empty:
            continue
        
        # Create figure with cartopy projection
        fig = plt.figure(figsize=(12, 9))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Add map features
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax.add_feature(cfeature.COASTLINE, linewidth=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.5)
        ax.add_feature(cfeature.LAKES, color='lightblue', alpha=0.5)
        ax.add_feature(cfeature.RIVERS, color='blue', alpha=0.5)
        
        # Create scatter plot
        scatter = ax.scatter(df_region['longitude'], df_region['latitude'], 
                           c=df_region['concentration'], s=30, alpha=0.8, 
                           cmap='turbo', vmin=global_vmin, vmax=global_vmax, 
                           transform=ccrs.PlateCarree(), zorder=5)
        
        # Add source and receptors
        source_lon, source_lat = -74.1, 40.7490
        ax.plot(source_lon, source_lat, 'r*', markersize=25, 
                label='Source Location', markeredgecolor='black', markeredgewidth=2, 
                transform=ccrs.PlateCarree(), zorder=6)
        
        receptor_coords = [(-73.98, 40.7490), (-74.00, 40.7490), (-74.02, 40.7490)]
        for i, (rec_lon, rec_lat) in enumerate(receptor_coords):
            ax.plot(rec_lon, rec_lat, 'bs', markersize=12, 
                    label='Receptors' if i == 0 else '', 
                    markeredgecolor='black', markeredgewidth=1, 
                    transform=ccrs.PlateCarree(), zorder=6)
        
        # Set extent
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, alpha=0.3)
        gl.top_labels = False
        gl.right_labels = False
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Concentration (Bq)', fontsize=14)
        
        # Title
        ax.set_title(f'NYC Particle Dispersion (Natural Earth) - Hour {hour}\n'
                     f'{len(df_region)} active particles | Time: {hour}:00', 
                     fontsize=16, fontweight='bold')
        
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
        
        # Save
        filename = os.path.join(output_dir, f'map_cartopy_frame_{hour:02d}.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved cartopy frame for hour {hour}: {filename}")

def create_both_map_animations():
    """Create both contextily and cartopy animation frames"""
    logs_dir = "../logs/ldm_logs"
    
    # Load particle data
    particle_data = load_particle_data(logs_dir)
    
    if not particle_data:
        print("No particle data found!")
        return
    
    print(f"Found particle data for {len(particle_data)} hours")
    
    # Create both types of maps
    print("\nCreating OpenStreetMap frames (contextily)...")
    create_contextily_animation_frames(particle_data, logs_dir)
    
    print("\nCreating Natural Earth frames (cartopy)...")
    create_cartopy_animation_frames(particle_data, logs_dir)
    
    print("\nBoth map animations created successfully!")
    print(f"Check {logs_dir}/ for map_*_frame_*.png files.")

if __name__ == "__main__":
    create_both_map_animations()