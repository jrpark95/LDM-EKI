#!/usr/bin/env python3
"""
Enhanced Particle Visualization with Geographic Context
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from matplotlib.patches import Rectangle

def load_particle_data(logs_dir="../logs/ldm_logs"):
    """Load particle position data from CSV files"""
    particle_files = glob.glob(os.path.join(logs_dir, "particles_hour_*.csv"))
    particle_data = {}
    
    for file in sorted(particle_files):
        filename = os.path.basename(file)
        hour = int(filename.split('_')[2].split('.')[0])
        
        try:
            df = pd.read_csv(file)
            if not df.empty:
                particle_data[hour] = df
                print(f"Loaded {len(df)} particles for hour {hour}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return particle_data

def create_geographic_styled_plots(particle_data, output_dir="../logs/ldm_logs"):
    """Create geographic-style plots with enhanced visualization"""
    os.makedirs(output_dir, exist_ok=True)
    
    # NYC area boundaries
    lon_min, lon_max = -74.3, -73.7
    lat_min, lat_max = 40.4, 41.0
    
    for hour, df in particle_data.items():
        if df.empty:
            continue
            
        # Filter particles in NYC area
        mask = ((df['longitude'] >= lon_min) & (df['longitude'] <= lon_max) & 
                (df['latitude'] >= lat_min) & (df['latitude'] <= lat_max))
        df_region = df[mask]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Main geographic plot
        if not df_region.empty:
            concentrations = df_region['concentration']
            
            # Use log normalization for better color distribution
            vmin, vmax = concentrations.min(), concentrations.max()
            
            scatter = ax1.scatter(df_region['longitude'], df_region['latitude'], 
                               c=concentrations, s=25, alpha=0.7, 
                               cmap='turbo', vmin=vmin, vmax=vmax)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label('Concentration (Bq)', fontsize=12)
            
            # Mark source location
            source_lon, source_lat = -74.1, 40.7490
            ax1.plot(source_lon, source_lat, 'r*', markersize=20, 
                    label='Source Location', markeredgecolor='black', markeredgewidth=2)
            
            # Mark receptor locations
            receptor_coords = [(-73.98, 40.7490), (-74.00, 40.7490), (-74.02, 40.7490)]
            for i, (rec_lon, rec_lat) in enumerate(receptor_coords):
                ax1.plot(rec_lon, rec_lat, 'bs', markersize=10, 
                        label='Receptors' if i == 0 else '', 
                        markeredgecolor='black', markeredgewidth=1)
            
            # Add some geographic context lines
            # Manhattan approximate boundaries
            manhattan_lons = [-73.93, -74.02, -74.02, -73.93, -73.93]
            manhattan_lats = [40.70, 40.70, 40.83, 40.83, 40.70]
            ax1.plot(manhattan_lons, manhattan_lats, 'k--', alpha=0.5, linewidth=2, label='Manhattan Approx.')
            
            ax1.set_xlim(lon_min, lon_max)
            ax1.set_ylim(lat_min, lat_max)
        
        ax1.set_xlabel('Longitude (degrees)', fontsize=12)
        ax1.set_ylabel('Latitude (degrees)', fontsize=12)
        ax1.set_title(f'Hour {hour}: Particle Distribution (NYC Area)\n{len(df_region)} particles in region', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Concentration heat map
        if not df_region.empty:
            # Create grid for heat map
            lon_bins = np.linspace(lon_min, lon_max, 50)
            lat_bins = np.linspace(lat_min, lat_max, 50)
            
            # Calculate concentration density
            H, xedges, yedges = np.histogram2d(df_region['longitude'], df_region['latitude'], 
                                             bins=[lon_bins, lat_bins], 
                                             weights=df_region['concentration'])
            
            # Particle count density
            H_count, _, _ = np.histogram2d(df_region['longitude'], df_region['latitude'], 
                                         bins=[lon_bins, lat_bins])
            
            # Average concentration per bin
            with np.errstate(divide='ignore', invalid='ignore'):
                H_avg = np.divide(H, H_count, out=np.zeros_like(H), where=H_count!=0)
            
            im = ax2.imshow(H_avg.T, origin='lower', aspect='auto', 
                          extent=[lon_min, lon_max, lat_min, lat_max],
                          cmap='turbo', alpha=0.8)
            
            plt.colorbar(im, ax=ax2, label='Avg Concentration (Bq)')
            
            # Mark locations on heat map too
            ax2.plot(source_lon, source_lat, 'r*', markersize=15, markeredgecolor='white')
            for rec_lon, rec_lat in receptor_coords:
                ax2.plot(rec_lon, rec_lat, 'ws', markersize=8, markeredgecolor='black')
        
        ax2.set_xlabel('Longitude (degrees)', fontsize=12)
        ax2.set_ylabel('Latitude (degrees)', fontsize=12)
        ax2.set_title(f'Concentration Heat Map', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Distance from source analysis
        if not df_region.empty:
            source_lon, source_lat = -74.1, 40.7490
            
            # Calculate distance from source (approximate)
            R = 6371  # Earth radius in km
            dlat = np.radians(df_region['latitude'] - source_lat)
            dlon = np.radians(df_region['longitude'] - source_lon)
            a = np.sin(dlat/2)**2 + np.cos(np.radians(source_lat)) * np.cos(np.radians(df_region['latitude'])) * np.sin(dlon/2)**2
            distance_km = 2 * R * np.arcsin(np.sqrt(a))
            
            # Plot concentration vs distance
            ax3.scatter(distance_km, df_region['concentration'], alpha=0.6, s=20)
            ax3.set_xlabel('Distance from Source (km)', fontsize=12)
            ax3.set_ylabel('Concentration (Bq)', fontsize=12)
            ax3.set_title('Concentration vs Distance from Source', fontsize=12, fontweight='bold')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)
        
        # Wind rose style plot (particle direction from source)
        if not df_region.empty:
            # Calculate bearing from source
            dlon = df_region['longitude'] - source_lon
            dlat = df_region['latitude'] - source_lat
            
            # Convert to polar coordinates
            angles = np.arctan2(dlon, dlat) * 180 / np.pi
            distances = np.sqrt(dlon**2 + dlat**2) * 111  # Approximate km conversion
            
            # Create polar plot
            ax4 = plt.subplot(2, 2, 4, projection='polar')
            scatter = ax4.scatter(np.radians(angles), distances, 
                               c=df_region['concentration'], s=20, alpha=0.7, 
                               cmap='turbo')
            
            ax4.set_theta_zero_location('N')
            ax4.set_theta_direction(1)
            ax4.set_title('Particle Distribution from Source\n(Polar View)', 
                         fontsize=12, fontweight='bold', pad=20)
            ax4.set_ylabel('Distance (km)', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        filename = os.path.join(output_dir, f'enhanced_particle_map_hour_{hour:02d}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved enhanced geographic plot for hour {hour}: {filename}")

def create_animation_frames(particle_data, output_dir="../logs/ldm_logs"):
    """Create individual frames for potential animation"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Common settings for all frames (zoomed out)
    lon_min, lon_max = -75.0, -73.0
    lat_min, lat_max = 40.0, 41.5
    
    # Find global concentration range for consistent coloring
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
        
        plt.figure(figsize=(12, 9))
        
        # Create scatter plot with consistent color scale
        scatter = plt.scatter(df_region['longitude'], df_region['latitude'], 
                           c=df_region['concentration'], s=30, alpha=0.7, 
                           cmap='turbo', vmin=global_vmin, vmax=global_vmax)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Concentration (Bq)', fontsize=14)
        
        # Mark source and receptors
        source_lon, source_lat = -74.1, 40.7490
        plt.plot(source_lon, source_lat, 'r*', markersize=25, 
                label='Source Location', markeredgecolor='black', markeredgewidth=2)
        
        receptor_coords = [(-73.98, 40.7490), (-74.00, 40.7490), (-74.02, 40.7490)]
        for i, (rec_lon, rec_lat) in enumerate(receptor_coords):
            plt.plot(rec_lon, rec_lat, 'bs', markersize=12, 
                    label='Receptors' if i == 0 else '', 
                    markeredgecolor='black', markeredgewidth=1)
        
        plt.xlim(lon_min, lon_max)
        plt.ylim(lat_min, lat_max)
        plt.xlabel('Longitude', fontsize=14)
        plt.ylabel('Latitude', fontsize=14)
        plt.title(f'Particle Dispersion - Hour {hour}\n'
                 f'{len(df_region)} active particles | Time: {hour}:00', 
                 fontsize=16, fontweight='bold')
        
        # Simple clean grid
        plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        plt.legend(loc='upper right', fontsize=12)
        
        # Add timestamp
        plt.text(0.02, 0.98, f'Hour {hour}', transform=plt.gca().transAxes, 
                fontsize=20, fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        # Save animation frame
        filename = os.path.join(output_dir, f'animation_frame_{hour:02d}.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved animation frame for hour {hour}: {filename}")

def main():
    """Main function"""
    print("Starting enhanced particle visualization...")
    
    # Load data
    logs_dir = "../logs/ldm_logs"
    particle_data = load_particle_data(logs_dir)
    
    if not particle_data:
        print("No particle data found!")
        return
    
    print(f"Found particle data for {len(particle_data)} hours")
    
    # Create enhanced visualizations
    print("Creating enhanced geographic plots...")
    create_geographic_styled_plots(particle_data, logs_dir)
    
    print("Creating animation frames...")
    create_animation_frames(particle_data, logs_dir)
    
    print("Enhanced visualization complete!")
    print(f"Check {logs_dir}/ for generated plots.")

def create_animation_frames_only():
    """Create only animation frames without other plots"""
    import os
    import glob
    
    logs_dir = "../logs/ldm_logs"
    
    # Load particle data
    particle_data = load_particle_data(logs_dir)
    
    if not particle_data:
        print("No particle data found!")
        return
    
    print(f"Found particle data for {len(particle_data)} hours")
    
    # Create only animation frames with zoomed out view
    print("Creating animation frames...")
    create_animation_frames(particle_data, logs_dir)
    
    print("Animation frames created successfully!")
    print(f"Check {logs_dir}/ for animation_frame_*.png files.")

if __name__ == "__main__":
    main()