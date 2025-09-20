#!/usr/bin/env python3
"""
OpenStreetMap grid visualization - 2x3 layout with zoom versions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
import os
import glob

def load_particle_data(logs_dir="../logs/ldm_logs"):
    """Load particle data from CSV files"""
    particle_data = {}
    
    particle_files = glob.glob(os.path.join(logs_dir, "particles_15min_*.csv"))
    
    for file in sorted(particle_files):
        try:
            df = pd.read_csv(file)
            quarter_hour = int(file.split('_15min_')[1].split('.')[0])
            particle_data[quarter_hour] = df
            print(f"Loaded {len(df)} particles for 15-min interval {quarter_hour}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return particle_data

def create_osm_grid_visualization(particle_data, output_dir="../logs/ldm_logs"):
    """Create 2x3 grid visualization with OpenStreetMap background"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Two zoom levels
    zoom_configs = {
        'zoomed_out': {
            'lon_min': -75.0, 'lon_max': -73.0,
            'lat_min': 40.0, 'lat_max': 41.5,
            'title_suffix': 'Wide View'
        },
        'zoomed_in': {
            'lon_min': -74.3, 'lon_max': -73.7,
            'lat_min': 40.4, 'lat_max': 41.0,
            'title_suffix': 'NYC Focus'
        }
    }
    
    for zoom_name, zoom_config in zoom_configs.items():
        lon_min, lon_max = zoom_config['lon_min'], zoom_config['lon_max']
        lat_min, lat_max = zoom_config['lat_min'], zoom_config['lat_max']
        
        # Find global concentration range for this zoom level
        all_concentrations = []
        filtered_data = {}
        
        for quarter_hour, df in particle_data.items():
            if df.empty:
                continue
            mask = ((df['longitude'] >= lon_min) & (df['longitude'] <= lon_max) & 
                    (df['latitude'] >= lat_min) & (df['latitude'] <= lat_max))
            df_region = df[mask]
            if not df_region.empty:
                all_concentrations.extend(df_region['concentration'].values)
                filtered_data[quarter_hour] = df_region
        
        if not all_concentrations:
            continue
            
        global_vmin, global_vmax = min(all_concentrations), max(all_concentrations)
        
        # Create 2x3 subplot grid
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'NYC Particle Dispersion - {zoom_config["title_suffix"]}\\n'
                     f'Time Evolution (Hourly snapshots)', fontsize=20, fontweight='bold')
        
        # Plot every 4th interval (1 hour intervals) for 6 hours
        sorted_keys = sorted(filtered_data.keys())
        # Select intervals 4, 8, 12, 16, 20, 24 (1hr, 2hr, 3hr, 4hr, 5hr, 6hr)
        hour_intervals = [4, 8, 12, 16, 20, 24]
        selected_keys = [k for k in sorted_keys if k in hour_intervals]
        for i, quarter_hour in enumerate(selected_keys):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            df_region = filtered_data[quarter_hour]
            
            # Create scatter plot
            scatter = ax.scatter(df_region['longitude'], df_region['latitude'], 
                               c=df_region['concentration'], s=20, alpha=0.7, 
                               cmap='turbo', vmin=global_vmin, vmax=global_vmax, zorder=5)
            
            # Add source and receptors
            source_lon, source_lat = -74.1, 40.7490
            ax.plot(source_lon, source_lat, 'r*', markersize=20, 
                    markeredgecolor='black', markeredgewidth=1.5, zorder=6)
            
            receptor_coords = [(-73.98, 40.7490), (-74.00, 40.7490), (-74.02, 40.7490)]
            for rec_lon, rec_lat in receptor_coords:
                ax.plot(rec_lon, rec_lat, 'bs', markersize=8, 
                        markeredgecolor='black', markeredgewidth=1, zorder=6)
            
            # Set extent
            ax.set_xlim(lon_min, lon_max)
            ax.set_ylim(lat_min, lat_max)
            
            # Add map background
            try:
                ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.7)
            except Exception as e:
                print(f"Could not add basemap for interval {quarter_hour}: {e}")
                ax.grid(True, alpha=0.3)
            
            # Labels and title for each subplot
            time_in_hours = quarter_hour * 0.25
            ax.set_title(f'Hour {int(time_in_hours)}\\n{len(df_region)} particles', fontsize=14, fontweight='bold')
            ax.set_xlabel('Longitude', fontsize=12)
            ax.set_ylabel('Latitude', fontsize=12)
            
            # Add time annotation
            ax.text(0.02, 0.98, f'{int(time_in_hours)}:00', transform=ax.transAxes, 
                    fontsize=16, fontweight='bold', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    verticalalignment='top')
        
        # Add colorbar to the right side
        cbar = fig.colorbar(scatter, ax=axes, shrink=0.8, aspect=30, pad=0.02)
        cbar.set_label('Concentration (Bq)', fontsize=16)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
                   markersize=15, markeredgecolor='black', label='Source'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', 
                   markersize=10, markeredgecolor='black', label='Receptors')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=2, 
                   fontsize=14, bbox_to_anchor=(0.5, 0.02))
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, bottom=0.08)
        
        # Save
        filename = os.path.join(output_dir, f'osm_grid_{zoom_name}.png')
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved OSM grid {zoom_name}: {filename}")

def create_individual_osm_frames(particle_data, output_dir="../logs/ldm_logs"):
    """Create individual OSM frames (replacing the old animation frames)"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Use zoomed out coordinates
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
        ax.set_title(f'NYC Particle Dispersion - Hour {hour}\\n'
                     f'{len(df_region)} active particles | Time: {hour}:00', 
                     fontsize=16, fontweight='bold')
        
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
        
        # Save (replace old animation frames)
        filename = os.path.join(output_dir, f'animation_frame_{hour:02d}.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved OSM frame for hour {hour}: {filename}")

def create_receptor_concentration_plot(particle_data, output_dir="../logs/ldm_logs"):
    """Create time series plot of receptor concentrations using LDM calculated data"""
    # Try to load LDM detailed 15-minute interval data first
    detailed_file = os.path.join(output_dir, 'receptor_concentrations.csv')
    hourly_file = os.path.join(output_dir, 'receptor_summary.csv')
    
    if os.path.exists(detailed_file):
        # Use LDM calculated detailed 15-minute data
        df_receptors = pd.read_csv(detailed_file)
        
        receptor_names = ['Receptor 1', 'Receptor 2', 'Receptor 3']
        colors = ['blue', 'red', 'green']
        
        plt.figure(figsize=(14, 8))
        
        for r in range(3):  # 3 receptors
            receptor_data = df_receptors[df_receptors['receptor'] == r]
            time_hours = receptor_data['time_hours'].values
            concentrations = receptor_data['concentration'].values
            
            plt.plot(time_hours, concentrations, marker='o', linewidth=2, 
                    markersize=6, color=colors[r], label=receptor_names[r])
        
        plt.xlabel('Time (hours)', fontsize=14)
        plt.ylabel('Concentration (Bq)', fontsize=14)
        plt.title('Receptor Concentration Time Series (15-minute intervals)\\nNYC Particle Dispersion Monitoring\\n(LDM Calculated Data)', 
                  fontsize=16, fontweight='bold')
        
        # Set x-axis ticks every hour but show 15-minute data
        plt.xticks(np.arange(0.25, 6.25, 1.0), [f'{i}:00' for i in range(1, 7)])
        plt.xlim(0, 6.25)
        
        # Add note about data source
        plt.text(0.02, 0.98, 'Data Source: LDM calculated concentrations\\n' +
                 'Time resolution: 15-minute intervals\\n' +
                 'Detection radius: ~1km (0.01°)\\n' +
                 'Same data used for EKI inversion',
                 transform=plt.gca().transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                 verticalalignment='top')
        
        print(f"Used LDM calculated detailed receptor data from {detailed_file}")
        
    elif os.path.exists(hourly_file):
        # Fallback to hourly data
        df_receptors = pd.read_csv(hourly_file)
        
        receptor_names = ['Receptor 1', 'Receptor 2', 'Receptor 3']
        colors = ['blue', 'red', 'green']
        
        plt.figure(figsize=(12, 8))
        
        for r in range(3):  # 3 receptors
            receptor_data = df_receptors[df_receptors['receptor'] == r]
            hours = receptor_data['hour'].values
            concentrations = receptor_data['concentration'].values
            
            plt.plot(hours, concentrations, marker='o', linewidth=2, 
                    markersize=8, color=colors[r], label=receptor_names[r])
        
        plt.xlabel('Time (hours)', fontsize=14)
        plt.ylabel('Concentration (Bq)', fontsize=14)
        plt.title('Receptor Concentration Time Series (hourly summary)\\nNYC Particle Dispersion Monitoring\\n(LDM Calculated Data)', 
                  fontsize=16, fontweight='bold')
        
        # Add note about data source
        plt.text(0.02, 0.98, 'Data Source: LDM calculated concentrations\\n' +
                 'Time resolution: Hourly summary\\n' +
                 'Detection radius: ~1km (0.01°)\\n' +
                 'Same data used for EKI inversion',
                 transform=plt.gca().transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                 verticalalignment='top')
        
        print(f"Used LDM calculated hourly receptor data from {hourly_file}")
        
    else:
        # Fallback: calculate from particle data (old method - for comparison)
        print(f"Warning: LDM receptor data not found at {receptor_file}")
        print("Using fallback calculation from particle data...")
        
        receptor_coords = [(-73.98, 40.7490), (-74.00, 40.7490), (-74.02, 40.7490)]
        receptor_names = ['Receptor 1', 'Receptor 2', 'Receptor 3']
        
        receptor_concentrations = {name: [] for name in receptor_names}
        hours = []
        
        for hour in sorted(particle_data.keys()):
            df = particle_data[hour]
            hours.append(hour)
            
            for i, (rec_lon, rec_lat) in enumerate(receptor_coords):
                # Use same detection radius as LDM (0.01 degrees)
                distance_threshold = 0.01
                dist_lat = np.abs(df['latitude'] - rec_lat)
                dist_lon = np.abs(df['longitude'] - rec_lon)
                distance = np.sqrt(dist_lat**2 + dist_lon**2)
                
                nearby_particles = df[distance <= distance_threshold]
                if len(nearby_particles) > 0:
                    # Sum concentration (same as LDM method)
                    total_concentration = nearby_particles['concentration'].sum()
                else:
                    total_concentration = 0.0
                
                receptor_concentrations[receptor_names[i]].append(total_concentration)
        
        plt.figure(figsize=(12, 8))
        colors = ['blue', 'red', 'green']
        
        for i, (name, concentrations) in enumerate(receptor_concentrations.items()):
            plt.plot(hours, concentrations, marker='o', linewidth=2, 
                    markersize=8, color=colors[i], label=name)
        
        plt.xlabel('Time (hours)', fontsize=14)
        plt.ylabel('Concentration (Bq)', fontsize=14)
        plt.title('Receptor Concentration Time Series\\nNYC Particle Dispersion Monitoring\\n(Fallback Calculation)', 
                  fontsize=16, fontweight='bold')
        
        # Add note about fallback method
        plt.text(0.02, 0.98, 'Data Source: Fallback particle calculation\\n' +
                 'Detection radius: ~1km (0.01°)\\n' +
                 'Method: Sum particles within radius',
                 transform=plt.gca().transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                 verticalalignment='top')
        
        # Set integer ticks for hours
        plt.xticks(hours)
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Save plot
    filename = os.path.join(output_dir, 'receptor_concentrations.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved receptor concentration plot: {filename}")

def create_osm_visualizations():
    """Create both grid and individual OSM visualizations"""
    logs_dir = "../logs/ldm_logs"
    
    # Load particle data
    particle_data = load_particle_data(logs_dir)
    
    if not particle_data:
        print("No particle data found!")
        return
    
    print(f"Found particle data for {len(particle_data)} hours")
    
    # Create grid visualizations
    print("\\nCreating OSM grid visualizations...")
    create_osm_grid_visualization(particle_data, logs_dir)
    
    # Create receptor concentration plot (commented out)
    # print("\\nCreating receptor concentration plot...")
    # create_receptor_concentration_plot(particle_data, logs_dir)
    
    # Create individual OSM frames (commented out)
    # print("\\nCreating individual OSM frames...")
    # create_individual_osm_frames(particle_data, logs_dir)
    
    print("\\nOSM visualizations completed!")
    print(f"Check {logs_dir}/ for:")
    print("  - osm_grid_zoomed_out.png (2x3 wide view)")
    print("  - osm_grid_zoomed_in.png (2x3 NYC focus)")

if __name__ == "__main__":
    create_osm_visualizations()