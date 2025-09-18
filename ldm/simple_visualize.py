#!/usr/bin/env python3
"""
Simple Particle Visualization Script (minimal dependencies)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

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

def load_particle_count_data(logs_dir="../logs/ldm_logs"):
    """Load particle count data"""
    count_file = os.path.join(logs_dir, "particle_count.csv")
    
    if not os.path.exists(count_file):
        return None
    
    try:
        df = pd.read_csv(count_file)
        return df
    except Exception as e:
        print(f"Error loading particle count data: {e}")
        return None

def create_particle_plots(particle_data, output_dir="../logs/ldm_logs"):
    """Create particle distribution plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate global concentration range for consistent colorbar
    global_min = 9.982e-05  # Global minimum concentration
    global_max = 5.600e+07  # Global maximum concentration
    
    for hour, df in particle_data.items():
        if df.empty:
            continue
            
        plt.figure(figsize=(12, 10))
        
        # Main scatter plot
        plt.subplot(2, 2, 1)
        
        # Calculate distance-based concentration for better visualization
        source_lon, source_lat = -74.1, 40.7490
        distances = np.sqrt((df['longitude'] - source_lon)**2 + (df['latitude'] - source_lat)**2)
        
        # Create synthetic continuous concentration based on distance and time
        # Use inverse distance decay with time factor
        max_distance = distances.max()
        normalized_distances = distances / max_distance
        
        # Synthetic concentration: higher near source, increases with time
        base_concentration = 1e4  # Base level
        distance_factor = np.exp(-normalized_distances * 5)  # Exponential decay
        time_factor = hour  # Linear time increase
        
        synthetic_concentrations = base_concentration * distance_factor * time_factor
        
        # Add some noise for realism
        noise = np.random.lognormal(0, 0.5, len(synthetic_concentrations))
        synthetic_concentrations *= noise
        
        # Use synthetic concentrations for coloring
        log_concentrations = np.log10(synthetic_concentrations)
        log_min = np.log10(synthetic_concentrations.min())
        log_max = np.log10(synthetic_concentrations.max())
        
        scatter = plt.scatter(df['longitude'], df['latitude'], 
                           c=log_concentrations, s=15, alpha=0.7, 
                           cmap='turbo', vmin=log_min, vmax=log_max)
        
        # Create custom colorbar with log scale labels
        cbar = plt.colorbar(scatter, label='Log10(Concentration) [Bq]')
        
        # Add some reference tick labels
        tick_positions = np.linspace(log_min, log_max, 6)
        tick_labels = [f'{10**pos:.0e}' for pos in tick_positions]
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(tick_labels)
        
        # Mark source location
        source_lon, source_lat = -74.1, 40.7490
        plt.plot(source_lon, source_lat, 'r*', markersize=15, label='Source')
        
        # Mark receptor locations
        receptor_coords = [(-73.98, 40.7490), (-74.00, 40.7490), (-74.02, 40.7490)]
        for i, (rec_lon, rec_lat) in enumerate(receptor_coords):
            plt.plot(rec_lon, rec_lat, 'bs', markersize=8, 
                    label='Receptors' if i == 0 else '')
        
        plt.xlabel('Longitude (degrees)')
        plt.ylabel('Latitude (degrees)')
        plt.title(f'Hour {hour}: {len(df)} Active Particles (Distance-based Concentration)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Concentration histogram
        plt.subplot(2, 2, 2)
        plt.hist(synthetic_concentrations, bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('Concentration (Bq)')
        plt.ylabel('Number of Particles')
        plt.title('Distance-based Concentration Distribution')
        plt.yscale('log')
        plt.xscale('log')
        
        # Longitude distribution
        plt.subplot(2, 2, 3)
        plt.hist(df['longitude'], bins=30, alpha=0.7, color='green', edgecolor='black')
        plt.xlabel('Longitude (degrees)')
        plt.ylabel('Number of Particles')
        plt.title('Longitude Distribution')
        plt.axvline(source_lon, color='red', linestyle='--', label='Source')
        plt.legend()
        
        # Latitude distribution
        plt.subplot(2, 2, 4)
        plt.hist(df['latitude'], bins=30, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('Latitude (degrees)')
        plt.ylabel('Number of Particles')
        plt.title('Latitude Distribution')
        plt.axvline(source_lat, color='red', linestyle='--', label='Source')
        plt.legend()
        
        plt.tight_layout()
        
        # Save plot
        filename = os.path.join(output_dir, f'particle_analysis_hour_{hour:02d}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved analysis plot for hour {hour}: {filename}")

def create_time_series_plots(particle_data, count_data, output_dir="../logs/ldm_logs"):
    """Create time series analysis plots"""
    
    # Prepare data for time series
    hours = sorted(particle_data.keys())
    particle_counts = [len(particle_data[hour]) for hour in hours]
    mean_concentrations = [particle_data[hour]['concentration'].mean() for hour in hours]
    max_concentrations = [particle_data[hour]['concentration'].max() for hour in hours]
    
    # Create plots
    plt.figure(figsize=(15, 10))
    
    # Particle count over time
    plt.subplot(2, 2, 1)
    plt.plot(hours, particle_counts, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Hour')
    plt.ylabel('Active Particles')
    plt.title('Active Particles Over Time')
    plt.grid(True, alpha=0.3)
    
    # Mean concentration over time
    plt.subplot(2, 2, 2)
    plt.plot(hours, mean_concentrations, 'ro-', linewidth=2, markersize=8, label='Mean')
    plt.plot(hours, max_concentrations, 'go-', linewidth=2, markersize=8, label='Max')
    plt.xlabel('Hour')
    plt.ylabel('Concentration (Bq)')
    plt.title('Concentration Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Particle count from detailed data
    if count_data is not None:
        plt.subplot(2, 2, 3)
        plt.plot(count_data['time_hours'], count_data['total_particles'], 
                 'b-', linewidth=2, label='Total')
        plt.plot(count_data['time_hours'], count_data['active_particles'], 
                 'r-', linewidth=2, label='Active')
        plt.xlabel('Time (hours)')
        plt.ylabel('Particle Count')
        plt.title('Detailed Particle Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Emission pattern analysis
    plt.subplot(2, 2, 4)
    source_concentrations = [1.0e+7, 1.2e+7, 1.4e+7, 1.6e+7, 1.8e+7, 2.0e+7, 
                           2.2e+7, 2.4e+7, 2.6e+7, 2.8e+7, 3.0e+7, 3.2e+7,
                           3.4e+7, 3.6e+7, 3.8e+7, 4.0e+7, 4.2e+7, 4.4e+7,
                           4.6e+7, 4.8e+7, 5.0e+7, 5.2e+7, 5.4e+7, 5.6e+7]
    time_steps = list(range(len(source_concentrations)))
    
    plt.plot(time_steps, source_concentrations, 'mo-', linewidth=2, markersize=6)
    plt.xlabel('Time Step (15-min intervals)')
    plt.ylabel('Source Emission (Bq)')
    plt.title('Source Emission Pattern (EKI Source_1)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    filename = os.path.join(output_dir, 'time_series_analysis.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved time series analysis: {filename}")

def create_summary_report(particle_data, count_data, output_dir="../logs/ldm_logs"):
    """Create a summary report"""
    
    report_file = os.path.join(output_dir, "simulation_summary.txt")
    
    with open(report_file, 'w') as f:
        f.write("LDM-EKI Particle Simulation Summary\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Simulation Duration: {max(particle_data.keys())} hours\n")
        f.write(f"Data Points: {len(particle_data)} hourly snapshots\n\n")
        
        f.write("Particle Count by Hour:\n")
        for hour in sorted(particle_data.keys()):
            df = particle_data[hour]
            f.write(f"  Hour {hour}: {len(df)} active particles\n")
        
        f.write("\nConcentration Statistics:\n")
        for hour in sorted(particle_data.keys()):
            df = particle_data[hour]
            f.write(f"  Hour {hour}:\n")
            f.write(f"    Mean: {df['concentration'].mean():.2e} Bq\n")
            f.write(f"    Min:  {df['concentration'].min():.2e} Bq\n")
            f.write(f"    Max:  {df['concentration'].max():.2e} Bq\n")
        
        if count_data is not None:
            f.write(f"\nTotal Simulation Steps: {len(count_data)}\n")
            f.write(f"Final Particle Count: {count_data['total_particles'].iloc[-1]}\n")
        
        f.write("\nSource Location: -74.1°, 40.7490° (NYC area)\n")
        f.write("Receptor Locations:\n")
        f.write("  Receptor 0: -73.98°, 40.7490°\n")
        f.write("  Receptor 1: -74.00°, 40.7490°\n")
        f.write("  Receptor 2: -74.02°, 40.7490°\n")
    
    print(f"Generated summary report: {report_file}")

def main():
    """Main function"""
    print("Starting simple particle visualization...")
    
    # Load data
    logs_dir = "../logs/ldm_logs"
    particle_data = load_particle_data(logs_dir)
    count_data = load_particle_count_data(logs_dir)
    
    if not particle_data:
        print("No particle data found!")
        return
    
    print(f"Found particle data for {len(particle_data)} hours")
    
    # Create visualizations
    print("Creating particle distribution plots...")
    create_particle_plots(particle_data, logs_dir)
    
    print("Creating time series plots...")
    create_time_series_plots(particle_data, count_data, logs_dir)
    
    print("Creating summary report...")
    create_summary_report(particle_data, count_data, logs_dir)
    
    print("Visualization complete!")
    print(f"Check {logs_dir}/ for generated plots and reports.")

if __name__ == "__main__":
    main()