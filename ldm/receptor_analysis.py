#!/usr/bin/env python3
"""
Receptor-based Particle Analysis
Analyzes particles within 0.1 degree radius of each receptor over time
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate approximate distance in degrees between two points"""
    return np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)

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

def analyze_receptor_measurements(particle_data, output_dir="../logs/ldm_logs"):
    """Analyze particles within 0.1 degree radius of each receptor"""
    
    # Receptor locations (from EKI config)
    receptors = [
        {"id": 1, "lat": 40.7490, "lon": -73.98, "name": "Receptor 1"},
        {"id": 2, "lat": 40.7490, "lon": -74.00, "name": "Receptor 2"}, 
        {"id": 3, "lat": 40.7490, "lon": -74.02, "name": "Receptor 3"}
    ]
    
    radius = 0.1  # 0.1 degree radius
    
    # Store results for each receptor
    receptor_results = {r["id"]: {"times": [], "particle_counts": [], "total_concentration": [], "avg_concentration": []} 
                       for r in receptors}
    
    # Analyze each hour
    for hour, df in sorted(particle_data.items()):
        if df.empty:
            continue
            
        print(f"Analyzing hour {hour}...")
        
        for receptor in receptors:
            rec_lat, rec_lon = receptor["lat"], receptor["lon"]
            rec_id = receptor["id"]
            
            # Calculate distances from receptor
            distances = calculate_distance(df['latitude'], df['longitude'], rec_lat, rec_lon)
            
            # Find particles within radius
            within_radius = distances <= radius
            particles_in_range = df[within_radius]
            
            # Calculate measurements
            particle_count = len(particles_in_range)
            total_conc = particles_in_range['concentration'].sum() if particle_count > 0 else 0
            avg_conc = particles_in_range['concentration'].mean() if particle_count > 0 else 0
            
            # Store results
            receptor_results[rec_id]["times"].append(hour)
            receptor_results[rec_id]["particle_counts"].append(particle_count)
            receptor_results[rec_id]["total_concentration"].append(total_conc)
            receptor_results[rec_id]["avg_concentration"].append(avg_conc)
            
            print(f"  Receptor {rec_id}: {particle_count} particles, Total: {total_conc:.2e} Bq, Avg: {avg_conc:.2e} Bq")
    
    return receptor_results, receptors

def create_receptor_plots(receptor_results, receptors, output_dir="../logs/ldm_logs"):
    """Create time series plots for receptor measurements"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create combined plot with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = ['blue', 'red', 'green']
    
    # Plot 1: Particle counts over time
    for i, receptor in enumerate(receptors):
        rec_id = receptor["id"]
        times = receptor_results[rec_id]["times"]
        counts = receptor_results[rec_id]["particle_counts"]
        
        ax1.plot(times, counts, 'o-', color=colors[i], linewidth=2, markersize=8, 
                label=f'{receptor["name"]} (Lon: {receptor["lon"]})')
    
    ax1.set_xlabel('Time (hours)', fontsize=12)
    ax1.set_ylabel('Particle Count (within 0.1° radius)', fontsize=12)
    ax1.set_title('Particle Count at Receptors Over Time', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(1, 7))
    
    # Plot 2: Total concentration over time
    for i, receptor in enumerate(receptors):
        rec_id = receptor["id"]
        times = receptor_results[rec_id]["times"]
        total_conc = receptor_results[rec_id]["total_concentration"]
        
        ax2.semilogy(times, total_conc, 'o-', color=colors[i], linewidth=2, markersize=8,
                    label=f'{receptor["name"]} (Lon: {receptor["lon"]})')
    
    ax2.set_xlabel('Time (hours)', fontsize=12)
    ax2.set_ylabel('Total Concentration (Bq)', fontsize=12)
    ax2.set_title('Total Concentration at Receptors Over Time', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(1, 7))
    
    # Plot 3: Average concentration over time
    for i, receptor in enumerate(receptors):
        rec_id = receptor["id"]
        times = receptor_results[rec_id]["times"]
        avg_conc = receptor_results[rec_id]["avg_concentration"]
        
        # Filter out zero values for log plot
        non_zero_mask = np.array(avg_conc) > 0
        times_filtered = np.array(times)[non_zero_mask]
        avg_conc_filtered = np.array(avg_conc)[non_zero_mask]
        
        if len(avg_conc_filtered) > 0:
            ax3.semilogy(times_filtered, avg_conc_filtered, 'o-', color=colors[i], 
                        linewidth=2, markersize=8, label=f'{receptor["name"]} (Lon: {receptor["lon"]})')
    
    ax3.set_xlabel('Time (hours)', fontsize=12)
    ax3.set_ylabel('Average Concentration (Bq)', fontsize=12)
    ax3.set_title('Average Concentration at Receptors Over Time', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(range(1, 7))
    
    # Plot 4: Cumulative particle detection
    for i, receptor in enumerate(receptors):
        rec_id = receptor["id"]
        times = receptor_results[rec_id]["times"]
        counts = receptor_results[rec_id]["particle_counts"]
        
        cumulative_counts = np.cumsum(counts)
        ax4.plot(times, cumulative_counts, 'o-', color=colors[i], linewidth=2, markersize=8,
                label=f'{receptor["name"]} (Lon: {receptor["lon"]})')
    
    ax4.set_xlabel('Time (hours)', fontsize=12)
    ax4.set_ylabel('Cumulative Particle Count', fontsize=12)
    ax4.set_title('Cumulative Particle Detection at Receptors', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(range(1, 7))
    
    plt.tight_layout()
    
    # Save plot
    filename = os.path.join(output_dir, 'receptor_measurements_over_time.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved receptor analysis plot: {filename}")
    
    # Create summary table
    create_summary_table(receptor_results, receptors, output_dir)

def create_summary_table(receptor_results, receptors, output_dir):
    """Create a summary table of receptor measurements"""
    
    summary_data = []
    
    for receptor in receptors:
        rec_id = receptor["id"]
        times = receptor_results[rec_id]["times"]
        counts = receptor_results[rec_id]["particle_counts"]
        total_conc = receptor_results[rec_id]["total_concentration"]
        avg_conc = receptor_results[rec_id]["avg_concentration"]
        
        # Calculate statistics
        max_particles = max(counts) if counts else 0
        total_particles = sum(counts) if counts else 0
        peak_concentration = max(total_conc) if total_conc else 0
        peak_time = times[total_conc.index(peak_concentration)] if total_conc and peak_concentration > 0 else 0
        
        summary_data.append({
            'Receptor': receptor["name"],
            'Longitude': receptor["lon"],
            'Total Particles Detected': total_particles,
            'Max Particles (single hour)': max_particles,
            'Peak Total Concentration (Bq)': f"{peak_concentration:.2e}",
            'Peak Time (hour)': peak_time
        })
    
    # Save summary as CSV
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, 'receptor_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved receptor summary: {summary_file}")
    
    return summary_df

def main():
    """Main function"""
    print("Starting receptor-based particle analysis...")
    print("Analyzing particles within 0.1 degree radius of each receptor")
    
    # Load data
    logs_dir = "../logs/ldm_logs"
    particle_data = load_particle_data(logs_dir)
    
    if not particle_data:
        print("No particle data found!")
        return
    
    print(f"Found particle data for {len(particle_data)} hours")
    
    # Analyze receptor measurements
    print("Analyzing receptor measurements...")
    receptor_results, receptors = analyze_receptor_measurements(particle_data, logs_dir)
    
    # Create plots
    print("Creating receptor analysis plots...")
    create_receptor_plots(receptor_results, receptors, logs_dir)
    
    print("Receptor analysis complete!")
    print(f"Check {logs_dir}/ for generated plots and summary.")

if __name__ == "__main__":
    main()