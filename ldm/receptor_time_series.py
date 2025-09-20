#!/usr/bin/env python3
"""
15분 간격 리셉터별 농도 측정 시계열 플롯
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def load_particle_data(logs_dir="../logs/ldm_logs"):
    """Load particle position data from CSV files"""
    particle_files = glob.glob(os.path.join(logs_dir, "particles_15min_*.csv"))
    particle_data = {}
    
    for file in sorted(particle_files):
        filename = os.path.basename(file)
        quarter_hour = int(filename.split('_')[2].split('.')[0])
        
        try:
            df = pd.read_csv(file)
            if not df.empty:
                particle_data[quarter_hour] = df
                print(f"Loaded {len(df)} particles for 15-min interval {quarter_hour}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return particle_data

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate approximate distance in degrees between two points"""
    return np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)

def analyze_receptor_time_series(particle_data, output_dir="../logs/ldm_logs"):
    """Analyze receptor measurements with 15-minute time resolution"""
    
    # Receptor locations (from EKI config)
    receptors = [
        {"id": 1, "lat": 40.7490, "lon": -73.98, "name": "Receptor 1"},
        {"id": 2, "lat": 40.7490, "lon": -74.00, "name": "Receptor 2"}, 
        {"id": 3, "lat": 40.7490, "lon": -74.02, "name": "Receptor 3"}
    ]
    
    radius = 10.0  # 10.0 degree radius - capture all particles
    
    # Store results for each receptor
    receptor_results = {r["id"]: {"times": [], "particle_counts": [], "total_concentration": [], "avg_concentration": []} 
                       for r in receptors}
    
    # Analyze each 15-minute interval
    for quarter_hour, df in sorted(particle_data.items()):
        if df.empty:
            continue
            
        print(f"Analyzing 15-min interval {quarter_hour}...")
        
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
            
            # Store results (convert to hours for plotting)
            time_in_hours = quarter_hour * 0.25
            receptor_results[rec_id]["times"].append(time_in_hours)
            receptor_results[rec_id]["particle_counts"].append(particle_count)
            receptor_results[rec_id]["total_concentration"].append(total_conc)
            receptor_results[rec_id]["avg_concentration"].append(avg_conc)
            
            print(f"  Receptor {rec_id}: {particle_count} particles, Total: {total_conc:.2e} Bq, Avg: {avg_conc:.2e} Bq")
    
    return receptor_results, receptors

def create_detailed_time_series_plots(receptor_results, receptors, output_dir="../logs/ldm_logs"):
    """Create detailed 15-minute resolution time series plots"""
    
    colors = ['blue', 'red', 'green']
    
    # Create comprehensive 4-panel plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
    
    # Plot 1: Particle count over time (15-min resolution)
    for i, receptor in enumerate(receptors):
        rec_id = receptor["id"]
        times = receptor_results[rec_id]["times"]
        counts = receptor_results[rec_id]["particle_counts"]
        
        ax1.plot(times, counts, 'o-', color=colors[i], linewidth=2, markersize=4, 
                label=f'{receptor["name"]} (Lon: {receptor["lon"]})')
    
    ax1.set_xlabel('Time (hours)', fontsize=12)
    ax1.set_ylabel('Particle Count (within 10° radius)', fontsize=12)
    ax1.set_title('Particle Count at Receptors Over Time (15-min resolution)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(np.arange(0, 6.25, 0.25))
    
    # Plot 2: Total concentration over time (15-min resolution)
    for i, receptor in enumerate(receptors):
        rec_id = receptor["id"]
        times = receptor_results[rec_id]["times"]
        total_conc = receptor_results[rec_id]["total_concentration"]
        
        ax2.semilogy(times, total_conc, 'o-', color=colors[i], linewidth=2, markersize=4,
                    label=f'{receptor["name"]} (Lon: {receptor["lon"]})')
    
    ax2.set_xlabel('Time (hours)', fontsize=12)
    ax2.set_ylabel('Total Concentration (Bq)', fontsize=12)
    ax2.set_title('Total Concentration at Receptors Over Time (15-min resolution)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(np.arange(0, 6.25, 0.25))
    
    # Plot 3: Average concentration over time (15-min resolution)
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
                        linewidth=2, markersize=4, label=f'{receptor["name"]} (Lon: {receptor["lon"]})')
    
    ax3.set_xlabel('Time (hours)', fontsize=12)
    ax3.set_ylabel('Average Concentration (Bq)', fontsize=12)
    ax3.set_title('Average Concentration at Receptors Over Time (15-min resolution)', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(np.arange(0, 6.25, 0.25))
    
    # Plot 4: Concentration rate of change
    for i, receptor in enumerate(receptors):
        rec_id = receptor["id"]
        times = receptor_results[rec_id]["times"]
        avg_conc = receptor_results[rec_id]["avg_concentration"]
        
        if len(times) > 1:
            # Calculate rate of change (difference between consecutive points)
            times_diff = times[1:]
            conc_diff = np.diff(avg_conc)
            time_step = 0.25  # 15 minutes = 0.25 hours
            rate_of_change = conc_diff / time_step  # Bq per hour
            
            ax4.plot(times_diff, rate_of_change, 'o-', color=colors[i], 
                    linewidth=2, markersize=4, label=f'{receptor["name"]} (Lon: {receptor["lon"]})')
    
    ax4.set_xlabel('Time (hours)', fontsize=12)
    ax4.set_ylabel('Concentration Rate of Change (Bq/hour)', fontsize=12)
    ax4.set_title('Concentration Rate of Change at Receptors (15-min resolution)', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(np.arange(0, 6.25, 0.25))
    ax4.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    
    # Save plot
    filename = os.path.join(output_dir, 'receptor_15min_time_series.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved detailed 15-min time series plot: {filename}")

def main():
    """Main function"""
    print("Starting 15-minute resolution receptor time series analysis...")
    print("Analyzing particles within 10.0 degree radius of each receptor (capturing all particles)")
    
    # Load data
    logs_dir = "../logs/ldm_logs"
    particle_data = load_particle_data(logs_dir)
    
    if not particle_data:
        print("No particle data found!")
        return
    
    print(f"Found particle data for {len(particle_data)} 15-minute intervals")
    
    # Analyze receptor measurements
    print("\\nAnalyzing receptor measurements...")
    receptor_results, receptors = analyze_receptor_time_series(particle_data, logs_dir)
    
    # Create detailed plots
    print("\\nCreating detailed 15-minute time series plots...")
    create_detailed_time_series_plots(receptor_results, receptors, logs_dir)
    
    print("\\n15-minute receptor time series analysis complete!")
    print(f"Check {logs_dir}/ for generated plots.")

if __name__ == "__main__":
    main()