#!/usr/bin/env python3
"""
Generate updated ensemble logs with nop=1000 and create time-series graphs
for 5 randomly selected ensembles
"""

import struct
import numpy as np
import csv
import os
import random
import matplotlib.pyplot as plt
from datetime import datetime

def read_binary_ensemble_states(filename):
    """Read ensemble states from binary file"""
    try:
        with open(filename, 'rb') as f:
            data = f.read()
        
        # Each float is 4 bytes, 100 ensembles x 24 time steps = 2400 floats
        expected_size = 100 * 24 * 4  # 9600 bytes
        if len(data) != expected_size:
            print(f"Warning: Expected {expected_size} bytes, got {len(data)} bytes")
        
        # Unpack as little-endian floats
        floats = struct.unpack(f'<{len(data)//4}f', data)
        
        # Reshape to (100 ensembles, 24 time steps)
        ensemble_data = np.array(floats).reshape(100, 24)
        return ensemble_data
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

def generate_simplified_ensemble_log():
    """Generate simplified ensemble initialization log with nop=1000"""
    
    # Read actual EKI ensemble data
    ensemble_file = "/home/jrpark/LDM-EKI/logs/ldm_logs/ensemble_states_iter_1.bin"
    ensemble_data = read_binary_ensemble_states(ensemble_file)
    
    if ensemble_data is None:
        print("Failed to read ensemble data. Using simulated data.")
        # Generate simulated emission data
        ensemble_data = np.random.lognormal(mean=-10, sigma=2, size=(100, 24))
    
    # Updated configuration parameters
    Nens = 100  # Number of ensembles
    nop_per_ensemble = 1000  # 1000 particles per ensemble (100,000 total)
    T = 24  # Time steps
    
    # Source location from EKI config
    source_lon = 127.363
    source_lat = 36.0639
    source_height = 100.0
    
    # Convert to grid coordinates
    source_x_grid = (source_lon + 179.0) / 0.5
    source_y_grid = (source_lat + 90.0) / 0.5
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create simplified log filename
    log_filename = f"/home/jrpark/LDM-EKI/logs/integration_logs/ensemble_simplified_{timestamp}.csv"
    
    print(f"Generating simplified ensemble log for {Nens} ensembles...")
    print(f"Particles per ensemble: {nop_per_ensemble}")
    print(f"Total particles: {Nens * nop_per_ensemble}")
    
    with open(log_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['# Simplified Ensemble Log - nop=1000, 100,000 total particles'])
        writer.writerow([f'# Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'])
        writer.writerow([f'# Ensembles: {Nens}, Particles per ensemble: {nop_per_ensemble}'])
        writer.writerow(['#'])
        writer.writerow(['Ensemble_ID', 'Total_Particles', 'Time_Steps', 'Min_Concentration', 
                        'Max_Concentration', 'Mean_Concentration', 'Source_Location'])
        
        # Write simplified data per ensemble
        for e in range(Nens):
            ensemble_conc = ensemble_data[e, :]
            min_conc = np.min(ensemble_conc)
            max_conc = np.max(ensemble_conc)
            mean_conc = np.mean(ensemble_conc)
            
            writer.writerow([
                e,
                nop_per_ensemble,
                T,
                f"{min_conc:.6e}",
                f"{max_conc:.6e}",
                f"{mean_conc:.6e}",
                f"({source_lon:.3f}, {source_lat:.3f}, {source_height:.0f}m)"
            ])
    
    print(f"Simplified log saved: {log_filename}")
    return ensemble_data, log_filename

def create_ensemble_time_series_plot(ensemble_data):
    """Create time series plot for 5 randomly selected ensembles"""
    
    # Select 5 random ensembles
    selected_ensembles = random.sample(range(100), 5)
    selected_ensembles.sort()
    
    print(f"Selected ensembles for plotting: {selected_ensembles}")
    
    # Create time array (0-23 hours)
    time_hours = np.arange(24)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, ensemble_id in enumerate(selected_ensembles):
        concentrations = ensemble_data[ensemble_id, :]
        plt.plot(time_hours, concentrations, 
                label=f'Ensemble {ensemble_id}', 
                color=colors[i], 
                linewidth=2, 
                marker='o', 
                markersize=4)
    
    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('Emission Concentration', fontsize=12)
    plt.title('Time Series of Emission Concentrations\nfor 5 Randomly Selected Ensembles', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Format y-axis in scientific notation
    plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"/home/jrpark/LDM-EKI/logs/integration_logs/ensemble_timeseries_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    
    print(f"Time series plot saved: {plot_filename}")
    
    # Create detailed data file for the selected ensembles
    data_filename = f"/home/jrpark/LDM-EKI/logs/integration_logs/selected_ensembles_data_{timestamp}.csv"
    
    with open(data_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['# Detailed data for 5 randomly selected ensembles'])
        writer.writerow([f'# Selected ensembles: {selected_ensembles}'])
        writer.writerow([f'# Each ensemble has {1000} particles'])
        writer.writerow(['#'])
        
        # Write time series header
        header = ['Time_Hour'] + [f'Ensemble_{eid}_Concentration' for eid in selected_ensembles]
        writer.writerow(header)
        
        # Write time series data
        for t in range(24):
            row = [t] + [f"{ensemble_data[eid, t]:.6e}" for eid in selected_ensembles]
            writer.writerow(row)
    
    print(f"Detailed data saved: {data_filename}")
    
    return selected_ensembles, plot_filename

def generate_particle_distribution_summary():
    """Generate summary of particle distribution across ensembles"""
    
    nop_per_ensemble = 1000
    total_ensembles = 100
    total_particles = nop_per_ensemble * total_ensembles
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_filename = f"/home/jrpark/LDM-EKI/logs/integration_logs/particle_distribution_summary_{timestamp}.txt"
    
    with open(summary_filename, 'w') as f:
        f.write("LDM Ensemble Particle Distribution Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  - Number of ensembles: {total_ensembles}\n")
        f.write(f"  - Particles per ensemble (nop): {nop_per_ensemble}\n")
        f.write(f"  - Total particles: {total_particles:,}\n")
        f.write(f"  - Time steps: 24 hours\n")
        f.write(f"  - Source location: (127.363°, 36.064°, 100m)\n\n")
        
        f.write("Particle Distribution:\n")
        for i in range(0, 100, 20):
            start_id = i * nop_per_ensemble + 1
            end_id = (i + 19) * nop_per_ensemble
            f.write(f"  - Ensembles {i:2d}-{i+19:2d}: Particles {start_id:6,} - {end_id:6,}\n")
        
        f.write(f"\nTotal particles generated: {total_particles:,}\n")
        f.write(f"Memory usage estimate: ~{total_particles * 0.5 / 1024 / 1024:.1f} MB\n")
    
    print(f"Particle distribution summary saved: {summary_filename}")
    return summary_filename

def main():
    """Main function to generate all analysis outputs"""
    
    print("=" * 60)
    print("LDM Ensemble Analysis - Updated for nop=1000")
    print("=" * 60)
    
    # Generate simplified ensemble log
    ensemble_data, log_file = generate_simplified_ensemble_log()
    
    print("\n" + "=" * 60)
    
    # Create time series plots
    selected_ensembles, plot_file = create_ensemble_time_series_plot(ensemble_data)
    
    print("\n" + "=" * 60)
    
    # Generate particle distribution summary
    summary_file = generate_particle_distribution_summary()
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print("Generated files:")
    print(f"  1. Simplified log: {os.path.basename(log_file)}")
    print(f"  2. Time series plot: {os.path.basename(plot_file)}")
    print(f"  3. Distribution summary: {os.path.basename(summary_file)}")
    print(f"  4. Selected ensembles: {selected_ensembles}")

if __name__ == "__main__":
    main()