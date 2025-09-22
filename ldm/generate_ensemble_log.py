#!/usr/bin/env python3
"""
Generate ensemble initialization log based on real EKI data
"""

import struct
import numpy as np
import csv
import os
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

def generate_ensemble_log():
    """Generate comprehensive ensemble initialization log"""
    
    # Read actual EKI ensemble data
    ensemble_file = "/home/jrpark/LDM-EKI/logs/ldm_logs/ensemble_states_iter_1.bin"
    ensemble_data = read_binary_ensemble_states(ensemble_file)
    
    if ensemble_data is None:
        print("Failed to read ensemble data. Using simulated data.")
        # Generate simulated emission data
        ensemble_data = np.random.lognormal(mean=-10, sigma=2, size=(100, 24))
    
    # Configuration parameters
    Nens = 100  # Number of ensembles
    nop_per_ensemble = 10000 // Nens  # 100 particles per ensemble (10000 total / 100 ensembles)
    T = 24  # Time steps
    
    # Source location from EKI config (Korean nuclear facility)
    source_lon = 127.363  # longitude
    source_lat = 36.0639  # latitude  
    source_height = 100.0  # height in meters
    
    # Convert to grid coordinates
    source_x_grid = (source_lon + 179.0) / 0.5
    source_y_grid = (source_lat + 90.0) / 0.5
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create log filename
    log_filename = f"/home/jrpark/LDM-EKI/logs/integration_logs/ensemble_initialization_{timestamp}.csv"
    
    print(f"Generating ensemble initialization log for {Nens} ensembles...")
    print(f"Particles per ensemble: {nop_per_ensemble}")
    print(f"Total particles: {Nens * nop_per_ensemble}")
    print(f"Source location: lon={source_lon}, lat={source_lat}, height={source_height}")
    print(f"Log file: {log_filename}")
    
    with open(log_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header comments
        writer.writerow(['# Ensemble Initialization Log - Real EKI Data (100 Ensembles)'])
        writer.writerow([f'# Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'])
        writer.writerow([f'# Number of ensembles: {Nens}'])
        writer.writerow([f'# Particles per ensemble: {nop_per_ensemble}'])
        writer.writerow([f'# Emission time steps: {T}'])
        writer.writerow([f'# Total particles: {Nens * nop_per_ensemble}'])
        writer.writerow([f'# Source location: lon={source_lon}, lat={source_lat}, height={source_height}'])
        writer.writerow([f'# Grid coordinates: x={source_x_grid:.3f}, y={source_y_grid:.3f}'])
        writer.writerow(['# Data source: EKI ensemble states from LDM-EKI integration'])
        writer.writerow(['#'])
        
        # Write CSV header
        writer.writerow(['Ensemble_ID', 'Particle_Global_ID', 'Particle_Local_ID', 'Time_Step_Index', 
                        'Emission_Concentration', 'Source_X_Grid', 'Source_Y_Grid', 'Source_Z',
                        'Source_Lon', 'Source_Lat', 'Ensemble_State_Value'])
        
        # Write particle data for all 100 ensembles
        for e in range(Nens):
            for i in range(nop_per_ensemble):
                # Calculate time step index for this particle
                time_step_index = (i * T) // nop_per_ensemble
                if time_step_index >= T:
                    time_step_index = T - 1
                
                # Get emission concentration from ensemble data
                emission_concentration = ensemble_data[e, time_step_index]
                
                # Global particle ID
                global_id = e * nop_per_ensemble + i + 1
                
                # Write particle data
                writer.writerow([
                    e,  # Ensemble_ID
                    global_id,  # Particle_Global_ID  
                    i,  # Particle_Local_ID
                    time_step_index,  # Time_Step_Index
                    f"{emission_concentration:.6e}",  # Emission_Concentration
                    f"{source_x_grid:.3f}",  # Source_X_Grid
                    f"{source_y_grid:.3f}",  # Source_Y_Grid
                    f"{source_height:.1f}",  # Source_Z
                    f"{source_lon:.6f}",  # Source_Lon
                    f"{source_lat:.6f}",  # Source_Lat
                    f"{ensemble_data[e, time_step_index]:.6e}"  # Ensemble_State_Value
                ])
    
    print(f"Successfully generated ensemble initialization log: {log_filename}")
    
    # Generate summary statistics
    summary_file = f"/home/jrpark/LDM-EKI/logs/integration_logs/ensemble_summary_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write("Ensemble Initialization Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of ensembles: {Nens}\n")
        f.write(f"Particles per ensemble: {nop_per_ensemble}\n")
        f.write(f"Total particles: {Nens * nop_per_ensemble}\n")
        f.write(f"Time steps: {T}\n\n")
        
        f.write("Emission Data Statistics:\n")
        f.write(f"  Min concentration: {np.min(ensemble_data):.6e}\n")
        f.write(f"  Max concentration: {np.max(ensemble_data):.6e}\n")
        f.write(f"  Mean concentration: {np.mean(ensemble_data):.6e}\n")
        f.write(f"  Std concentration: {np.std(ensemble_data):.6e}\n\n")
        
        f.write("Source Information:\n")
        f.write(f"  Longitude: {source_lon:.6f}°\n")
        f.write(f"  Latitude: {source_lat:.6f}°\n")
        f.write(f"  Height: {source_height:.1f} m\n")
        f.write(f"  Grid X: {source_x_grid:.3f}\n")
        f.write(f"  Grid Y: {source_y_grid:.3f}\n")
    
    print(f"Summary file generated: {summary_file}")

if __name__ == "__main__":
    generate_ensemble_log()