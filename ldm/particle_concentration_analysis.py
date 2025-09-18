#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def analyze_particle_concentrations():
    """
    Analyze particle concentrations over time to check if they follow the expected linear pattern
    """
    log_dir = "/home/jrpark/LDM-EKI/logs/ldm_logs"
    output_dir = log_dir
    
    # Find all particle CSV files
    particle_files = sorted(glob.glob(os.path.join(log_dir, "particles_hour_*.csv")))
    
    if not particle_files:
        print("No particle CSV files found!")
        return
    
    hours = []
    mean_concentrations = []
    median_concentrations = []
    std_concentrations = []
    min_concentrations = []
    max_concentrations = []
    particle_counts = []
    
    print("Analyzing particle concentrations over time...")
    
    for file_path in particle_files:
        # Extract hour from filename
        filename = os.path.basename(file_path)
        hour = int(filename.split("_hour_")[1].split(".csv")[0])
        
        # Read particle data
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                continue
                
            concentrations = df['concentration'].values
            
            hours.append(hour)
            mean_concentrations.append(np.mean(concentrations))
            median_concentrations.append(np.median(concentrations))
            std_concentrations.append(np.std(concentrations))
            min_concentrations.append(np.min(concentrations))
            max_concentrations.append(np.max(concentrations))
            particle_counts.append(len(concentrations))
            
            print(f"Hour {hour}: {len(concentrations)} particles, Mean conc: {np.mean(concentrations):.3e} Bq")
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    if not hours:
        print("No valid data found!")
        return
    
    # Create comprehensive analysis plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Mean and Median concentrations over time
    ax1.plot(hours, mean_concentrations, 'b-o', label='Mean Concentration', linewidth=2, markersize=8)
    ax1.plot(hours, median_concentrations, 'r-s', label='Median Concentration', linewidth=2, markersize=6)
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Concentration (Bq)')
    ax1.set_title('Mean and Median Particle Concentrations Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Plot 2: Concentration range (min/max) over time
    ax2.fill_between(hours, min_concentrations, max_concentrations, alpha=0.3, color='gray', label='Min-Max Range')
    ax2.plot(hours, mean_concentrations, 'b-o', label='Mean', linewidth=2)
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Concentration (Bq)')
    ax2.set_title('Concentration Range Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Plot 3: Standard deviation over time
    ax3.plot(hours, std_concentrations, 'g-^', label='Standard Deviation', linewidth=2, markersize=8)
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Concentration Standard Deviation (Bq)')
    ax3.set_title('Concentration Variability Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Plot 4: Particle count vs mean concentration
    ax4.scatter(particle_counts, mean_concentrations, c=hours, cmap='turbo', s=100, alpha=0.7)
    ax4.set_xlabel('Number of Particles')
    ax4.set_ylabel('Mean Concentration (Bq)')
    ax4.set_title('Mean Concentration vs Particle Count')
    cbar = plt.colorbar(ax4.scatter(particle_counts, mean_concentrations, c=hours, cmap='turbo', s=100, alpha=0.7), ax=ax4)
    cbar.set_label('Time (hours)')
    ax4.grid(True, alpha=0.3)
    ax4.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(output_dir, 'particle_concentration_time_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Concentration analysis plot saved to: {output_file}")
    plt.close()
    
    # Create a detailed time series plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Linear scale
    ax1.plot(hours, mean_concentrations, 'b-o', label='Mean Concentration', linewidth=3, markersize=10)
    ax1.plot(hours, median_concentrations, 'r-s', label='Median Concentration', linewidth=2, markersize=8)
    
    # Add expected linear trend line
    if len(hours) > 1:
        # Calculate linear fit
        z = np.polyfit(hours, mean_concentrations, 1)
        p = np.poly1d(z)
        ax1.plot(hours, p(hours), 'k--', alpha=0.7, linewidth=2, label=f'Linear Fit (slope: {z[0]:.2e})')
    
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Concentration (Bq)')
    ax1.set_title('Particle Concentration Time Series (Linear Scale)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Plot 2: Log scale for better visualization
    ax2.semilogy(hours, mean_concentrations, 'b-o', label='Mean Concentration', linewidth=3, markersize=10)
    ax2.semilogy(hours, median_concentrations, 'r-s', label='Median Concentration', linewidth=2, markersize=8)
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Concentration (Bq) - Log Scale')
    ax2.set_title('Particle Concentration Time Series (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the time series plot
    output_file2 = os.path.join(output_dir, 'particle_concentration_time_series.png')
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"Time series plot saved to: {output_file2}")
    plt.close()
    
    # Print summary statistics
    print("\n=== Concentration Analysis Summary ===")
    print(f"Time range: {min(hours)} to {max(hours)} hours")
    print(f"Initial mean concentration: {mean_concentrations[0]:.3e} Bq")
    print(f"Final mean concentration: {mean_concentrations[-1]:.3e} Bq")
    print(f"Concentration change: {mean_concentrations[-1] - mean_concentrations[0]:.3e} Bq")
    print(f"Percentage change: {((mean_concentrations[-1] - mean_concentrations[0]) / mean_concentrations[0] * 100):.2f}%")
    
    # Check if the trend is linear
    if len(hours) > 2:
        correlation = np.corrcoef(hours, mean_concentrations)[0, 1]
        print(f"Linear correlation coefficient: {correlation:.4f}")
        if correlation > 0.9:
            print("✓ Strong positive linear trend observed")
        elif correlation > 0.7:
            print("○ Moderate positive linear trend observed")
        else:
            print("✗ Weak or no linear trend observed")

if __name__ == "__main__":
    analyze_particle_concentrations()