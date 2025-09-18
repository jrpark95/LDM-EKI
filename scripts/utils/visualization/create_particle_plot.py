#!/usr/bin/env python3
"""
Simple particle count visualization for LDM-CRAM4
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil

def clear_output_directory():
    """Clear the cram_result directory before creating new outputs"""
    if os.path.exists('cram_result'):
        shutil.rmtree('cram_result')
    os.makedirs('cram_result', exist_ok=True)

def create_particle_plots():
    """Create particle count and time series plots"""
    # Read validation data
    df = pd.read_csv('validation/nuclide_totals.csv')
    
    time_hours = df['time(s)'] / 3600
    active_particles = df['active_particles']
    max_particles = 10000
    
    # Create particle count plot
    plt.figure(figsize=(12, 6))
    plt.plot(time_hours, active_particles, 'b-', linewidth=2, label='Active Particles')
    plt.fill_between(time_hours, active_particles, alpha=0.3, color='blue')
    
    plt.xlabel('Time (hours)')
    plt.ylabel('Number of Active Particles')
    plt.title('LDM-CRAM4: Active Particle Count Over Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add percentage annotations at key points
    for i in range(0, len(time_hours), max(1, len(time_hours)//8)):
        pct = (active_particles.iloc[i] / max_particles) * 100
        plt.annotate(f'{pct:.1f}%', 
                    xy=(time_hours.iloc[i], active_particles.iloc[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, alpha=0.8)
    
    plt.tight_layout()
    plt.savefig('cram_result/ldm_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create stacked-style plot showing particle growth
    plt.figure(figsize=(12, 8))
    
    # Create cumulative data
    plt.fill_between(time_hours, 0, active_particles, 
                     color='steelblue', alpha=0.7, label='Active Particles')
    plt.fill_between(time_hours, active_particles, max_particles,
                     color='lightgray', alpha=0.5, label='Inactive Particles')
    
    plt.xlabel('Time (hours)')
    plt.ylabel('Particle Count')
    plt.title('LDM-CRAM4: Particle Status Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add total particle line
    plt.axhline(y=max_particles, color='red', linestyle='--', alpha=0.7, label='Total Particles')
    
    plt.tight_layout()
    plt.savefig('cram_result/ldm_stacked.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary
    summary_data = [{
        'Metric': 'Initial Active Particles',
        'Value': active_particles.iloc[0]
    }, {
        'Metric': 'Final Active Particles', 
        'Value': active_particles.iloc[-1]
    }, {
        'Metric': 'Final Percentage Active',
        'Value': f"{(active_particles.iloc[-1] / max_particles * 100):.1f}%"
    }, {
        'Metric': 'Simulation Duration',
        'Value': f"{time_hours.iloc[-1]:.1f} hours"
    }]
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('cram_result/simulation_summary.csv', index=False)
    
    print("Particle visualization plots created successfully!")

if __name__ == "__main__":
    clear_output_directory()
    create_particle_plots()