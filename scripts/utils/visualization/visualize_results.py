#!/usr/bin/env python3
"""
LDM-CRAM4 Integrated Visualization Script
Automatically generates appropriate plots based on nuclides_config_60.txt file changes
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import os
import shutil

warnings.filterwarnings('ignore')

def clear_output_directory():
    """Clear the cram_result directory before creating new outputs"""
    if os.path.exists('cram_result'):
        shutil.rmtree('cram_result')
    os.makedirs('cram_result', exist_ok=True)

def read_nuclide_config():
    """Read nuclide configuration and identify enhanced nuclides"""
    config_path = 'data/input/nuclides_config_60.txt'
    enhanced_nuclides = []
    
    if not os.path.exists(config_path):
        return enhanced_nuclides
    
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split(',')
                if len(parts) >= 3:
                    nuclide = parts[0].strip()
                    try:
                        concentration = float(parts[2].strip())
                        if concentration > 0.1:  # Enhanced nuclides
                            enhanced_nuclides.append(nuclide)
                    except ValueError:
                        continue
    
    return enhanced_nuclides

def read_validation_data():
    """Read validation data from nuclide_totals.csv"""
    file_path = 'validation/nuclide_totals.csv'
    if not os.path.exists(file_path):
        return None
    
    return pd.read_csv(file_path)

def create_heatmap_from_real_data():
    """Create heatmap from real LDM simulation data"""
    print("Creating heatmap from real LDM data...")
    
    df = read_validation_data()
    if df is None:
        print("No validation data found")
        return
    
    enhanced_nuclides = read_nuclide_config()
    
    # Filter for only enhanced nuclides
    enhanced_cols = [col for col in df.columns if col != 'Time' and any(nuc in col for nuc in enhanced_nuclides)]
    
    if not enhanced_cols:
        print("No significant nuclides found for heatmap")
        return
    
    # Create time series data for heatmap
    time_hours = df['time(s)'] / 3600  # Convert to hours
    
    # Prepare data matrix
    heatmap_data = []
    nuclide_names = []
    
    for col in enhanced_cols:
        values = df[col].values
        if np.max(values) > 1.0:  # Only include significant concentrations
            heatmap_data.append(values)
            # Clean nuclide name
            name = col.replace('Total_', '').replace('_', '-')
            nuclide_names.append(name)
    
    if not heatmap_data:
        print("No significant nuclides found for heatmap")
        return
    
    heatmap_data = np.array(heatmap_data)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    im = plt.imshow(heatmap_data, aspect='auto', cmap='turbo', interpolation='bilinear')
    
    # Set up axes
    time_ticks = np.linspace(0, len(time_hours)-1, 10)
    time_labels = [f"{time_hours[int(i)]:.1f}h" for i in time_ticks]
    plt.xticks(time_ticks, time_labels)
    plt.yticks(range(len(nuclide_names)), nuclide_names)
    
    plt.xlabel('Time')
    plt.ylabel('Nuclides')
    plt.title('LDM-CRAM4: Multi-nuclide Concentration Heatmap')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Total Concentration')
    
    plt.tight_layout()
    plt.savefig('cram_result/ldm_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_stacked_plot_from_real_data():
    """Create stacked area plot from real LDM simulation data"""
    print("Creating stacked plot from real LDM data...")
    
    df = read_validation_data()
    if df is None:
        print("No validation data found")
        return
    
    enhanced_nuclides = read_nuclide_config()
    print(f"Enhanced nuclides from config: {enhanced_nuclides}")
    
    # Filter for enhanced nuclides
    enhanced_cols = [col for col in df.columns if col not in ['timestep', 'time(s)', 'active_particles', 'total_conc'] and any(nuc in col for nuc in enhanced_nuclides)]
    
    if not enhanced_cols:
        print("No significant nuclides found for stacked plot")
        return
    
    # Convert time to hours
    time_hours = df['time(s)'] / 3600
    
    # Prepare data for stacking
    plot_data = []
    plot_labels = []
    
    for col in enhanced_cols:
        values = df[col].values
        if np.max(values) > 1.0:  # Only significant concentrations
            plot_data.append(values)
            # Clean label
            label = col.replace('Total_', '').replace('_', '-')
            # Check if this is an enhanced nuclide
            is_enhanced = any(nuc in col for nuc in enhanced_nuclides)
            if is_enhanced:
                label += " (Enhanced)"
            plot_labels.append(label)
    
    if not plot_data:
        print("No significant nuclides found for stacked plot")
        return
    
    plot_data = np.array(plot_data)
    
    # Sort by maximum concentration (fastest decaying at bottom)
    max_concentrations = np.max(plot_data, axis=1)
    sort_indices = np.argsort(max_concentrations)
    plot_data = plot_data[sort_indices]
    plot_labels = [plot_labels[i] for i in sort_indices]
    
    # Create stacked plot
    plt.figure(figsize=(12, 8))
    
    # Use distinct colors for enhanced nuclides
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
    
    plt.stackplot(time_hours, *plot_data, labels=plot_labels, colors=colors[:len(plot_data)], alpha=0.7)
    
    plt.xlabel('Time (hours)')
    plt.ylabel('Total Concentration (all particles)')
    plt.title('LDM-CRAM4: Multi-nuclide Concentration Stack')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cram_result/ldm_stacked.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_report():
    """Create a summary report of the simulation"""
    print("Creating summary report...")
    
    df = read_validation_data()
    if df is None:
        return
    
    enhanced_nuclides = read_nuclide_config()
    print(f"Enhanced nuclides from config: {enhanced_nuclides}")
    
    # Calculate summary statistics
    summary_data = []
    
    for col in df.columns:
        if col not in ['timestep', 'time(s)', 'active_particles', 'total_conc']:
            values = df[col].values
            max_conc = np.max(values)
            final_conc = values[-1] if len(values) > 0 else 0
            
            if max_conc > 0.1:  # Only significant concentrations
                nuclide_name = col.replace('Total_', '').replace('_', '-')
                is_enhanced = any(nuc in col for nuc in enhanced_nuclides)
                
                summary_data.append({
                    'Nuclide': nuclide_name,
                    'Enhanced': is_enhanced,
                    'Max_Concentration': max_conc,
                    'Final_Concentration': final_conc,
                    'Concentration_Change': final_conc - values[0] if len(values) > 0 else 0
                })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Max_Concentration', ascending=False)
        summary_df.to_csv('cram_result/simulation_summary.csv', index=False)

def main():
    """Main execution function"""
    print("Starting LDM-CRAM4 integrated visualization...")
    print("=" * 50)
    
    # Clear output directory first
    clear_output_directory()
    
    # Create visualizations
    create_heatmap_from_real_data()
    create_stacked_plot_from_real_data()
    create_summary_report()
    
    print("=" * 50)
    print("Completed! Generated files:")
    
    # Check and report generated files
    files_to_check = [
        'cram_result/ldm_heatmap.png',
        'cram_result/ldm_stacked.png', 
        'cram_result/simulation_summary.csv'
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")

if __name__ == "__main__":
    main()