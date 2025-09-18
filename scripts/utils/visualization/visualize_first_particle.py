#!/usr/bin/env python3
"""
First Particle 60-Nuclide Concentration Visualization
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

def clear_output_directory():
    """Clear the cram_result directory before creating new outputs"""
    if os.path.exists('cram_result'):
        shutil.rmtree('cram_result')
    os.makedirs('cram_result', exist_ok=True)

def visualize_first_particle_concentrations():
    """Create comprehensive visualizations of first particle's 60 nuclide concentrations"""
    
    # Read the first particle concentration data
    if not os.path.exists('first_particle_concentrations.csv'):
        print("Error: first_particle_concentrations.csv not found")
        return
    
    df = pd.read_csv('first_particle_concentrations.csv')
    
    # Convert time to hours
    time_hours = df['time(s)'] / 3600
    
    # Get all nuclide columns (exclude timestep, time, total_conc)
    nuclide_cols = [col for col in df.columns if col not in ['timestep', 'time(s)', 'total_conc']]
    
    # 1. Create initial concentration bar chart
    initial_concentrations = df.iloc[0][nuclide_cols]
    enhanced_nuclides = initial_concentrations[initial_concentrations > 0]
    
    if len(enhanced_nuclides) > 0:
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(enhanced_nuclides)), enhanced_nuclides.values)
        plt.xticks(range(len(enhanced_nuclides)), enhanced_nuclides.index, rotation=45)
        plt.ylabel('Initial Concentration')
        plt.title('First Particle: Initial Enhanced Nuclide Concentrations')
        plt.grid(True, alpha=0.3)
        
        # Color enhanced nuclides differently
        for i, bar in enumerate(bars):
            bar.set_color('red')
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{enhanced_nuclides.values[i]:.2f}', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('cram_result/initial_concentrations.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Create time series for enhanced nuclides
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    enhanced_nuclide_names = enhanced_nuclides.index.tolist() if len(enhanced_nuclides) > 0 else []
    
    for i, nuclide in enumerate(enhanced_nuclide_names[:4]):  # Show top 4
        ax = axes[i]
        concentrations = df[nuclide]
        
        ax.plot(time_hours, concentrations, 'o-', linewidth=2, markersize=4, label=nuclide)
        ax.fill_between(time_hours, concentrations, alpha=0.3)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Concentration')
        ax.set_title(f'{nuclide} Concentration Over Time')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Highlight initial non-zero concentration
        if len(concentrations) > 0 and concentrations.iloc[0] > 0:
            ax.axhline(y=concentrations.iloc[0], color='red', linestyle='--', alpha=0.5, 
                      label=f'Initial: {concentrations.iloc[0]:.3f}')
    
    # Hide unused subplots
    for i in range(len(enhanced_nuclide_names), 4):
        axes[i].axis('off')
    
    plt.suptitle('First Particle: Enhanced Nuclides Concentration Time Series', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cram_result/ldm_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Create stacked area plot showing decay
    plt.figure(figsize=(12, 8))
    
    if len(enhanced_nuclides) > 0:
        # Prepare data for stacking
        stack_data = []
        labels = []
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
        
        for i, nuclide in enumerate(enhanced_nuclide_names):
            values = df[nuclide].values
            stack_data.append(values)
            labels.append(f'{nuclide} (Enhanced)')
        
        if stack_data:
            plt.stackplot(time_hours, *stack_data, labels=labels, 
                         colors=colors[:len(stack_data)], alpha=0.7)
            plt.legend(loc='upper right')
        
        plt.xlabel('Time (hours)')
        plt.ylabel('Concentration')
        plt.title('First Particle: Nuclear Decay of Enhanced Nuclides')
        plt.grid(True, alpha=0.3)
        
        # Add annotation about decay behavior
        plt.text(0.02, 0.95, 
                'Note: Concentrations should decay according to nuclear physics.\nIf all become 0, check CRAM initialization.', 
                transform=plt.gca().transAxes, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                verticalalignment='top')
    else:
        plt.text(0.5, 0.5, 'No enhanced nuclides found in first particle', 
                ha='center', va='center', transform=plt.gca().transAxes, 
                fontsize=16, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.savefig('cram_result/ldm_stacked.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Create summary table
    summary_data = []
    
    for nuclide in enhanced_nuclide_names:
        initial_conc = df[nuclide].iloc[0] if len(df) > 0 else 0
        final_conc = df[nuclide].iloc[-1] if len(df) > 0 else 0
        max_conc = df[nuclide].max() if len(df) > 0 else 0
        
        summary_data.append({
            'Nuclide': nuclide,
            'Initial_Concentration': initial_conc,
            'Final_Concentration': final_conc,
            'Max_Concentration': max_conc,
            'Decay_Rate': f"{((initial_conc - final_conc) / initial_conc * 100):.1f}%" if initial_conc > 0 else "N/A",
            'Status': 'Enhanced'
        })
    
    # Add simulation metadata
    summary_data.append({
        'Nuclide': 'SIMULATION_INFO',
        'Initial_Concentration': f'Duration: {time_hours.iloc[-1]:.1f} hours',
        'Final_Concentration': f'Timesteps: {len(df)}',
        'Max_Concentration': f'Enhanced_Nuclides: {len(enhanced_nuclides)}',
        'Decay_Rate': f'Total_Conc_Initial: {df["total_conc"].iloc[0]:.3f}',
        'Status': f'Total_Conc_Final: {df["total_conc"].iloc[-1]:.3f}'
    })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('cram_result/simulation_summary.csv', index=False)
    
    print(f"✓ First particle visualization completed!")
    print(f"✓ Enhanced nuclides found: {len(enhanced_nuclides)}")
    if len(enhanced_nuclides) > 0:
        print(f"✓ Enhanced nuclides: {list(enhanced_nuclides.index)}")
        print(f"✓ Initial concentrations: {enhanced_nuclides.values}")

if __name__ == "__main__":
    clear_output_directory()
    visualize_first_particle_concentrations()