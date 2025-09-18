#!/usr/bin/env python3
"""
Complete 60-Nuclide Visualization Tool for LDM-CRAM4
Show all nuclides concentration changes over time in multiple comprehensive formats
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import warnings
import os
import shutil

warnings.filterwarnings('ignore')

# Font settings for consistent display
plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 16

class Complete60NuclideVisualizer:
    def __init__(self, output_dir='cram_result'):
        self.data_file = 'validation/first_particle_concentrations.csv'
        self.output_dir = output_dir
        
        # Clear and create output directory
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)
    
    def load_data(self):
        """Load concentration data and include ALL 60 nuclides"""
        try:
            self.df = pd.read_csv(self.data_file)
            print(f"✓ Loaded {self.data_file}: {len(self.df)} timesteps")
            
            # Get all nuclide names (skip timestep, time, total_conc)
            self.nuclide_names = list(self.df.columns[3:])
            
            # Show statistics for all nuclides
            active_count = 0
            zero_count = 0
            for nuclide in self.nuclide_names:
                initial_value = self.df[nuclide].iloc[0]  # First timestep value
                max_value = self.df[nuclide].max()        # Maximum value across all time
                if initial_value > 0.0 or max_value > 1e-10:
                    active_count += 1
                    print(f"✅ Including {nuclide} (initial={initial_value:.2e}, max={max_value:.2e})")
                else:
                    zero_count += 1
                    print(f"📊 Including {nuclide} (zero throughout - will show as flat line)")
            
            self.num_nuclides = len(self.nuclide_names)
            print(f"📊 Total nuclides loaded: {self.num_nuclides}")
            print(f"✅ Active nuclides (with values): {active_count}")
            print(f"📊 Zero nuclides (flat lines): {zero_count}")
            
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def create_grid_plot(self):
        """Create a large grid showing active nuclides individually with log scale Y-axis"""
        print(f"📈 Creating {self.num_nuclides}-nuclide grid plot...")
        
        # Calculate grid dimensions based on actual number of nuclides
        import math
        cols = min(10, self.num_nuclides)  # Max 10 columns for better layout
        rows = math.ceil(self.num_nuclides / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
        
        # Handle single subplot case
        if self.num_nuclides == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Plot each nuclide
        for i, nuclide in enumerate(self.nuclide_names):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Get concentration values
            concentrations = self.df[nuclide].values
            time_hours = self.df['time(s)'].values / 3600.0  # Convert to hours
            
            # Plot line with different colors based on concentration level
            max_conc = max(concentrations)
            if max_conc > 1000:
                color = 'red'
                linewidth = 2
            elif max_conc > 1:
                color = 'orange'
                linewidth = 1.5
            elif max_conc > 0.01:
                color = 'blue'
                linewidth = 1.2
            else:
                color = 'gray'
                linewidth = 1
            
            # Handle zero concentrations for log scale by using a small positive value
            plot_concentrations = np.maximum(concentrations, 1e-50)  # Replace 0 with small positive
            if max_conc > 0:
                ax.semilogy(time_hours, plot_concentrations, 
                           color=color, linewidth=linewidth, marker='o', markersize=2)
            else:
                # If all values are zero, plot as flat line at bottom of log scale
                ax.semilogy(time_hours, np.full_like(time_hours, 1e-50), 
                           color='lightgray', linewidth=1, linestyle='--', alpha=0.5)
            
            # Force log scale for all plots
            ax.set_yscale('log')
            
            # Set title and labels
            ax.set_title(f'{nuclide}\n(Max: {max_conc:.2e})', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (hours)', fontsize=10)
            ax.set_ylabel('Concentration (log)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
        
        # Hide unused subplots
        total_subplots = rows * cols
        for i in range(self.num_nuclides, total_subplots):
            if i < len(axes):
                axes[i].set_visible(False)
        
        # Add legend
        legend_elements = [
            Patch(facecolor='red', label='Very High (>1000)'),
            Patch(facecolor='orange', label='High (1-1000)'),
            Patch(facecolor='blue', label='Medium (0.01-1)'),
            Patch(facecolor='gray', label='Low (<0.01)'),
            Patch(facecolor='lightgray', linestyle='--', alpha=0.5, label='Zero throughout')
        ]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.99))
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/all_60_nuclides_grid.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: {self.output_dir}/all_60_nuclides_grid.png")
    
    def create_stacked_plot(self):
        """Create stacked area plot showing all nuclides together"""
        print("📈 Creating stacked area plot...")
        
        fig, ax = plt.subplots(figsize=(18, 12))
        
        time_hours = self.df['time(s)'].values / 3600.0
        
        # Prepare data for stacking - show all nuclides with any measurable concentration
        nuclide_data = []
        
        for nuclide in self.nuclide_names:
            conc_values = self.df[nuclide].values
            max_conc = max(conc_values)
            # Lower threshold to show more nuclides
            if max_conc > 1e-10:  # Show almost all nuclides with any concentration
                nuclide_data.append((nuclide, conc_values, max_conc))
        
        # Sort by maximum concentration (highest first)
        nuclide_data.sort(key=lambda x: x[2], reverse=True)
        
        # Reorganize sorted data
        significant_nuclides = [data[0] for data in nuclide_data]
        concentrations_matrix = [data[1] for data in nuclide_data]
        
        if concentrations_matrix:
            concentrations_array = np.array(concentrations_matrix)
            
            # Use Set3 colormap like ldm-CRAM2
            base_colors = plt.cm.Set3(np.linspace(0, 1, len(significant_nuclides)))
            
            # Group every 5 colors with descending alpha
            colors = []
            alphas = []
            for i, color in enumerate(base_colors):
                # Group into 5s with alpha adjustment
                group_position = i % 5  # 0, 1, 2, 3, 4
                # Alpha: 1.0(opaque) → 0.4(transparent) in descending order
                alpha = 1.0 - (group_position * 0.15)  # 1.0, 0.85, 0.7, 0.55, 0.4
                
                colors.append(color)
                alphas.append(alpha)
            
            # Create stacked area plot with individual alpha values
            bottom = np.zeros(len(time_hours))
            
            for i, (conc_data, color, alpha) in enumerate(zip(concentrations_array, colors, alphas)):
                ax.fill_between(time_hours, bottom, bottom + conc_data, 
                              color=color, alpha=alpha, label=significant_nuclides[i])
                bottom += conc_data
            
            ax.set_xlabel('Time (hours)', fontsize=18)
            ax.set_ylabel('Concentration (arbitrary units)', fontsize=18)
            ax.grid(True, alpha=0.3)
            ax.set_title('LDM-CRAM4: First Particle Nuclear Decay Stack', fontsize=20, fontweight='bold')
            
            # Add legend with 2 columns to make it more compact
            ax.legend(bbox_to_anchor=(1.05, 1.05), loc='upper left', fontsize=12, ncol=2)
        else:
            # If no significant nuclides, show a message
            ax.text(0.5, 0.5, 'No nuclides with measurable concentrations found', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/ldm_stacked.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: {self.output_dir}/ldm_stacked.png")
    
    def create_heatmap_detailed(self):
        """Create detailed heatmaps showing all nuclides vs time with turbo colormap"""
        print("🔥 Creating detailed concentration heatmaps with turbo colormap...")
        
        # Prepare data matrix (nuclides × time)
        concentration_matrix = []
        for nuclide in self.nuclide_names:
            concentration_matrix.append(self.df[nuclide].values)
        
        concentration_matrix = np.array(concentration_matrix)
        
        # Create heatmap with log scale for better visualization
        # Add small value to avoid log(0)
        log_matrix = np.log10(concentration_matrix + 1e-10)
        
        # Use turbo colormap
        fig, ax = plt.subplots(figsize=(14, 20))
        
        im = ax.imshow(log_matrix, cmap='turbo', aspect='auto', interpolation='nearest', alpha=0.8)
        
        # Set labels with 3-hour intervals to avoid overlap
        time_hours = self.df['time(s)'].values / 3600.0  # Convert to hours
        # Show every 36 timesteps (approximately 3 hours)
        hour_indices = []
        hour_labels = []
        for i, hour in enumerate(time_hours):
            if i % 36 == 0:  # Every 36 timesteps
                hour_indices.append(i)
                hour_labels.append(f"{int(hour)}h")
        
        ax.set_xticks(hour_indices)
        ax.set_xticklabels(hour_labels, fontsize=14, rotation=0)
        ax.set_xlabel('Time (3-hour intervals)', fontsize=18)
        
        ax.set_yticks(range(len(self.nuclide_names)))
        ax.set_yticklabels(self.nuclide_names, fontsize=14)
        ax.set_ylabel('Nuclides', fontsize=18)
        
        ax.set_title('LDM-CRAM4: 60-Nuclide Concentration Heatmap (Log Scale)', 
                    fontsize=24, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('log₁₀(Concentration)', fontsize=18)
        
        plt.tight_layout()
        filename = f'{self.output_dir}/ldm_heatmap.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: {filename}")
    
    def create_summary_statistics(self):
        """Create comprehensive summary statistics"""
        print("📊 Creating comprehensive summary statistics...")
        
        # Calculate detailed statistics for each nuclide
        stats_data = []
        time_hours = self.df['time(s)'].values / 3600.0
        
        for nuclide in self.nuclide_names:
            conc_values = self.df[nuclide].values
            
            # Calculate decay rate (exponential decay fitting)
            non_zero_mask = conc_values > 0
            if np.sum(non_zero_mask) > 2:
                try:
                    # Fit exponential decay: C = C0 * exp(-lambda * t)
                    ln_conc = np.log(conc_values[non_zero_mask])
                    time_subset = time_hours[non_zero_mask]
                    slope, intercept = np.polyfit(time_subset, ln_conc, 1)
                    decay_rate = -slope  # lambda
                    half_life_calc = np.log(2) / decay_rate if decay_rate > 0 else np.inf
                except:
                    decay_rate = 0
                    half_life_calc = np.inf
            else:
                decay_rate = 0
                half_life_calc = np.inf
            
            # Calculate time to 50% and 10% of initial
            initial_conc = conc_values[0]
            time_to_50pct = np.inf
            time_to_10pct = np.inf
            
            if initial_conc > 0:
                for i, conc in enumerate(conc_values):
                    if conc <= 0.5 * initial_conc and time_to_50pct == np.inf:
                        time_to_50pct = time_hours[i]
                    if conc <= 0.1 * initial_conc and time_to_10pct == np.inf:
                        time_to_10pct = time_hours[i]
            
            stats_data.append({
                'Nuclide': nuclide,
                'Initial': conc_values[0],
                'Final': conc_values[-1],
                'Maximum': max(conc_values),
                'Minimum': min(conc_values),
                'Mean': np.mean(conc_values),
                'Std': np.std(conc_values),
                'Range': max(conc_values) - min(conc_values),
                'Change(%)': ((conc_values[-1] - conc_values[0]) / (conc_values[0] + 1e-10)) * 100,
                'Decay_Rate_hr': decay_rate,
                'Half_Life_calc_hr': half_life_calc,
                'Time_to_50pct_hr': time_to_50pct,
                'Time_to_10pct_hr': time_to_10pct,
                'Volatility': np.std(conc_values) / (np.mean(conc_values) + 1e-10)
            })
        
        stats_df = pd.DataFrame(stats_data)
        
        # Save comprehensive statistics to CSV
        stats_df.to_csv(f'{self.output_dir}/simulation_summary.csv', index=False)
        print(f"💾 Saved: {self.output_dir}/simulation_summary.csv")
        
        return stats_df
    
    def generate_all_visualizations(self):
        """Generate all visualization types"""
        print("🎨 Starting Complete 60-Nuclide Visualization Generation")
        print("=" * 60)
        
        if not self.load_data():
            return False
        
        # Generate all visualization types
        self.create_grid_plot()
        self.create_stacked_plot()
        self.create_heatmap_detailed()
        stats_df = self.create_summary_statistics()
        
        print("\n" + "=" * 60)
        print("✅ Complete 60-Nuclide Visualization Generation Finished!")
        print(f"📁 All files saved in: {self.output_dir}/")
        print("📊 Generated visualizations:")
        print("   • all_60_nuclides_grid.png - Individual plots with LOG SCALE Y-axis")
        print("   • ldm_stacked.png - Stacked area plot")
        print("   • ldm_heatmap.png - Heatmap with turbo colormap")
        print("   • simulation_summary.csv - Comprehensive statistical data")
        
        # Print enhanced nuclides summary
        enhanced = stats_df[stats_df['Initial'] > 0.1]
        if len(enhanced) > 0:
            print(f"\n✅ Enhanced nuclides detected: {len(enhanced)}")
            for _, row in enhanced.iterrows():
                print(f"   • {row['Nuclide']}: {row['Initial']:.3f} → {row['Final']:.3f} ({row['Change(%)']:.1f}%)")
        
        return True

if __name__ == "__main__":
    visualizer = Complete60NuclideVisualizer()
    visualizer.generate_all_visualizations()