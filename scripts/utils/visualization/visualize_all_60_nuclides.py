#!/usr/bin/env python3
"""
Complete 60-Nuclide Visualization Tool
Show all nuclides concentration changes over time in multiple comprehensive formats
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')
import os

# Font settings for consistent display
plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 16

class Complete60NuclideVisualizer:
    def __init__(self, output_dir='results'):
        self.data_file = 'first_particle_concentrations.csv'
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
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
        cols = min(6, self.num_nuclides)  # Max 6 columns
        rows = math.ceil(self.num_nuclides / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
        
        # Handle single subplot case
        if self.num_nuclides == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Plot each active nuclide
        for i, nuclide in enumerate(self.nuclide_names):
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
            ax.set_title(f'{nuclide}\n(Max: {max_conc:.2e})', fontsize=18, fontweight='bold')
            ax.set_xlabel('Time (hours)', fontsize=16)
            ax.set_ylabel('Concentration (log)', fontsize=16)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=14)
        
        # Hide unused subplots
        total_subplots = rows * cols
        for i in range(self.num_nuclides, total_subplots):
            if i < len(axes):
                axes[i].set_visible(False)
        
        # Add legend including zero nuclides
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
        
        fig, ax = plt.subplots(figsize=(18, 12))  # 크기 증가 for 범례 공간
        
        time_hours = self.df['time(s)'].values / 3600.0
        
        # Prepare data for stacking - show all nuclides with any measurable concentration
        nuclide_data = []
        
        for nuclide in self.nuclide_names:
            conc_values = self.df[nuclide].values
            max_conc = max(conc_values)
            # Lower threshold to show more nuclides (원래 0.1에서 1e-10으로 변경)
            if max_conc > 1e-10:  # Show almost all nuclides with any concentration
                nuclide_data.append((nuclide, conc_values, max_conc))
        
        # 최대 농도 순으로 정렬 (높은 것부터 낮은 것 순)
        nuclide_data.sort(key=lambda x: x[2], reverse=True)
        
        # 정렬된 순서로 리스트 재구성
        significant_nuclides = [data[0] for data in nuclide_data]
        concentrations_matrix = [data[1] for data in nuclide_data]
        
        if concentrations_matrix:
            concentrations_array = np.array(concentrations_matrix)
            
            # ldm3 폴더와 동일한 색상 조합 사용 (Set3 컬러맵)
            base_colors = plt.cm.Set3(np.linspace(0, 1, len(significant_nuclides)))
            
            # 인접한 5개씩 그룹으로 묶어서 투명도가 내림차순되도록 설정
            colors = []
            alphas = []
            for i, color in enumerate(base_colors):
                # 5개씩 그룹으로 묶어서 투명도 조정
                group_position = i % 5  # 0, 1, 2, 3, 4
                # 투명도: 1.0(불투명) → 0.4(반투명) 순으로 내림차순
                alpha = 1.0 - (group_position * 0.15)  # 1.0, 0.85, 0.7, 0.55, 0.4
                
                colors.append(color)
                alphas.append(alpha)
            
            # Create stacked area plot with individual alpha values
            # stackplot doesn't support individual alphas, so we'll create layers manually
            bottom = np.zeros(len(time_hours))
            
            for i, (conc_data, color, alpha) in enumerate(zip(concentrations_array, colors, alphas)):
                ax.fill_between(time_hours, bottom, bottom + conc_data, 
                              color=color, alpha=alpha, label=significant_nuclides[i])
                bottom += conc_data
            ax.set_xlabel('Time (hours)', fontsize=18)
            ax.set_ylabel('Concentration (arbitrary units)', fontsize=18)
            ax.grid(True, alpha=0.3)
            
            # Add legend with 2 columns to make it more compact
            ax.legend(bbox_to_anchor=(1.05, 1.05), loc='upper left', fontsize=14, ncol=2)
        else:
            # If no significant nuclides, show a message
            ax.text(0.5, 0.5, 'No nuclides with measurable concentrations found', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/all_60_nuclides_stacked.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: {self.output_dir}/all_60_nuclides_stacked.png")
    
    def create_log_overlay_plot(self):
        """Create overlay plot with all significant nuclides on log scale with broken axis (skipping 10^-42 to 10^-6)"""
        print("📈 Creating log-scale overlay plot with broken axis...")
        
        from matplotlib import gridspec
        
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.1)
        
        ax1 = fig.add_subplot(gs[0])  # Upper plot
        ax2 = fig.add_subplot(gs[1])  # Lower plot
        
        time_hours = self.df['time(s)'].values / 3600.0
        
        # Plot all nuclides with significant concentrations
        colors = plt.cm.tab20(np.linspace(0, 1, 20))  # Use distinct colors
        color_idx = 0
        
        plotted_nuclides = []
        for nuclide in self.nuclide_names:
            conc_values = self.df[nuclide].values
            max_conc = max(conc_values)
            
            # Only plot nuclides with measurable decay
            if max_conc > 1e-50 and np.any(conc_values > 0):
                # Filter out zero values for log plot
                non_zero_mask = conc_values > 0
                if np.any(non_zero_mask):
                    # Plot on upper axis (high values)
                    ax1.semilogy(time_hours[non_zero_mask], conc_values[non_zero_mask], 
                               color=colors[color_idx % 20], linewidth=2, 
                               label=f'{nuclide}', alpha=0.8)
                    # Plot on lower axis (low values)  
                    ax2.semilogy(time_hours[non_zero_mask], conc_values[non_zero_mask], 
                               color=colors[color_idx % 20], linewidth=2, alpha=0.8)
                    plotted_nuclides.append(nuclide)
                    color_idx += 1
        
        # Set different y-limits for the break  
        ax1.set_ylim(1e-6, 1e1)      # Upper part: above 10^-6
        ax2.set_ylim(1e-48, 1e-42)   # Lower part: below 10^-42
        
        # Hide the spines between ax1 and ax2
        ax1.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax1.xaxis.tick_top()
        ax1.tick_params(labeltop=False)  # don't put tick labels at the top
        ax2.xaxis.tick_bottom()
        
        # Add diagonal lines to indicate the break
        d = .015  # size of diagonal lines in axes coordinates
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
        ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
        
        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
        
        # Add "~" symbols to indicate break
        fig.text(0.02, 0.45, '~\n~\n~', ha='center', va='center', fontsize=20, fontweight='bold')
        fig.text(0.98, 0.45, '~\n~\n~', ha='center', va='center', fontsize=20, fontweight='bold')
        
        ax2.set_xlabel('Time (hours)', fontsize=18)
        ax1.set_ylabel('Concentration (log scale)', fontsize=18)
        ax2.set_ylabel('Concentration (log scale)', fontsize=18)
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # Add legend on upper plot with 2 columns
        if plotted_nuclides:
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, ncol=2)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/all_60_nuclides_log_overlay.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: {self.output_dir}/all_60_nuclides_log_overlay.png")
    
    def create_heatmap_detailed(self):
        """Create detailed heatmaps showing all nuclides vs time with multiple colormaps"""
        print("🔥 Creating detailed concentration heatmaps with multiple colormaps...")
        
        # Prepare data matrix (nuclides × time)
        concentration_matrix = []
        for nuclide in self.nuclide_names:
            concentration_matrix.append(self.df[nuclide].values)
        
        concentration_matrix = np.array(concentration_matrix)
        
        # Create heatmap with log scale for better visualization
        # Add small value to avoid log(0)
        log_matrix = np.log10(concentration_matrix + 1e-10)
        
        # Use only turbo colormap
        fig, ax = plt.subplots(figsize=(14, 20))
        
        im = ax.imshow(log_matrix, cmap='turbo', aspect='auto', interpolation='nearest', alpha=0.8)
        
        # Set labels with 3-hour intervals to avoid overlap
        time_hours = self.df['time(s)'].values / 3600.0  # Convert to hours
        # Show every 3-hour mark, but limit to reasonable intervals
        hour_indices = []
        hour_labels = []
        for i, hour in enumerate(time_hours):
            if i % 36 == 0:  # Every 36 timesteps (approximately 3 hours since timesteps are 5min)
                hour_indices.append(i)
                hour_labels.append(f"{int(hour)}h")
        
        ax.set_xticks(hour_indices)
        ax.set_xticklabels(hour_labels, fontsize=14, rotation=0)
        ax.set_xlabel('Time (3-hour intervals)', fontsize=18)
        
        ax.set_yticks(range(len(self.nuclide_names)))
        ax.set_yticklabels(self.nuclide_names, fontsize=14)
        ax.set_ylabel('Nuclides', fontsize=18)
        
        
        ax.set_title('60-Nuclide Concentration Heatmap (Log Scale)', 
                    fontsize=24, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('log₁₀(Concentration)', fontsize=18)
        
        plt.tight_layout()
        filename = f'{self.output_dir}/all_60_nuclides_heatmap_turbo_log.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: {filename}")
        
        # Create linear scale heatmap
        fig, ax = plt.subplots(figsize=(14, 20))
        
        im_linear = ax.imshow(concentration_matrix, cmap='turbo', aspect='auto', interpolation='nearest', alpha=0.8)
        
        # Set labels with 3-hour intervals to avoid overlap
        ax.set_xticks(hour_indices)
        ax.set_xticklabels(hour_labels, fontsize=14, rotation=0)
        ax.set_xlabel('Time (3-hour intervals)', fontsize=18)
        
        ax.set_yticks(range(len(self.nuclide_names)))
        ax.set_yticklabels(self.nuclide_names, fontsize=14)
        ax.set_ylabel('Nuclides', fontsize=18)
        
        ax.set_title('60-Nuclide Concentration Heatmap (Linear Scale)', 
                    fontsize=24, fontweight='bold')
        
        # Add colorbar
        cbar_linear = plt.colorbar(im_linear, ax=ax, shrink=0.8)
        cbar_linear.set_label('Concentration (linear)', fontsize=18)
        
        plt.tight_layout()
        filename_linear = f'{self.output_dir}/all_60_nuclides_heatmap_turbo_linear.png'
        plt.savefig(filename_linear, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: {filename_linear}")
        
        print(f"💾 Created heatmaps with turbo colormap (both log and linear scales)")
    
    def create_category_plots(self):
        """Create individual plots for each half-life category - 12 nuclides per category"""
        print("📊 Creating individual category-based plots by half-life (12 nuclides each)...")
        
        # Calculate half-life for each nuclide from decay data
        nuclide_half_lives = []
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
                    half_life_calc = np.inf
            else:
                half_life_calc = np.inf
            
            nuclide_half_lives.append((nuclide, half_life_calc))
        
        # Sort by half-life (shortest to longest)
        nuclide_half_lives.sort(key=lambda x: x[1])
        
        # Divide into 5 groups of 12 nuclides each
        categories = {}
        for i in range(5):
            start_idx = i * 12
            end_idx = min((i + 1) * 12, len(nuclide_half_lives))
            group_nuclides = nuclide_half_lives[start_idx:end_idx]
            
            if group_nuclides:
                shortest_half_life = group_nuclides[0][1]
                longest_half_life = group_nuclides[-1][1]
                
                # Format half-life ranges nicely
                if shortest_half_life == np.inf:
                    half_life_range = "Stable"
                elif longest_half_life == np.inf:
                    half_life_range = f"{shortest_half_life:.1e}h - Stable"
                else:
                    half_life_range = f"{shortest_half_life:.1e}h - {longest_half_life:.1e}h"
                
                category_name = f'Group {i+1} (Half-life: {half_life_range})'
                categories[category_name] = [nuc[0] for nuc in group_nuclides]
        
        print(f"Created {len(categories)} categories with 12 nuclides each, sorted by half-life")
        
        # Create separate plots for each category
        time_hours = self.df['time(s)'].values / 3600.0
        
        for idx, (category, nuclides) in enumerate(categories.items()):
            # Only use broken axis for category 1 (idx == 0)
            if idx == 0:
                from matplotlib import gridspec
                
                fig = plt.figure(figsize=(16, 12))
                gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.1)
                
                ax1 = fig.add_subplot(gs[0])  # Upper plot
                ax2 = fig.add_subplot(gs[1])  # Lower plot
                
                # Use a color cycle for multiple nuclides
                color_cycle = plt.cm.Set3(np.linspace(0, 1, 12))
                
                for j, nuclide in enumerate(nuclides):
                    concentrations = self.df[nuclide].values
                    non_zero_mask = concentrations > 0
                    if np.any(non_zero_mask):
                        # Plot on upper axis (high values)
                        ax1.semilogy(time_hours[non_zero_mask], concentrations[non_zero_mask], 
                                   label=nuclide, linewidth=2, marker='o', markersize=3, 
                                   alpha=0.8, color=color_cycle[j])
                        # Plot on lower axis (low values)  
                        ax2.semilogy(time_hours[non_zero_mask], concentrations[non_zero_mask], 
                                   linewidth=2, marker='o', markersize=3, 
                                   alpha=0.8, color=color_cycle[j])
                
                # Set different y-limits for the break  
                ax1.set_ylim(1e-6, 1e1)      # Upper part: above 10^-6
                ax2.set_ylim(1e-48, 1e-42)   # Lower part: below 10^-42
                
                # Hide the spines between ax1 and ax2
                ax1.spines['bottom'].set_visible(False)
                ax2.spines['top'].set_visible(False)
                ax1.xaxis.tick_top()
                ax1.tick_params(labeltop=False)  # don't put tick labels at the top
                ax2.xaxis.tick_bottom()
                
                # Add diagonal lines to indicate the break
                d = .015  # size of diagonal lines in axes coordinates
                kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
                ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
                ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
                
                kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
                ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
                ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
                
                # Add "~" symbols to indicate break
                fig.text(0.02, 0.45, '~\n~\n~', ha='center', va='center', fontsize=24, fontweight='bold')
                fig.text(0.98, 0.45, '~\n~\n~', ha='center', va='center', fontsize=24, fontweight='bold')
                
                ax2.set_xlabel('Time (hours)', fontsize=18)
                ax1.set_ylabel('Concentration (log scale)', fontsize=18)
                ax2.set_ylabel('Concentration (log scale)', fontsize=18)
                ax1.grid(True, alpha=0.3)
                ax2.grid(True, alpha=0.3)
                
                # Place legend outside plot area with 1 column
                ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, ncol=1)
            else:
                # Regular single plot for categories 2-5
                fig, ax = plt.subplots(figsize=(16, 10))
                
                # Use a color cycle for multiple nuclides
                color_cycle = plt.cm.Set3(np.linspace(0, 1, 12))
                
                for j, nuclide in enumerate(nuclides):
                    concentrations = self.df[nuclide].values
                    non_zero_mask = concentrations > 0
                    if np.any(non_zero_mask):
                        ax.semilogy(time_hours[non_zero_mask], concentrations[non_zero_mask], 
                                   label=nuclide, linewidth=2, marker='o', markersize=3, 
                                   alpha=0.8, color=color_cycle[j])
                
                ax.set_xlabel('Time (hours)', fontsize=18)
                ax.set_ylabel('Concentration (log scale)', fontsize=18)
                ax.grid(True, alpha=0.3)
                
                # Place legend outside plot area with 1 column
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, ncol=1)
            
            plt.tight_layout()
            filename = f'{self.output_dir}/category_{idx+1}_half_life_group_{idx+1}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"💾 Saved: {filename}")
        
        # Print category summary
        print("\n📋 Nuclide Category Summary (12 per group, sorted by half-life):")
        for i, (category, nuclides) in enumerate(categories.items()):
            print(f"  {category}: {len(nuclides)} nuclides")
            # Show half-life for each nuclide in the group
            group_info = []
            for nuclide in nuclides:
                for nuc_name, half_life in nuclide_half_lives:
                    if nuc_name == nuclide:
                        if half_life == np.inf:
                            group_info.append(f"{nuclide}(stable)")
                        else:
                            group_info.append(f"{nuclide}({half_life:.1e}h)")
                        break
            print(f"    {', '.join(group_info)}")
            print()
    
    def create_summary_statistics(self):
        """Create comprehensive summary statistics with useful metrics"""
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
        
        # Create comprehensive plots (3x2 grid)
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # Plot 1: Top 20 by Maximum Concentration
        top_20_max = stats_df.nlargest(20, 'Maximum')
        axes[0,0].bar(range(len(top_20_max)), top_20_max['Maximum'], color='red', alpha=0.7)
        axes[0,0].set_xticks(range(len(top_20_max)))
        axes[0,0].set_xticklabels(top_20_max['Nuclide'], rotation=45, ha='right', fontsize=14)
        axes[0,0].set_ylabel('Maximum Concentration')
        axes[0,0].set_yscale('log')
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Top 20 by Decay Rate
        valid_decay = stats_df[stats_df['Decay_Rate_hr'] > 0].nlargest(20, 'Decay_Rate_hr')
        if len(valid_decay) > 0:
            axes[0,1].bar(range(len(valid_decay)), valid_decay['Decay_Rate_hr'], color='orange', alpha=0.7)
            axes[0,1].set_xticks(range(len(valid_decay)))
            axes[0,1].set_xticklabels(valid_decay['Nuclide'], rotation=45, ha='right', fontsize=14)
            axes[0,1].set_ylabel('Decay Rate (hr⁻¹)')
            axes[0,1].set_yscale('log')
            axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Half-Life Distribution
        valid_half_life = stats_df[stats_df['Half_Life_calc_hr'] < 1000]['Half_Life_calc_hr']
        if len(valid_half_life) > 0:
            axes[1,0].hist(valid_half_life, bins=30, color='blue', alpha=0.7, edgecolor='black')
            axes[1,0].set_xlabel('Calculated Half-Life (hours)')
            axes[1,0].set_ylabel('Number of Nuclides')
            axes[1,0].set_xscale('log')
            axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Concentration Volatility (stability)
        top_20_volatile = stats_df.nlargest(20, 'Volatility')
        axes[1,1].bar(range(len(top_20_volatile)), top_20_volatile['Volatility'], color='green', alpha=0.7)
        axes[1,1].set_xticks(range(len(top_20_volatile)))
        axes[1,1].set_xticklabels(top_20_volatile['Nuclide'], rotation=45, ha='right', fontsize=14)
        axes[1,1].set_ylabel('Volatility (σ/μ)')
        axes[1,1].grid(True, alpha=0.3)
        
        # Plot 5: Initial vs Final (Log-Log)
        axes[2,0].scatter(stats_df['Initial'], stats_df['Final'], alpha=0.6, c='purple', s=50)
        axes[2,0].plot([1e-15, 1e3], [1e-15, 1e3], 'r--', label='No Change Line', linewidth=2)
        axes[2,0].set_xlabel('Initial Concentration')
        axes[2,0].set_ylabel('Final Concentration')
        axes[2,0].set_xscale('log')
        axes[2,0].set_yscale('log')
        axes[2,0].grid(True, alpha=0.3)
        axes[2,0].legend()
        
        # Plot 6: Time to 50% reduction
        valid_time_50 = stats_df[stats_df['Time_to_50pct_hr'] < 20]  # Within simulation time
        if len(valid_time_50) > 0:
            axes[2,1].bar(range(len(valid_time_50)), valid_time_50['Time_to_50pct_hr'], color='cyan', alpha=0.7)
            axes[2,1].set_xticks(range(len(valid_time_50)))
            axes[2,1].set_xticklabels(valid_time_50['Nuclide'], rotation=45, ha='right', fontsize=14)
            axes[2,1].set_ylabel('Time to 50% Reduction (hours)')
            axes[2,1].grid(True, alpha=0.3)
        else:
            axes[2,1].text(0.5, 0.5, 'No nuclides reached\n50% reduction\nwithin simulation time', 
                          ha='center', va='center', transform=axes[2,1].transAxes, fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/all_60_nuclides_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: {self.output_dir}/all_60_nuclides_statistics.png")
        
        # Create additional specific analysis plots
        self.create_decay_analysis(stats_df)
        
        # Save comprehensive statistics to CSV
        stats_df.to_csv(f'{self.output_dir}/60_nuclides_comprehensive_statistics.csv', index=False)
        print(f"💾 Saved: {self.output_dir}/60_nuclides_comprehensive_statistics.csv")
    
    def create_decay_analysis(self, stats_df):
        """Create specific decay analysis plots"""
        print("📈 Creating decay analysis plots...")
        
        # Decay efficiency analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Fastest vs Slowest decaying
        fastest_10 = stats_df[stats_df['Decay_Rate_hr'] > 0].nlargest(10, 'Decay_Rate_hr')
        slowest_10 = stats_df[stats_df['Decay_Rate_hr'] > 0].nsmallest(10, 'Decay_Rate_hr')
        
        x_fast = range(len(fastest_10))
        x_slow = range(len(slowest_10))
        
        axes[0,0].bar(x_fast, fastest_10['Decay_Rate_hr'], color='red', alpha=0.7, label='Fastest 10')
        axes[0,0].set_xticks(x_fast)
        axes[0,0].set_xticklabels(fastest_10['Nuclide'], rotation=45, ha='right', fontsize=14)
        axes[0,0].set_ylabel('Decay Rate (hr⁻¹)')
        axes[0,0].set_yscale('log')
        axes[0,0].grid(True, alpha=0.3)
        
        axes[0,1].bar(x_slow, slowest_10['Decay_Rate_hr'], color='blue', alpha=0.7, label='Slowest 10')
        axes[0,1].set_xticks(x_slow)
        axes[0,1].set_xticklabels(slowest_10['Nuclide'], rotation=45, ha='right', fontsize=14)
        axes[0,1].set_ylabel('Decay Rate (hr⁻¹)')
        axes[0,1].set_yscale('log')
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Percentage change vs Initial concentration
        axes[1,0].scatter(stats_df['Initial'], stats_df['Change(%)'], alpha=0.6, c='green', s=50)
        axes[1,0].set_xlabel('Initial Concentration')
        axes[1,0].set_ylabel('Percentage Change (%)')
        axes[1,0].set_xscale('log')
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Stability ranking (least volatile = most stable)
        most_stable = stats_df.nsmallest(15, 'Volatility')
        axes[1,1].bar(range(len(most_stable)), most_stable['Volatility'], color='orange', alpha=0.7)
        axes[1,1].set_xticks(range(len(most_stable)))
        axes[1,1].set_xticklabels(most_stable['Nuclide'], rotation=45, ha='right', fontsize=14)
        axes[1,1].set_ylabel('Volatility (lower = more stable)')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/decay_analysis_detailed.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: {self.output_dir}/decay_analysis_detailed.png")
    
    def generate_all_visualizations(self):
        """Generate all visualization types"""
        print("🎨 Starting Complete 60-Nuclide Visualization Generation")
        print("=" * 60)
        
        if not self.load_data():
            return False
        
        # Generate all visualization types
        self.create_grid_plot()
        self.create_stacked_plot()
        self.create_log_overlay_plot()
        self.create_heatmap_detailed()
        self.create_category_plots()
        self.create_summary_statistics()
        
        print("\n" + "=" * 60)
        print("✅ Complete 60-Nuclide Visualization Generation Finished!")
        print(f"📁 All files saved in: {self.output_dir}/")
        print("📊 Generated visualizations:")
        print("   • all_60_nuclides_grid.png - Individual plots with LOG SCALE Y-axis")
        print("   • all_60_nuclides_stacked.png - Stacked area plot with units")
        print("   • all_60_nuclides_log_overlay.png - Overlay with broken axis (물결무늬로 10⁻⁴² to 10⁻⁶ 구간 생략)")
        print("   • all_60_nuclides_heatmap_turbo.png - Heatmap with turbo colormap")
        print("   • category_*_half_life_group_*.png - 5 half-life sorted groups (12 nuclides each)")
        print("   • all_60_nuclides_statistics.png - Comprehensive statistical analysis")
        print("   • decay_analysis_detailed.png - Detailed decay analysis")
        print("   • 60_nuclides_comprehensive_statistics.csv - Comprehensive statistical data")
        return True

if __name__ == "__main__":
    visualizer = Complete60NuclideVisualizer()
    visualizer.generate_all_visualizations()