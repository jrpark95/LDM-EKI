#!/usr/bin/env python3
"""
핵종별 시간에 따른 농도 3D 가시화
3D Visualization of Nuclide Concentrations over Time
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def create_nuclide_time_3d_surface(csv_file, output_file=None):
    """Create 3D surface plot of nuclide concentrations over time"""
    
    # Read the nuclide ratios data
    df = pd.read_csv(csv_file)
    
    # Extract time and nuclide data
    time = df['time(s)'].values
    timesteps = df['timestep'].values
    
    # Get nuclide columns (ratio_Q_0 to ratio_Q_59)
    nuclide_cols = [col for col in df.columns if col.startswith('ratio_Q_')]
    nuclide_data = df[nuclide_cols].values
    
    # Create nuclide indices
    nuclide_indices = np.arange(len(nuclide_cols))
    
    # Create meshgrid for 3D surface
    T, N = np.meshgrid(time, nuclide_indices)
    Z = nuclide_data.T  # Transpose so nuclides are rows, time is columns
    
    # Create 3D figure
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create surface plot
    surf = ax.plot_surface(T, N, Z, cmap='plasma', alpha=0.8, 
                          linewidth=0, antialiased=True, shade=True)
    
    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.8, aspect=20, pad=0.1)
    cbar.set_label('핵종 농도 비율 (Nuclide Concentration Ratio)', 
                   rotation=270, labelpad=25, fontsize=12)
    
    # Labels and title
    ax.set_xlabel('시간 (Time, seconds)', fontsize=12)
    ax.set_ylabel('핵종 번호 (Nuclide Index)', fontsize=12)
    ax.set_zlabel('농도 비율 (Concentration Ratio)', fontsize=12)
    ax.set_title('핵종별 시간에 따른 농도 변화 - 3D Surface\nNuclide Concentration Evolution Over Time', 
                 fontsize=14, pad=20)
    
    # Set better viewing angle
    ax.view_init(elev=30, azim=45)
    
    # Add statistics
    max_conc = np.max(Z)
    min_conc = np.min(Z)
    mean_conc = np.mean(Z)
    
    stats_text = f"""
    시뮬레이션 통계:
    최대 농도: {max_conc:.6f}
    최소 농도: {min_conc:.6f}
    평균 농도: {mean_conc:.6f}
    총 핵종 수: {len(nuclide_cols)}
    시간 범위: {time.min():.0f}-{time.max():.0f}초
    """
    ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, 
              verticalalignment='top', 
              bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8),
              fontsize=10)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"3D 핵종 농도 표면도 저장: {output_file}")
    
    return fig, ax

def create_nuclide_time_3d_wireframe(csv_file, output_file=None):
    """Create 3D wireframe plot of nuclide concentrations"""
    
    df = pd.read_csv(csv_file)
    time = df['time(s)'].values
    
    # Get nuclide columns
    nuclide_cols = [col for col in df.columns if col.startswith('ratio_Q_')]
    nuclide_data = df[nuclide_cols].values
    nuclide_indices = np.arange(len(nuclide_cols))
    
    # Create meshgrid
    T, N = np.meshgrid(time, nuclide_indices)
    Z = nuclide_data.T
    
    # Create 3D wireframe
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create wireframe
    wire = ax.plot_wireframe(T, N, Z, cmap='viridis', alpha=0.7, linewidth=0.8)
    
    # Add some scatter points for key nuclides
    key_nuclides = [0, 10, 20, 30, 40, 50]  # Sample key nuclides
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    for i, (nuc_idx, color) in enumerate(zip(key_nuclides, colors)):
        if nuc_idx < len(nuclide_cols):
            ax.plot(time, [nuc_idx]*len(time), nuclide_data[:, nuc_idx], 
                   color=color, linewidth=2, alpha=0.9, 
                   label=f'Nuclide {nuc_idx}')
    
    ax.set_xlabel('시간 (Time, seconds)', fontsize=12)
    ax.set_ylabel('핵종 번호 (Nuclide Index)', fontsize=12)
    ax.set_zlabel('농도 비율 (Concentration Ratio)', fontsize=12)
    ax.set_title('핵종별 농도 변화 - 3D Wireframe\nNuclide Concentration Wireframe', 
                 fontsize=14, pad=20)
    
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
    ax.view_init(elev=25, azim=60)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"3D 핵종 농도 와이어프레임 저장: {output_file}")
    
    return fig, ax

def create_nuclide_time_3d_scatter(csv_file, output_file=None):
    """Create 3D scatter plot showing nuclide concentrations evolution"""
    
    df = pd.read_csv(csv_file)
    time = df['time(s)'].values
    
    # Get nuclide columns
    nuclide_cols = [col for col in df.columns if col.startswith('ratio_Q_')]
    nuclide_data = df[nuclide_cols].values
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create scatter plot with varying sizes and colors
    for t_idx, t in enumerate(time):
        concentrations = nuclide_data[t_idx, :]
        nuclide_indices = np.arange(len(concentrations))
        
        # Size proportional to concentration
        sizes = 20 + 200 * (concentrations / np.max(concentrations))
        
        scatter = ax.scatter(
            [t] * len(concentrations),  # Time coordinate
            nuclide_indices,            # Nuclide index
            concentrations,             # Concentration (height)
            c=concentrations,           # Color by concentration
            s=sizes,                    # Size by concentration
            cmap='plasma',
            alpha=0.6
        )
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('농도 비율 (Concentration Ratio)', rotation=270, labelpad=20, fontsize=12)
    
    ax.set_xlabel('시간 (Time, seconds)', fontsize=12)
    ax.set_ylabel('핵종 번호 (Nuclide Index)', fontsize=12)
    ax.set_zlabel('농도 비율 (Concentration Ratio)', fontsize=12)
    ax.set_title('핵종별 농도 변화 - 3D Scatter\nNuclide Concentration Scatter Plot', 
                 fontsize=14, pad=20)
    
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"3D 핵종 농도 산점도 저장: {output_file}")
    
    return fig, ax

def create_top_nuclides_3d_evolution(csv_file, output_file=None, top_n=10):
    """Create 3D plot focusing on top N nuclides by concentration"""
    
    df = pd.read_csv(csv_file)
    time = df['time(s)'].values
    
    # Get nuclide columns
    nuclide_cols = [col for col in df.columns if col.startswith('ratio_Q_')]
    nuclide_data = df[nuclide_cols].values
    
    # Find top N nuclides by maximum concentration
    max_concentrations = np.max(nuclide_data, axis=0)
    top_indices = np.argsort(max_concentrations)[-top_n:]
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color map for different nuclides
    colors = plt.cm.tab10(np.linspace(0, 1, top_n))
    
    # Plot each top nuclide as a line in 3D space
    for i, nuc_idx in enumerate(top_indices):
        concentrations = nuclide_data[:, nuc_idx]
        ax.plot(time, [nuc_idx]*len(time), concentrations, 
               color=colors[i], linewidth=3, alpha=0.8,
               label=f'핵종 {nuc_idx} (max: {max_concentrations[nuc_idx]:.4f})')
        
        # Add scatter points at key time intervals
        key_times = np.arange(0, len(time), max(1, len(time)//10))
        ax.scatter(time[key_times], [nuc_idx]*len(key_times), 
                  concentrations[key_times], 
                  color=colors[i], s=50, alpha=0.9)
    
    ax.set_xlabel('시간 (Time, seconds)', fontsize=12)
    ax.set_ylabel('핵종 번호 (Nuclide Index)', fontsize=12)
    ax.set_zlabel('농도 비율 (Concentration Ratio)', fontsize=12)
    ax.set_title(f'상위 {top_n}개 핵종의 농도 변화\nTop {top_n} Nuclides Concentration Evolution', 
                 fontsize=14, pad=20)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"상위 핵종 3D 분석 저장: {output_file}")
    
    return fig, ax

def main():
    """Main function for nuclide-time 3D visualizations"""
    
    print("=== 핵종별 시간에 따른 농도 3D 가시화 ===")
    print("=== Nuclide Concentration over Time - 3D Visualization ===\n")
    
    # Create output directory
    os.makedirs("cram_result/nuclide_time_3d", exist_ok=True)
    
    # Use the nuclide ratios data
    data_file = "all_particles_nuclide_ratios.csv"
    
    if not os.path.exists(data_file):
        print(f"파일을 찾을 수 없습니다: {data_file}")
        print("핵종 농도 데이터가 필요합니다.")
        return
    
    print(f"데이터 파일 사용: {data_file}")
    
    # 1. Surface plot
    print("1. 3D 표면도 생성 중...")
    create_nuclide_time_3d_surface(data_file, "cram_result/nuclide_time_3d/nuclide_surface.png")
    
    # 2. Wireframe plot
    print("2. 3D 와이어프레임 생성 중...")
    create_nuclide_time_3d_wireframe(data_file, "cram_result/nuclide_time_3d/nuclide_wireframe.png")
    
    # 3. Scatter plot
    print("3. 3D 산점도 생성 중...")
    create_nuclide_time_3d_scatter(data_file, "cram_result/nuclide_time_3d/nuclide_scatter.png")
    
    # 4. Top nuclides focus
    print("4. 상위 핵종 분석 생성 중...")
    create_top_nuclides_3d_evolution(data_file, "cram_result/nuclide_time_3d/top_nuclides.png", top_n=15)
    
    print("\n=== 3D 핵종 농도 가시화 완료! ===")
    print("생성된 파일들:")
    print("- cram_result/nuclide_time_3d/nuclide_surface.png")
    print("- cram_result/nuclide_time_3d/nuclide_wireframe.png") 
    print("- cram_result/nuclide_time_3d/nuclide_scatter.png")
    print("- cram_result/nuclide_time_3d/top_nuclides.png")

if __name__ == "__main__":
    main()