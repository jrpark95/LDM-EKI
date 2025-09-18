import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# CSV 파일 읽기
df = pd.read_csv('validation/concentration_grid_00720.csv')

# 기본 정보 출력
print("데이터 형태:", df.shape)
print("\n컬럼 정보:")
print(df.info())
print("\n기본 통계:")
print(df.describe())

# 1. 2D 히트맵 - 특정 고도에서의 농도 분포
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 각 고도별 농도 히트맵
altitudes = sorted(df['alt'].unique())[:4]
for i, alt in enumerate(altitudes):
    row, col = i // 2, i % 2
    subset = df[df['alt'] == alt]
    pivot_data = subset.pivot_table(values='concentration', index='lat', columns='lon', aggfunc='mean')
    
    im = axes[row, col].imshow(pivot_data, cmap='viridis', aspect='auto')
    axes[row, col].set_title(f'농도 분포 (고도 {alt}m)')
    axes[row, col].set_xlabel('경도')
    axes[row, col].set_ylabel('위도')
    plt.colorbar(im, ax=axes[row, col])

plt.tight_layout()
plt.savefig('concentration_heatmap_by_altitude.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. 3D 산점도 - 농도가 0이 아닌 점들
non_zero = df[df['concentration'] > 0]
if not non_zero.empty:
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(non_zero['lon'], non_zero['lat'], non_zero['alt'], 
                        c=non_zero['concentration'], cmap='plasma', 
                        s=50, alpha=0.6)
    
    ax.set_xlabel('경도 (Longitude)')
    ax.set_ylabel('위도 (Latitude)')
    ax.set_zlabel('고도 (Altitude, m)')
    ax.set_title('3D 농도 분포 (농도 > 0인 지점)')
    
    plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    plt.savefig('concentration_3d_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()

# 3. 고도별 농도 프로파일
plt.figure(figsize=(10, 6))
altitude_profile = df.groupby('alt')['concentration'].mean()
plt.plot(altitude_profile.values, altitude_profile.index, 'o-', linewidth=2, markersize=6)
plt.xlabel('평균 농도')
plt.ylabel('고도 (m)')
plt.title('고도별 평균 농도 프로파일')
plt.grid(True, alpha=0.3)
plt.savefig('concentration_altitude_profile.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. 입자 수 vs 농도 관계
plt.figure(figsize=(10, 6))
plt.scatter(df['particle_count'], df['concentration'], alpha=0.5, s=10)
plt.xlabel('입자 수 (Particle Count)')
plt.ylabel('농도 (Concentration)')
plt.title('입자 수와 농도의 관계')
plt.grid(True, alpha=0.3)

# 상관계수 계산
correlation = df['particle_count'].corr(df['concentration'])
plt.text(0.05, 0.95, f'상관계수: {correlation:.3f}', transform=plt.gca().transAxes, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.savefig('particle_count_vs_concentration.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. 농도 히스토그램
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df['concentration'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('농도')
plt.ylabel('빈도')
plt.title('농도 분포 (전체)')
plt.yscale('log')

plt.subplot(1, 2, 2)
non_zero_conc = df[df['concentration'] > 0]['concentration']
if not non_zero_conc.empty:
    plt.hist(non_zero_conc, bins=30, edgecolor='black', alpha=0.7, color='orange')
    plt.xlabel('농도')
    plt.ylabel('빈도')
    plt.title('농도 분포 (농도 > 0)')
    plt.yscale('log')

plt.tight_layout()
plt.savefig('concentration_histogram.png', dpi=300, bbox_inches='tight')
plt.show()

print("시각화 완료! 다음 파일들이 생성되었습니다:")
print("- concentration_heatmap_by_altitude.png")
print("- concentration_3d_scatter.png") 
print("- concentration_altitude_profile.png")
print("- particle_count_vs_concentration.png")
print("- concentration_histogram.png")