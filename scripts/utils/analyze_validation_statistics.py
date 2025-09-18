import pandas as pd
import numpy as np
import json

# CSV 파일 읽기
df = pd.read_csv('validation/concentration_grid_00720.csv')

print("=" * 80)
print("LDM-CRAM4 검증용 통계 지표 (Timestep 720)")
print("=" * 80)

# 1. 전체 입자 통계
total_particles = df['particle_count'].sum()
active_cells = (df['particle_count'] > 0).sum()
total_cells = len(df)

print("\n1. 전체 입자 분포 통계")
print("-" * 40)
print(f"총 입자 수: {total_particles}")
print(f"활성 격자 수: {active_cells} / {total_cells} ({100*active_cells/total_cells:.2f}%)")
print(f"평균 입자/격자: {df['particle_count'].mean():.2f}")
print(f"표준편차: {df['particle_count'].std():.2f}")
print(f"최대 입자 수: {df['particle_count'].max()}")
print(f"최소 입자 수 (>0): {df[df['particle_count']>0]['particle_count'].min()}")

# 2. 공간 분포 통계
print("\n2. 공간 분포 통계")
print("-" * 40)

# 중심점 계산 (입자 가중 평균)
weighted_lon = (df['lon'] * df['particle_count']).sum() / total_particles
weighted_lat = (df['lat'] * df['particle_count']).sum() / total_particles
weighted_alt = (df['alt'] * df['particle_count']).sum() / total_particles

print(f"입자 분포 중심점:")
print(f"  경도: {weighted_lon:.4f}°")
print(f"  위도: {weighted_lat:.4f}°")
print(f"  고도: {weighted_alt:.1f}m")

# 표준편차 (spread)
lon_std = np.sqrt(((df['lon'] - weighted_lon)**2 * df['particle_count']).sum() / total_particles)
lat_std = np.sqrt(((df['lat'] - weighted_lat)**2 * df['particle_count']).sum() / total_particles)
alt_std = np.sqrt(((df['alt'] - weighted_alt)**2 * df['particle_count']).sum() / total_particles)

print(f"\n분산 정도 (표준편차):")
print(f"  경도 방향: {lon_std:.4f}°")
print(f"  위도 방향: {lat_std:.4f}°")
print(f"  고도 방향: {alt_std:.1f}m")

# 3. 고도별 분포
print("\n3. 고도별 입자 분포")
print("-" * 40)
altitude_dist = df.groupby('alt')['particle_count'].sum().sort_index()
for alt, count in altitude_dist.items():
    if count > 0:
        print(f"  {alt:4.0f}m: {count:5d} ({100*count/total_particles:5.2f}%)")

# 4. 격자별 입자 수 분위수
print("\n4. 격자별 입자 수 분위수 (활성 격자만)")
print("-" * 40)
active_counts = df[df['particle_count'] > 0]['particle_count']
percentiles = [10, 25, 50, 75, 90, 95, 99]
for p in percentiles:
    val = np.percentile(active_counts, p)
    print(f"  {p:2d}th percentile: {val:.0f}")

# 5. 공간 집중도 지표
print("\n5. 공간 집중도 지표")
print("-" * 40)

# Gini coefficient 계산
sorted_counts = np.sort(df['particle_count'].values)
n = len(sorted_counts)
index = np.arange(1, n + 1)
gini = (2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n
print(f"Gini 계수: {gini:.4f} (0=완전균등, 1=완전집중)")

# Top N% 격자가 차지하는 입자 비율
top_percentages = [1, 5, 10, 20]
sorted_df = df.sort_values('particle_count', ascending=False)
for top_p in top_percentages:
    n_cells = int(len(df) * top_p / 100)
    top_particles = sorted_df.head(n_cells)['particle_count'].sum()
    print(f"상위 {top_p:2d}% 격자의 입자 비율: {100*top_particles/total_particles:.1f}%")

# 6. 3D 공간 모멘트
print("\n6. 3D 공간 모멘트")
print("-" * 40)

# 간단한 비대칭성 지표 (평균-중앙값 비교)
active_df = df[df['particle_count'] > 0]

def weighted_median(values, weights):
    sorted_indices = np.argsort(values.values)
    sorted_values = values.values[sorted_indices]
    sorted_weights = weights.values[sorted_indices]
    cumsum = np.cumsum(sorted_weights)
    cutoff = sorted_weights.sum() / 2.0
    return sorted_values[cumsum >= cutoff][0]

lon_median = weighted_median(active_df['lon'], active_df['particle_count'])
lat_median = weighted_median(active_df['lat'], active_df['particle_count'])
alt_median = weighted_median(active_df['alt'], active_df['particle_count'])

print(f"분포 비대칭성 (평균-중앙값):")
print(f"  경도: {weighted_lon - lon_median:.4f}")
print(f"  위도: {weighted_lat - lat_median:.4f}")
print(f"  고도: {weighted_alt - alt_median:.1f}")

# 7. 검증용 핵심 지표 요약 (JSON 저장)
validation_metrics = {
    "timestep": 720,
    "total_particles": int(total_particles),
    "active_cells": int(active_cells),
    "center_of_mass": {
        "lon": float(weighted_lon),
        "lat": float(weighted_lat),
        "alt": float(weighted_alt)
    },
    "spatial_spread": {
        "lon_std": float(lon_std),
        "lat_std": float(lat_std),
        "alt_std": float(alt_std)
    },
    "altitude_distribution": {
        int(alt): int(count) for alt, count in altitude_dist.items() if count > 0
    },
    "percentiles": {
        f"p{p}": float(np.percentile(active_counts, p)) for p in [25, 50, 75, 90]
    },
    "gini_coefficient": float(gini),
    "top_10pct_concentration": float(sorted_df.head(int(len(df)*0.1))['particle_count'].sum() / total_particles)
}

# JSON 파일로 저장
with open('validation_reference_metrics.json', 'w') as f:
    json.dump(validation_metrics, f, indent=2)

print("\n" + "=" * 80)
print("검증 지표가 'validation_reference_metrics.json'에 저장되었습니다.")
print("=" * 80)

# 허용 오차 범위 제안
print("\n권장 허용 오차 범위 (몬테카를로 시뮬레이션 특성 고려):")
print("-" * 40)
print("- 총 입자 수: ±5%")
print("- 활성 격자 수: ±10%")
print("- 분포 중심점: ±0.01° (경도/위도), ±50m (고도)")
print("- 공간 분산: ±15%")
print("- Gini 계수: ±0.05")
print("- 분위수: ±20%")