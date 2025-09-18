#!/usr/bin/env python3
"""
LDM-CRAM4 검증 도구
새로운 시뮬레이션 결과를 기준 데이터와 비교하여 일치성을 확인합니다.
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path

class ValidationChecker:
    def __init__(self, reference_file='validation_reference_metrics.json'):
        """기준 데이터를 로드합니다."""
        try:
            with open(reference_file, 'r') as f:
                self.reference = json.load(f)
            print(f"✓ 기준 데이터 로드: {reference_file}")
        except FileNotFoundError:
            print(f"❌ 기준 데이터 파일을 찾을 수 없습니다: {reference_file}")
            sys.exit(1)
    
    def calculate_metrics(self, csv_file):
        """CSV 파일로부터 검증 지표를 계산합니다."""
        try:
            df = pd.read_csv(csv_file)
        except FileNotFoundError:
            print(f"❌ 시뮬레이션 결과 파일을 찾을 수 없습니다: {csv_file}")
            return None
        
        # 기본 통계
        total_particles = df['particle_count'].sum()
        active_cells = (df['particle_count'] > 0).sum()
        
        # 질량 중심
        weighted_lon = (df['lon'] * df['particle_count']).sum() / total_particles
        weighted_lat = (df['lat'] * df['particle_count']).sum() / total_particles
        weighted_alt = (df['alt'] * df['particle_count']).sum() / total_particles
        
        # 공간 분산
        lon_std = np.sqrt(((df['lon'] - weighted_lon)**2 * df['particle_count']).sum() / total_particles)
        lat_std = np.sqrt(((df['lat'] - weighted_lat)**2 * df['particle_count']).sum() / total_particles)
        alt_std = np.sqrt(((df['alt'] - weighted_alt)**2 * df['particle_count']).sum() / total_particles)
        
        # 고도별 분포
        altitude_dist = df.groupby('alt')['particle_count'].sum().to_dict()
        
        # 분위수 (활성 격자만)
        active_counts = df[df['particle_count'] > 0]['particle_count']
        percentiles = {
            f"p{p}": float(np.percentile(active_counts, p)) for p in [25, 50, 75, 90]
        }
        
        # Gini 계수
        sorted_counts = np.sort(df['particle_count'].values)
        n = len(sorted_counts)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n
        
        # Top 10% 집중도
        sorted_df = df.sort_values('particle_count', ascending=False)
        top_10pct_particles = sorted_df.head(int(len(df)*0.1))['particle_count'].sum()
        top_10pct_concentration = top_10pct_particles / total_particles
        
        return {
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
            "altitude_distribution": {int(k): int(v) for k, v in altitude_dist.items()},
            "percentiles": percentiles,
            "gini_coefficient": float(gini),
            "top_10pct_concentration": float(top_10pct_concentration)
        }
    
    def compare_metrics(self, current_metrics):
        """현재 지표를 기준 지표와 비교합니다."""
        if current_metrics is None:
            return False, []
        
        results = []
        all_passed = True
        
        # 1. 총 입자 수 (±5%)
        ref_particles = self.reference["total_particles"]
        cur_particles = current_metrics["total_particles"]
        diff_pct = abs(cur_particles - ref_particles) / ref_particles * 100
        passed = diff_pct <= 5.0
        results.append({
            "metric": "총 입자 수",
            "reference": ref_particles,
            "current": cur_particles,
            "diff_pct": diff_pct,
            "threshold": 5.0,
            "passed": passed
        })
        if not passed: all_passed = False
        
        # 2. 활성 격자 수 (±10%)
        ref_cells = self.reference["active_cells"]
        cur_cells = current_metrics["active_cells"]
        diff_pct = abs(cur_cells - ref_cells) / ref_cells * 100
        passed = diff_pct <= 10.0
        results.append({
            "metric": "활성 격자 수",
            "reference": ref_cells,
            "current": cur_cells,
            "diff_pct": diff_pct,
            "threshold": 10.0,
            "passed": passed
        })
        if not passed: all_passed = False
        
        # 3. 분포 중심점
        ref_cm = self.reference["center_of_mass"]
        cur_cm = current_metrics["center_of_mass"]
        
        # 경도 (±0.01°)
        lon_diff = abs(cur_cm["lon"] - ref_cm["lon"])
        passed = lon_diff <= 0.01
        results.append({
            "metric": "중심점 경도",
            "reference": ref_cm["lon"],
            "current": cur_cm["lon"],
            "diff_abs": lon_diff,
            "threshold": 0.01,
            "passed": passed
        })
        if not passed: all_passed = False
        
        # 위도 (±0.01°)
        lat_diff = abs(cur_cm["lat"] - ref_cm["lat"])
        passed = lat_diff <= 0.01
        results.append({
            "metric": "중심점 위도",
            "reference": ref_cm["lat"],
            "current": cur_cm["lat"],
            "diff_abs": lat_diff,
            "threshold": 0.01,
            "passed": passed
        })
        if not passed: all_passed = False
        
        # 고도 (±50m)
        alt_diff = abs(cur_cm["alt"] - ref_cm["alt"])
        passed = alt_diff <= 50.0
        results.append({
            "metric": "중심점 고도",
            "reference": ref_cm["alt"],
            "current": cur_cm["alt"],
            "diff_abs": alt_diff,
            "threshold": 50.0,
            "passed": passed
        })
        if not passed: all_passed = False
        
        # 4. 공간 분산 (±15%)
        ref_spread = self.reference["spatial_spread"]
        cur_spread = current_metrics["spatial_spread"]
        
        for coord in ["lon_std", "lat_std", "alt_std"]:
            diff_pct = abs(cur_spread[coord] - ref_spread[coord]) / ref_spread[coord] * 100
            passed = diff_pct <= 15.0
            results.append({
                "metric": f"분산 {coord}",
                "reference": ref_spread[coord],
                "current": cur_spread[coord],
                "diff_pct": diff_pct,
                "threshold": 15.0,
                "passed": passed
            })
            if not passed: all_passed = False
        
        # 5. Gini 계수 (±0.05)
        ref_gini = self.reference["gini_coefficient"]
        cur_gini = current_metrics["gini_coefficient"]
        diff_abs = abs(cur_gini - ref_gini)
        passed = diff_abs <= 0.05
        results.append({
            "metric": "Gini 계수",
            "reference": ref_gini,
            "current": cur_gini,
            "diff_abs": diff_abs,
            "threshold": 0.05,
            "passed": passed
        })
        if not passed: all_passed = False
        
        # 6. 분위수 (±20%)
        ref_percentiles = self.reference["percentiles"]
        cur_percentiles = current_metrics["percentiles"]
        
        for p in ["p25", "p50", "p75", "p90"]:
            diff_pct = abs(cur_percentiles[p] - ref_percentiles[p]) / ref_percentiles[p] * 100
            passed = diff_pct <= 20.0
            results.append({
                "metric": f"분위수 {p}",
                "reference": ref_percentiles[p],
                "current": cur_percentiles[p],
                "diff_pct": diff_pct,
                "threshold": 20.0,
                "passed": passed
            })
            if not passed: all_passed = False
        
        return all_passed, results
    
    def print_results(self, results, all_passed):
        """검증 결과를 출력합니다."""
        print("\n" + "="*80)
        print("검증 결과")
        print("="*80)
        
        passed_count = sum(1 for r in results if r["passed"])
        total_count = len(results)
        
        print(f"전체 검증: {'✓ PASSED' if all_passed else '❌ FAILED'}")
        print(f"통과율: {passed_count}/{total_count} ({100*passed_count/total_count:.1f}%)")
        print()
        
        for result in results:
            status = "✓" if result["passed"] else "❌"
            metric = result["metric"]
            ref = result["reference"]
            cur = result["current"]
            
            if "diff_pct" in result:
                diff = result["diff_pct"]
                threshold = result["threshold"]
                print(f"{status} {metric:15s}: {cur:8.3f} (기준: {ref:8.3f}, 차이: {diff:5.1f}%, 허용: ±{threshold}%)")
            else:
                diff = result["diff_abs"]
                threshold = result["threshold"]
                print(f"{status} {metric:15s}: {cur:8.3f} (기준: {ref:8.3f}, 차이: {diff:8.3f}, 허용: ±{threshold})")
        
        return all_passed
    
    def validate(self, csv_file):
        """전체 검증 과정을 실행합니다."""
        print(f"시뮬레이션 결과 검증: {csv_file}")
        
        # 지표 계산
        current_metrics = self.calculate_metrics(csv_file)
        if current_metrics is None:
            return False
        
        # 비교 수행
        all_passed, results = self.compare_metrics(current_metrics)
        
        # 결과 출력
        return self.print_results(results, all_passed)

def main():
    """명령행에서 사용할 수 있는 메인 함수"""
    if len(sys.argv) != 2:
        print("사용법: python validation_checker.py <시뮬레이션_결과.csv>")
        print("예시: python validation_checker.py validation/concentration_grid_00720.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    checker = ValidationChecker()
    passed = checker.validate(csv_file)
    
    sys.exit(0 if passed else 1)

if __name__ == "__main__":
    main()