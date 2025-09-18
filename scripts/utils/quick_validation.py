#!/usr/bin/env python3
"""
빠른 검증 도구 - 핵심 지표만 빠르게 확인
"""

import pandas as pd
import numpy as np
import json
import sys

def quick_check(csv_file):
    """핵심 지표만 빠르게 확인"""
    try:
        # 기준 데이터 로드
        with open('validation_reference_metrics.json', 'r') as f:
            ref = json.load(f)
        
        # 현재 데이터 로드
        df = pd.read_csv(csv_file)
        
        # 핵심 지표 계산
        total_particles = df['particle_count'].sum()
        active_cells = (df['particle_count'] > 0).sum()
        weighted_lon = (df['lon'] * df['particle_count']).sum() / total_particles
        weighted_lat = (df['lat'] * df['particle_count']).sum() / total_particles
        
        # 빠른 비교
        particle_diff = abs(total_particles - ref["total_particles"]) / ref["total_particles"] * 100
        cells_diff = abs(active_cells - ref["active_cells"]) / ref["active_cells"] * 100
        lon_diff = abs(weighted_lon - ref["center_of_mass"]["lon"])
        lat_diff = abs(weighted_lat - ref["center_of_mass"]["lat"])
        
        # 결과 출력
        print(f"빠른 검증: {csv_file}")
        print(f"  총 입자: {total_particles} (차이: {particle_diff:.1f}%)")
        print(f"  활성 격자: {active_cells} (차이: {cells_diff:.1f}%)")
        print(f"  중심점: ({weighted_lon:.3f}, {weighted_lat:.3f}) (차이: {lon_diff:.4f}°, {lat_diff:.4f}°)")
        
        # 간단한 pass/fail
        if particle_diff <= 5 and cells_diff <= 10 and lon_diff <= 0.01 and lat_diff <= 0.01:
            print("  상태: ✓ 정상")
            return True
        else:
            print("  상태: ❌ 이상")
            return False
            
    except Exception as e:
        print(f"오류: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python quick_validation.py <결과파일.csv>")
        sys.exit(1)
    
    success = quick_check(sys.argv[1])
    sys.exit(0 if success else 1)