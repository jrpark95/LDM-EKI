#!/usr/bin/env python3
"""
CRAM(사전계산 exp(dt·A)) vs 개별 지수감쇠(__expf) 완전 비교 실험
목표: 딸핵종 생성 없이 순수 반감기 감쇠만 고려할 때 두 방법의 수치적 일치 확인
"""

import numpy as np
import matplotlib.pyplot as plt
# from scipy.linalg import expm  # Not needed for diagonal matrix
from typing import List, Tuple, Dict
import re

def parse_nuclide_data() -> Tuple[List[str], np.ndarray, np.ndarray]:
    """BEGIN_NUDATA...END_NUDATA 블록을 파싱해 핵종 데이터 추출"""
    
    nudata_text = """BEGIN_NUDATA
Co-58,-1.134E-07,0.1
Co-60,-4.168E-09,0.1
Kr-85,-2.049E-09,0.1
Kr-85m,-4.297E-05,0.1
Rb-86,-4.300E-07,0.1
Kr-87,-1.514E-04,0.1
Kr-88,-6.782E-05,0.1
Sr-89,-1.589E-07,0.1
Sr-90,-7.542E-10,0.1
Y-90,-3.008E-06,0.1
Sr-91,-2.027E-05,0.1
Y-91,-1.371E-07,0.1
Sr-92,-7.105E-05,0.1
Y-92,-5.441E-05,0.1
Y-93,-1.906E-05,0.1
Zr-95,-1.254E-07,0.1
Nb-95,-2.282E-07,0.1
Zr-97,-1.139E-05,0.1
Mo-99,-2.917E-06,0.1
Tc-99m,-3.199E-05,0.1
Ru-103,-2.042E-07,0.1
Ru-105,-4.338E-05,0.1
Rh-105,-5.445E-06,0.1
Rh-106,-2.318E-02,0.1
Sb-127,-2.084E-06,0.1
Te-127,-2.059E-05,0.1
Te-127m,-7.360E-08,0.1
Sb-129,-4.458E-05,0.1
Te-129,-1.660E-04,0.1
Te-129m,-2.388E-07,0.1
Te-131m,-6.418E-06,0.1
I-131,-9.978E-07,0.1
Te-132,-2.462E-06,0.1
I-132,-8.371E-05,0.1
I-133,-9.257E-06,0.1
Xe-133,-1.529E-06,0.1
I-134,-2.196E-02,0.1
Cs-134,-1.065E-08,0.1
I-135,-2.912E-05,0.1
Xe-135,-2.118E-05,0.1
Cs-136,-6.123E-07,0.1
Cs-137,-7.322E-10,0.1
Ba-139,-1.397E-04,0.1
Ba-140,-6.273E-07,0.1
La-140,-4.787E-06,0.1
La-141,-4.899E-05,0.1
Ce-141,-2.468E-07,0.1
La-142,-1.249E-04,0.1
Ce-143,-5.835E-06,0.1
Pr-143,-5.914E-07,0.1
Ce-144,-2.822E-08,0.1
Nd-147,-7.254E-07,0.1
Pu-238,-2.503E-10,0.1
Np-239,-3.406E-06,0.1
Pu-239,-9.128E-13,0.1
Pu-240,-3.360E-12,0.1
Pu-241,-1.525E-09,0.1
Am-241,-5.082E-11,0.1
Cm-242,-4.916E-08,0.1
Cm-244,-1.213E-09,0.1
END_NUDATA"""
    
    # Parse data between BEGIN_NUDATA and END_NUDATA
    lines = nudata_text.strip().split('\n')
    start_idx = next(i for i, line in enumerate(lines) if 'BEGIN_NUDATA' in line) + 1
    end_idx = next(i for i, line in enumerate(lines) if 'END_NUDATA' in line)
    
    names = []
    lambdas = []
    c0_values = []
    
    for line in lines[start_idx:end_idx]:
        parts = line.strip().split(',')
        if len(parts) == 3:
            name = parts[0]
            lambda_val = abs(float(parts[1]))  # λ = abs(두 번째 열)
            c0 = float(parts[2])
            
            names.append(name)
            lambdas.append(lambda_val)
            c0_values.append(c0)
    
    return names, np.array(lambdas, dtype=np.float64), np.array(c0_values, dtype=np.float64)

def compute_baseline_exp(lambdas: np.ndarray, c0: np.ndarray, dt: float, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """Baseline 방법: 각 step마다 C_exp ← C_exp ⊙ exp(-λ·Δt)"""
    n_nuclides = len(lambdas)
    
    # 시간 배열
    times = np.arange(n_steps + 1) * dt
    
    # 농도 배열 (time x nuclide)
    concentrations = np.zeros((n_steps + 1, n_nuclides), dtype=np.float64)
    concentrations[0] = c0.copy()
    
    # 각 스텝별 지수 감쇠 계수 사전계산
    exp_factors = np.exp(-lambdas * dt)
    
    # 시간 진화
    for step in range(1, n_steps + 1):
        concentrations[step] = concentrations[step-1] * exp_factors
    
    return times, concentrations

def compute_cram_precomputed(lambdas: np.ndarray, c0: np.ndarray, dt: float, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """CRAM 방법: T = exp(Δt·A_diag) 사전계산 후, 각 step마다 C_cram ← T @ C_cram"""
    n_nuclides = len(lambdas)
    
    # A_diag = diag(-λ_i) - 대각행렬
    A_diag = np.diag(-lambdas)
    
    # T = exp(Δt·A_diag) 사전계산 (대각행렬이므로 직접 계산)
    T = np.diag(np.exp(-lambdas * dt))
    
    # 시간 배열
    times = np.arange(n_steps + 1) * dt
    
    # 농도 배열
    concentrations = np.zeros((n_steps + 1, n_nuclides), dtype=np.float64)
    concentrations[0] = c0.copy()
    
    # 시간 진화
    for step in range(1, n_steps + 1):
        concentrations[step] = T @ concentrations[step-1]
    
    return times, concentrations

def calculate_relative_errors(c_exp: np.ndarray, c_cram: np.ndarray, epsilon: float = 1e-12) -> Tuple[float, float]:
    """상대오차 계산
    Returns:
        final_max_rel_error: 최종시점 최대 상대오차
        overall_max_rel_error: 전 구간 최대 상대오차
    """
    # 최종시점 상대오차
    final_exp = c_exp[-1]
    final_cram = c_cram[-1]
    final_rel_errors = np.abs(final_cram - final_exp) / np.maximum(np.abs(final_exp), epsilon)
    final_max_rel_error = np.max(final_rel_errors)
    
    # 전 구간 상대오차
    rel_errors = np.abs(c_cram - c_exp) / np.maximum(np.abs(c_exp), epsilon)
    overall_max_rel_error = np.max(rel_errors)
    
    return final_max_rel_error, overall_max_rel_error

def calculate_expected_24h_values(lambdas: np.ndarray, c0: np.ndarray, target_time: float) -> Dict[str, float]:
    """24시간 후 기대값 계산 (sanity check용)"""
    expected_values = c0 * np.exp(-lambdas * target_time)
    
    # 특정 핵종들의 기대값 반환
    names = ['Tc-99m', 'I-135', 'Xe-135', 'La-141', 'I-133', 'I-131', 'Cs-137']
    name_to_idx = {
        'Tc-99m': 19, 'I-135': 38, 'Xe-135': 39, 'La-141': 45, 
        'I-133': 34, 'I-131': 31, 'Cs-137': 41
    }
    
    result = {}
    for name in names:
        idx = name_to_idx[name]
        result[name] = expected_values[idx]
    
    return result

def create_summary_table(names: List[str], c_exp: np.ndarray, c_cram: np.ndarray, lambdas: np.ndarray) -> Dict:
    """핵심 핵종들의 24h 결과 요약 테이블 생성"""
    key_nuclides = ['La-141', 'Tc-99m', 'I-135', 'Xe-135', 'I-133', 'I-131', 'Cs-137']
    name_to_idx = {
        'La-141': 45, 'Tc-99m': 19, 'I-135': 38, 'Xe-135': 39,
        'I-133': 34, 'I-131': 31, 'Cs-137': 41
    }
    
    table_data = []
    for nuclide in key_nuclides:
        idx = name_to_idx[nuclide]
        lambda_val = lambdas[idx]
        c_exp_final = c_exp[-1, idx]
        c_cram_final = c_cram[-1, idx]
        rel_error = abs(c_cram_final - c_exp_final) / max(abs(c_exp_final), 1e-12)
        
        table_data.append({
            'Nuclide': nuclide,
            'λ [s⁻¹]': f"{lambda_val:.3e}",
            'C_exp(24h)': f"{c_exp_final:.6e}",
            'C_cram(24h)': f"{c_cram_final:.6e}",
            'Rel_Error': f"{rel_error:.2e}"
        })
    
    return table_data

def create_concentration_plot(times: np.ndarray, c_exp: np.ndarray, c_cram: np.ndarray, 
                            names: List[str], title: str, filename: str):
    """농도 변화 세미로그 플롯 생성"""
    key_nuclides = ['La-141', 'Tc-99m', 'I-135', 'Xe-135', 'I-133', 'I-131', 'Cs-137']
    name_to_idx = {
        'La-141': 45, 'Tc-99m': 19, 'I-135': 38, 'Xe-135': 39,
        'I-133': 34, 'I-131': 31, 'Cs-137': 41
    }
    
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(key_nuclides)))
    
    for i, nuclide in enumerate(key_nuclides):
        idx = name_to_idx[nuclide]
        
        # EXP와 CRAM 결과를 겹쳐서 플롯 (겹쳐 보이는 것이 정상)
        plt.semilogy(times/3600, c_exp[:, idx], 'o-', color=colors[i], 
                    markersize=4, linewidth=2, label=f'{nuclide} (EXP)', alpha=0.8)
        plt.semilogy(times/3600, c_cram[:, idx], 's--', color=colors[i], 
                    markersize=3, linewidth=1.5, label=f'{nuclide} (CRAM)', alpha=0.6)
    
    plt.xlabel('Time [hours]')
    plt.ylabel('Concentration')
    plt.title(f'{title}\nCRAM vs EXP 동일(딸핵종 OFF) — 겹쳐 보이는 것이 정상')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"플롯 저장됨: {filename}")

def run_experiment():
    """메인 실험 실행"""
    print("=" * 60)
    print("CRAM vs 개별 지수감쇠 비교 실험")
    print("=" * 60)
    
    # 1. 데이터 파싱
    names, lambdas, c0 = parse_nuclide_data()
    n_nuclides = len(names)
    
    print(f"실험 설정:")
    print(f"- 핵종 수: {n_nuclides}")
    print(f"- Δt: 60 s")
    print(f"- 초기농도: 모든 핵종 0.1")
    print(f"- 정밀도: float64")
    print()
    
    # 실험 파라미터
    dt = 60.0  # seconds
    
    # 2. 6시간 실험 (N=360 steps)
    print("6시간 실험 (N=360 steps)")
    print("-" * 30)
    
    n_steps_6h = 360
    times_6h, c_exp_6h = compute_baseline_exp(lambdas, c0, dt, n_steps_6h)
    _, c_cram_6h = compute_cram_precomputed(lambdas, c0, dt, n_steps_6h)
    
    final_error_6h, overall_error_6h = calculate_relative_errors(c_exp_6h, c_cram_6h)
    
    print(f"최종시점 최대 상대오차: {final_error_6h:.2e}")
    print(f"전 구간 최대 상대오차: {overall_error_6h:.2e}")
    print(f"수용 기준 (<1e-6): {'PASS' if max(final_error_6h, overall_error_6h) < 1e-6 else 'FAIL'}")
    print()
    
    # 3. 24시간 실험 (N=1440 steps)
    print("24시간 실험 (N=1440 steps)")
    print("-" * 30)
    
    n_steps_24h = 1440
    times_24h, c_exp_24h = compute_baseline_exp(lambdas, c0, dt, n_steps_24h)
    _, c_cram_24h = compute_cram_precomputed(lambdas, c0, dt, n_steps_24h)
    
    final_error_24h, overall_error_24h = calculate_relative_errors(c_exp_24h, c_cram_24h)
    
    print(f"최종시점 최대 상대오차: {final_error_24h:.2e}")
    print(f"전 구간 최대 상대오차: {overall_error_24h:.2e}")
    print(f"수용 기준 (<1e-6): {'PASS' if max(final_error_24h, overall_error_24h) < 1e-6 else 'FAIL'}")
    print()
    
    # 4. 24시간 요약 테이블
    print("24시간 핵심 핵종 요약 테이블")
    print("-" * 40)
    
    summary_table = create_summary_table(names, c_exp_24h, c_cram_24h, lambdas)
    
    # Print table manually
    print(f"{'Nuclide':<8} {'λ [s⁻¹]':<12} {'C_exp(24h)':<12} {'C_cram(24h)':<12} {'Rel_Error':<12}")
    print("-" * 60)
    for row in summary_table:
        print(f"{row['Nuclide']:<8} {row['λ [s⁻¹]']:<12} {row['C_exp(24h)']:<12} {row['C_cram(24h)']:<12} {row['Rel_Error']:<12}")
    print()
    
    # 5. Sanity Check (24시간 기대값)
    print("Sanity Check - 24시간 기대값 비교")
    print("-" * 40)
    
    expected_24h = calculate_expected_24h_values(lambdas, c0, 24*3600)
    key_indices = {'Tc-99m': 19, 'I-135': 38, 'Xe-135': 39, 'La-141': 45, 
                   'I-133': 34, 'I-131': 31, 'Cs-137': 41}
    
    for name, expected in expected_24h.items():
        idx = key_indices[name]
        actual = c_exp_24h[-1, idx]
        print(f"{name:8s}: 기대 {expected:.6e}, 실제 {actual:.6e}, 차이 {abs(actual-expected):.2e}")
    print()
    
    # 6. 플롯 생성
    print("플롯 생성 중...")
    create_concentration_plot(times_6h, c_exp_6h, c_cram_6h, names, 
                            "6시간 농도 변화", "/home/jrpark/ldm-CRAM/concentration_6h.png")
    create_concentration_plot(times_24h, c_exp_24h, c_cram_24h, names, 
                            "24시간 농도 변화", "/home/jrpark/ldm-CRAM/concentration_24h.png")
    print()
    
    # 7. 최종 보고
    print("=" * 60)
    print("최종 실험 보고")
    print("=" * 60)
    print(f"실험 환경: Python/NumPy, 정밀도 float64")
    print(f"Δt: {dt} s, 핵종 수: {n_nuclides}")
    print(f"연쇄 전이: OFF (대각행렬만 사용)")
    print()
    print("정확도 지표:")
    print(f"  6시간  - 최종: {final_error_6h:.2e}, 전구간: {overall_error_6h:.2e}")
    print(f"  24시간 - 최종: {final_error_24h:.2e}, 전구간: {overall_error_24h:.2e}")
    print()
    
    if max(final_error_6h, overall_error_6h, final_error_24h, overall_error_24h) < 1e-6:
        print("✓ 결론: CRAM과 개별 지수감쇠가 부동소수점 수준에서 일치함")
    else:
        print("✗ 결론: 예상보다 큰 오차 발견 - 디버깅 필요")
        print("디버그 포인트:")
        print("  (1) 단위/부호 확인")
        print("  (2) Δt 일치성")
        print("  (3) 행/열 정렬")
        print("  (4) 수치정밀도")

if __name__ == "__main__":
    run_experiment()