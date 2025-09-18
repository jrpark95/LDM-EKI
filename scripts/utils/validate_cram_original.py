#!/usr/bin/env python3
"""
추가 검증: CRAM48 직접 구현 vs 대각행렬 사전계산 비교
목표: 대각행렬에서 CRAM48 알고리즘과 직접 exp 계산이 동일한지 확인
"""

import numpy as np
import matplotlib.pyplot as plt

# CRAM48 상수들 (원본과 동일)
CRAM48_ALPHA0 = 2.258038182743983e-47
cram48_alpha_re = np.array([
    6.387380733878774e+2, 1.909896179065730e+2, 4.236195226571914e+2, 4.645770595258726e+2,
    7.765163276752433e+2, 1.907115136768522e+3, 2.909892685603256e+3, 1.944772206620450e+2,
    1.382799786972332e+5, 5.628442079602433e+3, 2.151681283794220e+2, 1.324720240514420e+3,
    1.617548476343347e+4, 1.112729040439685e+2, 1.074624783191125e+2, 8.835727765158191e+1,
    9.354078136054179e+1, 9.418142823531573e+1, 1.040012390717851e+2, 6.861882624343235e+1,
    8.766654491283722e+1, 1.056007619389650e+2, 7.738987569039419e+1, 1.041366366475571e+2
])

cram48_alpha_im = np.array([
    -6.743912502859256e+2, -3.973203432721332e+2, -2.041233768918671e+3, -1.652917287299683e+3,
    -1.783617639907328e+4, -5.887068595142284e+4, -9.953255345514560e+3, -1.427131226068449e+3,
    -3.256885197214938e+6, -2.924284515884309e+4, -1.121774011188224e+3, -6.370088443140973e+4,
    -1.008798413156542e+6, -8.837109731680418e+1, -1.457246116408180e+2, -6.388286188419360e+1,
    -2.195424319460237e+2, -6.719055740098035e+2, -1.693747595553868e+2, -1.177598523430493e+1,
    -4.596464999363902e+3, -1.738294585524067e+3, -4.311715386228984e+1, -2.777743732451969e+2
])

cram48_theta_re = np.array([
    -4.465731934165702e+1, -5.284616241568964e+0, -8.867715667624458e+0, 3.493013124279215e+0,
    1.564102508858634e+1, 1.742097597385893e+1, -2.834466755180654e+1, 1.661569367939544e+1,
    8.011836167974721e+0, -2.056267541998229e+0, 1.449208170441839e+1, 1.853807176907916e+1,
    9.932562704505182e+0, -2.244223871767187e+1, 8.590014121680897e-1, -1.286192925744479e+1,
    1.164596909542055e+1, 1.806076684783089e+1, 5.870672154659249e+0, -3.542938819659747e+1,
    1.901323489060250e+1, 1.885508331552577e+1, -1.734689708174982e+1, 1.316284237125190e+1
])

cram48_theta_im = np.array([
    6.233225190695437e+1, 4.057499381311059e+1, 4.325515754166724e+1, 3.281615453173585e+1,
    1.558061616372237e+1, 1.076629305714420e+1, 5.492841024648724e+1, 1.316994930024688e+1,
    2.780232111309410e+1, 3.794824788914354e+1, 1.799988210051809e+1, 5.974332563100539e+0,
    2.532823409972962e+1, 5.179633600312162e+1, 3.536456194294350e+1, 4.600304902833652e+1,
    2.287153304140217e+1, 8.368200580099821e+0, 3.029700159040121e+1, 5.834381701800013e+1,
    1.194282058271408e+0, 3.583428564427879e+0, 4.883941101108207e+1, 2.042951874827759e+1
])

def gauss_solve(A, b):
    """Gaussian elimination solver"""
    n = len(b)
    A_copy = A.copy()
    x = b.copy()
    
    for k in range(n):
        # Find pivot
        piv = k
        pmax = abs(A_copy[k, k])
        for i in range(k+1, n):
            if abs(A_copy[i, k]) > pmax:
                pmax = abs(A_copy[i, k])
                piv = i
        
        if pmax < 1e-20:
            continue
        
        # Swap rows
        if piv != k:
            A_copy[[k, piv]] = A_copy[[piv, k]]
            x[k], x[piv] = x[piv], x[k]
        
        # Scale pivot row
        pivot_val = A_copy[k, k]
        A_copy[k, k:] /= pivot_val
        x[k] /= pivot_val
        
        # Eliminate column
        for i in range(n):
            if i != k and abs(A_copy[i, k]) > 1e-20:
                factor = A_copy[i, k]
                A_copy[i, k:] -= factor * A_copy[k, k:]
                x[i] -= factor * x[k]
    
    return x

def compute_exp_cram48_full(A, dt):
    """CRAM48 전체 구현 (복소수 시스템 포함)"""
    n = len(A)
    dim = 2 * n
    
    # B = dt * A (주의: 부호 확인)
    B = dt * A
    
    # Result matrix initialization (identity)
    result = np.eye(n, dtype=np.float64)
    
    # CRAM48 iteration for each column
    for col in range(n):
        column_result = np.zeros(n, dtype=np.float64)
        column_result[col] = 1.0  # Unit vector for this column
        
        for k in range(24):
            tr = cram48_theta_re[k]
            ti = cram48_theta_im[k]
            ar = cram48_alpha_re[k]
            ai = cram48_alpha_im[k]
            
            # Build complex system matrix M
            M = np.zeros((dim, dim), dtype=np.float64)
            for i in range(n):
                for j in range(n):
                    M[i, j] = B[i, j]
                    M[i+n, j+n] = B[i, j]
                M[i, i] -= tr
                M[i+n, i+n] -= tr
                M[i, i+n] = ti
                M[i+n, i] = -ti
            
            # Build RHS vector
            b_vec = np.zeros(dim, dtype=np.float64)
            for i in range(n):
                b_vec[i] = column_result[i]
            
            # Solve system
            x_vec = gauss_solve(M, b_vec)
            
            # Update result
            for i in range(n):
                re = ar * x_vec[i] - ai * x_vec[i+n]
                column_result[i] += 2.0 * re
        
        # Apply ALPHA0 and store
        result[:, col] = column_result * CRAM48_ALPHA0
    
    return result

def test_diagonal_matrix_cram():
    """대각행렬에서 CRAM48 vs 직접 exp 비교"""
    print("=" * 60)
    print("대각행렬 CRAM48 vs 직접 exp 비교 테스트")
    print("=" * 60)
    
    # 테스트 케이스들
    test_cases = [
        ("단일 λ", np.array([[-1e-5]])),
        ("2x2 대각", np.array([[-1e-5, 0], [0, -2e-6]])),
        ("3x3 대각", np.array([[-1e-5, 0, 0], [0, -2e-6, 0], [0, 0, -5e-7]])),
    ]
    
    dt = 60.0  # 60초
    
    for test_name, A in test_cases:
        print(f"\n{test_name} 테스트")
        print("-" * 30)
        
        # 직접 exp 계산 (대각행렬이므로)
        lambdas = -np.diag(A)  # λ = -A_ii
        direct_exp = np.diag(np.exp(-lambdas * dt))
        
        # CRAM48 계산
        cram48_result = compute_exp_cram48_full(A, dt)
        
        # 비교
        max_abs_error = np.max(np.abs(cram48_result - direct_exp))
        max_rel_error = np.max(np.abs(cram48_result - direct_exp) / 
                              np.maximum(np.abs(direct_exp), 1e-12))
        
        print(f"최대 절대오차: {max_abs_error:.2e}")
        print(f"최대 상대오차: {max_rel_error:.2e}")
        
        # 세부 비교 (첫 번째 대각 원소)
        print(f"Direct exp[0,0]: {direct_exp[0,0]:.12e}")
        print(f"CRAM48[0,0]:     {cram48_result[0,0]:.12e}")
        print(f"차이:           {abs(cram48_result[0,0] - direct_exp[0,0]):.2e}")
        
        if max_rel_error < 1e-10:
            print("✓ PASS - 매우 정확함")
        elif max_rel_error < 1e-6:
            print("✓ PASS - 허용 오차 내")
        else:
            print("✗ FAIL - 오차가 큼")

def test_sign_convention():
    """부호 규약 테스트"""
    print("\n" + "=" * 60)
    print("부호 규약 검증")
    print("=" * 60)
    
    # A60.csv의 첫 번째 원소 (-1.134E-07)
    A_negative = np.array([[-1.134e-7]])  # A는 음수
    dt = 60.0
    
    print(f"A = {A_negative[0,0]:.3e} (음수)")
    print(f"dt = {dt} s")
    print()
    
    # 방법 1: B = dt * A (원본 cram_runtime.cu 방식)
    print("방법 1: B = dt * A (원본 방식)")
    cram48_wrong = compute_exp_cram48_full(A_negative, dt)
    print(f"T = exp(dt*A) = {cram48_wrong[0,0]:.6f}")
    print(f"물리적 의미: {'성장' if cram48_wrong[0,0] > 1.0 else '감쇠'}")
    print()
    
    # 방법 2: B = -dt * A (수정된 방식)
    print("방법 2: B = -dt * A (수정된 방식)")
    cram48_correct = compute_exp_cram48_full(-A_negative, dt)  # A에 마이너스
    print(f"T = exp(-dt*A) = {cram48_correct[0,0]:.6f}")
    print(f"물리적 의미: {'성장' if cram48_correct[0,0] > 1.0 else '감쇠'}")
    print()
    
    # 기대값 (직접 계산)
    lambda_val = 1.134e-7  # λ = |A|
    expected = np.exp(-lambda_val * dt)
    print(f"기대값 exp(-λ*dt) = {expected:.6f} (감쇠)")
    print()
    
    print("결론:")
    if abs(cram48_correct[0,0] - expected) < 1e-10:
        print("✓ 방법 2 (B = -dt*A)가 올바름")
    else:
        print("✗ 둘 다 문제있음")
    
    if abs(cram48_wrong[0,0] - expected) < 1e-10:
        print("? 방법 1도 맞음 (예상과 다름)")
    else:
        print("✗ 방법 1은 틀림")

if __name__ == "__main__":
    test_sign_convention()
    test_diagonal_matrix_cram()