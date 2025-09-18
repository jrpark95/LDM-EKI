#include "cram_runtime.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define N_NUCLIDES 60

// External function from ldm_cram2.cuh
extern bool compute_exp_matrix(float dt, bool use_cram48);


// CRAM48 constants hardcoded to avoid device memory issues
const double h_CRAM48_ALPHA0 = 2.258038182743983e-47;
const double h_cram48_alpha_re[24] = {
    6.387380733878774e+2, 1.909896179065730e+2, 4.236195226571914e+2, 4.645770595258726e+2,
    7.765163276752433e+2, 1.907115136768522e+3, 2.909892685603256e+3, 1.944772206620450e+2,
    1.382799786972332e+5, 5.628442079602433e+3, 2.151681283794220e+2, 1.324720240514420e+3,
    1.617548476343347e+4, 1.112729040439685e+2, 1.074624783191125e+2, 8.835727765158191e+1,
    9.354078136054179e+1, 9.418142823531573e+1, 1.040012390717851e+2, 6.861882624343235e+1,
    8.766654491283722e+1, 1.056007619389650e+2, 7.738987569039419e+1, 1.041366366475571e+2
};

// Host version of Gaussian elimination
void host_cram_gauss_solve(double* A, double* b, double* x, int n) {
    for(int i = 0; i < n; i++) x[i] = b[i];
    
    for(int k = 0; k < n; k++) {
        int piv = k;
        double pmax = fabs(A[k*n + k]);
        for(int i = k+1; i < n; i++) {
            double v = fabs(A[i*n + k]);
            if(v > pmax) { pmax = v; piv = i; }
        }
        if(pmax < 1e-20) return;
        
        if(piv != k) {
            for(int j = k; j < n; j++) {
                double temp = A[k*n + j];
                A[k*n + j] = A[piv*n + j];
                A[piv*n + j] = temp;
            }
            double temp = x[k]; x[k] = x[piv]; x[piv] = temp;
        }
        
        double pivVal = A[k*n + k];
        for(int j = k; j < n; j++) A[k*n + j] /= pivVal;
        x[k] /= pivVal;
        
        for(int i = 0; i < n; i++) {
            if(i != k && fabs(A[i*n + k]) > 1e-20) {
                double factor = A[i*n + k];
                for(int j = k; j < n; j++) A[i*n + j] -= factor * A[k*n + j];
                x[i] -= factor * x[k];
            }
        }
    }
}

// Simple CRAM precompute function
bool cram_precompute_transition(float dt) {
    std::cout << "CRAM precompute transition called with dt=" << dt << std::endl;
    std::cout << "[DEBUG] About to call compute_exp_matrix..." << std::endl;
    bool result = compute_exp_matrix(dt, true);
    std::cout << "[DEBUG] compute_exp_matrix returned: " << (result ? "true" : "false") << std::endl;
    return result;
}