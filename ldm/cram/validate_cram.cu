// =============================================================
//  Build: nvcc -O2 -std=c++17 validate_cram.cu -o validate_cram
//  Usage: ./validate_cram
// =============================================================

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <complex>
#include <cuda_runtime.h>


// =============================================================
// CRAM_SOLVER.txt
// =============================================================

#define N_NUCLIDES 60

// CRAM16 constants
__constant__ double CRAM16_ALPHA0 = 2.124853710495224e-16;
__constant__ double cram16_alpha_re[8] = {
    5.464930576870210e+3, 9.045112476907548e+1, 2.344818070467641e+2, 9.453304067358312e+1,
    7.283792954673409e+2, 3.648229059594851e+1, 2.547321630156819e+1, 2.394538338734709e+1
};
__constant__ double cram16_alpha_im[8] = {
    -3.797983575308356e+4, -1.115537522430261e+3, -4.228020157070496e+2, -2.951294291446048e+2,
    -1.205646080220011e+5, -1.155509621409682e+2, -2.639500283021502e+1, -5.650522971778156e+0
};
__constant__ double cram16_theta_re[8] = {
    3.509103608414918, 5.948152268951177, -5.264971343442647, 1.419375897185666,
    6.416177699099435, 4.993174737717997, -1.413928462488886, -10.843917078696990
};
__constant__ double cram16_theta_im[8] = {
    8.436198985884374, 3.587457362018322, 16.220221473167930, 10.925363484496720,
    1.194122393370139, 5.996881713603942, 13.497725698892750, 19.277446167181650
};

// CRAM48 constants
__constant__ double CRAM48_ALPHA0 = 2.258038182743983e-47;
__constant__ double cram48_alpha_re[24] = {
    6.387380733878774e+2, 1.909896179065730e+2, 4.236195226571914e+2, 4.645770595258726e+2,
    7.765163276752433e+2, 1.907115136768522e+3, 2.909892685603256e+3, 1.944772206620450e+2,
    1.382799786972332e+5, 5.628442079602433e+3, 2.151681283794220e+2, 1.324720240514420e+3,
    1.617548476343347e+4, 1.112729040439685e+2, 1.074624783191125e+2, 8.835727765158191e+1,
    9.354078136054179e+1, 9.418142823531573e+1, 1.040012390717851e+2, 6.861882624343235e+1,
    8.766654491283722e+1, 1.056007619389650e+2, 7.738987569039419e+1, 1.041366366475571e+2
};
__constant__ double cram48_alpha_im[24] = {
    -6.743912502859256e+2, -3.973203432721332e+2, -2.041233768918671e+3, -1.652917287299683e+3,
    -1.783617639907328e+4, -5.887068595142284e+4, -9.953255345514560e+3, -1.427131226068449e+3,
    -3.256885197214938e+6, -2.924284515884309e+4, -1.121774011188224e+3, -6.370088443140973e+4,
    -1.008798413156542e+6, -8.837109731680418e+1, -1.457246116408180e+2, -6.388286188419360e+1,
    -2.195424319460237e+2, -6.719055740098035e+2, -1.693747595553868e+2, -1.177598523430493e+1,
    -4.596464999363902e+3, -1.738294585524067e+3, -4.311715386228984e+1, -2.777743732451969e+2
};
__constant__ double cram48_theta_re[24] = {
    -4.465731934165702e+1, -5.284616241568964e+0, -8.867715667624458e+0, 3.493013124279215e+0,
    1.564102508858634e+1, 1.742097597385893e+1, -2.834466755180654e+1, 1.661569367939544e+1,
    8.011836167974721e+0, -2.056267541998229e+0, 1.449208170441839e+1, 1.853807176907916e+1,
    9.932562704505182e+0, -2.244223871767187e+1, 8.590014121680897e-1, -1.286192925744479e+1,
    1.164596909542055e+1, 1.806076684783089e+1, 5.870672154659249e+0, -3.542938819659747e+1,
    1.901323489060250e+1, 1.885508331552577e+1, -1.734689708174982e+1, 1.316284237125190e+1
};
__constant__ double cram48_theta_im[24] = {
    6.233225190695437e+1, 4.057499381311059e+1, 4.325515754166724e+1, 3.281615453173585e+1,
    1.558061616372237e+1, 1.076629305714420e+1, 5.492841024648724e+1, 1.316994930024688e+1,
    2.780232111309410e+1, 3.794824788914354e+1, 1.799988210051809e+1, 5.974332563100539e+0,
    2.532823409972962e+1, 5.179633600312162e+1, 3.536456194294350e+1, 4.600304902833652e+1,
    2.287153304140217e+1, 8.368200580099821e+0, 3.029700159040121e+1, 5.834381701800013e+1,
    1.194282058271408e+0, 3.583428564427879e+0, 4.883941101108207e+1, 2.042951874827759e+1
};

// A Matrix Loading Functions
bool load_A_matrix(const char* filename, float* A_matrix) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;
    
    std::string line;
    for(int i = 0; i < N_NUCLIDES; i++) {
        if(std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            for(int j = 0; j < N_NUCLIDES; j++) {
                if(std::getline(ss, cell, ',')) {
                    A_matrix[i*N_NUCLIDES + j] = cell.empty() ? 0.0f : std::stof(cell);
                } else {
                    A_matrix[i*N_NUCLIDES + j] = 0.0f;
                }
            }
        } else {
            for(int k = i*N_NUCLIDES; k < N_NUCLIDES*N_NUCLIDES; k++) {
                A_matrix[k] = 0.0f;
            }
            break;
        }
    }
    file.close();
    return true;
}

bool initialize_A_system(const char* A60_filename, float** d_A_matrix) {
    float* h_A_matrix = new float[N_NUCLIDES * N_NUCLIDES];
    
    if (!load_A_matrix(A60_filename, h_A_matrix)) {
        delete[] h_A_matrix;
        return false;
    }
    
    cudaError_t err = cudaMalloc(d_A_matrix, N_NUCLIDES * N_NUCLIDES * sizeof(float));
    if (err != cudaSuccess) {
        delete[] h_A_matrix;
        return false;
    }
    
    err = cudaMemcpy(*d_A_matrix, h_A_matrix, 
                     N_NUCLIDES * N_NUCLIDES * sizeof(float), 
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(*d_A_matrix);
        delete[] h_A_matrix;
        return false;
    }
    
    delete[] h_A_matrix;
    return true;
}

void cleanup_nuclear_system(float* d_A_matrix) {
    if (d_A_matrix) {
        cudaFree(d_A_matrix);
    }
}

// Gaussian Elimination
__device__ void cram_gauss_solve(double* A, double* b, double* x, int n) {
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

// CRAM16 Function
__device__ void nuclear_decay_cram16(float* A_matrix, float* concentration, float dt) {
    const int n = N_NUCLIDES;
    const int dim = 2 * n;
    
    double* B = (double*)malloc(n * n * sizeof(double));
    double* M = (double*)malloc(dim * dim * sizeof(double));
    double* bVec = (double*)malloc(dim * sizeof(double));
    double* xVec = (double*)malloc(dim * sizeof(double));
    double* result = (double*)malloc(n * sizeof(double));
    
    if(!B || !M || !bVec || !xVec || !result) {
        if(B) free(B); if(M) free(M); if(bVec) free(bVec); 
        if(xVec) free(xVec); if(result) free(result);
        return;
    }
    
    double dt_d = (double)dt;
    for(int i = 0; i < n; i++) result[i] = (double)concentration[i];
    for(int i = 0; i < n*n; i++) B[i] = dt_d * (double)A_matrix[i];
    
    for(int k = 0; k < 8; k++) {
        double tr = cram16_theta_re[k], ti = cram16_theta_im[k];
        double ar = cram16_alpha_re[k], ai = cram16_alpha_im[k];
        
        for(int i = 0; i < dim*dim; i++) M[i] = 0.0;
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                M[i*dim + j] = B[i*n + j];
                M[(i+n)*dim + (j+n)] = B[i*n + j];
            }
            M[i*dim + i] -= tr;
            M[(i+n)*dim + (i+n)] -= tr;
            M[i*dim + (i+n)] = ti;
            M[(i+n)*dim + i] = -ti;
        }
        
        for(int i = 0; i < n; i++) { bVec[i] = result[i]; bVec[i+n] = 0.0; }
        cram_gauss_solve(M, bVec, xVec, dim);
        
        for(int i = 0; i < n; i++) {
            double re = ar * xVec[i] - ai * xVec[i+n];
            result[i] += 2.0 * re;
        }
    }
    
    for(int i = 0; i < n; i++) {
        concentration[i] = (float)(result[i] * CRAM16_ALPHA0);
    }
    
    free(B); free(M); free(bVec); free(xVec); free(result);
}

// CRAM48 Function
__device__ void nuclear_decay_cram48(float* A_matrix, float* concentration, float dt) {
    const int n = N_NUCLIDES;
    const int dim = 2 * n;
    
    double* B = (double*)malloc(n * n * sizeof(double));
    double* M = (double*)malloc(dim * dim * sizeof(double));
    double* bVec = (double*)malloc(dim * sizeof(double));
    double* xVec = (double*)malloc(dim * sizeof(double));
    double* result = (double*)malloc(n * sizeof(double));
    
    if(!B || !M || !bVec || !xVec || !result) {
        if(B) free(B); if(M) free(M); if(bVec) free(bVec); 
        if(xVec) free(xVec); if(result) free(result);
        return;
    }
    
    double dt_d = (double)dt;
    for(int i = 0; i < n; i++) result[i] = (double)concentration[i];
    for(int i = 0; i < n*n; i++) B[i] = dt_d * (double)A_matrix[i];
    
    for(int k = 0; k < 24; k++) {
        double tr = cram48_theta_re[k], ti = cram48_theta_im[k];
        double ar = cram48_alpha_re[k], ai = cram48_alpha_im[k];
        
        for(int i = 0; i < dim*dim; i++) M[i] = 0.0;
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                M[i*dim + j] = B[i*n + j];
                M[(i+n)*dim + (j+n)] = B[i*n + j];
            }
            M[i*dim + i] -= tr;
            M[(i+n)*dim + (i+n)] -= tr;
            M[i*dim + (i+n)] = ti;
            M[(i+n)*dim + i] = -ti;
        }
        
        for(int i = 0; i < n; i++) { bVec[i] = result[i]; bVec[i+n] = 0.0; }
        cram_gauss_solve(M, bVec, xVec, dim);
        
        for(int i = 0; i < n; i++) {
            double re = ar * xVec[i] - ai * xVec[i+n];
            result[i] += 2.0 * re;
        }
    }
    
    for(int i = 0; i < n; i++) {
        concentration[i] = (float)(result[i] * CRAM48_ALPHA0);
    }
    
    free(B); free(M); free(bVec); free(xVec); free(result);
}

// Main Interface
__device__ void nuclear_decay(float* A_matrix, float* concentration, 
                             float dt, bool use_cram48 = false) {
    if(use_cram48) {
        nuclear_decay_cram48(A_matrix, concentration, dt);
    } else {
        nuclear_decay_cram16(A_matrix, concentration, dt);
    }
}

// =============================================================
// 검증 테스트 코드
// =============================================================

__global__ void test_cram_kernel(float* d_A, float* d_n0, float dt, 
                                float* d_result16, float* d_result48) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx == 0) {
        // CRAM16
        for(int i = 0; i < N_NUCLIDES; i++) d_result16[i] = d_n0[i];
        nuclear_decay_cram16(d_A, d_result16, dt);
        
        // CRAM48
        for(int i = 0; i < N_NUCLIDES; i++) d_result48[i] = d_n0[i];
        nuclear_decay_cram48(d_A, d_result48, dt);
    }
}

void print_results(const std::vector<float>& initial, 
                   const std::vector<float>& cram16, 
                   const std::vector<float>& cram48, 
                   float dt) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "CRAM Results for dt = " << dt << " seconds" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    // 각 핵종별 농도 출력
    std::cout << "\nNuclide Concentrations:" << std::endl;
    std::cout << "Index\tInitial\t\tCRAM16\t\tCRAM48\t\tDiff16\t\tDiff48" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    for(int i = 0; i < N_NUCLIDES; i++) {
        double diff16 = cram16[i] - initial[i];
        double diff48 = cram48[i] - initial[i];
        
        std::cout << std::setw(5) << i << "\t" 
                  << std::fixed << std::setprecision(3) << std::setw(10) << initial[i] << "\t"
                  << std::fixed << std::setprecision(3) << std::setw(10) << cram16[i] << "\t"
                  << std::fixed << std::setprecision(3) << std::setw(10) << cram48[i] << "\t"
                  << std::fixed << std::setprecision(3) << std::setw(10) << diff16 << "\t"
                  << std::fixed << std::setprecision(3) << std::setw(10) << diff48 << std::endl;
    }
    
    // 요약 통계
    double total_initial = 0, total_cram16 = 0, total_cram48 = 0;
    for(int i = 0; i < N_NUCLIDES; i++) {
        total_initial += initial[i];
        total_cram16 += cram16[i];
        total_cram48 += cram48[i];
    }
    
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "TOTAL\t" 
              << std::fixed << std::setprecision(3) << std::setw(10) << total_initial << "\t"
              << std::fixed << std::setprecision(3) << std::setw(10) << total_cram16 << "\t"
              << std::fixed << std::setprecision(3) << std::setw(10) << total_cram48 << "\t"
              << std::fixed << std::setprecision(3) << std::setw(10) << (total_cram16 - total_initial) << "\t"
              << std::fixed << std::setprecision(3) << std::setw(10) << (total_cram48 - total_initial) << std::endl;
    
    std::cout << "\nSummary:" << std::endl;
    std::cout << "  Total initial mass: " << std::fixed << std::setprecision(1) << total_initial << std::endl;
    std::cout << "  CRAM16 final mass:  " << std::fixed << std::setprecision(1) << total_cram16 << std::endl;
    std::cout << "  CRAM48 final mass:  " << std::fixed << std::setprecision(1) << total_cram48 << std::endl;
    std::cout << "  CRAM16 conservation: " << std::fixed << std::setprecision(4) 
              << (total_cram16/total_initial)*100.0 << "%" << std::endl;
    std::cout << "  CRAM48 conservation: " << std::fixed << std::setprecision(4) 
              << (total_cram48/total_initial)*100.0 << "%" << std::endl;
}

void n0_test(std::vector<float>& n0) {
    n0.resize(N_NUCLIDES, 1000.0f);
}

int main() {
    std::cout << "=== CRAM GPU Results Validation ===" << std::endl;
    
    int device_count;
    cudaGetDeviceCount(&device_count);
    if(device_count == 0) {
        std::cerr << "No CUDA devices available!" << std::endl;
        return 1;
    }
    
    float* d_A_matrix = nullptr;
    std::vector<float> n0;
    bool use_real_data = false;
    
    std::cout << "\nTrying to load A60.csv..." << std::endl;
    if(initialize_A_system("A60.csv", &d_A_matrix)) {
        std::cout << "A60.csv loaded successfully!" << std::endl;
        use_real_data = true;
        
        n0_test(n0);
        
    } else {
        std::cout << "A60.csv not found" << std::endl;
        std::cout << "Please ensure A60.csv is in the current directory." << std::endl;
        return 1;
    }
    
    float *d_n0, *d_result16, *d_result48;
    cudaMalloc(&d_n0, N_NUCLIDES * sizeof(float));
    cudaMalloc(&d_result16, N_NUCLIDES * sizeof(float));
    cudaMalloc(&d_result48, N_NUCLIDES * sizeof(float));
    
    cudaMemcpy(d_n0, n0.data(), N_NUCLIDES * sizeof(float), cudaMemcpyHostToDevice);
    
    std::vector<float> test_times = {0.1f, 1.0f, 5.0f, 10.0f};
    
    for(float dt : test_times) {
        test_cram_kernel<<<1, 1>>>(d_A_matrix, d_n0, dt, d_result16, d_result48);
        cudaDeviceSynchronize();
        
        std::vector<float> gpu_result16(N_NUCLIDES), gpu_result48(N_NUCLIDES);
        cudaMemcpy(gpu_result16.data(), d_result16, N_NUCLIDES * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(gpu_result48.data(), d_result48, N_NUCLIDES * sizeof(float), cudaMemcpyDeviceToHost);
        
        print_results(n0, gpu_result16, gpu_result48, dt);
    }
    
    cleanup_nuclear_system(d_A_matrix);
    cudaFree(d_n0);
    cudaFree(d_result16);
    cudaFree(d_result48);
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Validation completed!" << std::endl;
    
    return 0;
}