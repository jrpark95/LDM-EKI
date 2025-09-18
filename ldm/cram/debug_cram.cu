// Debug version to check CRAM implementation
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <cuda_runtime.h>

#define N_NUCLIDES 60

// CRAM48 constants (simplified for debugging)
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

// Load A matrix
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
        }
    }
    file.close();
    return true;
}

// Simple host Gaussian elimination
void host_gauss_solve(double* A, double* b, double* x, int n) {
    for(int i = 0; i < n; i++) x[i] = b[i];
    
    for(int k = 0; k < n; k++) {
        // Find pivot
        int piv = k;
        double pmax = fabs(A[k*n + k]);
        for(int i = k+1; i < n; i++) {
            double v = fabs(A[i*n + k]);
            if(v > pmax) { pmax = v; piv = i; }
        }
        if(pmax < 1e-20) {
            std::cout << "Singular matrix at step " << k << ", pmax = " << pmax << std::endl;
            return;
        }
        
        // Swap rows if needed
        if(piv != k) {
            for(int j = k; j < n; j++) {
                double temp = A[k*n + j];
                A[k*n + j] = A[piv*n + j];
                A[piv*n + j] = temp;
            }
            double temp = x[k]; x[k] = x[piv]; x[piv] = temp;
        }
        
        // Scale pivot row
        double pivVal = A[k*n + k];
        for(int j = k; j < n; j++) A[k*n + j] /= pivVal;
        x[k] /= pivVal;
        
        // Eliminate
        for(int i = 0; i < n; i++) {
            if(i != k && fabs(A[i*n + k]) > 1e-20) {
                double factor = A[i*n + k];
                for(int j = k; j < n; j++) A[i*n + j] -= factor * A[k*n + j];
                x[i] -= factor * x[k];
            }
        }
    }
}

// Simplified CRAM48
void simple_cram48(const float* A_matrix, double* concentration, double dt) {
    const int n = N_NUCLIDES;
    const int dim = 2 * n;
    
    std::vector<double> B(n * n);
    std::vector<double> M(dim * dim);
    std::vector<double> bVec(dim);
    std::vector<double> xVec(dim);
    std::vector<double> result(n);
    
    // Initialize
    for(int i = 0; i < n; i++) result[i] = concentration[i];
    for(int i = 0; i < n*n; i++) B[i] = dt * (double)A_matrix[i];
    
    // Copy constants from device
    double h_cram48_alpha_re[24], h_cram48_alpha_im[24];
    double h_cram48_theta_re[24], h_cram48_theta_im[24];
    double h_CRAM48_ALPHA0;
    
    cudaMemcpyFromSymbol(h_cram48_alpha_re, cram48_alpha_re, 24 * sizeof(double));
    cudaMemcpyFromSymbol(h_cram48_alpha_im, cram48_alpha_im, 24 * sizeof(double));
    cudaMemcpyFromSymbol(h_cram48_theta_re, cram48_theta_re, 24 * sizeof(double));
    cudaMemcpyFromSymbol(h_cram48_theta_im, cram48_theta_im, 24 * sizeof(double));
    cudaMemcpyFromSymbol(&h_CRAM48_ALPHA0, &CRAM48_ALPHA0, sizeof(double));
    
    std::cout << "Initial concentration sum: " << std::accumulate(result.begin(), result.end(), 0.0) << std::endl;
    std::cout << "CRAM48_ALPHA0: " << h_CRAM48_ALPHA0 << std::endl;
    
    // Check first few iterations for debugging
    for(int k = 0; k < std::min(3, 24); k++) {
        double tr = h_cram48_theta_re[k], ti = h_cram48_theta_im[k];
        double ar = h_cram48_alpha_re[k], ai = h_cram48_alpha_im[k];
        
        std::cout << "Step " << k << ": theta_re=" << tr << ", theta_im=" << ti 
                  << ", alpha_re=" << ar << ", alpha_im=" << ai << std::endl;
        
        // Initialize M matrix
        std::fill(M.begin(), M.end(), 0.0);
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
        
        // Set up RHS
        for(int i = 0; i < n; i++) { bVec[i] = result[i]; bVec[i+n] = 0.0; }
        
        // Copy for solving (since Gauss elimination modifies the matrix)
        std::vector<double> M_copy = M;
        host_gauss_solve(M_copy.data(), bVec.data(), xVec.data(), dim);
        
        // Check for NaN
        bool has_nan = false;
        for(int i = 0; i < dim; i++) {
            if(std::isnan(xVec[i]) || std::isinf(xVec[i])) {
                has_nan = true;
                break;
            }
        }
        
        if(has_nan) {
            std::cout << "NaN detected in solution at step " << k << std::endl;
            break;
        }
        
        // Update result
        for(int i = 0; i < n; i++) {
            double re = ar * xVec[i] - ai * xVec[i+n];
            result[i] += 2.0 * re;
        }
        
        double sum_after = std::accumulate(result.begin(), result.end(), 0.0);
        std::cout << "Sum after step " << k << ": " << sum_after << std::endl;
    }
    
    // Apply final scaling
    for(int i = 0; i < n; i++) {
        concentration[i] = result[i] * h_CRAM48_ALPHA0;
    }
    
    std::cout << "Final concentration sum: " << std::accumulate(concentration, concentration + n, 0.0) << std::endl;
}

int main() {
    std::cout << "=== Debug CRAM Implementation ===" << std::endl;
    
    // Load A matrix
    std::vector<float> A_matrix(N_NUCLIDES * N_NUCLIDES);
    if(!load_A_matrix("A60.csv", A_matrix.data())) {
        std::cerr << "Failed to load A60.csv!" << std::endl;
        return 1;
    }
    
    std::cout << "A matrix loaded successfully!" << std::endl;
    
    // Print a few elements of A matrix
    std::cout << "A[0,0] = " << A_matrix[0] << std::endl;
    std::cout << "A[0,1] = " << A_matrix[1] << std::endl;
    std::cout << "A[1,1] = " << A_matrix[N_NUCLIDES + 1] << std::endl;
    
    // Test with simple unit vector
    std::vector<double> concentration(N_NUCLIDES, 0.0);
    concentration[0] = 1000.0;  // Put mass in first nuclide
    
    std::cout << "\nTesting CRAM48 with unit vector..." << std::endl;
    std::cout << "Initial: " << concentration[0] << " in first nuclide" << std::endl;
    
    double dt = 0.1;
    simple_cram48(A_matrix.data(), concentration.data(), dt);
    
    std::cout << "\nFinal concentrations (first 10):" << std::endl;
    for(int i = 0; i < 10; i++) {
        std::cout << "  [" << i << "] = " << concentration[i] << std::endl;
    }
    
    return 0;
}