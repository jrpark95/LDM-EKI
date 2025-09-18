#include <iostream>
#include <cmath>
#include <random>
#include <chrono>

// CRAM48 accuracy verification test constants
const double HOST_CRAM48_ALPHA_ZERO = 2.258038182743983e-47;
const double host_cram48_alpha_real_coefficients[24] = {
    6.387380733878774e+2, 1.909896179065730e+2, 4.236195226571914e+2, 4.645770595258726e+2,
    7.765163276752433e+2, 1.907115136768522e+3, 2.909892685603256e+3, 1.944772206620450e+2,
    1.382799786972332e+5, 5.628442079602433e+3, 2.151681283794220e+2, 1.324720240514420e+3,
    1.617548476343347e+4, 1.112729040439685e+2, 1.074624783191125e+2, 8.835727765158191e+1,
    9.354078136054179e+1, 9.418142823531573e+1, 1.040012390717851e+2, 6.861882624343235e+1,
    8.766654491283722e+1, 1.056007619389650e+2, 7.738987569039419e+1, 1.041366366475571e+2
};
const double host_cram48_alpha_imaginary_coefficients[24] = {
    -6.743912502859256e+2, -3.973203432721332e+2, -2.041233768918671e+3, -1.652917287299683e+3,
    -1.783617639907328e+4, -5.887068595142284e+4, -9.953255345514560e+3, -1.427131226068449e+3,
    -3.256885197214938e+6, -2.924284515884309e+4, -1.121774011188224e+3, -6.370088443140973e+4,
    -1.008798413156542e+6, -8.837109731680418e+1, -1.457246116408180e+2, -6.388286188419360e+1,
    -2.195424319460237e+2, -6.719055740098035e+2, -1.693747595553868e+2, -1.177598523430493e+1,
    -4.596464999363902e+3, -1.738294585524067e+3, -4.311715386228984e+1, -2.777743732451969e+2
};
const double host_cram48_theta_real_coefficients[24] = {
    -4.465731934165702e+1, -5.284616241568964e+0, -8.867715667624458e+0, 3.493013124279215e+0,
    1.564102508858634e+1, 1.742097597385893e+1, -2.834466755180654e+1, 1.661569367939544e+1,
    8.011836167974721e+0, -2.056267541998229e+0, 1.449208170441839e+1, 1.853807176907916e+1,
    9.932562704505182e+0, -2.244223871767187e+1, 8.590014121680897e-1, -1.286192925744479e+1,
    1.164596909542055e+1, 1.806076684783089e+1, 5.870672154659249e+0, -3.542938819659747e+1,
    1.901323489060250e+1, 1.885508331552577e+1, -1.734689708174982e+1, 1.316284237125190e+1
};
const double host_cram48_theta_imaginary_coefficients[24] = {
    6.233225190695437e+1, 4.057499381311059e+1, 4.325515754166724e+1, 3.281615453173585e+1,
    1.558061616372237e+1, 1.076629305714420e+1, 5.492841024648724e+1, 1.316994930024688e+1,
    2.780232111309410e+1, 3.794824788914354e+1, 1.799988210051809e+1, 5.974332563100539e+0,
    2.532823409972962e+1, 5.179633600312162e+1, 3.536456194294350e+1, 4.600304902833652e+1,
    2.287153304140217e+1, 8.368200580099821e+0, 3.029700159040121e+1, 5.834381701800013e+1,
    1.194282058271408e+0, 3.583428564427879e+0, 4.883941101108207e+1, 2.042951874827759e+1
};

void solve_linear_system_gaussian_elimination(double* coefficient_matrix, double* right_hand_side, double* solution_vector, int matrix_size) {
    for(int i = 0; i < matrix_size; i++) solution_vector[i] = right_hand_side[i];
    
    for(int k = 0; k < matrix_size; k++) {
        int piv = k;
        double pmax = fabs(coefficient_matrix[k*matrix_size + k]);
        for(int i = k+1; i < matrix_size; i++) {
            double v = fabs(coefficient_matrix[i*matrix_size + k]);
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

// CRAM48 implementation
void compute_exp_cram48(double* A, double dt, int n, double* result) {
    const int dim = 2 * n;
    
    // Initialize result as identity matrix
    for(int i = 0; i < n*n; i++) result[i] = 0.0;
    for(int i = 0; i < n; i++) result[i*n + i] = 1.0;
    
    double* B = new double[n * n];
    double* M = new double[dim * dim];
    double* bVec = new double[dim];
    double* xVec = new double[dim];
    
    // B = dt * A 
    for(int i = 0; i < n*n; i++) B[i] = dt * A[i];
    
    // Process each column to get exp(dt*A)
    for(int col = 0; col < n; col++) {
        double* column_result = new double[n];
        for(int i = 0; i < n; i++) column_result[i] = (i == col) ? 1.0 : 0.0;
        
        for(int k = 0; k < 24; k++) {
            double tr = h_cram48_theta_re[k], ti = h_cram48_theta_im[k];
            double ar = h_cram48_alpha_re[k], ai = h_cram48_alpha_im[k];
            
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
            
            for(int i = 0; i < n; i++) { bVec[i] = column_result[i]; bVec[i+n] = 0.0; }
            host_cram_gauss_solve(M, bVec, xVec, dim);
            
            for(int i = 0; i < n; i++) {
                double re = ar * xVec[i] - ai * xVec[i+n];
                column_result[i] += 2.0 * re;
            }
        }
        
        // Apply ALPHA0 and store in result matrix
        for(int i = 0; i < n; i++) {
            result[i*n + col] = column_result[i] * h_CRAM48_ALPHA0;
        }
        
        delete[] column_result;
    }
    
    delete[] B; delete[] M; delete[] bVec; delete[] xVec;
}

// Exact exponential for 1x1 case
double exact_exp_1x1(double a, double dt) {
    return exp(dt * a);
}

// Test cases
void test_1x1_case() {
    std::cout << "\n=== 1x1 Test Case ===" << std::endl;
    
    double A = -1.13e-7;  // From A60.csv diagonal
    double dt = 60.0;
    
    double result_cram[1];
    compute_exp_cram48(&A, dt, 1, result_cram);
    
    double exact = exact_exp_1x1(A, dt);
    double rel_error = fabs(result_cram[0] - exact) / fabs(exact);
    
    std::cout << "A = " << A << ", dt = " << dt << std::endl;
    std::cout << "CRAM48: " << result_cram[0] << std::endl;
    std::cout << "Exact:  " << exact << std::endl;
    std::cout << "Relative error: " << rel_error << std::endl;
    std::cout << "Target < 1e-7: " << (rel_error < 1e-7 ? "PASS" : "FAIL") << std::endl;
}

void test_2x2_chain() {
    std::cout << "\n=== 2x2 Chain Test Case ===" << std::endl;
    
    // Simple decay chain: 1 -> 2
    double A[4] = {
        -1e-5,  0,
         1e-5, -1e-6
    };
    double dt = 60.0;
    
    double result_cram[4];
    compute_exp_cram48(A, dt, 2, result_cram);
    
    // Analytical solution for 2x2 case
    double lambda1 = 1e-5, lambda2 = 1e-6;
    double exp1 = exp(-lambda1 * dt);
    double exp2 = exp(-lambda2 * dt);
    
    double exact[4];
    exact[0] = exp1;  // exp(-lambda1*dt)
    exact[1] = 0.0;
    exact[2] = lambda1/(lambda2-lambda1) * (exp1 - exp2);  // Transfer term
    exact[3] = exp2;  // exp(-lambda2*dt)
    
    std::cout << "CRAM48 matrix:" << std::endl;
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {
            std::cout << result_cram[i*2 + j] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "Exact matrix:" << std::endl;
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {
            std::cout << exact[i*2 + j] << " ";
        }
        std::cout << std::endl;
    }
    
    // Calculate relative errors
    double max_rel_error = 0.0;
    for(int i = 0; i < 4; i++) {
        if(fabs(exact[i]) > 1e-15) {
            double rel_err = fabs(result_cram[i] - exact[i]) / fabs(exact[i]);
            max_rel_error = std::max(max_rel_error, rel_err);
        }
    }
    
    std::cout << "Max relative error: " << max_rel_error << std::endl;
    std::cout << "Target < 1e-7: " << (max_rel_error < 1e-7 ? "PASS" : "FAIL") << std::endl;
}

void test_sign_convention() {
    std::cout << "\n=== Sign Convention Test ===" << std::endl;
    
    // Test: if A has negative diagonals, exp(dt*A) should decay
    double A = -1e-5;  // Negative (decay)
    double dt = 60.0;
    
    double result_cram[1];
    compute_exp_cram48(&A, dt, 1, result_cram);
    
    std::cout << "A = " << A << " (negative = decay)" << std::endl;
    std::cout << "exp(dt*A) = " << result_cram[0] << std::endl;
    std::cout << "Should be < 1 for decay: " << (result_cram[0] < 1.0 ? "PASS" : "FAIL") << std::endl;
    
    // Test positive (growth)
    A = 1e-5;  // Positive (growth)
    compute_exp_cram48(&A, dt, 1, result_cram);
    
    std::cout << "A = " << A << " (positive = growth)" << std::endl;
    std::cout << "exp(dt*A) = " << result_cram[0] << std::endl;
    std::cout << "Should be > 1 for growth: " << (result_cram[0] > 1.0 ? "PASS" : "FAIL") << std::endl;
}

int main() {
    std::cout << "CRAM48 Accuracy Verification Tests" << std::endl;
    std::cout << "===================================" << std::endl;
    
    test_sign_convention();
    test_1x1_case();
    test_2x2_chain();
    
    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "All tests completed. Check individual PASS/FAIL status above." << std::endl;
    
    return 0;
}