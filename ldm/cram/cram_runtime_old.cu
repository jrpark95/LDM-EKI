#include "cram_runtime.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define N_NUCLIDES 60

// External function from ldm_cram.cuh
extern bool compute_exp_matrix_host_cram(float dt, bool use_cram48);


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
const double h_cram48_alpha_im[24] = {
    -6.743912502859256e+2, -3.973203432721332e+2, -2.041233768918671e+3, -1.652917287299683e+3,
    -1.783617639907328e+4, -5.887068595142284e+4, -9.953255345514560e+3, -1.427131226068449e+3,
    -3.256885197214938e+6, -2.924284515884309e+4, -1.121774011188224e+3, -6.370088443140973e+4,
    -1.008798413156542e+6, -8.837109731680418e+1, -1.457246116408180e+2, -6.388286188419360e+1,
    -2.195424319460237e+2, -6.719055740098035e+2, -1.693747595553868e+2, -1.177598523430493e+1,
    -4.596464999363902e+3, -1.738294585524067e+3, -4.311715386228984e+1, -2.777743732451969e+2
};
const double h_cram48_theta_re[24] = {
    -4.465731934165702e+1, -5.284616241568964e+0, -8.867715667624458e+0, 3.493013124279215e+0,
    1.564102508858634e+1, 1.742097597385893e+1, -2.834466755180654e+1, 1.661569367939544e+1,
    8.011836167974721e+0, -2.056267541998229e+0, 1.449208170441839e+1, 1.853807176907916e+1,
    9.932562704505182e+0, -2.244223871767187e+1, 8.590014121680897e-1, -1.286192925744479e+1,
    1.164596909542055e+1, 1.806076684783089e+1, 5.870672154659249e+0, -3.542938819659747e+1,
    1.901323489060250e+1, 1.885508331552577e+1, -1.734689708174982e+1, 1.316284237125190e+1
};
const double h_cram48_theta_im[24] = {
    6.233225190695437e+1, 4.057499381311059e+1, 4.325515754166724e+1, 3.281615453173585e+1,
    1.558061616372237e+1, 1.076629305714420e+1, 5.492841024648724e+1, 1.316994930024688e+1,
    2.780232111309410e+1, 3.794824788914354e+1, 1.799988210051809e+1, 5.974332563100539e+0,
    2.532823409972962e+1, 5.179633600312162e+1, 3.536456194294350e+1, 4.600304902833652e+1,
    2.287153304140217e+1, 8.368200580099821e+0, 3.029700159040121e+1, 5.834381701800013e+1,
    1.194282058271408e+0, 3.583428564427879e+0, 4.883941101108207e+1, 2.042951874827759e+1
};


// Host version of Gaussian elimination
void host_cram_gauss_solve(double* A, double* b, double* x, int n) {
    for(int i = 0; i < n; i++) x[i] = b[i];
    
    for(int k = 0; k < n; k++) {
        int pivot_row = current_row;
        double max_pivot_value = fabs(coefficient_matrix[current_row*matrix_size + current_row]);
        
        // Find the row with maximum pivot element
        for(int row = current_row+1; row < matrix_size; row++) {
            double pivot_candidate = fabs(coefficient_matrix[row*matrix_size + current_row]);
            if(pivot_candidate > max_pivot_value) { 
                max_pivot_value = pivot_candidate; 
                pivot_row = row; 
            }
        }
        
        // Check for singular matrix
        if(max_pivot_value < 1e-20) return;
        
        // Swap rows if necessary
        if(pivot_row != current_row) {
            for(int column = current_row; column < matrix_size; column++) {
                double temp = coefficient_matrix[current_row*matrix_size + column];
                coefficient_matrix[current_row*matrix_size + column] = coefficient_matrix[pivot_row*matrix_size + column];
                coefficient_matrix[pivot_row*matrix_size + column] = temp;
            }
            double temp = solution_vector[current_row]; 
            solution_vector[current_row] = solution_vector[pivot_row]; 
            solution_vector[pivot_row] = temp;
        }
        
        // Scale pivot row
        double pivot_value = coefficient_matrix[current_row*matrix_size + current_row];
        for(int column = current_row; column < matrix_size; column++) {
            coefficient_matrix[current_row*matrix_size + column] /= pivot_value;
        }
        solution_vector[current_row] /= pivot_value;
        
        // Eliminate column entries
        for(int row = 0; row < matrix_size; row++) {
            if(row != current_row && fabs(coefficient_matrix[row*matrix_size + current_row]) > 1e-20) {
                double elimination_factor = coefficient_matrix[row*matrix_size + current_row];
                for(int column = current_row; column < matrix_size; column++) {
                    coefficient_matrix[row*matrix_size + column] -= elimination_factor * coefficient_matrix[current_row*matrix_size + column];
                }
                solution_vector[row] -= elimination_factor * solution_vector[current_row];
            }
        }
    }
}

// Host implementation of CRAM48 algorithm for nuclear transition matrix computation
// Computes transition matrix T = exp(A*timestep) using 48-pole CRAM approximation
void compute_nuclear_transition_matrix_cram48(double* nuclear_decay_matrix, double timestep_size, float* transition_matrix) {
    const int nuclide_count = MAX_NUMBER_OF_NUCLIDES;
    const int extended_dimension = 2 * nuclide_count;
    
    // Initialize transition matrix as identity matrix
    for(int i = 0; i < nuclide_count*nuclide_count; i++) transition_matrix[i] = 0.0f;
    for(int i = 0; i < nuclide_count; i++) transition_matrix[i*nuclide_count + i] = 1.0f;
    
    // Allocate temporary matrices for CRAM48 computation
    double* scaled_decay_matrix = new double[nuclide_count * nuclide_count];
    double* extended_matrix = new double[extended_dimension * extended_dimension];
    double* right_hand_side_vector = new double[extended_dimension];
    double* solution_vector = new double[extended_dimension];
    double* temporary_result = new double[nuclide_count * nuclide_count];
    
    // Check memory allocation success
    if(!scaled_decay_matrix || !extended_matrix || !right_hand_side_vector || !solution_vector || !temporary_result) {
        delete[] scaled_decay_matrix; delete[] extended_matrix; delete[] right_hand_side_vector; delete[] solution_vector; delete[] temporary_result;
        return;
    }
    
    // Initialize temp_result as identity
    for(int i = 0; i < n*n; i++) temp_result[i] = 0.0;
    for(int i = 0; i < n; i++) temp_result[i*n + i] = 1.0;
    
    double dt_d = dt;
    for(int i = 0; i < n*n; i++) B[i] = dt_d * A_matrix[i];
    
    // Process each column of the identity matrix to get exp(dt*A)
    for(int col = 0; col < n; col++) {
        // Reset column result to identity column
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
        
        // Apply CRAM48_ALPHA0 and store in result matrix
        for(int i = 0; i < n; i++) {
            result_matrix[i*n + col] = (float)(column_result[i] * h_CRAM48_ALPHA0);
        }
        
        delete[] column_result;
    }
    
    delete[] B; delete[] M; delete[] bVec; delete[] xVec; delete[] temp_result;
}


bool cram_precompute_transition(float dt) {
    // Load A matrix from A60.csv
    std::ifstream file("cram/A60.csv");
    if (!file.is_open()) {
        std::cerr << "[ERROR] Failed to open A60.csv for CRAM precomputation" << std::endl;
        return false;
    }
    
    // Load A matrix in double precision for accuracy
    double* A_matrix = new double[N_NUCLIDES * N_NUCLIDES];
    
    std::string line;
    for(int i = 0; i < N_NUCLIDES && std::getline(file, line); i++) {
        std::stringstream ss(line);
        std::string cell;
        for(int j = 0; j < N_NUCLIDES; j++) {
            if(std::getline(ss, cell, ',')) {
                A_matrix[i*N_NUCLIDES + j] = cell.empty() ? 0.0 : std::stod(cell);
            } else {
                A_matrix[i*N_NUCLIDES + j] = 0.0;
            }
        }
    }
    file.close();
    
    
    // Create result matrix for T = exp(A*dt) using CRAM48
    float* T_matrix = new float[N_NUCLIDES * N_NUCLIDES];
    
    // Call proper CRAM48 host computation: T = exp(A*dt) where A already has correct signs
    host_nuclear_decay_cram48(A_matrix, (double)dt, T_matrix);
    
    
    // Upload to GPU using unified pointer management
    extern float* d_exp_matrix_global;
    extern float current_dt;
    
    if (d_exp_matrix_global == nullptr) {
        cudaError_t malloc_err = cudaMalloc(&d_exp_matrix_global, N_NUCLIDES * N_NUCLIDES * sizeof(float));
        if (malloc_err != cudaSuccess) {
            std::cerr << "[ERROR] Failed to allocate GPU memory: " << cudaGetErrorString(malloc_err) << std::endl;
            delete[] A_matrix;
            delete[] T_matrix;
            return false;
        }
    }
    
    cudaError_t copy_err = cudaMemcpy(d_exp_matrix_global, T_matrix, 
                                      N_NUCLIDES * N_NUCLIDES * sizeof(float), 
                                      cudaMemcpyHostToDevice);
    
    if (copy_err != cudaSuccess) {
        std::cerr << "[ERROR] Failed to upload transition matrix: " << cudaGetErrorString(copy_err) << std::endl;
        delete[] A_matrix;
        delete[] T_matrix;
        return false;
    }
    
    // Update tracking variables
    current_dt = dt;
    
    // Cleanup
    delete[] A_matrix;
    delete[] T_matrix;
    
    return true;
}


