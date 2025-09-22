#include <iostream>
#include <fstream>
#include <iomanip>
#include "src/include/ldm.cuh"

__global__ void testRandomSeeds(LDM::LDMpart* d_part, float t0) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= 10) return;  // Only test first 10 particles
    
    int ensemble_id = d_part[idx].ensemble_id;
    unsigned long long seed = static_cast<unsigned long long>((t0 + idx * 0.001f + ensemble_id * 1000.0f) * ULLONG_MAX);
    
    curandState ss;
    curand_init(seed, idx, 0, &ss);
    
    float random_value = curand_uniform(&ss);
    
    printf("Particle %d: ensemble_id=%d, seed=%llu, random=%.6f\n", 
           idx, ensemble_id, seed, random_value);
}

int main() {
    // Test ensemble diversity
    std::cout << "Testing ensemble random seed diversity..." << std::endl;
    
    // Create test particles with different ensemble IDs
    LDM::LDMpart* h_particles = new LDM::LDMpart[10];
    for (int i = 0; i < 10; i++) {
        h_particles[i].ensemble_id = i / 5;  // First 5 particles: ensemble 0, next 5: ensemble 1
        h_particles[i].x = -73.965f;  // Same initial position
        h_particles[i].y = 40.749f;
        h_particles[i].z = 10.0f;
    }
    
    // Copy to GPU
    LDM::LDMpart* d_particles;
    cudaMalloc(&d_particles, 10 * sizeof(LDM::LDMpart));
    cudaMemcpy(d_particles, h_particles, 10 * sizeof(LDM::LDMpart), cudaMemcpyHostToDevice);
    
    // Test random seeds
    testRandomSeeds<<<1, 10>>>(d_particles, 300.0f);  // t0 = 300 seconds
    cudaDeviceSynchronize();
    
    // Cleanup
    cudaFree(d_particles);
    delete[] h_particles;
    
    return 0;
}