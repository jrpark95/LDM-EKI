#include "ldm.cuh"
#include "ldm_ensemble_init.cuh"

// Ensemble activation kernel implementation
__global__ void update_particle_flags_ensembles(LDM::LDMpart* d_part,
                                               int nop_per_ensemble,
                                               int Nens,
                                               float activationRatio) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nop_per_ensemble * Nens;
    
    if (idx >= total) return;
    
    int local_i = idx % nop_per_ensemble;
    int maxLocal = static_cast<int>(nop_per_ensemble * activationRatio);
    
    d_part[idx].flag = (local_i < maxLocal) ? 1 : 0;
}

// Sanity check kernel implementation
__global__ void count_active_particles_per_ensemble(const LDM::LDMpart* d_part,
                                                   int nop_per_ensemble,
                                                   int Nens,
                                                   int* active_counts) {
    int ensemble_idx = blockIdx.x;
    if (ensemble_idx >= Nens) return;
    
    int tid = threadIdx.x;
    int local_count = 0;
    
    // Count active particles in this ensemble
    for (int i = tid; i < nop_per_ensemble; i += blockDim.x) {
        int global_idx = ensemble_idx * nop_per_ensemble + i;
        if (d_part[global_idx].flag == 1) {
            local_count++;
        }
    }
    
    // Reduce within block
    __shared__ int sdata[256];
    sdata[tid] = local_count;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        active_counts[ensemble_idx] = sdata[0];
    }
}