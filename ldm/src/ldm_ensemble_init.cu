#include "ldm_ensemble_init.cuh"
#include "ldm_eki.cuh"
#include "ldm_nuclides.cuh"
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <sstream>

// Compilation: nvcc -O3 -std=c++17 -arch=sm_80 ldm_ensemble_init.cu -o ldm_ensemble_init

// Deterministic RNG for ensembles
uint64_t splitmix64(uint64_t& z) {
    z += 0x9e3779b97f4a7c15;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

bool LDM::initializeParticlesEnsembles(int Nens,
                                      const std::vector<std::vector<float>>& ensemble_matrix,
                                      const std::vector<Source>& sources,
                                      int nop_per_ensemble) {
    
    // Convert 2D ensemble matrix to flattened format
    const int T = static_cast<int>(ensemble_matrix.size());
    std::vector<float> emission_flat(Nens * T);
    
    // Apply ensemble-specific emission rates
    for (int e = 0; e < Nens; ++e) {
        for (int t = 0; t < T; ++t) {
            emission_flat[e * T + t] = ensemble_matrix[t][e];  // Use ensemble-specific values
        }
    }
    
    // Call overload 2
    return initializeParticlesEnsemblesFlat(Nens, emission_flat, sources, nop_per_ensemble);
}

bool LDM::initializeParticlesEnsemblesFlat(int Nens,
                                           const std::vector<float>& emission_flat,
                                           const std::vector<Source>& sources,
                                           int nop_per_ensemble) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (Nens <= 0 || nop_per_ensemble <= 0) {
        std::cerr << "[ERROR] Invalid ensemble parameters: Nens=" << Nens 
                  << ", nop_per_ensemble=" << nop_per_ensemble << std::endl;
        return false;
    }
    
    if (sources.empty()) {
        std::cerr << "[ERROR] No sources provided for ensemble initialization" << std::endl;
        return false;
    }
    
    if (emission_flat.empty()) {
        std::cerr << "[ERROR] Empty emission data" << std::endl;
        return false;
    }
    
    const int T = static_cast<int>(emission_flat.size()) / Nens;
    const int d_nop_total = Nens * nop_per_ensemble;
    
    DEBUG_LOG("Initializing ensembles: Nens=%d, nop_per_ensemble=%d, T=%d, total_particles=%d", 
              Nens, nop_per_ensemble, T, d_nop_total);
    
    // Get nuclide configuration
    NuclideConfig* nucConfig = NuclideConfig::getInstance();
    if (!nucConfig) {
        std::cerr << "[ERROR] Failed to get nuclide configuration" << std::endl;
        return false;
    }
    
    const int num_nuclides = nucConfig->getNumNuclides();
    
    // Use first source for initialization
    const Source& source = sources[0];
    const float source_x = (source.lon + 179.0f) / 0.5f;
    const float source_y = (source.lat + 90.0f) / 0.5f;
    const float source_z = source.height;
    
    // Resize particle vector
    part.clear();
    part.reserve(d_nop_total);
    
    // Base deterministic seed
    const uint64_t base_seed = 1337;
    
    DEBUG_LOG("Starting particle initialization with base_seed=%llu", base_seed);
    
    // Initialize particles for each ensemble
    for (int e = 0; e < Nens; ++e) {
        const uint64_t seed_e = 1337 + 10007ULL * e;
        EnsembleRNG::XORShift128Plus rng(seed_e);
        
        for (int i = 0; i < nop_per_ensemble; ++i) {
            // Calculate time step index for this particle (0-23)
            int time_step_index = (i * T) / nop_per_ensemble;
            if (time_step_index >= T) time_step_index = T - 1;
            
            // Get ensemble-specific emission concentration for this time step
            const float base_concentration = emission_flat[e * T + time_step_index];
            
            // Generate random radius using normal distribution
            const float random_radius = rng.normal(g_mpi.particleSizes[mpiRank], 
                                                  g_mpi.sizeStandardDeviations[mpiRank]);
            
            // Global particle ID
            const int global_id = e * nop_per_ensemble + i + 1;
            
            // Create particle
            part.emplace_back(source_x, source_y, source_z,
                             g_mpi.decayConstants[mpiRank],
                             base_concentration,
                             g_mpi.depositionVelocities[mpiRank],
                             random_radius,
                             g_mpi.particleDensities[mpiRank],
                             global_id);
            
            LDMpart& current_particle = part.back();
            current_particle.timeidx = i; // Local time index within ensemble
            current_particle.flag = 0;    // Initially inactive
            
            // Initialize multi-nuclide concentrations
            for (int nuc = 0; nuc < MAX_NUCLIDES; ++nuc) {
                if (nuc < num_nuclides) {
                    const float initial_ratio = nucConfig->getInitialRatio(nuc);
                    current_particle.concentrations[nuc] = base_concentration * initial_ratio;
                } else {
                    current_particle.concentrations[nuc] = 0.0f;
                }
            }
            
            // Debug logging for first few particles
            if constexpr (LDM_DEBUG_ENS) {
                if (e < 2 && i < 3) {
                    DEBUG_LOG("Ensemble %d, Particle %d: time_step_index=%d, base_conc=%.3e, "
                             "first_nuclide_conc=%.3e, timeidx=%d, global_id=%d",
                             e, i, time_step_index, base_concentration,
                             current_particle.concentrations[0], current_particle.timeidx, global_id);
                }
            }
        }
    }
    
    auto cpu_time = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_time - start_time);
    DEBUG_LOG("CPU initialization completed in %ld ms", cpu_duration.count());
    
    // Update global particle count
    nop = d_nop_total;
    
    // Update device constant memory
    cudaError_t err = cudaMemcpyToSymbol(d_nop, &nop, sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] Failed to copy nop to device: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Allocate GPU memory
    if (d_part != nullptr) {
        cudaFree(d_part);
        d_part = nullptr;
    }
    
    const size_t total_size = d_nop_total * sizeof(LDMpart);
    err = cudaMalloc((void**)&d_part, total_size);
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] Failed to allocate device memory: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Copy particles to GPU using async transfer
    cudaStream_t stream;
    err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] Failed to create CUDA stream: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_part);
        d_part = nullptr;
        return false;
    }
    
    err = cudaMemcpyAsync(d_part, part.data(), total_size, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] Failed to copy particles to device: " << cudaGetErrorString(err) << std::endl;
        cudaStreamDestroy(stream);
        cudaFree(d_part);
        d_part = nullptr;
        return false;
    }
    
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] Failed to synchronize stream: " << cudaGetErrorString(err) << std::endl;
        cudaStreamDestroy(stream);
        cudaFree(d_part);
        d_part = nullptr;
        return false;
    }
    
    cudaStreamDestroy(stream);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    DEBUG_LOG("Total ensemble initialization completed in %ld ms (target: <4000ms)", total_duration.count());
    
    if (total_duration.count() > 4000) {
        DEBUG_LOG("PERFORMANCE WARNING: Initialization took longer than 4s target. "
                 "Optimization hints: 1) Use cudaMallocHost for pinned memory, "
                 "2) Parallelize CPU initialization, 3) Use multiple streams for large transfers");
    }
    
    // Save initialization values to integration_logs
    saveEnsembleInitializationLog(Nens, nop_per_ensemble, emission_flat, T);
    
    std::cout << "[INFO] Ensemble initialization successful: " << Nens << " ensembles, "
              << nop_per_ensemble << " particles each, total " << d_nop_total << " particles" << std::endl;
    
    return true;
}

// Function to save ensemble initialization log
void saveEnsembleInitializationLog(int Nens, int nop_per_ensemble, 
                                   const std::vector<float>& emission_flat, int T) {
    // Create timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto tm = *std::localtime(&time_t);
    
    std::ostringstream timestamp;
    timestamp << std::put_time(&tm, "%Y%m%d_%H%M%S");
    
    // Create filename
    std::string filename = "/home/jrpark/LDM-EKI/logs/integration_logs/ensemble_initialization_" 
                          + timestamp.str() + ".csv";
    
    std::ofstream logFile(filename);
    if (!logFile.is_open()) {
        std::cerr << "[ERROR] Failed to create ensemble initialization log: " << filename << std::endl;
        return;
    }
    
    // Write header
    logFile << "# Ensemble Initialization Log\n";
    logFile << "# Generated at: " << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << "\n";
    logFile << "# Number of ensembles: " << Nens << "\n";
    logFile << "# Particles per ensemble: " << nop_per_ensemble << "\n";
    logFile << "# Time steps: " << T << "\n";
    logFile << "# Total particles: " << (Nens * nop_per_ensemble) << "\n";
    logFile << "#\n";
    logFile << "# Format: Ensemble_ID,Particle_ID,Time_Step,Emission_Concentration,Source_X,Source_Y,Source_Z\n";
    logFile << "Ensemble_ID,Particle_ID,Time_Step,Emission_Concentration,Source_X,Source_Y,Source_Z\n";
    
    // Write particle initialization data
    for (int e = 0; e < Nens; ++e) {
        for (int i = 0; i < nop_per_ensemble; ++i) {
            const int time_step_index = (i * T) / nop_per_ensemble;
            const float base_concentration = emission_flat[e * T + time_step_index];
            const int global_id = e * nop_per_ensemble + i + 1;
            
            // Use actual source positions from EKI config
            EKIConfig* ekiConfig = EKIConfig::getInstance();
            const float source_x = (ekiConfig->getSourceLon() + 179.0f) / 0.5f;
            const float source_y = (ekiConfig->getSourceLat() + 90.0f) / 0.5f;
            const float source_z = ekiConfig->getSourceAlt();
            
            logFile << e << "," << global_id << "," << time_step_index << ","
                   << std::scientific << std::setprecision(6) << base_concentration << ","
                   << std::fixed << std::setprecision(3) << source_x << ","
                   << source_y << "," << source_z << "\n";
        }
    }
    
    logFile.close();
    std::cout << "[INFO] Ensemble initialization log saved: " << filename << std::endl;
}

