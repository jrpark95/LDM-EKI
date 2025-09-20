#pragma once

#include <mpi.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <cstdio>
#include <vector>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <sstream>
#include <math.h>
#include <limits>
#include <float.h>
#include <chrono>
#include <random>
#include <tuple>
#include <future>

// Forward declarations to avoid circular dependency


#include "ldm_struct.cuh"
#include "ldm_config.cuh" 
//#include "ldm_cram.cuh"

#include <math_constants.h>
#include <vector_types.h>

#ifdef _WIN32
    #include <direct.h>
#else
    #include <sys/types.h>
    #include <sys/stat.h>
#endif

// Physics model host variables (loaded from setting.txt)
extern int g_turb_switch;
extern int g_drydep;
extern int g_wetdep;
extern int g_raddecay;

// Physics model device constant memory (for kernels)
__constant__ int d_turb_switch;
__constant__ int d_drydep;
__constant__ int d_wetdep;
__constant__ int d_raddecay;

// Backward compatibility macros (for kernels)
#define TURB_SWITCH d_turb_switch
#define DRYDEP d_drydep  
#define WETDEP d_wetdep
#define RADDECAY d_raddecay

// CRAM Debug and Decay-Only Mode
// #define DECAY_ONLY 1

// Modern configuration structures
struct SimulationConfig {
    float timeEnd;
    float deltaTime;
    int outputFrequency;
    int numParticles;
    int gfsIndex = 0;
    bool isRural;
    bool isPasquillGifford;
    bool isGFS;
    float settlingVelocity = 0.0;
    float cunninghamFactor = 0.0;
};

struct MPIConfig {
    int rank = 1;
    int size = 1;
    std::string species[4];
    float decayConstants[4];
    float depositionVelocities[4];
    float particleSizes[4];
    float particleDensities[4];
    float sizeStandardDeviations[4];
};

SimulationConfig g_sim;
MPIConfig g_mpi;
ConfigReader g_config;

// 물리/격자 상수 정리 (계산 로직 변경 없음)
namespace Constants {
    // 격자 상수
    constexpr float LDAPS_E = 132.36f, LDAPS_W = 121.06f, LDAPS_N = 43.13f, LDAPS_S = 32.20f;
    constexpr int dimX = 602, dimY = 781, dimZ_pres = 24, dimZ_etas = 71;
    constexpr int dimX_GFS = 720, dimY_GFS = 361, dimZ_GFS = 26;
    constexpr int time_interval = 10800;
    
    // 물리 상수
    constexpr float d_trop = 50.0f, d_strat = 0.1f, turbmesoscale = 0.16f, r_earth = 6371000.0f;
    constexpr float _myl = 1.81e-5f, _nyl = 0.15e-4f, _lam = 6.53e-8f, _kb = 1.38e-23f;
    constexpr float _eps = 1.2e-38f, _Tr = 293.15f, _rair = 287.05f, _ga = 9.81f, _href = 15.0f;
    constexpr int _nspec = 19;
    constexpr int NI = 11;
    constexpr int N_NUCLIDES = 60; 
}

using namespace Constants;

// Global variables (like LDM-CRAM4)
// float time_end;
// float dt;
// int freq_output;
// int nop;
// int gfs_idx = 0;
// bool isRural;
// bool isPG;
// bool isGFS;
// float vsetaver = 0.0;
// float cunningham = 0.0;
#define time_end (g_sim.timeEnd)
#define dt (g_sim.deltaTime)
#define freq_output (g_sim.outputFrequency)
#define nop (g_sim.numParticles)
#define gfs_idx (g_sim.gfsIndex)
#define isRural (g_sim.isRural)
#define isPG (g_sim.isPasquillGifford)
#define isGFS (g_sim.isGFS)
#define vsetaver (g_sim.settlingVelocity)
#define cunningham (g_sim.cunninghamFactor)


__constant__ float d_time_end;
__constant__ float d_dt;
__constant__ int d_freq_output;  // Used in initialization
__constant__ int d_nop;  // Used in kernels
__constant__ bool d_isRural;  // Used in initialization
__constant__ bool d_isPG;    // Used in initialization
__constant__ bool d_isGFS;   // Used in initialization
__constant__ float d_vsetaver;
__constant__ float d_cunningham;

__constant__ float d_start_lat;  // Used in grid output functions
__constant__ float d_start_lon;  // Used in grid output functions
__constant__ float d_lat_step;   // Used in grid output functions
__constant__ float d_lon_step;   // Used in grid output functions


std::vector<float> flex_hgt;  // Used in flex functions


__constant__ float d_flex_hgt[50];  // Used in flex functions
__constant__ float T_const[N_NUCLIDES * N_NUCLIDES];

__device__ double d_uWind[512];
__device__ double d_vWind[512];

class LDM
{
private:

    PresData* device_meteorological_data_pres;  // Used in constructor/destructor
    UnisData* device_meteorological_data_unis;  // Used in constructor/destructor
    EtasData* device_meteorological_data_etas;  // Used in constructor/destructor

    FlexUnis* device_meteorological_flex_unis0;
    FlexPres* device_meteorological_flex_pres0;
    FlexUnis* device_meteorological_flex_unis1;
    FlexPres* device_meteorological_flex_pres1;
    FlexUnis* device_meteorological_flex_unis2;
    FlexPres* device_meteorological_flex_pres2;

    float4* host_unisA0; // HMIX, USTR, WSTR, OBKL
    float4* host_unisB0; // VDEP, LPREC, CPREC, TCC
    float4* host_unisA1;
    float4* host_unisB1;
    float4* host_unisA2;
    float4* host_unisB2;

    cudaArray* d_unisArrayA0; //
    cudaArray* d_unisArrayB0; //
    cudaArray* d_unisArrayA1; //
    cudaArray* d_unisArrayB1; //
    cudaArray* d_unisArrayA2; //
    cudaArray* d_unisArrayB2; //

    cudaChannelFormatDesc channelDesc2D = cudaCreateChannelDesc<float4>();
    cudaChannelFormatDesc channelDesc3D = cudaCreateChannelDesc<float4>();

    float4* host_presA0; // DRHO, RHO, TT, QV
    float4* host_presB0; // UU, VV, WW, 0.0 
    float4* host_presA1;
    float4* host_presB1;
    float4* host_presA2;
    float4* host_presB2;

    cudaArray* d_presArrayA0; //
    cudaArray* d_presArrayB0; //
    cudaArray* d_presArrayA1; //
    cudaArray* d_presArrayB1; //
    cudaArray* d_presArrayA2; //
    cudaArray* d_presArrayB2; //

    std::vector<Source> sources;  // Used in initialization
    std::vector<float> decayConstants;  // Used in initialization
    std::vector<float> drydepositionVelocity;  // Used in initialization
    std::vector<Concentration> concentrations;  // Used in initialization

    cudaTextureObject_t m_texUnisA0 = 0;
    cudaTextureObject_t m_texUnisA1 = 0;
    cudaTextureObject_t m_texUnisA2 = 0;

    cudaTextureObject_t m_texUnisB0 = 0;
    cudaTextureObject_t m_texUnisB1 = 0;
    cudaTextureObject_t m_texUnisB2 = 0;

    cudaTextureObject_t m_texPresA0 = 0;
    cudaTextureObject_t m_texPresA1 = 0;
    cudaTextureObject_t m_texPresA2 = 0;

    cudaTextureObject_t m_texPresB0 = 0;
    cudaTextureObject_t m_texPresB1 = 0;
    cudaTextureObject_t m_texPresB2 = 0;
    
public:

    LDM();
    ~LDM();

    float minX, minY, maxX, maxY;
    float *d_minX, *d_minY, *d_maxX, *d_maxY;

    // std::vector<float> U_flat;  // Unused - only used in lkd/val functions
    // std::vector<float> V_flat;  // Unused - only used in lkd/val functions
    // std::vector<float> W_flat;  // Unused - only used in lkd/val functions

    // float* d_U_flat = nullptr;  // Unused - only used in lkd/val functions
    // float* d_V_flat = nullptr;  // Unused - only used in lkd/val functions
    // float* d_W_flat = nullptr;  // Unused - only used in lkd/val functions

    // std::vector<float> U_flat_next;  // Unused - only used in lkd/val functions
    // std::vector<float> V_flat_next;  // Unused - only used in lkd/val functions
    // std::vector<float> W_flat_next;  // Unused - only used in lkd/val functions

    // float* d_U_flat_next = nullptr;  // Unused - only used in lkd/val functions
    // float* d_V_flat_next = nullptr;  // Unused - only used in lkd/val functions
    // float* d_W_flat_next = nullptr;  // Unused - only used in lkd/val functions

    std::chrono::high_resolution_clock::time_point timerStart, timerEnd;
    std::chrono::high_resolution_clock::time_point stepStart, stepEnd;


    __device__ __host__ struct LDMpart{

        float x, y, z;                       // Essential: Position coordinates
        float u, v, w;                       // Essential: Velocity components
        float up, vp, wp;                    // Essential: Turbulent velocity components (used in kernels)
        // float um, vm, wm;                // Memory save: Mean velocity components (rarely used)
        float decay_const;                   // Legacy: Single nuclide decay constant (still used for compatibility)
        float conc;                          // Legacy: Single nuclide concentration (still used for output)
        float concentrations[N_NUCLIDES];  // Essential: Multi-nuclide concentration vector (60 nuclides)
        float age;                           // Essential: Particle age for decay calculations
        float virtual_distance;              // Essential: Used in dispersion calculations
        float u_wind, v_wind, w_wind;        // Essential: Meteorological wind components
        float sigma_z, sigma_h;              // Essential: Turbulent dispersion parameters
        float drydep_vel;                    // Essential: Dry deposition velocity
        float radi, prho;                    // Essential: Particle radius and density (used in settling)
        // float dsig;                      // Memory save: Size standard deviation (less critical)
        curandState* randState;              // Essential: Random number generator state

        int timeidx;                         // Essential: Time index for particle tracking
        int flag;                            // Essential: Particle active flag (heavily used)
        int dir;                             // Essential: Direction flag (used in reflection)

        LDMpart() :
            x(0.0f), y(0.0f), z(0.0f), 
            u(0.0f), v(0.0f), w(0.0f), 
            up(0.0f), vp(0.0f), wp(0.0f),
            decay_const(0.0f), 
            conc(0.0f), 
            age(0.0f), 
            virtual_distance(1e-5), 
            u_wind(0.0f), v_wind(0.0f), w_wind(0.0f),
            sigma_z(0.0f), sigma_h(0.0f),
            drydep_vel(0.0f), radi(0.0f), prho(0.0f),
            randState(0),
            timeidx(0), flag(0), dir(1){
                // Initialize multi-nuclide concentrations to zero
                for(int i = 0; i < N_NUCLIDES; i++) concentrations[i] = 0.0f;
            }

        LDMpart(float _x, float _y, float _z, 
                float _decayConstant, float _concentration, 
                float _drydep_vel, int _timeidx)  : 
                x(_x), y(_y), z(_z), 
                u(0.0f), v(0.0f), w(0.0f), 
                up(0.0f), vp(0.0f), wp(0.0f),
                decay_const(_decayConstant), 
                conc(_concentration), 
                age(0.0f),
                virtual_distance(1e-5), 
                u_wind(0.0f), v_wind(0.0f), w_wind(0.0f),
                sigma_z(0.0f), sigma_h(0.0f),
                drydep_vel(_drydep_vel), radi(0.0f), prho(0.0f),
                randState(0),
                timeidx(_timeidx), flag(0), dir(1){
                    // Initialize multi-nuclide concentrations to zero 
                    // (will be properly set in initialization functions)
                    for(int i = 0; i < N_NUCLIDES; i++) concentrations[i] = 0.0f;
                }

        LDMpart(float _x, float _y, float _z, 
                float _decayConstant, float _concentration, 
                float _drydep_vel, float radi, float prho, int _timeidx)  : 
                x(_x), y(_y), z(_z), 
                u(0.0f), v(0.0f), w(0.0f), 
                up(0.0f), vp(0.0f), wp(0.0f),
                decay_const(_decayConstant), 
                conc(_concentration), 
                age(0.0f),
                virtual_distance(1e-5), 
                u_wind(0.0f), v_wind(0.0f), w_wind(0.0f),
                sigma_z(0.0f), sigma_h(0.0f),
                drydep_vel(_drydep_vel), radi(radi), prho(prho),
                randState(0),
                timeidx(_timeidx), flag(0), dir(1){
                    // Initialize multi-nuclide concentrations to zero 
                    // (will be properly set in initialization functions)
                    for(int i = 0; i < N_NUCLIDES; i++) concentrations[i] = 0.0f;
                }

    };

    std::vector<LDMpart> part;
    LDMpart* d_part = nullptr;

    // ldm_func.cuh
    void printParticleData();
    void allocateGPUMemory();
    void runSimulation();

    void findBoundingBox();
    void startTimer();   // Used in main
    void stopTimer();     // Used in main

    void createTextureObjects(); 
    void destroyTextureObjects();

    // ldm_init.cuh
    void loadSimulationConfiguration();  // Used in main
    void cleanOutputDirectory();  // Used in loadSimulationConfiguration
    void calculateAverageSettlingVelocity();  // Used in main
    void initializeParticles();
    void calculateSettlingVelocity();
    
    // EKI time-based particle emission functions
    void releaseParticlesForCurrentTime(float current_time_seconds);
    void updateParticleConcentrationsFromEKI(float current_time_seconds);
    void updateGPUParticleMemory();
    
    // Particle data logging and visualization functions
    void logParticlePositionsForVisualization(int timestep, float currentTime);
    void logParticleCountData(int timestep, float currentTime);
    
    // Ensemble initialization methods  
    bool initializeParticlesEnsembles(int Nens,
                                     const std::vector<float>& emission_time_series,
                                     const std::vector<Source>& sources,
                                     int nop_per_ensemble);
    
    bool initializeParticlesEnsemblesFlat(int Nens,
                                          const std::vector<float>& emission_flat,
                                          const std::vector<Source>& sources,
                                          int nop_per_ensemble);
    
    // Integration methods
    bool writeObservationsSingle(const std::string& dir, const std::string& tag);
    void freeGPUMemory();
    bool runSimulationEnsembles(int Nens);
    bool writeObservationsEnsembles(const std::string& dir, const std::string& tag);
    bool writeIntegrationDebugLogs(const std::string& dir, const std::string& tag);
    
    // Ensemble state
    bool ensemble_mode_active = false;
    int current_Nens = 1;
    int current_nop_per_ensemble = 0;

    // ldm_mdata.cuh
    void initializeFlexGFSData();  // Used in main
    void loadFlexGFSData();  // Used in time_update_mpi
    void loadMeteorologicalHeightData();
    void loadFlexHeightData();

    // ldm_plot.cuh
    int countActiveParticles();
    void swapByteOrder(float& value);
    void outputParticlesBinaryMPI(int timestep);  // Used in time_update_mpi

    // // Concentration tracking functions
    void log_first_particle_concentrations(int timestep, float currentTime);
    void log_all_particles_nuclide_ratios(int timestep, float currentTime);
    void log_first_particle_cram_detail(int timestep, float currentTime, float dt_used);
    void log_first_particle_decay_analysis(int timestep, float currentTime);
    
    // // Validation functions for CRAM4 reference data
    void exportValidationData(int timestep, float currentTime);
    void exportConcentrationGrid(int timestep, float currentTime);
    void exportNuclideTotal(int timestep, float currentTime);

    // ldm_cram2.cuh
    // static void gauss_solve_pivot_inplace(double* A, double* b, double* x, int N);
    // static void cram48_expm_times_ej_host(const std::vector<double>& A, int j, std::vector<double>& out_col);
    // bool build_T_on_host_and_upload(const char* A60_csv_path);
    // void debug_print_T_head(int k);

    bool load_A_csv(const char* path, std::vector<double>& A_out);
    void gauss_solve_inplace(std::vector<double>& M, std::vector<double>& b, int n);
    void cram48_expm_times_ej_host(const std::vector<double>& A, int j, std::vector<double>& col_out);
    bool build_T_matrix_and_upload(const char* A60_csv_path);
    bool initialize_cram_system(const char* A60_csv_path);
};

#define LDM_CLASS_DECLARED 1
// Global MPI variables - modernized
extern int mpiRank, mpiSize;

// Ensemble activation kernel declarations (must be global)
__global__ void update_particle_flags_ensembles(LDM::LDMpart* d_part,
                                               int nop_per_ensemble,
                                               int Nens,
                                               float activationRatio);

__global__ void count_active_particles_per_ensemble(const LDM::LDMpart* d_part,
                                                   int nop_per_ensemble,
                                                   int Nens,
                                                   int* active_counts);

// Debug macros
#define LDM_DEBUG_ENS 0

// Ensemble mode global variables
extern bool ensemble_mode_active;
extern int Nens;
extern int nop_per_ensemble;

// Global nuclide count variable
extern int g_num_nuclides;

// Legacy aliases for compatibility
#define __rank mpiRank
#define __size mpiSize

// Note: Core constants are now compile-time constants for CUDA compatibility

struct GridConfig {
    float start_lat{36.0f};
    float start_lon{140.0f};
    float end_lat{37.0f};
    float end_lon{141.0f};
    float lat_step{0.5f};
    float lon_step{0.5f};
};


GridConfig loadGridConfig() {
    GridConfig config;
    
    std::string source_file_path = g_config.getString("input_base_path", "./data/input/") + "source.txt";
    std::ifstream file(source_file_path);
    
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open source.txt for grid config, using defaults" << std::endl;
        return config;
    }
    
    std::string line;
    bool in_grid_section = false;
    
    while (std::getline(file, line)) {
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        
        if (line.empty() || line[0] == '#') continue;
        
        if (line == "[GRID_CONFIG]") {
            in_grid_section = true;
            continue;
        }
        
        if (line.find('[') != std::string::npos) {
            in_grid_section = false;
            continue;
        }
        
        if (in_grid_section) {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 1);
                
                key.erase(0, key.find_first_not_of(" \t"));
                key.erase(key.find_last_not_of(" \t") + 1);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t") + 1);
                
                try {
                    if (key == "start_lat") config.start_lat = std::stof(value);
                    else if (key == "start_lon") config.start_lon = std::stof(value);
                    else if (key == "end_lat") config.end_lat = std::stof(value);
                    else if (key == "end_lon") config.end_lon = std::stof(value);
                    else if (key == "lat_step") config.lat_step = std::stof(value);
                    else if (key == "lon_step") config.lon_step = std::stof(value);
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Invalid value for " << key << ": " << value << std::endl;
                }
            }
        }
    }
    
    return config;
}

// Global variable definitions
int mpiRank = 1, mpiSize = 1;

// Ensemble mode global variables implementation
bool ensemble_mode_active = false;
int Nens = 1;
int nop_per_ensemble = 0;

// void loadRadionuclideData() {
//     std::vector<std::string> species_names = g_config.getStringArray("species_names");
//     std::vector<float> decay_constants = g_config.getFloatArray("decay_constants");
//     std::vector<float> deposition_velocities = g_config.getFloatArray("deposition_velocities");
//     std::vector<float> particle_sizes = g_config.getFloatArray("particle_sizes");
//     std::vector<float> particle_densities = g_config.getFloatArray("particle_densities");
//     std::vector<float> size_standard_deviations = g_config.getFloatArray("size_standard_deviations");
    
//     for (int i = 0; i < 4 && i < species_names.size(); i++) {
//         g_mpi.species[i] = species_names[i];
//         g_mpi.decayConstants[i] = (i < decay_constants.size()) ? decay_constants[i] : 1.00e-6f;
//         g_mpi.depositionVelocities[i] = (i < deposition_velocities.size()) ? deposition_velocities[i] : 0.01f;
//         g_mpi.particleSizes[i] = (i < particle_sizes.size()) ? particle_sizes[i] : 0.6f;
//         g_mpi.particleDensities[i] = (i < particle_densities.size()) ? particle_densities[i] : 2500.0f;
//         g_mpi.sizeStandardDeviations[i] = (i < size_standard_deviations.size()) ? size_standard_deviations[i] : 0.01f;
//     }
// }
// float __Z[720] = {0.0f, };


#include "ldm_cram2.cuh"
#include "ldm_kernels.cuh" 
#include "ldm_init.cuh" 
#include "ldm_mdata.cuh" 
#include "ldm_ensemble_init.cuh"
// #include "ldm_socket.cuh"
#include "ldm_func.cuh" 
#include "ldm_plot.cuh"