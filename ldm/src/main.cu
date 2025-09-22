#include "ldm.cuh"
#include "ldm_nuclides.cuh"
#include "ldm_eki.cuh"
#include "ldm_eki_logger.cuh"
#include <cstdio>  // for remove() function
#include <fstream>
#include <sstream>
#include <iomanip>

// Physics model global variables (will be loaded from setting.txt)
int g_num_nuclides = 1;   // Single nuclide: Co-60
int g_turb_switch = 0;    // Default values, overwritten by setting.txt
int g_drydep = 0;
int g_wetdep = 0; 
int g_raddecay = 0;       // Radioactive decay disabled for simple simulation

// Function declarations
void saveEnsembleInitializationLog(int Nens, const std::vector<float>& emission_time_series, const std::vector<Source>& sources, int nop_per_ensemble);

// Proper ensemble particle activation kernel
__global__ void update_particle_flags_ensembles(LDM::LDMpart* d_part, int nop_per_ensemble, int Nens, float activationRatio) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nop_per_ensemble * Nens;
    
    if (idx >= total) return;
    
    // Calculate which ensemble this particle belongs to
    int ensemble_id = idx / nop_per_ensemble;
    int particle_idx_in_ensemble = idx % nop_per_ensemble;
    
    // Each ensemble should activate particles based on ensemble-local indices
    int maxActiveTimeidx = int(nop_per_ensemble * activationRatio);
    if (particle_idx_in_ensemble <= maxActiveTimeidx) {
        d_part[idx].flag = 1;
    } else {
        d_part[idx].flag = 0;
    }
    
    // Debug: Print activation info for first few particles of first few ensembles
    if (ensemble_id < 3 && particle_idx_in_ensemble < 5) {
        printf("[KERNEL] E%d_P%d: idx_in_ens=%d, maxActive=%d, sim_ratio=%.3f, flag=%d\n",
               ensemble_id, particle_idx_in_ensemble, particle_idx_in_ensemble, 
               maxActiveTimeidx, activationRatio, d_part[idx].flag);
    }
}

// Sequential workflow functions
bool runSingleModeLDM(LDM& ldm);
bool runPostProcessing();
bool runEKIEstimationStep();
bool loadEKIEnsembleResults(std::vector<std::vector<float>>& ensemble_matrix, int& time_intervals, int& ensemble_size);
bool runEnsembleLDM(LDM& ldm, const std::vector<std::vector<float>>& ensemble_matrix, int time_intervals, int ensemble_size);

// Ensemble simulation functions
bool initializeEnsembleParticles(LDM& ldm, const std::vector<std::vector<float>>& ensemble_matrix, int time_intervals, int ensemble_size, const std::vector<Source>& sources);
void saveEnsembleParticleSnapshot(int timestep, const std::vector<LDM::LDMpart>& particles, int ensemble_size, int nop_per_ensemble);
bool calculateEnsembleObservations(float ensemble_observations[100][24][3], int ensemble_size, int time_intervals, const std::vector<LDM::LDMpart>& particles);
void saveEnsembleObservationsToEKI(const float ensemble_observations[100][24][3], int ensemble_size, int time_intervals, int iteration);

// Function to clean all log directories
void cleanAllLogDirectories() {
    std::cout << "[INFO] Cleaning all log directories..." << std::endl;
    
    // Clean EKI logs
    system("rm -f /home/jrpark/LDM-EKI/logs/eki_logs/*.bin");
    system("rm -f /home/jrpark/LDM-EKI/logs/eki_logs/*.csv"); 
    system("rm -f /home/jrpark/LDM-EKI/logs/eki_logs/*.log");
    
    // Clean integration logs
    system("rm -f /home/jrpark/LDM-EKI/logs/integration_logs/*.csv");
    system("rm -f /home/jrpark/LDM-EKI/logs/integration_logs/*.txt");
    system("rm -f /home/jrpark/LDM-EKI/logs/integration_logs/*.png");
    
    // Clean LDM logs (keep directory structure)
    system("rm -f /home/jrpark/LDM-EKI/logs/ldm_logs/*.csv");
    system("rm -f /home/jrpark/LDM-EKI/logs/ldm_logs/*.png");
    system("rm -f /home/jrpark/LDM-EKI/logs/ldm_logs/*.bin");
    system("rm -rf /home/jrpark/LDM-EKI/logs/ldm_logs/ensemble_results");
    
    std::cout << "[INFO] All log directories cleaned successfully!" << std::endl;
}

int main(int argc, char** argv) {

    mpiRank = 1;
    mpiSize = 1;

    // Clean all previous log files before starting
    cleanAllLogDirectories();

    std::cout << "==================================================================" << std::endl;
    std::cout << "       LDM-EKI Sequential Workflow - Integrated Simulation        " << std::endl;
    std::cout << "==================================================================" << std::endl;
    std::cout << "Workflow Steps:" << std::endl;
    std::cout << "  1. Single Mode LDM Execution" << std::endl;
    std::cout << "  2. Post-processing and Observation Data Generation" << std::endl;
    std::cout << "  3. EKI Estimation" << std::endl;
    std::cout << "  4. Ensemble Generation from EKI Results" << std::endl;
    std::cout << "  5. Ensemble Mode LDM Execution" << std::endl;
    std::cout << "==================================================================" << std::endl;

    // Load nuclide configuration (single Co-60 nuclide)
    NuclideConfig* nucConfig = NuclideConfig::getInstance();
    std::string nuclide_config_file = "./data/input/nuclides_config_1.txt";
    
    if (!nucConfig->loadFromFile(nuclide_config_file)) {
        std::cerr << "[ERROR] Failed to load nuclide configuration" << std::endl;
        return 1;
    }
    
    // Update global nuclide count
    g_num_nuclides = nucConfig->getNumNuclides();
    
    // Load EKI configuration
    EKIConfig* ekiConfig = EKIConfig::getInstance();
    std::string eki_config_file = "../eki/config/input_data";
    
    if (!ekiConfig->loadFromFile(eki_config_file)) {
        std::cerr << "[ERROR] Failed to load EKI configuration" << std::endl;
        return 1;
    }
    
    printEKIConfiguration(ekiConfig);
    writeEKIConfigLog(ekiConfig);
    
    // Initialize LDM instance
    LDM ldm;
    ldm.loadSimulationConfiguration();
    printSystemInfo();
    ldm.calculateAverageSettlingVelocity();
    
    cleanLogDirectory();

    // =================================================================
    // STEP 1: Single Mode LDM Execution
    // =================================================================
    std::cout << "\n[STEP 1] Starting Single Mode LDM Execution..." << std::endl;
    
    if (!runSingleModeLDM(ldm)) {
        std::cerr << "[ERROR] Single mode LDM execution failed" << std::endl;
        return 1;
    }
    
    std::cout << "[STEP 1] Single mode LDM execution completed successfully" << std::endl;

    // =================================================================
    // STEP 2: Post-processing and Observation Data Generation
    // =================================================================
    std::cout << "\n[STEP 2] Starting Post-processing..." << std::endl;
    
    if (!runPostProcessing()) {
        std::cerr << "[ERROR] Post-processing failed" << std::endl;
        return 1;
    }
    
    std::cout << "[STEP 2] Post-processing completed successfully" << std::endl;

    // =================================================================
    // STEP 3: EKI Estimation
    // =================================================================
    std::cout << "\n[STEP 3] Starting EKI Estimation..." << std::endl;
    
    if (!runEKIEstimationStep()) {
        std::cerr << "[ERROR] EKI estimation failed" << std::endl;
        return 1;
    }
    
    std::cout << "[STEP 3] EKI estimation completed successfully" << std::endl;

    // =================================================================
    // STEP 4: Load EKI Ensemble Results
    // =================================================================
    std::cout << "\n[STEP 4] Loading EKI Ensemble Results..." << std::endl;
    
    std::vector<std::vector<float>> ensemble_matrix;
    int time_intervals, ensemble_size;
    
    if (!loadEKIEnsembleResults(ensemble_matrix, time_intervals, ensemble_size)) {
        std::cerr << "[ERROR] Failed to load EKI ensemble results" << std::endl;
        return 1;
    }
    
    std::cout << "[STEP 4] Successfully loaded ensemble matrix [" 
              << time_intervals << " x " << ensemble_size << "]" << std::endl;

    // =================================================================
    // STEP 5: Ensemble Mode LDM Execution
    // =================================================================
    std::cout << "\n[STEP 5] Starting Ensemble Mode LDM Execution..." << std::endl;
    
    if (!runEnsembleLDM(ldm, ensemble_matrix, time_intervals, ensemble_size)) {
        std::cerr << "[ERROR] Ensemble mode LDM execution failed" << std::endl;
        return 1;
    }
    
    std::cout << "[STEP 5] Ensemble mode LDM execution completed successfully" << std::endl;

    // =================================================================
    // Final Cleanup and Summary
    // =================================================================
    std::cout << "\n==================================================================" << std::endl;
    std::cout << "       LDM-EKI Sequential Workflow Completed Successfully         " << std::endl;
    std::cout << "==================================================================" << std::endl;

    // Clean up intermediate particle files (keep 0 and 24, remove 1-23)
    std::cout << "\n[CLEANUP] Cleaning up intermediate particle files..." << std::endl;
    int files_removed = 0;
    for (int i = 1; i <= 23; i++) {
        std::string filename = "/home/jrpark/LDM-EKI/logs/ldm_logs/particles_15min_" + std::to_string(i) + ".csv";
        if (remove(filename.c_str()) == 0) {
            files_removed++;
        }
    }
    std::cout << "[CLEANUP] Intermediate particle files cleaned (" << files_removed 
              << " files removed, kept particles_15min_0.csv and particles_15min_24.csv)" << std::endl;

    return 0;
}

// =================================================================
// Implementation of Sequential Workflow Functions
// =================================================================

bool runSingleModeLDM(LDM& ldm) {
    std::cout << "  [1.1] Initializing particles in single mode..." << std::endl;
    
    // Ensure single mode (no ensemble)
    ensemble_mode_active = false;
    ldm.ensemble_mode_active = false;
    Nens = 1;
    
    // Reset nop to correct single mode value from config file
    nop = g_config.getInt("Total_number_of_particle", 1000);
    
    // Initialize particles in single mode
    ldm.initializeParticles();
    
    std::cout << "  [1.2] Loading height and GFS data..." << std::endl;
    ldm.loadFlexHeightData();
    ldm.initializeFlexGFSData();
    
    std::cout << "  [1.3] Allocating GPU memory..." << std::endl;
    ldm.allocateGPUMemory();
    
    std::cout << "  [1.4] Running single mode simulation..." << std::endl;
    ldm.startTimer();
    ldm.runSimulation();
    ldm.stopTimer();
    
    std::cout << "  [1.5] Single mode simulation completed" << std::endl;
    
    // Ensure d_nop is properly set to single mode value after simulation
    cudaError_t err = cudaMemcpyToSymbol("d_nop", &nop, sizeof(int));
    if (err == cudaSuccess) {
        std::cout << "  [1.5.1] Confirmed d_nop reset to single mode value: " << nop << std::endl;
    }
    
    return true;
}

bool runPostProcessing() {
    std::cout << "  [2.1] Preparing observation data..." << std::endl;
    
    EKIConfig* ekiConfig = EKIConfig::getInstance();
    prepareObservationData(ekiConfig);
    
    std::cout << "  [2.2] Running visualization..." << std::endl;
    runVisualization();
    
    std::cout << "  [2.3] Post-processing completed" << std::endl;
    
    return true;
}

bool runEKIEstimationStep() {
    std::cout << "  [3.1] Executing EKI estimation algorithm..." << std::endl;
    
    printSimulationStatus("Running EKI estimation...");
    ::runEKIEstimation();  // Call global function
    
    std::cout << "  [3.2] EKI estimation algorithm completed" << std::endl;
    
    return true;
}

bool loadEKIEnsembleResults(std::vector<std::vector<float>>& ensemble_matrix, 
                           int& time_intervals, int& ensemble_size) {
    std::cout << "  [4.1] Loading ensemble states from EKI..." << std::endl;
    
    // Try to load ensemble from latest iteration
    for (int iter = 5; iter >= 1; iter--) {
        if (loadEKIEnsembleStates(ensemble_matrix, time_intervals, ensemble_size, iter)) {
            std::cout << "  [4.2] Loaded ensemble from iteration " << iter << std::endl;
            
            // Validate ensemble data
            if (ensemble_size != 100 || time_intervals != 24) {
                std::cerr << "  [4.2] WARNING: Unexpected ensemble dimensions: " 
                         << time_intervals << "x" << ensemble_size 
                         << " (expected 24x100)" << std::endl;
            }
            
            // Display sample values
            float sample_value = ensemble_matrix[0][0];
            std::cout << "  [4.3] Sample ensemble value [time=0, ensemble=0]: " << sample_value << std::endl;
            std::cout << "  [4.4] Memory allocated: " << (time_intervals * ensemble_size * sizeof(float)) << " bytes" << std::endl;
            
            return true;
        }
    }
    
    std::cerr << "  [4.1] No ensemble files found from any iteration" << std::endl;
    return false;
}

bool runEnsembleLDM(LDM& ldm, const std::vector<std::vector<float>>& ensemble_matrix, 
                   int time_intervals, int ensemble_size) {
    std::cout << "  [5.1] Configuring ensemble mode..." << std::endl;
    
    // Set ensemble mode parameters
    const int Nens = ensemble_size;  // 100 ensembles
    const int nop_per_ensemble = 1000;  // 1000 particles per ensemble as specified
    
    ensemble_mode_active = true;
    ::Nens = Nens;
    ::nop_per_ensemble = nop_per_ensemble;
    
    std::cout << "  [5.2] Ensemble configuration: " << Nens << " ensembles, " 
              << nop_per_ensemble << " particles each, total " << (Nens * nop_per_ensemble) << std::endl;
    
    // Set ensemble total particles (enop = Nens * nop_per_ensemble)
    enop = Nens * nop_per_ensemble;
    
    // The ensemble matrix already contains ensemble-specific data
    std::cout << "  [5.3] Using ensemble-specific emission data..." << std::endl;
    
    // No need for a single emission time series - each ensemble has its own data in ensemble_matrix
    // This will be handled in initializeEnsembleParticles function
    
    // Create source from EKI configuration
    std::vector<Source> sources;
    EKIConfig* ekiConfig = EKIConfig::getInstance();
    Source eki_source;
    eki_source.lat = ekiConfig->getSourceLat();
    eki_source.lon = ekiConfig->getSourceLon();
    eki_source.height = ekiConfig->getSourceAlt();
    sources.push_back(eki_source);
    
    std::cout << "  [5.4] Initializing ensemble particles..." << std::endl;
    
    // Use new ensemble-specific initialization
    bool ensemble_init_success = initializeEnsembleParticles(ldm, ensemble_matrix, time_intervals, ensemble_size, sources);
    
    if (!ensemble_init_success) {
        std::cerr << "  [5.4] Ensemble particle initialization failed" << std::endl;
        return false;
    }
    
    // Save ensemble initialization log with sample emission data
    std::vector<float> sample_emission(time_intervals);
    for (int t = 0; t < time_intervals; t++) {
        sample_emission[t] = ensemble_matrix[t][0]; // Use first ensemble as sample
    }
    saveEnsembleInitializationLog(Nens, sample_emission, sources, nop_per_ensemble);
    
    std::cout << "  [5.5] Running ensemble simulation..." << std::endl;
    
    // Activate ensemble mode for LDM class
    ldm.ensemble_mode_active = true;
    ldm.current_Nens = Nens;
    ldm.current_nop_per_ensemble = nop_per_ensemble;
    
    // Make sure enop is accessible in LDM class
    std::cout << "  [5.5.1] Setting ensemble total particles: " << enop << std::endl;
    
    // Force d_nop to be set correctly for ensemble mode by reloading configuration
    std::cout << "  [5.5.2] Reloading simulation configuration for ensemble mode..." << std::endl;
    ldm.loadSimulationConfiguration();
    std::cout << "  [5.5.3] Simulation configuration reloaded for ensemble mode" << std::endl;
    
    // Run ensemble simulation
    ldm.startTimer();
    ldm.runSimulation();
    ldm.stopTimer();
    
    // Deactivate ensemble mode after simulation
    ensemble_mode_active = false;
    ldm.ensemble_mode_active = false;
    
    // Reset d_nop back to single mode value
    cudaError_t reset_err = cudaMemcpyToSymbol("d_nop", &nop, sizeof(int));
    if (reset_err == cudaSuccess) {
        std::cout << "  [5.6.1] Reset d_nop back to single mode value: " << nop << std::endl;
    }
    
    std::cout << "  [5.6] Ensemble simulation completed" << std::endl;
    
    // Calculate ensemble observations for EKI feedback
    std::cout << "  [5.7] Calculating ensemble observations..." << std::endl;
    
    static float ensemble_observations[100][24][3];
    bool obs_success = calculateEnsembleObservations(ensemble_observations, ensemble_size, time_intervals, ldm.part);
    
    if (!obs_success) {
        std::cerr << "  [5.7] Failed to calculate ensemble observations" << std::endl;
        return false;
    }
    
    // Save observations for EKI
    saveEnsembleObservationsToEKI(ensemble_observations, ensemble_size, time_intervals, 1);
    
    std::cout << "  [5.7] Ensemble observations calculated and saved for EKI feedback" << std::endl;
    
    return true;
}

// Function to save ensemble initialization log
void saveEnsembleInitializationLog(int Nens, 
                                   const std::vector<float>& emission_time_series,
                                   const std::vector<Source>& sources, 
                                   int nop_per_ensemble) {
    // Create timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto tm = *std::localtime(&time_t);
    
    std::ostringstream timestamp;
    timestamp << std::put_time(&tm, "%Y%m%d_%H%M%S");
    
    // Create filename
    std::string filename = "/home/jrpark/LDM-EKI/logs/integration_logs/ensemble_sequential_" 
                          + timestamp.str() + ".csv";
    
    std::ofstream logFile(filename);
    if (!logFile.is_open()) {
        std::cerr << "[ERROR] Failed to create ensemble initialization log: " << filename << std::endl;
        return;
    }
    
    const int T = static_cast<int>(emission_time_series.size());
    const Source& source = sources[0];
    const float source_x = (source.lon + 179.0f) / 0.5f;
    const float source_y = (source.lat + 90.0f) / 0.5f;
    const float source_z = source.height;
    
    // Write header
    logFile << "# Sequential LDM-EKI Ensemble Initialization Log\n";
    logFile << "# Generated at: " << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << "\n";
    logFile << "# Workflow: Single LDM → Post-processing → EKI → Ensemble LDM\n";
    logFile << "# Number of ensembles: " << Nens << "\n";
    logFile << "# Particles per ensemble: " << nop_per_ensemble << "\n";
    logFile << "# Emission time steps: " << T << "\n";
    logFile << "# Total particles: " << (Nens * nop_per_ensemble) << "\n";
    logFile << "# Source location: lon=" << source.lon << ", lat=" << source.lat << ", height=" << source.height << "\n";
    logFile << "#\n";
    logFile << "Ensemble_ID,Total_Particles,Mean_Concentration,Source_Location\n";
    
    // Write ensemble summary data
    for (int e = 0; e < Nens; ++e) {
        float mean_concentration = 0.0f;
        for (int t = 0; t < T; ++t) {
            mean_concentration += emission_time_series[t];
        }
        mean_concentration /= T;
        
        logFile << e << "," << nop_per_ensemble << ","
               << std::scientific << std::setprecision(6) << mean_concentration << ","
               << "(" << source.lon << "," << source.lat << "," << source.height << ")\n";
    }
    
    logFile.close();
    std::cout << "  [5.4] Ensemble initialization log saved: " << filename << std::endl;
}

// =================================================================
// Ensemble Simulation Functions
// =================================================================

bool initializeEnsembleParticles(LDM& ldm, const std::vector<std::vector<float>>& ensemble_matrix, 
                                int time_intervals, int ensemble_size, const std::vector<Source>& sources) {
    
    std::cout << "  [ENSEMBLE] Initializing " << ensemble_size << " ensembles with " 
              << "1000 particles each..." << std::endl;
    
    // Log EKI ensemble matrix sample data
    std::cout << "  [EKI_MATRIX] Sample ensemble matrix data:" << std::endl;
    for (int e = 0; e < std::min(5, ensemble_size); e++) {
        std::cout << "    Ensemble " << e << ": ";
        for (int t = 0; t < std::min(5, time_intervals); t++) {
            std::cout << "T" << t << "=" << std::scientific << std::setprecision(2) 
                     << ensemble_matrix[t][e] << " ";
        }
        std::cout << "... (total " << time_intervals << " time steps)" << std::endl;
    }
    std::cout << "    ... (total " << ensemble_size << " ensembles)" << std::endl;
    
    const int nop_per_ensemble = 1000;
    const Source& source = sources[0];
    const float source_x = (source.lon + 179.0f) / 0.5f;
    const float source_y = (source.lat + 90.0f) / 0.5f;
    const float source_z = source.height;
    
    // Clear existing particles
    ldm.part.clear();
    ldm.part.reserve(enop);  // Use enop for ensemble mode
    
    // Create detailed initialization log file
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto tm = *std::localtime(&time_t);
    
    std::ostringstream timestamp;
    timestamp << std::put_time(&tm, "%Y%m%d_%H%M%S");
    
    std::string detail_log_filename = "/home/jrpark/LDM-EKI/logs/integration_logs/ensemble_particle_details_" 
                                    + timestamp.str() + ".csv";
    
    std::ofstream detail_log(detail_log_filename);
    if (detail_log.is_open()) {
        detail_log << "# Detailed EKI Ensemble Particle Initialization Log\n";
        detail_log << "# Generated at: " << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << "\n";
        detail_log << "# EKI Ensemble Matrix Dimensions: " << time_intervals << " x " << ensemble_size << "\n";
        detail_log << "# Total particles: " << (ensemble_size * nop_per_ensemble) << "\n";
        detail_log << "#\n";
        detail_log << "ensemble_id,particle_idx,global_id,timeidx,time_step_index,eki_emission_rate,final_concentration,source_x,source_y,source_z\n";
    }
    
    // Initialize particles for each ensemble
    for (int e = 0; e < ensemble_size; e++) {
        std::cout << "  [ENSEMBLE " << e << "] Initializing " << nop_per_ensemble << " particles..." << std::endl;
        
        for (int i = 0; i < nop_per_ensemble; i++) {
            int global_idx = e * nop_per_ensemble + i;
            
            // Calculate time step index for this particle (0-23)
            int time_step_index = (i * time_intervals) / nop_per_ensemble;
            if (time_step_index >= time_intervals) time_step_index = time_intervals - 1;
            
            // Get ensemble-specific emission rate
            float ensemble_emission_rate = ensemble_matrix[time_step_index][e];
            
            // Create particle
            LDM::LDMpart particle;
            particle.x = source_x;
            particle.y = source_y;
            particle.z = source_z;
            particle.timeidx = i;  // Use ensemble-local index for individual diffusion
            particle.flag = 0;  // Initially inactive
            particle.ensemble_id = e;
            particle.global_id = global_idx + 1;
            
            // Set ensemble-specific concentration
            particle.concentrations[0] = ensemble_emission_rate;
            particle.conc = ensemble_emission_rate;  // Legacy field
            
            // Add particle physics properties
            particle.decay_const = g_mpi.decayConstants[mpiRank];
            particle.drydep_vel = g_mpi.depositionVelocities[mpiRank];
            particle.radi = g_mpi.particleSizes[mpiRank];
            particle.prho = g_mpi.particleDensities[mpiRank];
            
            // Log detailed information for first 5 particles of first 3 ensembles
            if (e < 3 && i < 5) {
                std::cout << "    [DEBUG] E" << e << "_P" << i << ": global_id=" << particle.global_id 
                         << " timeidx=" << particle.timeidx << " time_step=" << time_step_index 
                         << " eki_rate=" << std::scientific << ensemble_emission_rate 
                         << " conc=" << particle.concentrations[0] << std::endl;
            }
            
            // Write to detailed log file
            if (detail_log.is_open()) {
                detail_log << e << "," << i << "," << particle.global_id << "," << particle.timeidx << ","
                          << time_step_index << "," << std::scientific << std::setprecision(6) 
                          << ensemble_emission_rate << "," << particle.concentrations[0] << ","
                          << particle.x << "," << particle.y << "," << particle.z << "\n";
            }
            
            ldm.part.push_back(particle);
        }
        
        // Log ensemble statistics
        if (e < 5 || e % 20 == 0) {
            float total_conc = 0.0f;
            int start_idx = e * nop_per_ensemble;
            int end_idx = (e + 1) * nop_per_ensemble;
            
            for (int p = start_idx; p < end_idx; p++) {
                total_conc += ldm.part[p].concentrations[0];
            }
            float mean_conc = total_conc / nop_per_ensemble;
            
            std::cout << "  [ENSEMBLE " << e << "] Mean concentration: " << std::scientific 
                     << mean_conc << " (range: " << ldm.part[start_idx].concentrations[0] 
                     << " to " << ldm.part[end_idx-1].concentrations[0] << ")" << std::endl;
        }
    }
    
    if (detail_log.is_open()) {
        detail_log.close();
        std::cout << "  [ENSEMBLE] Detailed initialization log saved: " << detail_log_filename << std::endl;
    }
    
    std::cout << "  [ENSEMBLE] Initialized " << ldm.part.size() << " particles across " 
              << ensemble_size << " ensembles" << std::endl;
    
    // Update device memory
    if (ldm.d_part != nullptr) {
        cudaFree(ldm.d_part);
    }
    
    const size_t total_size = enop * sizeof(LDM::LDMpart);
    cudaError_t err = cudaMalloc((void**)&ldm.d_part, total_size);
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] Failed to allocate ensemble device memory: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    err = cudaMemcpy(ldm.d_part, ldm.part.data(), total_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] Failed to copy ensemble particles to device: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    return true;
}

void saveEnsembleParticleSnapshot(int timestep, const std::vector<LDM::LDMpart>& particles, 
                                 int ensemble_size, int nop_per_ensemble) {
    
    // Create ensemble results directory
    std::string ensemble_dir = "/home/jrpark/LDM-EKI/logs/ldm_logs/ensemble_results";
    system(("mkdir -p " + ensemble_dir).c_str());
    
    // Save particles for each ensemble separately
    for (int e = 0; e < ensemble_size; e++) {
        std::string filename = ensemble_dir + "/ensemble_" + std::to_string(e) + 
                              "_particles_15min_" + std::to_string(timestep / 9 + 1) + ".csv";
        
        std::ofstream file(filename);
        if (!file.is_open()) continue;
        
        file << "particle_id,ensemble_id,x,y,z,concentration,flag,timeidx\n";
        
        // Write particles for this ensemble
        int start_idx = e * nop_per_ensemble;
        int end_idx = (e + 1) * nop_per_ensemble;
        
        for (int i = start_idx; i < end_idx && i < particles.size(); i++) {
            if (particles[i].flag == 1) {  // Only active particles
                file << particles[i].global_id << ","
                     << particles[i].ensemble_id << ","
                     << particles[i].x << "," << particles[i].y << "," << particles[i].z << ","
                     << particles[i].concentrations[0] << ","
                     << particles[i].flag << ","
                     << particles[i].timeidx << "\n";
            }
        }
    }
}

bool calculateEnsembleObservations(float ensemble_observations[100][24][3], 
                                  int ensemble_size, int time_intervals, 
                                  const std::vector<LDM::LDMpart>& particles) {
    
    std::cout << "  [ENSEMBLE] Calculating observations for " << ensemble_size 
              << " ensembles over " << time_intervals << " time intervals..." << std::endl;
    
    // Load EKI receptor configuration
    EKIConfig* ekiConfig = EKIConfig::getInstance();
    if (!ekiConfig) {
        std::cerr << "[ERROR] Failed to get EKI configuration for receptors" << std::endl;
        return false;
    }
    
    // Get receptor positions
    std::vector<Receptor> receptors;
    for (int r = 0; r < 3; r++) {
        Receptor receptor = ekiConfig->getReceptor(r);
        receptors.push_back(receptor);
    }
    
    // Initialize observation matrix
    for (int e = 0; e < ensemble_size; e++) {
        for (int t = 0; t < time_intervals; t++) {
            for (int r = 0; r < 3; r++) {
                ensemble_observations[e][t][r] = 0.0f;
            }
        }
    }
    
    // Use particle data directly from memory (much more efficient than CSV reading)
    std::cout << "  [ENSEMBLE] Using " << particles.size() << " particles from simulation memory" << std::endl;
    
    // Count active particles per ensemble and check ensemble_id distribution
    std::vector<int> active_counts(ensemble_size, 0);
    std::vector<int> total_counts(ensemble_size, 0);
    int total_active = 0;
    
    for (const auto& particle : particles) {
        if (particle.ensemble_id < ensemble_size) {
            total_counts[particle.ensemble_id]++;
            if (particle.flag == 1) {
                active_counts[particle.ensemble_id]++;
                total_active++;
            }
        }
    }
    
    std::cout << "  [ENSEMBLE] Particle distribution by ensemble:" << std::endl;
    for (int e = 0; e < 5; e++) {
        std::cout << "    Ensemble " << e << ": " << total_counts[e] << " total, " 
                 << active_counts[e] << " active" << std::endl;
    }
    
    std::cout << "  [ENSEMBLE] Total active particles: " << total_active << " out of " << particles.size() << std::endl;
    
    // Calculate receptor concentrations for each ensemble
    // Note: At simulation end, all particles should contribute regardless of flag
    for (const auto& particle : particles) {
        // Force all particles to be considered active for final observation calculation
        // if (particle.flag != 1) continue; // Skip inactive particles
        
        int e = particle.ensemble_id;
        if (e >= ensemble_size) continue; // Skip invalid ensemble IDs
        
        // Convert particle coordinates from grid to degrees
        // particle.x and particle.y are in grid coordinates, need to convert to lon/lat
        float particle_lon = (particle.x * 0.5f) - 179.0f; // Convert grid x to longitude
        float particle_lat = (particle.y * 0.5f) - 90.0f;  // Convert grid y to latitude
        
        // Calculate contribution to each receptor using simple rectangular grid
        for (int r = 0; r < 3; r++) {
            // Simple rectangular grid check (10 degrees as specified)
            float dlat = abs(particle_lat - receptors[r].lat);
            float dlon = abs(particle_lon - receptors[r].lon);
            
            // Debug: Log some calculations for verification
            static int debug_count = 0;
            if (e == 0 && r == 0 && debug_count < 5) {
                debug_count++;
                std::cout << "  [DEBUG] Particle " << particle.global_id 
                         << ": ensemble=" << e << " timeidx=" << particle.timeidx
                         << " pos(" << particle_lon << "," << particle_lat 
                         << ") receptor(" << receptors[r].lon << "," << receptors[r].lat
                         << ") dlat=" << dlat << " dlon=" << dlon << " concentration=" << particle.concentrations[0] << std::endl;
            }
            
            // Check if particle is within 10-degree rectangular grid
            if (dlat <= 5.0f && dlon <= 5.0f) { // 10-degree area (±5 degrees from receptor)
                float contribution = particle.concentrations[0];
                
                // Calculate which time step this particle corresponds to based on timeidx
                // Each ensemble has 1000 particles with timeidx 0-999
                // Map timeidx to time intervals (0-23)
                int particle_time_step = (particle.timeidx * time_intervals) / 1000;
                if (particle_time_step >= time_intervals) particle_time_step = time_intervals - 1;
                
                // Only apply contribution from this time step onwards (particle activated)
                for (int t = particle_time_step; t < time_intervals; t++) {
                    ensemble_observations[e][t][r] += contribution;
                }
            }
        }
    }
    
    // Log some statistics
    float total_obs = 0.0f;
    int non_zero_count = 0;
    
    for (int e = 0; e < ensemble_size; e++) {
        for (int t = 0; t < time_intervals; t++) {
            for (int r = 0; r < 3; r++) {
                total_obs += ensemble_observations[e][t][r];
                if (ensemble_observations[e][t][r] > 0.0f) non_zero_count++;
            }
        }
    }
    
    std::cout << "  [ENSEMBLE] Generated " << non_zero_count << " non-zero observations out of " 
              << (ensemble_size * time_intervals * 3) << " total" << std::endl;
    std::cout << "  [ENSEMBLE] Average observation value: " << (total_obs / (ensemble_size * time_intervals * 3)) << std::endl;
    
    return true;
}

void saveEnsembleObservationsToEKI(const float ensemble_observations[100][24][3], 
                                  int ensemble_size, int time_intervals, int iteration) {
    
    // Create EKI directory if it doesn't exist
    system("mkdir -p /home/jrpark/LDM-EKI/logs/eki_logs");
    
    // Save binary file for EKI
    std::string binary_filename = "/home/jrpark/LDM-EKI/logs/eki_logs/ensemble_observations_iter_" + 
                                 std::to_string(iteration) + ".bin";
    
    std::ofstream binary_file(binary_filename, std::ios::binary);
    if (binary_file.is_open()) {
        binary_file.write(reinterpret_cast<const char*>(ensemble_observations), 
                         ensemble_size * time_intervals * 3 * sizeof(float));
        binary_file.close();
        std::cout << "  [ENSEMBLE] Saved binary observations: " << binary_filename << std::endl;
    }
    
    // Save CSV file for analysis
    std::string csv_filename = "/home/jrpark/LDM-EKI/logs/eki_logs/ensemble_observations_iter_" + 
                              std::to_string(iteration) + ".csv";
    
    std::ofstream csv_file(csv_filename);
    if (csv_file.is_open()) {
        csv_file << "# Ensemble Observations Matrix (Iteration " << iteration << ")\n";
        csv_file << "# Generated: " << std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count() << "\n";
        csv_file << "# Dimensions: " << ensemble_size << " ensembles × " << time_intervals 
                 << " time steps × 3 receptors\n";
        csv_file << "# Format: ensemble_id,time_step,receptor_id,concentration\n";
        csv_file << "ensemble_id,time_step,receptor_id,concentration\n";
        
        for (int e = 0; e < ensemble_size; e++) {
            for (int t = 0; t < time_intervals; t++) {
                for (int r = 0; r < 3; r++) {
                    csv_file << e << "," << t << "," << r << "," 
                            << std::scientific << std::setprecision(6) 
                            << ensemble_observations[e][t][r] << "\n";
                }
            }
        }
        csv_file.close();
        std::cout << "  [ENSEMBLE] Saved CSV observations: " << csv_filename << std::endl;
    }
}