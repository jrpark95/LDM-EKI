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

// Sequential workflow functions
bool runSingleModeLDM(LDM& ldm);
bool runPostProcessing();
bool runEKIEstimation();
bool loadEKIEnsembleResults(std::vector<std::vector<float>>& ensemble_matrix, int& time_intervals, int& ensemble_size);
bool runEnsembleLDM(LDM& ldm, const std::vector<std::vector<float>>& ensemble_matrix, int time_intervals, int ensemble_size);

int main(int argc, char** argv) {

    mpiRank = 1;
    mpiSize = 1;

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
    
    if (!runEKIEstimation()) {
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
    Nens = 1;
    
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

bool runEKIEstimation() {
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
    const int nop_per_ensemble = nop / Nens;  // Divide particles among ensembles
    
    ensemble_mode_active = true;
    ::Nens = Nens;
    ::nop_per_ensemble = nop_per_ensemble;
    
    std::cout << "  [5.2] Ensemble configuration: " << Nens << " ensembles, " 
              << nop_per_ensemble << " particles each, total " << (Nens * nop_per_ensemble) << std::endl;
    
    // Convert ensemble matrix to emission time series
    std::cout << "  [5.3] Converting ensemble data to emission time series..." << std::endl;
    
    // For now, use the first ensemble's data as the emission time series
    // In a full implementation, this would be more sophisticated
    std::vector<float> emission_time_series;
    for (int t = 0; t < time_intervals; t++) {
        emission_time_series.push_back(ensemble_matrix[t][0]);  // Use first ensemble as baseline
    }
    
    // Create source from EKI configuration
    std::vector<Source> sources;
    EKIConfig* ekiConfig = EKIConfig::getInstance();
    Source eki_source;
    eki_source.lat = ekiConfig->getSourceLat();
    eki_source.lon = ekiConfig->getSourceLon();
    eki_source.height = ekiConfig->getSourceAlt();
    sources.push_back(eki_source);
    
    std::cout << "  [5.4] Initializing ensemble particles..." << std::endl;
    
    // Initialize particles using ensemble method
    bool ensemble_init_success = ldm.initializeParticlesEnsembles(
        Nens, emission_time_series, sources, nop_per_ensemble);
        
    if (!ensemble_init_success) {
        std::cerr << "  [5.4] Ensemble particle initialization failed" << std::endl;
        return false;
    }
    
    // Save ensemble initialization log
    saveEnsembleInitializationLog(Nens, emission_time_series, sources, nop_per_ensemble);
    
    std::cout << "  [5.5] Running ensemble simulation..." << std::endl;
    
    // Run ensemble simulation
    ldm.startTimer();
    ldm.runSimulation();
    ldm.stopTimer();
    
    std::cout << "  [5.6] Ensemble simulation completed" << std::endl;
    
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