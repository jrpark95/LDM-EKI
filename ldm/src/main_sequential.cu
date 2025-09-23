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
// saveEnsembleInitializationLog is handled in ldm_ensemble_init.cu

// Sequential workflow functions
bool runSingleModeLDM(LDM& ldm);
bool runPostProcessing();
bool runEKIEstimationSequential();
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
    
    if (!runEKIEstimationSequential()) {
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

bool runEKIEstimationSequential() {
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
    
    // Use ensemble matrix directly (no conversion needed)
    std::cout << "  [5.3] Using ensemble matrix with ensemble-specific emission rates..." << std::endl;
    
    // Use source from LDM's loaded source.txt configuration
    // LDM object already has sources loaded from source.txt in loadSimulationConfiguration()
    if (ldm.getSources().empty()) {
        std::cerr << "[ERROR] No sources loaded from source.txt" << std::endl;
        return 1;
    }
    std::vector<Source> sources = ldm.getSources();  // Use sources from source.txt
    
    std::cout << "  [5.4] Initializing ensemble particles..." << std::endl;
    
    // Initialize particles using ensemble method with full matrix
    bool ensemble_init_success = ldm.initializeParticlesEnsembles(
        Nens, ensemble_matrix, sources, nop_per_ensemble);
        
    if (!ensemble_init_success) {
        std::cerr << "  [5.4] Ensemble particle initialization failed" << std::endl;
        return false;
    }
    
    // Ensemble initialization log is saved automatically in initializeParticlesEnsembles
    
    std::cout << "  [5.5] Running ensemble simulation..." << std::endl;
    
    // Run ensemble simulation
    ldm.startTimer();
    ldm.runSimulation();
    ldm.stopTimer();
    
    std::cout << "  [5.6] Ensemble simulation completed" << std::endl;
    
    return true;
}

// Function to save ensemble initialization log
// Function removed - ensemble initialization logging is handled in ldm_ensemble_init.cu