#include "ldm.cuh"
#include "ldm_nuclides.cuh"
#include "ldm_eki.cuh"
#include "ldm_eki_logger.cuh"
#include <thread>
#include <chrono>
#include <vector>
#include <fstream>
//#include "ldm_cram.cuh"
//#include "cram_runtime.h"

// Physics model global variables (will be loaded from setting.txt)
int g_num_nuclides = 1;   // Single nuclide: Co-60
int g_turb_switch = 0;    // Default values, overwritten by setting.txt
int g_drydep = 0;
int g_wetdep = 0; 
int g_raddecay = 0;       // Radioactive decay disabled for simple simulation


int main(int argc, char** argv) {

    mpiRank = 1;
    mpiSize = 1;


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
    
    // Load EKI input_config file
    std::string eki_input_config_file = "../eki/config/input_config";
    if (!ekiConfig->loadInputConfigFromFile(eki_input_config_file)) {
        std::cerr << "[ERROR] Failed to load EKI input_config" << std::endl;
        return 1;
    }
    
    printEKIConfiguration(ekiConfig);
    
    // Write EKI configuration to log file
    writeEKIConfigLog(ekiConfig);
    
    LDM ldm;

    ldm.loadSimulationConfiguration();

    // CRAM system disabled for single nuclide (Co-60) simulation
    // std::cout << "[DEBUG] Initializing CRAM system..." << std::endl;
    // if (!ldm.initialize_cram_system("./cram/A60.csv")) {
    //     std::cerr << "[ERROR] CRAM system initialization failed" << std::endl;
    //     return 1;
    // }
    // std::cout << "[DEBUG] CRAM system initialization completed" << std::endl;
    printSystemInfo();

    ldm.calculateAverageSettlingVelocity();
    ldm.initializeParticles();
    
    ldm.loadFlexHeightData();    // Load height data FIRST
    ldm.initializeFlexGFSData(); // Then calculate DRHO using height data
    
    cleanLogDirectory();

    ldm.allocateGPUMemory();

    ldm.startTimer();
    ldm.runSimulation();
    ldm.stopTimer();

    // MPI_Finalize();
    
    prepareObservationData(ekiConfig);
    runVisualization();
    printSimulationStatus("Preparing data for EKI...");
    runEKIEstimation();
    
    // Test ensemble loading functionality
    printSimulationStatus("Testing ensemble loading from EKI...");
    std::vector<std::vector<float>> ensemble_matrix;
    int time_intervals, ensemble_size;
    
    // Try to load ensemble from first iteration
    if (loadEKIEnsembleStates(ensemble_matrix, time_intervals, ensemble_size, 1)) {
        std::cout << "[INFO] Successfully loaded and logged ensemble matrix [" 
                  << time_intervals << " x " << ensemble_size << "]" << std::endl;
        
        // Display some basic info about the loaded ensemble
        float sample_value = ensemble_matrix[0][0];
        std::cout << "[INFO] Sample ensemble value [time=0, ensemble=0]: " << sample_value << std::endl;
        
        // Test memory allocation verification
        std::cout << "[INFO] Memory allocation test:" << std::endl;
        std::cout << "  - Allocated " << ensemble_matrix.size() << " time rows" << std::endl;
        std::cout << "  - Each row has " << ensemble_matrix[0].size() << " ensemble columns" << std::endl;
        std::cout << "  - Total memory: " << (time_intervals * ensemble_size * sizeof(float)) << " bytes" << std::endl;
    } else {
        std::cout << "[INFO] No ensemble file found - this is expected on first run" << std::endl;
    }
    
    
    return 0;
}