#include "include/ldm.cuh"
#include "include/ldm_eki.cuh"
#include "include/ldm_eki_logger.cuh"

int main(int argc, char *argv[]) {
    std::cout << "[INFO] Starting LDM single mode execution..." << std::endl;
    
    // Load EKI configuration
    EKIConfig* ekiConfig = EKIConfig::getInstance();
    std::string eki_config_file = "../eki/config/input_data";
    
    if (!ekiConfig->loadFromFile(eki_config_file)) {
        std::cerr << "[ERROR] Failed to load EKI configuration" << std::endl;
        return 1;
    }
    
    // Clean log directories
    cleanLogDirectory();
    
    // Write EKI configuration log
    writeEKIConfigLog(ekiConfig);
    
    // Print configuration
    printEKIConfiguration(ekiConfig);
    
    // Initialize LDM
    LDM ldm;
    ldm.loadSimulationConfiguration();
    ldm.calculateAverageSettlingVelocity();
    
    std::cout << "[INFO] Single mode - initializing particles..." << std::endl;
    ldm.initializeParticles();
    
    ldm.loadFlexHeightData();
    ldm.initializeFlexGFSData();
    
    // Run simulation
    std::cout << "[INFO] Starting simulation..." << std::endl;
    ldm.runSimulation();
    
    // Prepare observation data for EKI
    prepareObservationData(ekiConfig);
    
    // Run visualization
    runVisualization();
    
    // Run EKI estimation
    runEKIEstimation();
    
    std::cout << "[INFO] LDM single mode execution completed successfully." << std::endl;
    return 0;
}