#include "ldm.cuh"
#include "ldm_nuclides.cuh"
#include "ldm_eki.cuh"
#include "ldm_eki_logger.cuh"
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
    
    std::cout << "[DEBUG] EKI configuration loaded successfully:" << std::endl;
    std::cout << "[DEBUG]   Receptors: " << ekiConfig->getNumReceptors() << std::endl;
    std::cout << "[DEBUG]   Source emission time steps: " << ekiConfig->getSourceEmission().num_time_steps << std::endl;
    std::cout << "[DEBUG]   Prior source time steps: " << ekiConfig->getPriorSource().num_time_steps << std::endl;
    
    // Display receptor positions
    for (int i = 0; i < ekiConfig->getNumReceptors(); i++) {
        Receptor r = ekiConfig->getReceptor(i);
        std::cout << "[DEBUG]   Receptor " << i << ": lat=" << r.lat << ", lon=" << r.lon << ", alt=" << r.alt << std::endl;
    }
    
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
    std::cout << "[DEBUG] Single nuclide mode: CRAM system disabled" << std::endl;

    ldm.calculateAverageSettlingVelocity();
    ldm.initializeParticles();
    
    ldm.loadFlexHeightData();    // Load height data FIRST
    ldm.initializeFlexGFSData(); // Then calculate DRHO using height data
    ldm.allocateGPUMemory();

    ldm.startTimer();
    ldm.runSimulation();
    ldm.stopTimer();

    // MPI_Finalize();
    
    // Automatically run visualization scripts after simulation
    std::cout << "\n[INFO] Simulation completed. Running visualization scripts..." << std::endl;
    
    // Run simple visualization (particle analysis plots)
    std::cout << "[INFO] Running simple particle visualization..." << std::endl;
    int result1 = system("cd /home/jrpark/LDM-EKI/ldm && python3 simple_visualize.py");
    if (result1 == 0) {
        std::cout << "[INFO] Simple visualization completed successfully." << std::endl;
    } else {
        std::cout << "[WARNING] Simple visualization failed with exit code: " << result1 << std::endl;
    }
    
    // Run enhanced visualization
    std::cout << "[INFO] Running enhanced particle visualization..." << std::endl;
    int result2 = system("cd /home/jrpark/LDM-EKI/ldm && python3 enhanced_visualize.py");
    if (result2 == 0) {
        std::cout << "[INFO] Enhanced visualization completed successfully." << std::endl;
    } else {
        std::cout << "[WARNING] Enhanced visualization failed with exit code: " << result2 << std::endl;
    }
    
    // Run receptor analysis
    std::cout << "[INFO] Running receptor analysis..." << std::endl;
    int result3 = system("cd /home/jrpark/LDM-EKI/ldm && python3 receptor_analysis.py");
    if (result3 == 0) {
        std::cout << "[INFO] Receptor analysis completed successfully." << std::endl;
    } else {
        std::cout << "[WARNING] Receptor analysis failed with exit code: " << result3 << std::endl;
    }
    
    // Run particle concentration analysis
    std::cout << "[INFO] Running particle concentration analysis..." << std::endl;
    int result4 = system("cd /home/jrpark/LDM-EKI/ldm && python3 particle_concentration_analysis.py");
    if (result4 == 0) {
        std::cout << "[INFO] Particle concentration analysis completed successfully." << std::endl;
    } else {
        std::cout << "[WARNING] Particle concentration analysis failed with exit code: " << result4 << std::endl;
    }
    
    std::cout << "[INFO] All visualization scripts completed. Check logs/ldm_logs/ for results." << std::endl;
    
    return 0;
}