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
    
    // Check if ensemble mode should be enabled (default: 100 ensembles for EKI)
    const int target_ensembles = 100;  // Standard EKI ensemble count
    bool use_ensemble_mode = target_ensembles > 1;
    
    if (use_ensemble_mode) {
        std::cout << "[INFO] Ensemble mode enabled with " << target_ensembles << " ensembles" << std::endl;
        
        // Set global ensemble variables
        ensemble_mode_active = true;
        Nens = target_ensembles;
        nop_per_ensemble = nop / Nens;  // Divide particles among ensembles
        
        std::cout << "[INFO] Each ensemble will have " << nop_per_ensemble << " particles" << std::endl;
        
        // Get emission time series from EKI
        const SourceEmission& source_emission = ekiConfig->getSourceEmission();
        std::vector<float> emission_time_series = source_emission.time_series;
        
        // Create source from EKI configuration
        std::vector<Source> sources;
        Source eki_source;
        eki_source.lat = ekiConfig->getSourceLat();
        eki_source.lon = ekiConfig->getSourceLon();
        eki_source.height = ekiConfig->getSourceAlt();
        sources.push_back(eki_source);
        
        // Initialize particles using ensemble method
        bool ensemble_init_success = ldm.initializeParticlesEnsembles(
            Nens, emission_time_series, sources, nop_per_ensemble);
            
        if (!ensemble_init_success) {
            std::cerr << "[ERROR] Ensemble particle initialization failed" << std::endl;
            return 1;
        }
        
        std::cout << "[INFO] Ensemble particle initialization completed successfully" << std::endl;
    } else {
        std::cout << "[INFO] Single ensemble mode" << std::endl;
        ldm.initializeParticles();
    }
    
    ldm.loadFlexHeightData();    // Load height data FIRST
    ldm.initializeFlexGFSData(); // Then calculate DRHO using height data
    // Clean log directory before starting simulation
    std::cout << "[INFO] Cleaning log directory..." << std::endl;
    int clean_result = system("rm -f /home/jrpark/LDM-EKI/logs/ldm_logs/*.csv /home/jrpark/LDM-EKI/logs/ldm_logs/*.txt /home/jrpark/LDM-EKI/logs/ldm_logs/*.png");
    if (clean_result == 0) {
        std::cout << "[INFO] Log directory cleaned successfully" << std::endl;
    } else {
        std::cout << "[WARNING] Failed to clean log directory" << std::endl;
    }

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
    
    // Run OpenStreetMap visualization
    std::cout << "[INFO] Running OpenStreetMap visualization..." << std::endl;
    int result5 = system("cd /home/jrpark/LDM-EKI/ldm && python3 osm_grid_visualization.py");
    if (result5 == 0) {
        std::cout << "[INFO] OpenStreetMap visualization completed successfully." << std::endl;
    } else {
        std::cout << "[WARNING] OpenStreetMap visualization failed with exit code: " << result5 << std::endl;
    }
    
    // Run 15-minute receptor time series analysis
    std::cout << "[INFO] Running 15-minute receptor time series analysis..." << std::endl;
    int result6 = system("cd /home/jrpark/LDM-EKI/ldm && python3 receptor_time_series.py");
    if (result6 == 0) {
        std::cout << "[INFO] 15-minute receptor time series analysis completed successfully." << std::endl;
    } else {
        std::cout << "[WARNING] 15-minute receptor time series analysis failed with exit code: " << result6 << std::endl;
    }
    
    std::cout << "[INFO] All visualization scripts completed. Check logs/ldm_logs/ for results." << std::endl;
    
    // Clean up intermediate particle files (keep only the 24th one)
    std::cout << "\n[INFO] Cleaning up intermediate particle files..." << std::endl;
    int cleanup_result = system("cd /home/jrpark/LDM-EKI/logs/ldm_logs && find . -name 'particles_15min_*.csv' ! -name 'particles_15min_24.csv' -delete 2>/dev/null || true");
    if (cleanup_result == 0) {
        std::cout << "[INFO] Intermediate particle files cleaned (kept particles_15min_24.csv)" << std::endl;
    } else {
        std::cout << "[WARNING] Failed to clean intermediate files" << std::endl;
    }
    
    return 0;
}