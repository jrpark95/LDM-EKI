#include "ldm.cuh"
#include "ldm_nuclides.cuh"
#include "ldm_eki.cuh"
#include "ldm_eki_logger.cuh"
#include <cstdio>  // for remove() function
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
    
    printEKIConfiguration(ekiConfig);
    
    // Write EKI configuration to log file
    writeEKIConfigLog(ekiConfig);
    
    LDM ldm;

    ldm.loadSimulationConfiguration();

    // CRAM system disabled for single nuclide (Co-60) simulation
    printSystemInfo();

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
    
    cleanLogDirectory();

    ldm.allocateGPUMemory();

    ldm.startTimer();
    ldm.runSimulation();
    ldm.stopTimer();

    // MPI_Finalize();
    
    // EKI Integration Workflow
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
    
    if(0){
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
        
    }

    // Clean up intermediate particle files (keep 0 and 24, remove 1-23)
    std::cout << "\n[INFO] Cleaning up intermediate particle files..." << std::endl;
    int files_removed = 0;
    for (int i = 1; i <= 23; i++) {
        std::string filename = "/home/jrpark/LDM-EKI/logs/ldm_logs/particles_15min_" + std::to_string(i) + ".csv";
        if (remove(filename.c_str()) == 0) {
            files_removed++;
        }
    }
    std::cout << "[INFO] Intermediate particle files cleaned (" << files_removed << " files removed from particles_15min_1.csv to particles_15min_23.csv, kept particles_15min_0.csv and particles_15min_24.csv)" << std::endl;

    

    return 0;
}