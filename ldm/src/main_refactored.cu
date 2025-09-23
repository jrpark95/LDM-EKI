#include "ldm.cuh"
#include "ldm_nuclides.cuh"
#include "ldm_eki.cuh"
#include "ldm_eki_logger.cuh"
#include "ldm_integration.cuh"
#include <sys/stat.h>
#include <errno.h>
#include <cstring>

// Physics model global variables
int g_num_nuclides = 1;
int g_turb_switch = 0;
int g_drydep = 0;
int g_wetdep = 0;
int g_raddecay = 0;

// Utility functions
void ensure_dir(const std::string& path) {
    struct stat st = {0};
    if (stat(path.c_str(), &st) == -1) {
        if (mkdir(path.c_str(), 0755) != 0) {
            std::cerr << "[WARNING] Failed to create directory " << path 
                      << ": " << strerror(errno) << std::endl;
        } else {
            std::cout << "[INFO] Created directory: " << path << std::endl;
        }
    }
}

int main(int argc, char** argv) {
    auto program_start = std::chrono::high_resolution_clock::now();
    
    // Setup MPI
    mpiRank = 1;
    mpiSize = 1;
    
    // Create all necessary directories first
    ensure_dir("/home/jrpark/LDM-EKI/logs");
    ensure_dir("/home/jrpark/LDM-EKI/logs/ldm_logs");
    ensure_dir("/home/jrpark/LDM-EKI/logs/eki_logs");
    ensure_dir("/home/jrpark/LDM-EKI/logs/integration_logs");

    // Load nuclide configuration
    NuclideConfig* nucConfig = NuclideConfig::getInstance();
    std::string nuclide_config_file = "./data/input/nuclides_config_1.txt";
    
    if (!nucConfig->loadFromFile(nuclide_config_file)) {
        std::cerr << "[ERROR] Failed to load nuclide configuration" << std::endl;
        return 1;
    }
    
    g_num_nuclides = nucConfig->getNumNuclides();
    
    // Load EKI configuration
    EKIConfig* ekiConfig = EKIConfig::getInstance();
    std::string eki_config_file = "../eki/config/input_data";
    
    if (!ekiConfig->loadFromFile(eki_config_file)) {
        std::cerr << "[ERROR] Failed to load EKI configuration" << std::endl;
        return 1;
    }
    
    writeEKIConfigLog(ekiConfig);
    
    LDM ldm;
    ldm.loadSimulationConfiguration();
    ldm.calculateAverageSettlingVelocity();

    // =============================================
    // Phase 1: LDM Single Mode
    // =============================================
    std::cout << "\n=== Phase 1: LDM Single Mode ===" << std::endl;
    
    ldm.initializeParticles();
    ldm.loadFlexHeightData();
    ldm.initializeFlexGFSData();
    ldm.allocateGPUMemory();
    ldm.runSimulation();
    ldm.writeObservationsSingle("/home/jrpark/LDM-EKI/logs/ldm_logs", "iter000");
    
    // Keep visualization scripts but disable file deletion
    std::cout << "[INFO] Running visualization scripts (keeping all files)..." << std::endl;
    system("cd /home/jrpark/LDM-EKI/ldm && python3 simple_visualize.py");
    system("cd /home/jrpark/LDM-EKI/ldm && python3 enhanced_visualize.py");
    system("cd /home/jrpark/LDM-EKI/ldm && python3 receptor_analysis.py");
    system("cd /home/jrpark/LDM-EKI/ldm && python3 particle_concentration_analysis.py");
    system("cd /home/jrpark/LDM-EKI/ldm && python3 osm_grid_visualization.py");
    system("cd /home/jrpark/LDM-EKI/ldm && python3 receptor_time_series.py");

    // =============================================
    // Phase 2: EKI Once
    // =============================================
    std::cout << "\n=== Phase 2: EKI Once ===" << std::endl;
    
    bool use_system_call = false;
    if (use_system_call) {
        system("cd ../eki && python3 run_eki_once.py --in /home/jrpark/LDM-EKI/logs/ldm_logs --out /home/jrpark/LDM-EKI/logs/eki_logs --iter 000");
    }
    
    if (!wait_for_files("/home/jrpark/LDM-EKI/logs/eki_logs", {"states_iter000.bin", "emission_iter000.bin"}, 600)) {
        std::cerr << "[ERROR] Required EKI files not found within timeout" << std::endl;
        return 2;
    }

    // =============================================
    // Phase 3: LDM Ensemble Mode
    // =============================================
    std::cout << "\n=== Phase 3: LDM Ensemble Mode ===" << std::endl;
    
    ldm.freeGPUMemory();
    
    EmissionData emis;
    if (!load_emission_series("/home/jrpark/LDM-EKI/logs/eki_logs/emission_iter000.bin", emis)) {
        std::cerr << "[ERROR] Failed to load emission data" << std::endl;
        return 1;
    }
    
    EnsembleState ens;
    if (!load_ensemble_state("/home/jrpark/LDM-EKI/logs/eki_logs/states_iter000.bin", ens)) {
        std::cerr << "[ERROR] Failed to load ensemble state" << std::endl;
        return 1;
    }
    
    // Consistency checks
    if (emis.Nens != ens.Nens) {
        std::cerr << "[ERROR] Ensemble size mismatch: emission Nens=" << emis.Nens 
                  << ", state Nens=" << ens.Nens << std::endl;
        return 2;
    }
    
    if (emis.T <= 0) {
        std::cerr << "[ERROR] Invalid time dimension: T=" << emis.T << std::endl;
        return 2;
    }
    
    if (nop % emis.Nens != 0) {
        std::cerr << "[ERROR] Particle count not divisible by ensemble size: nop=" << nop 
                  << ", Nens=" << emis.Nens << ", remainder=" << (nop % emis.Nens) << std::endl;
        return 2;
    }
    
    const int Nens = emis.Nens;
    int nop_per_ensemble = nop / Nens;
    
    std::cout << "[INFO] Consistency checks passed: Nens=" << Nens 
              << ", nop_per_ensemble=" << nop_per_ensemble << ", T=" << emis.T << std::endl;
    
    // Use source from LDM's loaded source.txt configuration
    if (ldm.getSources().empty()) {
        std::cerr << "[ERROR] No sources loaded from source.txt" << std::endl;
        return 1;
    }
    std::vector<Source> sources = ldm.getSources();  // Use sources from source.txt
    
    if (!ldm.initializeParticlesEnsemblesFlat(Nens, emis.flat, sources, nop_per_ensemble)) {
        std::cerr << "[ERROR] Ensemble initialization failed" << std::endl;
        return 1;
    }
    
    ldm.allocateGPUMemory();
    
    if (!ldm.runSimulationEnsembles(Nens)) {
        std::cerr << "[ERROR] Ensemble simulation failed" << std::endl;
        return 1;
    }
    
    if (!ldm.writeObservationsEnsembles("/home/jrpark/LDM-EKI/logs/eki_logs", "iter000")) {
        std::cerr << "[ERROR] Failed to write ensemble observations" << std::endl;
        return 1;
    }
    
    if (!ldm.writeIntegrationDebugLogs("/home/jrpark/LDM-EKI/logs/integration_logs", "iter000")) {
        std::cerr << "[ERROR] Failed to write integration debug logs" << std::endl;
        return 1;
    }
    
    std::cout << "\n=== All Phases Completed Successfully ===" << std::endl;
    return 0;
}