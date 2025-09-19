#include "ldm_eki_logger.cuh"

void writeEKIConfigLog(EKIConfig* ekiConfig) {
    std::cout << "[DEBUG] Writing EKI configuration to log file..." << std::endl;
    
    std::ofstream logFile("../logs/eki_logs/eki_config.log");
    if (!logFile.is_open()) {
        std::cerr << "[ERROR] Failed to create EKI configuration log file" << std::endl;
        return;
    }
    
    // Get current time
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    
    logFile << "========================================" << std::endl;
    logFile << "EKI Configuration Log" << std::endl;
    logFile << "Generated: " << std::ctime(&time_t_now);
    logFile << "========================================" << std::endl << std::endl;
    
    // Receptor positions
    logFile << "nreceptor : " << ekiConfig->getNumReceptors() << std::endl;
    logFile << "receptor_position: [" << std::endl;
    logFile << "  # " << ekiConfig->getNumReceptors() << " receptors with same latitude as source (40.7490), different longitudes" << std::endl;
    for (int i = 0; i < ekiConfig->getNumReceptors(); i++) {
        Receptor r = ekiConfig->getReceptor(i);
        logFile << "  [" << std::fixed << std::setprecision(4) << r.lat << ", " 
                << std::fixed << std::setprecision(2) << r.lon << ", " 
                << std::fixed << std::setprecision(1) << r.alt << "]";
        if (i < ekiConfig->getNumReceptors() - 1) {
            logFile << ",   # Receptor " << i << " - lon: " << std::fixed << std::setprecision(2) << r.lon;
        } else {
            logFile << "    # Receptor " << i << " - lon: " << std::fixed << std::setprecision(2) << r.lon;
        }
        logFile << std::endl;
    }
    logFile << "]" << std::endl << std::endl;
    
    // Source_1 emission data
    const SourceEmission& emission = ekiConfig->getSourceEmission();
    const PriorSource& prior = ekiConfig->getPriorSource();
    
    logFile << "Source_1: [" << std::scientific << std::setprecision(2) << prior.decay_constant << ", " 
            << std::scientific << std::setprecision(2) << prior.uncertainty << ", ["
            << std::fixed << std::setprecision(4) << prior.position[0] << ", "
            << std::fixed << std::setprecision(4) << prior.position[1] << ", "
            << std::fixed << std::setprecision(1) << prior.position[2] << "], [" << std::endl;
    
    if (emission.num_time_steps > 0 && emission.time_series.size() >= emission.num_time_steps) {
        for (int i = 0; i < emission.num_time_steps; i++) {
            int hours = (i * 15) / 60;
            int minutes = (i * 15) % 60;
            logFile << "  " << std::scientific << std::setprecision(1) << emission.time_series[i];
            if (i < emission.num_time_steps - 1) {
                logFile << ",   # " << std::setw(1) << hours << ":" 
                        << std::setw(2) << std::setfill('0') << minutes << std::setfill(' ') 
                        << "-" << std::setw(1) << ((i+1) * 15) / 60 << ":" 
                        << std::setw(2) << std::setfill('0') << ((i+1) * 15) % 60 << std::setfill(' ');
                if (i == 0) logFile << " - Start: " << (int)(emission.time_series[i] / 1e6) << "M";
                else if (i == emission.num_time_steps - 1) logFile << " - End: " << (int)(emission.time_series[i] / 1e6) << "M";
                else logFile << " - Linear increase: " << (int)(emission.time_series[i] / 1e6) << "M";
            } else {
                logFile << "    # " << std::setw(1) << hours << ":" 
                        << std::setw(2) << std::setfill('0') << minutes << std::setfill(' ') 
                        << "-6:00 - End: " << (int)(emission.time_series[i] / 1e6) << "M";
            }
            logFile << std::endl;
        }
    }
    logFile << "  ], 0.0e-0, 0.0e-0, 'Co-60'] # LINEAR INCREASE: " 
            << (int)(emission.time_series[0] / 1e6) << "M to " 
            << (int)(emission.time_series[emission.num_time_steps-1] / 1e6) << "M over 6 hours" << std::endl << std::endl;
    
    // Prior_Source_1 data
    logFile << "Prior_Source_1: [" << std::scientific << std::setprecision(2) << prior.decay_constant << ", " 
            << std::scientific << std::setprecision(2) << prior.uncertainty << ", [["
            << std::fixed << std::setprecision(4) << prior.position[0] << ", "
            << std::fixed << std::setprecision(4) << prior.position[1] << ", "
            << std::fixed << std::setprecision(1) << prior.position[2] << "],["
            << std::fixed << std::setprecision(1) << prior.position_std << "]], [[" << std::endl;
    
    if (prior.num_time_steps > 0 && prior.prior_values.size() >= prior.num_time_steps) {
        for (int i = 0; i < prior.num_time_steps; i++) {
            int hours = (i * 15) / 60;
            int minutes = (i * 15) % 60;
            logFile << "  " << std::scientific << std::setprecision(1) << prior.prior_values[i];
            if (i < prior.num_time_steps - 1) {
                logFile << ",   # " << std::setw(1) << hours << ":" 
                        << std::setw(2) << std::setfill('0') << minutes << std::setfill(' ') 
                        << "-" << std::setw(1) << ((i+1) * 15) / 60 << ":" 
                        << std::setw(2) << std::setfill('0') << ((i+1) * 15) % 60 << std::setfill(' ') 
                        << " - Prior: " << (int)(prior.prior_values[i] / 1e6) << "M";
            } else {
                logFile << "    # " << std::setw(1) << hours << ":" 
                        << std::setw(2) << std::setfill('0') << minutes << std::setfill(' ') 
                        << "-6:00 - Prior: " << (int)(prior.prior_values[i] / 1e6) << "M";
            }
            logFile << std::endl;
        }
    }
    logFile << "  ],[" << std::fixed << std::setprecision(1) << prior.value_std << "]], '" 
            << prior.nuclide_name << "'] # Constant prior: " 
            << (int)(prior.prior_values[0] / 1e6) << "M with very low std for tight control" << std::endl;
    
    logFile << std::endl << "========================================" << std::endl;
    
    logFile.close();
    std::cout << "[DEBUG] EKI configuration log written to: ../logs/eki_logs/eki_config.log" << std::endl;
}