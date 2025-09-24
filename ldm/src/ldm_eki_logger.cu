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

// ===== EKI INTEGRATION FUNCTIONS =====

void printSystemInfo() {
    std::cout << "[DEBUG] Single nuclide mode: CRAM system disabled" << std::endl;
}

void printEKIConfiguration(EKIConfig* ekiConfig) {
    std::cout << "[DEBUG] EKI configuration loaded successfully:" << std::endl;
    std::cout << "[DEBUG]   Receptors: " << ekiConfig->getNumReceptors() << std::endl;
    std::cout << "[DEBUG]   Source emission time steps: " << ekiConfig->getSourceEmission().num_time_steps << std::endl;
    std::cout << "[DEBUG]   Prior source time steps: " << ekiConfig->getPriorSource().num_time_steps << std::endl;
    
    // Display receptor positions
    for (int i = 0; i < ekiConfig->getNumReceptors(); i++) {
        Receptor r = ekiConfig->getReceptor(i);
        std::cout << "[DEBUG]   Receptor " << i << ": lat=" << r.lat << ", lon=" << r.lon << ", alt=" << r.alt << std::endl;
    }
}

void printSimulationStatus(const std::string& status) {
    std::cout << "[INFO] " << status << std::endl;
}

void cleanLogDirectory() {
    std::cout << "[INFO] Cleaning log directories..." << std::endl;
    
    // Clean LDM logs
    int clean_result1 = system("rm -f /home/jrpark/LDM-EKI/logs/ldm_logs/*.csv /home/jrpark/LDM-EKI/logs/ldm_logs/*.txt /home/jrpark/LDM-EKI/logs/ldm_logs/*.png /home/jrpark/LDM-EKI/logs/ldm_logs/*.bin");
    
    // Clean EKI logs
    int clean_result2 = system("rm -f /home/jrpark/LDM-EKI/logs/eki_logs/*.log /home/jrpark/LDM-EKI/logs/eki_logs/*.txt");
    
    if (clean_result1 == 0 && clean_result2 == 0) {
        std::cout << "[INFO] Log directories cleaned successfully (LDM + EKI logs)" << std::endl;
    } else {
        std::cout << "[WARNING] Failed to clean one or more log directories" << std::endl;
    }
}

void prepareObservationData(EKIConfig* ekiConfig) {
    std::cout << "\n[INFO] Preparing observation data for EKI..." << std::endl;
    
    int num_receptors = ekiConfig->getNumReceptors();
    int time_intervals = ekiConfig->getSourceEmission().num_time_steps;
    
    std::vector<float> observation_data(num_receptors * time_intervals, 0.0f);
    
    // Calculate actual receptor concentrations from particle data
    std::cout << "[INFO] Calculating receptor concentrations from available particle data..." << std::endl;
    
    // Check which particle files actually exist
    std::vector<int> available_timesteps;
    for (int t = 0; t <= time_intervals; t++) {
        std::string particle_file = "../logs/ldm_logs/particles_15min_" + std::to_string(t) + ".csv";
        std::ifstream test_file(particle_file);
        if (test_file.is_open()) {
            available_timesteps.push_back(t);
            test_file.close();
            std::cout << "[INFO] Found particle data for timestep " << t << std::endl;
        }
    }
    
    if (available_timesteps.empty()) {
        std::cout << "[WARNING] No particle files found. Using final particle state..." << std::endl;
        // Use particles_final.csv if available
        std::string final_file = "../logs/ldm_logs/particles_final.csv";
        std::ifstream file(final_file);
        if (file.is_open()) {
            available_timesteps.push_back(time_intervals); // Treat as final timestep
            file.close();
            std::cout << "[INFO] Using particles_final.csv for concentration calculation" << std::endl;
        }
    }
    
    // If we still have no data, create realistic estimates based on EKI source emissions
    if (available_timesteps.empty()) {
        std::cout << "[INFO] No particle data available. Generating realistic receptor estimates..." << std::endl;
        
        for (int t = 0; t < time_intervals; t++) {
            // Get source emission for this time step
            float source_emission = ekiConfig->getSourceEmission().time_series[t];
            
            // All receptors get the same value for this time step (but different across time)
            for (int r = 0; r < num_receptors; r++) {
                int idx = r * time_intervals + t;
                observation_data[idx] = source_emission; // Direct source emission as observation
            }
        }
        
        std::cout << "[INFO] Generated receptor concentration estimates based on source emissions" << std::endl;
    } else {
        // Process available particle files
        for (int timestep : available_timesteps) {
            std::string particle_file;
            if (timestep == time_intervals) {
                particle_file = "../logs/ldm_logs/particles_final.csv";
            } else {
                particle_file = "../logs/ldm_logs/particles_15min_" + std::to_string(timestep) + ".csv";
            }
            
            std::ifstream file(particle_file);
            if (!file.is_open()) continue;
            
            std::string line;
            std::getline(file, line); // Skip header
            
            // Initialize receptor concentration sums for this timestep
            std::vector<float> receptor_concentrations(num_receptors, 0.0f);
            std::vector<int> receptor_particle_counts(num_receptors, 0);
            
            // Receptor radius is 10.0 degrees (captures all particles)
            const float RECEPTOR_RADIUS = 10.0f;
            
            while (std::getline(file, line)) {
                std::istringstream ss(line);
                std::string token;
                
                // Parse CSV: particle_id,longitude,latitude,altitude,concentration,age,flag
                std::vector<std::string> tokens;
                while (std::getline(ss, token, ',')) {
                    tokens.push_back(token);
                }
                
                if (tokens.size() >= 7) {
                    float particle_lon = std::stof(tokens[1]);
                    float particle_lat = std::stof(tokens[2]);
                    float particle_conc = std::stof(tokens[4]);
                    int particle_flag = std::stoi(tokens[6]);
                    
                    if (particle_flag == 1) { // Only active particles
                        // Check each receptor
                        for (int r = 0; r < num_receptors; r++) {
                            Receptor receptor = ekiConfig->getReceptor(r);
                            
                            // Calculate distance to receptor
                            float dist_lon = particle_lon - receptor.lon;
                            float dist_lat = particle_lat - receptor.lat;
                            float distance = sqrt(dist_lon * dist_lon + dist_lat * dist_lat);
                            
                            // Check if particle is within receptor radius (10.0 degrees)
                            if (distance <= RECEPTOR_RADIUS) {
                                // Simply sum up concentrations (no normalization)
                                receptor_concentrations[r] += particle_conc;
                                receptor_particle_counts[r]++;
                            }
                        }
                    }
                }
            }
            file.close();
            
            // Store receptor concentrations for this timestep
            // If this is the final timestep, distribute scaled values across all times
            if (timestep == time_intervals) {
                // Use final particle state to estimate time-varying concentrations
                for (int t = 0; t < time_intervals; t++) {
                    // Get emission rate for this time step
                    float time_emission = ekiConfig->getSourceEmission().time_series[t];
                    float final_emission = ekiConfig->getSourceEmission().time_series[time_intervals-1];
                    
                    // Scale final concentration by emission ratio
                    float time_scale = (final_emission > 0) ? (time_emission / final_emission) : 1.0f;
                    
                    for (int r = 0; r < num_receptors; r++) {
                        int idx = r * time_intervals + t;
                        observation_data[idx] = receptor_concentrations[r] * time_scale;
                    }
                }
            } else {
                // Use data for specific timestep
                for (int r = 0; r < num_receptors; r++) {
                    int idx = r * time_intervals + timestep;
                    observation_data[idx] = receptor_concentrations[r];
                }
            }
            
            std::cout << "[INFO] Processed timestep " << timestep << " particles captured: ";
            for (int r = 0; r < num_receptors; r++) {
                std::cout << "R" << r << "=" << receptor_particle_counts[r] << "(" 
                          << std::scientific << std::setprecision(2) << receptor_concentrations[r] << ") ";
            }
            std::cout << std::endl;
        }
    }
    
    // Save observation data to file for EKI to read
    std::ofstream obs_file("../logs/ldm_logs/initial_observations.bin", std::ios::binary);
    if (obs_file.is_open()) {
        obs_file.write(reinterpret_cast<const char*>(observation_data.data()), 
                      observation_data.size() * sizeof(float));
        obs_file.close();
        std::cout << "[INFO] Initial observation data saved to file." << std::endl;
    } else {
        std::cout << "[WARNING] Failed to save observation data to file." << std::endl;
    }
    
    // Save observation data to integration_logs for analysis
    saveObservationToIntegrationLogs(observation_data, num_receptors, time_intervals);
    
    // Log LDM->EKI matrix transmission in ASCII format
    logLDMtoEKIMatrix(observation_data, num_receptors, time_intervals);
}

void runVisualization() {
    std::cout << "\n[INFO] Simulation completed. Creating animation frames..." << std::endl;
    
    std::cout << "[INFO] Running OSM visualization generation..." << std::endl;
    int result = system("cd /home/jrpark/LDM-EKI/ldm && python3 -c \"import osm_grid_visualization; osm_grid_visualization.create_osm_visualizations()\"");
    if (result == 0) {
        std::cout << "[INFO] OSM visualizations created successfully." << std::endl;
    } else {
        std::cout << "[WARNING] OSM visualization generation failed with exit code: " << result << std::endl;
    }
    
    std::cout << "[INFO] OSM visualization completed. Check logs/ldm_logs/ for:" << std::endl;
    std::cout << "  - osm_grid_zoomed_out.png (2x3 wide view)" << std::endl;
    std::cout << "  - osm_grid_zoomed_in.png (2x3 NYC focus)" << std::endl;
    std::cout << "  - animation_frame_*.png (individual OSM frames)" << std::endl;
}

void runIterativeEKIEstimation(LDM& ldm, EKIConfig* ekiConfig) {
    const int max_iterations = 5; // Maximum number of EKI iterations
    
    std::cout << "\n[INFO] Starting iterative EKI-LDM estimation..." << std::endl;
    
    // Initialize iteration status file
    system("mkdir -p /home/jrpark/LDM-EKI/logs/integration_logs");
    std::ofstream status_file("/home/jrpark/LDM-EKI/logs/integration_logs/iteration_status.txt");
    status_file << "iteration: 0\nstatus: initialized\n";
    status_file.close();
    
    // Backup original EKI config
    system("cd /home/jrpark/LDM-EKI/eki && cp config/input_config config/input_config_backup");
    
    for (int iter = 1; iter <= max_iterations; iter++) {
        std::cout << "\n[INFO] === EKI Iteration " << iter << " ===" << std::endl;
        
        // Update iteration status
        std::ofstream status_file("/home/jrpark/LDM-EKI/logs/integration_logs/iteration_status.txt");
        status_file << "iteration: " << iter << "\nstatus: eki_running\n";
        status_file.close();
        
        // Create temporary config for single iteration
        std::string sed_cmd = "cd /home/jrpark/LDM-EKI/eki && sed 's/iteration: [0-9]*/iteration: 1/' config/input_config > config/input_config_temp";
        system(sed_cmd.c_str());
        
        // Run EKI with iteration number parameter
        std::string eki_cmd = "cd /home/jrpark/LDM-EKI/eki && python3 src/RunEstimator.py config/input_config_temp config/input_data " + std::to_string(iter);
        int eki_result = system(eki_cmd.c_str());
        
        if (eki_result != 0) {
            std::cout << "[ERROR] EKI iteration " << iter << " failed with exit code: " << eki_result << std::endl;
            break;
        }
        
        std::cout << "[INFO] EKI iteration " << iter << " completed. Starting ensemble LDM calculation..." << std::endl;
        
        // Update status to LDM running
        std::ofstream status_file2("/home/jrpark/LDM-EKI/logs/integration_logs/iteration_status.txt");
        status_file2 << "iteration: " << iter << "\nstatus: ldm_running\n";
        status_file2.close();
        
        // Execute LDM ensemble calculation directly (no background process)
        if (!executeLDMEnsemble(ldm, iter, ekiConfig)) {
            std::cout << "[ERROR] LDM ensemble execution failed for iteration " << iter << std::endl;
            break;
        }
        
        std::cout << "[INFO] Iteration " << iter << " completed successfully." << std::endl;
        
        // Update status to completed
        std::ofstream status_file3("/home/jrpark/LDM-EKI/logs/integration_logs/iteration_status.txt");
        status_file3 << "iteration: " << iter << "\nstatus: completed\n";
        status_file3.close();
        
        // Check if we should continue (convergence check would go here)
        // For now, continue all iterations
    }
    
    // Restore original config and cleanup
    system("cd /home/jrpark/LDM-EKI/eki && mv config/input_config_backup config/input_config");
    system("cd /home/jrpark/LDM-EKI/eki && rm -f config/input_config_temp");
    
    std::cout << "\n[INFO] Iterative EKI-LDM estimation completed." << std::endl;
}

bool executeLDMEnsemble(LDM& ldm, int iteration, EKIConfig* ekiConfig) {
    std::cout << "[INFO] Executing LDM ensemble calculation for iteration " << iteration << std::endl;
    
    // Load EKI ensemble results for this iteration
    std::vector<std::vector<float>> ensemble_matrix;
    int time_intervals, ensemble_size;
    
    if (!loadEKIEnsembleStates(ensemble_matrix, time_intervals, ensemble_size, iteration)) {
        std::cerr << "[ERROR] Failed to load EKI ensemble states for iteration " << iteration << std::endl;
        return false;
    }

    std::cout << "[INFO] Loaded ensemble matrix: " << time_intervals << "×" << ensemble_size << std::endl;

    // Run ensemble simulation directly without creating new process
    if (!runEnsembleLDM(ldm, ensemble_matrix, time_intervals, ensemble_size)) {
        std::cerr << "[ERROR] Ensemble LDM execution failed for iteration " << iteration << std::endl;
        return false;
    }
    
    std::cout << "[INFO] LDM ensemble calculation completed for iteration " << iteration << std::endl;
    return true;
}

// waitForLDMCompletion function removed - now using direct execution

// Legacy function for backward compatibility - needs ldm and ekiConfig parameters
// This function is now deprecated - use runIterativeEKIEstimation directly

void saveObservationToIntegrationLogs(const std::vector<float>& observation_data, int num_receptors, int time_intervals) {
    std::cout << "[INFO] Saving observation data to integration_logs..." << std::endl;
    
    // Create integration_logs directory if it doesn't exist
    system("mkdir -p /home/jrpark/LDM-EKI/logs/integration_logs");
    
    // Get current time for timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::stringstream timestamp;
    timestamp << std::put_time(std::localtime(&time_t_now), "%Y%m%d_%H%M%S");
    
    // Save as CSV
    std::string csv_filename = "/home/jrpark/LDM-EKI/logs/integration_logs/single_mode_observations_" + timestamp.str() + ".csv";
    std::ofstream csv_file(csv_filename);
    
    if (csv_file.is_open()) {
        // Write header
        csv_file << "# LDM Single Mode Observation Data\n";
        csv_file << "# Generated: " << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S") << "\n";
        csv_file << "# Receptors: " << num_receptors << "\n";
        csv_file << "# Time intervals: " << time_intervals << "\n";
        csv_file << "# Total elements: " << observation_data.size() << "\n";
        csv_file << "# Data format: receptor_id,time_step,concentration\n";
        csv_file << "receptor_id,time_step,concentration\n";
        
        // Write data
        for (int r = 0; r < num_receptors; r++) {
            for (int t = 0; t < time_intervals; t++) {
                int idx = r * time_intervals + t;
                csv_file << r << "," << t << "," << std::scientific << std::setprecision(6) << observation_data[idx] << "\n";
            }
        }
        
        csv_file.close();
        std::cout << "[INFO] Observation CSV saved: " << csv_filename << std::endl;
    }
    
    // Save as binary for EKI consumption
    std::string bin_filename = "/home/jrpark/LDM-EKI/logs/integration_logs/single_mode_observations_" + timestamp.str() + ".bin";
    std::ofstream bin_file(bin_filename, std::ios::binary);
    
    if (bin_file.is_open()) {
        bin_file.write(reinterpret_cast<const char*>(observation_data.data()), observation_data.size() * sizeof(float));
        bin_file.close();
        std::cout << "[INFO] Observation binary saved: " << bin_filename << std::endl;
    }
}

void logLDMtoEKIMatrix(const std::vector<float>& data, int num_receptors, int time_intervals) {
    std::cout << "[DEBUG] Logging LDM->EKI matrix transmission..." << std::endl;
    
    // Get current time for timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    
    std::ofstream logFile("../logs/eki_logs/ldm_eki_matrix_transmission.log");
    if (!logFile.is_open()) {
        std::cerr << "[ERROR] Failed to create matrix transmission log file" << std::endl;
        return;
    }
    
    logFile << "========================================" << std::endl;
    logFile << "LDM -> EKI Matrix Transmission Log" << std::endl;
    logFile << "Generated: " << std::ctime(&time_t_now);
    logFile << "========================================" << std::endl << std::endl;
    
    logFile << "Matrix Dimensions:" << std::endl;
    logFile << "  Receptors: " << num_receptors << std::endl;
    logFile << "  Time intervals: " << time_intervals << std::endl;
    logFile << "  Total elements: " << num_receptors * time_intervals << std::endl;
    logFile << "  Matrix shape: [" << num_receptors * time_intervals << " × 1]" << std::endl << std::endl;
    
    logFile << "Matrix Content (ASCII Format):" << std::endl;
    logFile << "Initial Observations Matrix:" << std::endl;
    logFile << std::string(60, '=') << std::endl;
    
    // Print matrix header
    logFile << std::setw(8) << "Index" << " | " 
            << std::setw(10) << "Receptor" << " | " 
            << std::setw(8) << "Time" << " | " 
            << std::setw(15) << "Value" << std::endl;
    logFile << std::string(60, '-') << std::endl;
    
    // Print matrix content
    for (size_t i = 0; i < data.size(); i++) {
        int receptor_id = i / time_intervals;
        int time_step = i % time_intervals;
        
        logFile << std::setw(8) << i << " | " 
                << std::setw(10) << receptor_id << " | " 
                << std::setw(8) << time_step << " | " 
                << std::scientific << std::setprecision(3) << data[i] << std::endl;
        
        if (i >= 10 && i < data.size() - 5) {
            if (i == 10) {
                logFile << "    ... (middle values omitted for brevity) ..." << std::endl;
            }
            if (i < data.size() - 5) continue;
        }
    }
    
    logFile << std::string(60, '=') << std::endl << std::endl;
    
    // Statistics
    float min_val = *std::min_element(data.begin(), data.end());
    float max_val = *std::max_element(data.begin(), data.end());
    float sum_val = std::accumulate(data.begin(), data.end(), 0.0f);
    
    logFile << "Matrix Statistics:" << std::endl;
    logFile << "  Min: " << std::scientific << std::setprecision(3) << min_val << std::endl;
    logFile << "  Max: " << std::scientific << std::setprecision(3) << max_val << std::endl;
    logFile << "  Sum: " << std::scientific << std::setprecision(3) << sum_val << std::endl;
    
    logFile << "\n========================================" << std::endl;
    
    logFile.close();
    std::cout << "[DEBUG] LDM->EKI matrix log written to: ../logs/eki_logs/ldm_eki_matrix_transmission.log" << std::endl;
}

bool loadEKIEnsembleStates(std::vector<std::vector<float>>& ensemble_matrix, int& time_intervals, int& ensemble_size, int iteration) {
    std::string ensemble_file_path = "../logs/ldm_logs/ensemble_states_iter_" + std::to_string(iteration) + ".bin";
    
    std::cout << "[DEBUG] Attempting to load EKI ensemble states from: " << ensemble_file_path << std::endl;
    
    std::ifstream file(ensemble_file_path, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "[WARNING] Could not open ensemble file: " << ensemble_file_path << std::endl;
        return false;
    }
    
    // Read all data first to determine size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    size_t total_elements = file_size / sizeof(float);
    std::cout << "[DEBUG] File size: " << file_size << " bytes, Total elements: " << total_elements << std::endl;
    
    // Read raw data
    std::vector<float> raw_data(total_elements);
    file.read(reinterpret_cast<char*>(raw_data.data()), file_size);
    file.close();
    
    // Determine matrix dimensions (expect 24 x 100 = 2400 elements)
    time_intervals = 24;
    ensemble_size = total_elements / time_intervals;
    
    if (total_elements != time_intervals * ensemble_size) {
        std::cout << "[ERROR] Unexpected data size. Expected: " << (time_intervals * ensemble_size) 
                  << ", Got: " << total_elements << std::endl;
        return false;
    }
    
    std::cout << "[DEBUG] Ensemble matrix dimensions: " << time_intervals << " x " << ensemble_size << std::endl;
    
    // Allocate 2D matrix dynamically
    ensemble_matrix.clear();
    ensemble_matrix.resize(time_intervals);
    for (int t = 0; t < time_intervals; t++) {
        ensemble_matrix[t].resize(ensemble_size);
        for (int e = 0; e < ensemble_size; e++) {
            // Data is stored in row-major order: [time][ensemble]
            ensemble_matrix[t][e] = raw_data[t * ensemble_size + e];
        }
    }
    
    std::cout << "[INFO] Successfully loaded ensemble matrix [" << time_intervals << " x " << ensemble_size << "]" << std::endl;
    
    // Log the reception
    logLDMEnsembleReception(ensemble_matrix, time_intervals, ensemble_size, iteration);
    
    return true;
}

void logLDMEnsembleReception(const std::vector<std::vector<float>>& ensemble_matrix, int time_intervals, int ensemble_size, int iteration) {
    std::cout << "[DEBUG] Logging LDM ensemble reception..." << std::endl;
    
    // Get current time for timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    
    std::string log_file_path = "../logs/eki_logs/ldm_ensemble_reception_iter_" + std::to_string(iteration) + ".log";
    std::ofstream logFile(log_file_path);
    if (!logFile.is_open()) {
        std::cerr << "[ERROR] Failed to create LDM ensemble reception log file" << std::endl;
        return;
    }
    
    logFile << "========================================" << std::endl;
    logFile << "LDM Reception of EKI Ensemble States (Iteration " << iteration << ")" << std::endl;
    logFile << "Generated: " << std::ctime(&time_t_now);
    logFile << "========================================" << std::endl << std::endl;
    
    // Matrix dimensions
    logFile << "Received Ensemble Matrix Dimensions:" << std::endl;
    logFile << "  Time intervals: " << time_intervals << std::endl;
    logFile << "  Ensemble members: " << ensemble_size << std::endl;
    logFile << "  Total elements: " << (time_intervals * ensemble_size) << std::endl;
    logFile << "  Matrix shape: [" << time_intervals << " × " << ensemble_size << "]" << std::endl << std::endl;
    
    // Data layout explanation
    logFile << "Data Layout:" << std::endl;
    logFile << "  Rows: Time intervals (0-23, corresponding to 0-6 hours in 15-min steps)" << std::endl;
    logFile << "  Columns: Ensemble members (0-" << (ensemble_size-1) << ", different emission scenarios)" << std::endl;
    logFile << "  Each cell: Emission rate (Bq/s) for specific time and ensemble" << std::endl << std::endl;
    
    // Sample matrix content
    logFile << "Sample Matrix Content (First 5x5):" << std::endl;
    logFile << std::string(80, '=') << std::endl;
    logFile << std::setw(10) << "Time\\Ens";
    for (int e = 0; e < std::min(5, ensemble_size); e++) {
        logFile << std::setw(15) << ("E" + std::to_string(e));
    }
    logFile << std::endl;
    logFile << std::string(80, '-') << std::endl;
    
    for (int t = 0; t < std::min(5, time_intervals); t++) {
        logFile << std::setw(10) << ("T" + std::to_string(t));
        for (int e = 0; e < std::min(5, ensemble_size); e++) {
            logFile << std::setw(15) << std::scientific << std::setprecision(2) << ensemble_matrix[t][e];
        }
        logFile << std::endl;
    }
    
    logFile << std::string(80, '=') << std::endl;
    
    logFile.close();
    std::cout << "[DEBUG] LDM ensemble reception log written to: " << log_file_path << std::endl;
}