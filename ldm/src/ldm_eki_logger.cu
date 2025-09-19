#include "ldm_eki_logger.cuh"

// ===== DEBUG AND STATUS FUNCTIONS =====

void printSystemInfo() {
    std::cout << "[DEBUG] Single nuclide mode: CRAM system disabled" << std::endl;
}

void printEKIConfiguration(EKIConfig* ekiConfig) {
    std::cout << "[DEBUG] EKI configuration loaded successfully:" << std::endl;
    std::cout << "[DEBUG]   Receptors: " << ekiConfig->getNumReceptors() << std::endl;
    std::cout << "[DEBUG]   Source emission time steps: " << ekiConfig->getSourceEmission().num_time_steps << std::endl;
    std::cout << "[DEBUG]   Prior source time steps: " << ekiConfig->getPriorSource().num_time_steps << std::endl;
    std::cout << "[DEBUG]   Ensemble size (sample_ctrl): " << ekiConfig->sample_ctrl << std::endl;
    std::cout << "[DEBUG]   Iterations: " << ekiConfig->iteration << std::endl;
    
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
    std::cout << "[INFO] Cleaning log directory..." << std::endl;
    int clean_result = system("rm -f /home/jrpark/LDM-EKI/logs/ldm_logs/*.csv /home/jrpark/LDM-EKI/logs/ldm_logs/*.txt /home/jrpark/LDM-EKI/logs/ldm_logs/*.png");
    if (clean_result == 0) {
        std::cout << "[INFO] Log directory cleaned successfully" << std::endl;
    } else {
        std::cout << "[WARNING] Failed to clean log directory" << std::endl;
    }
}

// ===== DATA EXPORT FUNCTIONS =====

void prepareObservationData(EKIConfig* ekiConfig) {
    std::cout << "\n[INFO] Preparing observation data for EKI..." << std::endl;
    
    int num_receptors = ekiConfig->getNumReceptors();
    int time_intervals = ekiConfig->getSourceEmission().num_time_steps;
    
    std::vector<float> observation_data(num_receptors * time_intervals);
    
    // Fill with dummy data for testing
    for (int r = 0; r < num_receptors; r++) {
        for (int t = 0; t < time_intervals; t++) {
            int idx = r * time_intervals + t;
            observation_data[idx] = (r + 1) * (t + 1) * 1e-8;
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

void runEKIEstimation() {
    std::cout << "\n[INFO] Starting EKI source estimation..." << std::endl;
    std::cout << "[INFO] Running EKI RunEstimator.py with config files..." << std::endl;
    
    int eki_result = system("cd /home/jrpark/LDM-EKI/eki && python3 src/RunEstimator.py config/input_config config/input_data");
    if (eki_result == 0) {
        std::cout << "[INFO] EKI source estimation completed successfully." << std::endl;
        std::cout << "[INFO] Check eki/results/ for estimation results and plots." << std::endl;
    } else {
        std::cout << "[WARNING] EKI estimation failed with exit code: " << eki_result << std::endl;
    }
}

// ===== MATRIX LOGGING FUNCTIONS =====

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
    
    logFile << "Data Layout (Row-major order):" << std::endl;
    logFile << "  Elements 0-" << (time_intervals-1) << ": Receptor 0, Times 0-" << (time_intervals-1) << std::endl;
    logFile << "  Elements " << time_intervals << "-" << (2*time_intervals-1) << ": Receptor 1, Times 0-" << (time_intervals-1) << std::endl;
    logFile << "  Elements " << (2*time_intervals) << "-" << (3*time_intervals-1) << ": Receptor 2, Times 0-" << (time_intervals-1) << std::endl << std::endl;
    
    logFile << "Matrix Content (ASCII Format):" << std::endl;
    logFile << "Initial Observations Matrix [" << num_receptors * time_intervals << " × 1]:" << std::endl;
    logFile << std::string(60, '=') << std::endl;
    
    // Print matrix header
    logFile << std::setw(8) << "Index" << " | " 
            << std::setw(10) << "Receptor" << " | " 
            << std::setw(8) << "Time" << " | " 
            << std::setw(15) << "Value" << std::endl;
    logFile << std::string(60, '-') << std::endl;
    
    // Print matrix content
    for (int i = 0; i < data.size(); i++) {
        int receptor_id = i / time_intervals;
        int time_step = i % time_intervals;
        
        logFile << std::setw(8) << i << " | " 
                << std::setw(10) << receptor_id << " | " 
                << std::setw(8) << time_step << " | " 
                << std::scientific << std::setprecision(3) << std::setw(15) << data[i] << std::endl;
    }
    
    logFile << std::string(60, '=') << std::endl << std::endl;
    
    // Print matrix in receptor-wise blocks
    logFile << "Matrix by Receptor Blocks:" << std::endl;
    for (int r = 0; r < num_receptors; r++) {
        logFile << "\nReceptor " << r << " (Times 0-" << (time_intervals-1) << "):" << std::endl;
        logFile << "[ ";
        for (int t = 0; t < time_intervals; t++) {
            int idx = r * time_intervals + t;
            logFile << std::scientific << std::setprecision(2) << data[idx];
            if (t < time_intervals - 1) logFile << ", ";
        }
        logFile << " ]" << std::endl;
    }
    
    logFile << "\n" << std::string(60, '=') << std::endl;
    logFile << "Matrix Statistics:" << std::endl;
    
    float min_val = *std::min_element(data.begin(), data.end());
    float max_val = *std::max_element(data.begin(), data.end());
    float sum_val = std::accumulate(data.begin(), data.end(), 0.0f);
    float mean_val = sum_val / data.size();
    
    logFile << "  Min value: " << std::scientific << std::setprecision(3) << min_val << std::endl;
    logFile << "  Max value: " << std::scientific << std::setprecision(3) << max_val << std::endl;
    logFile << "  Mean value: " << std::scientific << std::setprecision(3) << mean_val << std::endl;
    logFile << "  Sum: " << std::scientific << std::setprecision(3) << sum_val << std::endl;
    
    logFile << "\n========================================" << std::endl;
    
    logFile.close();
    std::cout << "[DEBUG] LDM->EKI matrix log written to: ../logs/eki_logs/ldm_eki_matrix_transmission.log" << std::endl;
}

// ===== ENSEMBLE HANDLING FUNCTIONS =====

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
    
    // Full matrix content with ensembles as rows
    logFile << "Full Matrix Content (Ensembles as Rows):" << std::endl;
    logFile << std::string(150, '=') << std::endl;
    
    // Header row - show all 24 time intervals
    logFile << std::setw(5) << "Ens";
    for (int t = 0; t < time_intervals; t++) {
        logFile << std::setw(10) << ("T" + std::to_string(t));
    }
    logFile << std::endl;
    logFile << std::string(150, '-') << std::endl;
    
    // Data rows - all ensembles
    for (int e = 0; e < ensemble_size; e++) {
        logFile << std::setw(5) << e;
        for (int t = 0; t < time_intervals; t++) {
            logFile << std::scientific << std::setprecision(2) << std::setw(10) << ensemble_matrix[t][e];
        }
        logFile << std::endl;
    }
    
    logFile << std::string(150, '=') << std::endl << std::endl;
    
    // Full matrix by time intervals (for easy reading)
    logFile << "Full Matrix by Time Intervals:" << std::endl;
    for (int t = 0; t < time_intervals; t++) {
        int hours = (t * 15) / 60;
        int minutes = (t * 15) % 60;
        logFile << std::fixed << std::setprecision(0);
        logFile << "Time " << std::setw(2) << t << " (" << hours << ":" << std::setw(2) << std::setfill('0') << minutes << std::setfill(' ') << "): [";
        
        // Show first 5 ensemble values
        for (int e = 0; e < std::min(5, ensemble_size); e++) {
            if (e > 0) logFile << ", ";
            logFile << std::scientific << std::setprecision(2) << ensemble_matrix[t][e];
        }
        if (ensemble_size > 5) {
            logFile << ", ... " << (ensemble_size - 5) << " more ensembles";
        }
        logFile << "]" << std::endl;
    }
    
    // Statistics by time
    logFile << std::endl << std::string(80, '=') << std::endl;
    logFile << "Statistics by Time Interval:" << std::endl;
    logFile << std::setw(6) << "Time" << " | " 
            << std::setw(12) << "Min" << " | " 
            << std::setw(12) << "Max" << " | " 
            << std::setw(12) << "Mean" << " | " 
            << std::setw(12) << "Std" << std::endl;
    logFile << std::string(80, '-') << std::endl;
    
    for (int t = 0; t < time_intervals; t++) {
        // Calculate statistics for this time interval
        float min_val = ensemble_matrix[t][0];
        float max_val = ensemble_matrix[t][0];
        float sum_val = 0.0f;
        
        for (int e = 0; e < ensemble_size; e++) {
            float val = ensemble_matrix[t][e];
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
            sum_val += val;
        }
        
        float mean_val = sum_val / ensemble_size;
        
        // Calculate standard deviation
        float variance = 0.0f;
        for (int e = 0; e < ensemble_size; e++) {
            float diff = ensemble_matrix[t][e] - mean_val;
            variance += diff * diff;
        }
        float std_val = std::sqrt(variance / ensemble_size);
        
        logFile << std::setw(6) << t << " | " 
                << std::scientific << std::setprecision(2) << std::setw(12) << min_val << " | " 
                << std::scientific << std::setprecision(2) << std::setw(12) << max_val << " | " 
                << std::scientific << std::setprecision(2) << std::setw(12) << mean_val << " | " 
                << std::scientific << std::setprecision(2) << std::setw(12) << std_val << std::endl;
    }
    
    // Overall statistics
    logFile << std::endl << std::string(80, '=') << std::endl;
    logFile << "Overall Matrix Statistics:" << std::endl;
    
    float global_min = ensemble_matrix[0][0];
    float global_max = ensemble_matrix[0][0];
    float global_sum = 0.0f;
    
    for (int t = 0; t < time_intervals; t++) {
        for (int e = 0; e < ensemble_size; e++) {
            float val = ensemble_matrix[t][e];
            global_min = std::min(global_min, val);
            global_max = std::max(global_max, val);
            global_sum += val;
        }
    }
    
    float global_mean = global_sum / (time_intervals * ensemble_size);
    
    logFile << "  Global Min: " << std::scientific << std::setprecision(3) << global_min << std::endl;
    logFile << "  Global Max: " << std::scientific << std::setprecision(3) << global_max << std::endl;
    logFile << "  Global Mean: " << std::scientific << std::setprecision(3) << global_mean << std::endl;
    
    logFile << std::endl << "========================================" << std::endl;
    
    logFile.close();
    std::cout << "[DEBUG] LDM ensemble reception log written to: " << log_file_path << std::endl;
}

// ===== CONFIGURATION LOGGING FUNCTIONS =====

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