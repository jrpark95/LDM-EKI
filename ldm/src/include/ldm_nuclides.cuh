#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cuda_runtime.h>

// Maximum number of nuclides supported
#define MAX_NUCLIDES 60

// Decay constants array in constant memory - initialized with test values
__constant__ float d_decay_constants[MAX_NUCLIDES];

// Nuclide information structure
struct NuclideInfo {
    char name[32];
    float half_life_hours;
    float decay_constant;  // Calculated from half-life
    float initial_ratio;
};

// Global nuclide configuration
class NuclideConfig {
private:
    static NuclideConfig* instance;
    std::vector<NuclideInfo> nuclides;
    int num_nuclides;
    
    // Device memory for decay constants
    float* d_decay_constants;
    
public:
    NuclideConfig() : num_nuclides(0), d_decay_constants(nullptr) {}
    
    ~NuclideConfig() {
        if (d_decay_constants) {
            cudaFree(d_decay_constants);
        }
    }
    
    static NuclideConfig* getInstance() {
        if (!instance) {
            instance = new NuclideConfig();
        }
        return instance;
    }
    
    bool loadFromFile(const std::string& filename) {
        std::cout << "[DEBUG] Loading nuclide config from: " << filename << std::endl;
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "[ERROR] Cannot open nuclides config file: " << filename << std::endl;
            return false;
        }
        std::cout << "[DEBUG] Successfully opened nuclide config file" << std::endl;
        
        // Debug: read and show first few lines
        std::string debug_line;
        int line_count = 0;
        while (std::getline(file, debug_line) && line_count < 3) {
            std::cout << "[DEBUG] Line " << line_count << ": " << debug_line << std::endl;
            line_count++;
        }
        
        // Close and reopen file to ensure clean state
        file.close();
        file.open(filename);
        if (!file.is_open()) {
            std::cerr << "[ERROR] Cannot reopen nuclides config file: " << filename << std::endl;
            return false;
        }
        
        nuclides.clear();
        std::string line;
        
        while (std::getline(file, line)) {
            // Skip comments and empty lines
            if (line.empty() || line[0] == '#') continue;
            
            std::cout << "[DEBUG] Processing line: '" << line << "'" << std::endl;
            
            std::istringstream iss(line);
            std::string name;
            float decay_const_from_file, ratio;
            
            // Parse CSV manually for better control
            std::istringstream iss2(line);
            std::string token;
            std::vector<std::string> tokens;
            
            while (std::getline(iss2, token, ',')) {
                tokens.push_back(token);
            }
            
            std::cout << "[DEBUG] Parsed " << tokens.size() << " tokens" << std::endl;
            for (size_t i = 0; i < tokens.size(); i++) {
                std::cout << "[DEBUG] Token[" << i << "]: '" << tokens[i] << "'" << std::endl;
            }
            
            if (tokens.size() == 3) {
                name = tokens[0];
                decay_const_from_file = std::stof(tokens[1]);
                ratio = std::stof(tokens[2]);
                
                NuclideInfo info;
                strncpy(info.name, name.c_str(), 31);
                info.name[31] = '\0';
                
                // File contains decay constants (λ in s⁻¹), not half-lives
                // Take absolute value of the decay constant from file
                info.decay_constant = fabs(decay_const_from_file);
                
                // Calculate half-life from decay constant: t½ = ln(2) / λ
                // Result will be in seconds, convert to hours
                if (info.decay_constant > 0) {
                    info.half_life_hours = 0.693147f / info.decay_constant / 3600.0f;
                } else {
                    info.half_life_hours = 1e10f; // Very large value for stable nuclides
                }
                
                info.initial_ratio = ratio;
                
                nuclides.push_back(info);
            }
        }
        
        file.close();
        num_nuclides = nuclides.size();
        std::cout << "[DEBUG] Loaded " << num_nuclides << " nuclides from config file" << std::endl;
        
        if (num_nuclides == 0) {
            std::cerr << "[ERROR] No nuclides loaded from config file" << std::endl;
            return false;
        }
        
        if (num_nuclides > MAX_NUCLIDES) {
            std::cerr << "[WARNING] Too many nuclides (" << num_nuclides 
                     << "), limiting to " << MAX_NUCLIDES << std::endl;
            num_nuclides = MAX_NUCLIDES;
            nuclides.resize(MAX_NUCLIDES);
        }
        
        // Normalize initial ratios if they don't sum to 1
        float total_ratio = 0.0f;
        for (const auto& nuc : nuclides) {
            total_ratio += nuc.initial_ratio;
        }
        
        if (std::abs(total_ratio - 1.0f) > 0.001f) {
            for (auto& nuc : nuclides) {
                nuc.initial_ratio /= total_ratio;
            }
        }
        
        // Allocate and copy decay constants to device
        return copyToDevice();
    }
    
    bool copyToDevice() {
        if (d_decay_constants) {
            cudaFree(d_decay_constants);
        }
        
        // Prepare decay constants array
        std::vector<float> decay_constants(num_nuclides);
        for (int i = 0; i < num_nuclides; i++) {
            decay_constants[i] = nuclides[i].decay_constant;
        }
        
        // Allocate device memory
        cudaError_t err = cudaMalloc(&d_decay_constants, num_nuclides * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "[ERROR] Failed to allocate device memory for decay constants: " 
                     << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        // Copy to device
        err = cudaMemcpy(d_decay_constants, decay_constants.data(), 
                        num_nuclides * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "[ERROR] Failed to copy decay constants to device: " 
                     << cudaGetErrorString(err) << std::endl;
            cudaFree(d_decay_constants);
            d_decay_constants = nullptr;
            return false;
        }
        
        // Try to copy to constant memory (may fail due to symbol visibility)
        err = cudaMemcpyToSymbol(d_decay_constants, decay_constants.data(), 
                                num_nuclides * sizeof(float));
        if (err != cudaSuccess) {
            // Global memory fallback is already allocated and copied above
        }
        return true;
    }
    
    int getNumNuclides() const { return num_nuclides; }
    const std::vector<NuclideInfo>& getNuclides() const { return nuclides; }
    float* getDeviceDecayConstants() { return d_decay_constants; }
    
    // Get decay constant for a specific nuclide
    float getDecayConstant(int index) const {
        if (index >= 0 && index < num_nuclides) {
            return nuclides[index].decay_constant;
        }
        return 0.0f;
    }
    
    // Get initial ratio for a specific nuclide
    float getInitialRatio(int index) const {
        if (index >= 0 && index < num_nuclides) {
            return nuclides[index].initial_ratio;
        }
        return 0.0f;
    }
    
    // Get nuclide name
    const char* getNuclideName(int index) const {
        if (index >= 0 && index < num_nuclides) {
            return nuclides[index].name;
        }
        return "Unknown";
    }
    
    // Get half-life for a specific nuclide (in hours)
    float getHalfLife(int index) const {
        if (index >= 0 && index < num_nuclides) {
            return nuclides[index].half_life_hours;
        }
        return 0.0f;
    }
};

// Initialize static member
NuclideConfig* NuclideConfig::instance = nullptr;