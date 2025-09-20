#pragma once

#include <vector>
#include <string>

// Data structures for integration
struct EmissionData {
    int Nens;
    int T;
    std::vector<float> flat;
    EmissionData() : Nens(0), T(0) {}
};

struct EnsembleState {
    int Nens;
    int state_dim;
    std::vector<float> X;
    EnsembleState() : Nens(0), state_dim(0) {}
};

// Utility functions
bool file_exists(const std::string& filepath);
bool wait_for_files(const std::string& dir, const std::vector<std::string>& filenames, int timeout_sec);
bool load_emission_series(const std::string& filepath, EmissionData& emis);
bool load_ensemble_state(const std::string& filepath, EnsembleState& ens);