#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <regex>

// Maximum number of time steps for source emission (24 time steps = 6 hours with 15-minute intervals)
#define MAX_TIME_STEPS 24

// Maximum number of receptors
#define MAX_RECEPTORS 10

// Receptor structure
struct Receptor {
    float lat;  // Latitude
    float lon;  // Longitude  
    float alt;  // Altitude (meters)
};

// Source emission data structure
struct SourceEmission {
    std::vector<float> time_series;  // Time-dependent emission concentrations
    int num_time_steps;              // Number of time steps
};

// Prior source data structure  
struct PriorSource {
    float decay_constant;
    float uncertainty;
    float position[3];  // [lon, lat, alt]
    float position_std;
    std::vector<float> prior_values;  // Prior concentration values for each time step
    float value_std;
    std::string nuclide_name;
    int num_time_steps;
};

// Global EKI configuration data
class EKIConfig {
private:
    static EKIConfig* instance;
    
public:
    // Time configuration
    float time_day;
    int time_interval_min;
    float inverse_time_interval_hour;
    int ave_t;
    
    // Grid configuration
    int size_alt, size_lat, size_lon;
    float alt_spacing, lat_spacing, lon_spacing;
    
    // Wind configuration
    int size_alt_wind, size_lat_wind, size_lon_wind;
    std::string wind_init_mode;
    float wind_constant_value_x, wind_constant_value_y, wind_constant_value_z;
    int wind_grid_interval;
    int grid_space_size_lat, grid_space_size_lon;
    
    // Simulation parameters
    float puff_concentration_threshold;
    float R_max;
    
    // Receptor data
    int num_receptors;
    std::vector<Receptor> receptors;
    float nreceptor_err;
    float nreceptor_MDA;
    
    // Source configuration
    std::string source_location;
    int nsource;
    std::vector<std::string> source_names;
    
    // Source emission data
    SourceEmission source_emission;
    
    // Prior source data
    PriorSource prior_source;
    
    // Additional prior data
    float prior_source1_concentration;
    float prior_source1_std;
    float prior_source1_decay_constant;
    
    // Boundary data
    float real_source1_boundary_min;
    float real_source1_boundary_max;
    
    // Input config parameters
    int sample_ctrl;  // ensemble size
    int iteration;    // number of iterations
    
    EKIConfig() : num_receptors(0), nsource(0), time_day(0), time_interval_min(0), 
                  inverse_time_interval_hour(0), ave_t(0), size_alt(0), size_lat(0), size_lon(0),
                  alt_spacing(0), lat_spacing(0), lon_spacing(0), size_alt_wind(0), size_lat_wind(0),
                  size_lon_wind(0), wind_constant_value_x(0), wind_constant_value_y(0), wind_constant_value_z(0),
                  wind_grid_interval(0), grid_space_size_lat(0), grid_space_size_lon(0), sample_ctrl(0), iteration(0),
                  puff_concentration_threshold(0), R_max(0), nreceptor_err(0), nreceptor_MDA(0),
                  prior_source1_concentration(0), prior_source1_std(0), prior_source1_decay_constant(0),
                  real_source1_boundary_min(0), real_source1_boundary_max(0) {}
    
    static EKIConfig* getInstance() {
        if (!instance) {
            instance = new EKIConfig();
        }
        return instance;
    }
    
    bool loadFromFile(const std::string& filename);
    bool loadInputConfigFromFile(const std::string& input_config_filename);
    
    // Accessor methods
    int getNumReceptors() const { return num_receptors; }
    const std::vector<Receptor>& getReceptors() const { return receptors; }
    const SourceEmission& getSourceEmission() const { return source_emission; }
    const PriorSource& getPriorSource() const { return prior_source; }
    
    // Get specific receptor
    Receptor getReceptor(int index) const {
        if (index >= 0 && index < num_receptors) {
            return receptors[index];
        }
        return {0.0f, 0.0f, 0.0f};
    }
    
    // Get source emission concentration for specific time step
    float getSourceEmission(int time_step) const {
        if (time_step >= 0 && time_step < source_emission.num_time_steps) {
            return source_emission.time_series[time_step];
        }
        return 0.0f;
    }
    
    // Get prior source value for specific time step
    float getPriorSourceValue(int time_step) const {
        if (time_step >= 0 && time_step < prior_source.num_time_steps) {
            return prior_source.prior_values[time_step];
        }
        return 0.0f;
    }
    
    // Get current time step based on simulation time (15-minute intervals)
    int getCurrentTimeStep(float current_time_seconds) const {
        int time_step = (int)(current_time_seconds / 900.0f);  // 900 seconds = 15 minutes
        return (time_step < source_emission.num_time_steps) ? time_step : source_emission.num_time_steps - 1;
    }
    
    // Get source position
    float getSourceLon() const { return prior_source.position[0]; }
    float getSourceLat() const { return prior_source.position[1]; }
    float getSourceAlt() const { return prior_source.position[2]; }
};

// Static member declaration (definition in ldm_eki.cu)
// EKIConfig* EKIConfig::instance = nullptr;

// Parse receptor data from line
std::vector<Receptor> parseReceptorPositions(const std::string& receptor_data);

// Parse source emission data from line  
SourceEmission parseSourceEmission(const std::string& source_line);

// Parse prior source data from line
PriorSource parsePriorSource(const std::string& prior_line);

// Helper function to extract array data from string
std::vector<float> extractFloatArray(const std::string& str);