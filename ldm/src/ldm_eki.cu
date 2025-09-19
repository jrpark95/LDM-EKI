#include "ldm_eki.cuh"

// Initialize static member
EKIConfig* EKIConfig::instance = nullptr;

// Helper function to safely parse value from line
std::string parseValue(const std::string& line, const std::string& key) {
    std::size_t pos = line.find(":");
    if (pos != std::string::npos) {
        std::string value_str = line.substr(pos + 1);
        value_str.erase(0, value_str.find_first_not_of(" \t"));
        value_str.erase(value_str.find_last_not_of(" \t") + 1);
        // Remove comment part
        std::size_t comment_pos = value_str.find('#');
        if (comment_pos != std::string::npos) {
            value_str = value_str.substr(0, comment_pos);
            value_str.erase(value_str.find_last_not_of(" \t") + 1);
        }
        return value_str;
    }
    return "";
}

// Safe conversion functions
float safeStof(const std::string& str, float defaultValue = 0.0f) {
    try {
        if (!str.empty()) {
            return std::stof(str);
        }
    } catch (const std::exception& e) {
        std::cout << "[DEBUG] Failed to parse float: '" << str << "'" << std::endl;
    }
    return defaultValue;
}

int safeStoi(const std::string& str, int defaultValue = 0) {
    try {
        if (!str.empty()) {
            return std::stoi(str);
        }
    } catch (const std::exception& e) {
        std::cout << "[DEBUG] Failed to parse int: '" << str << "'" << std::endl;
    }
    return defaultValue;
}

bool EKIConfig::loadFromFile(const std::string& filename) {
    std::cout << "[DEBUG] Loading EKI config from: " << filename << std::endl;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open EKI config file: " << filename << std::endl;
        return false;
    }
    
    std::string line;
    bool reading_receptor_positions = false;
    std::string receptor_data = "";
    
    while (std::getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') continue;
        
        // Remove leading/trailing whitespace
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);
        
        // Parse configuration parameters using safe functions
        if (line.find("time:") != std::string::npos && line.find("time_interval:") == std::string::npos) {
            time_day = safeStof(parseValue(line, "time"));
        }
        else if (line.find("time_interval:") != std::string::npos && line.find("inverse_time_interval:") == std::string::npos) {
            time_interval_min = safeStoi(parseValue(line, "time_interval"));
        }
        else if (line.find("inverse_time_interval:") != std::string::npos) {
            inverse_time_interval_hour = safeStof(parseValue(line, "inverse_time_interval"));
        }
        else if (line.find("ave_t:") != std::string::npos) {
            ave_t = safeStoi(parseValue(line, "ave_t"));
        }
        else if (line.find("size_alt :") != std::string::npos) {
            size_alt = safeStoi(parseValue(line, "size_alt"));
        }
        else if (line.find("size_lat :") != std::string::npos) {
            size_lat = safeStoi(parseValue(line, "size_lat"));
        }
        else if (line.find("size_lon :") != std::string::npos) {
            size_lon = safeStoi(parseValue(line, "size_lon"));
        }
        else if (line.find("alt_spacing :") != std::string::npos) {
            alt_spacing = safeStof(parseValue(line, "alt_spacing"));
        }
        else if (line.find("lat_spacing :") != std::string::npos) {
            lat_spacing = safeStof(parseValue(line, "lat_spacing"));
        }
        else if (line.find("lon_spacing :") != std::string::npos) {
            lon_spacing = safeStof(parseValue(line, "lon_spacing"));
        }
        else if (line.find("size_alt_wind:") != std::string::npos) {
            size_alt_wind = safeStoi(parseValue(line, "size_alt_wind"));
        }
        else if (line.find("size_lat_wind:") != std::string::npos) {
            size_lat_wind = safeStoi(parseValue(line, "size_lat_wind"));
        }
        else if (line.find("size_lon_wind:") != std::string::npos) {
            size_lon_wind = safeStoi(parseValue(line, "size_lon_wind"));
        }
        else if (line.find("wind_init_mode:") != std::string::npos) {
            wind_init_mode = parseValue(line, "wind_init_mode");
        }
        else if (line.find("wind_constant_value_x:") != std::string::npos) {
            wind_constant_value_x = safeStof(parseValue(line, "wind_constant_value_x"));
        }
        else if (line.find("wind_constant_value_y:") != std::string::npos) {
            wind_constant_value_y = safeStof(parseValue(line, "wind_constant_value_y"));
        }
        else if (line.find("wind_constant_value_z:") != std::string::npos) {
            wind_constant_value_z = safeStof(parseValue(line, "wind_constant_value_z"));
        }
        else if (line.find("wind_grid_interval:") != std::string::npos) {
            wind_grid_interval = safeStoi(parseValue(line, "wind_grid_interval"));
        }
        else if (line.find("grid_space_size_lat:") != std::string::npos) {
            grid_space_size_lat = safeStoi(parseValue(line, "grid_space_size_lat"));
        }
        else if (line.find("grid_space_size_lon:") != std::string::npos) {
            grid_space_size_lon = safeStoi(parseValue(line, "grid_space_size_lon"));
        }
        else if (line.find("puff_concentration_threshold:") != std::string::npos) {
            puff_concentration_threshold = safeStof(parseValue(line, "puff_concentration_threshold"));
        }
        else if (line.find("R_max:") != std::string::npos) {
            R_max = safeStof(parseValue(line, "R_max"));
        }
        else if (line.find("nreceptor_err :") != std::string::npos) {
            nreceptor_err = safeStof(parseValue(line, "nreceptor_err"));
        }
        else if (line.find("nreceptor_MDA :") != std::string::npos) {
            nreceptor_MDA = safeStof(parseValue(line, "nreceptor_MDA"));
        }
        else if (line.find("Source_location:") != std::string::npos) {
            source_location = parseValue(line, "Source_location");
            // Remove quotes
            source_location.erase(0, source_location.find_first_not_of("'\""));
            source_location.erase(source_location.find_last_not_of("'\"") + 1);
        }
        else if (line.find("nsource :") != std::string::npos) {
            nsource = safeStoi(parseValue(line, "nsource"));
        }
        else if (line.find("source_name :") != std::string::npos) {
            std::size_t start_pos = line.find("[");
            std::size_t end_pos = line.find("]");
            if (start_pos != std::string::npos && end_pos != std::string::npos) {
                std::string names_str = line.substr(start_pos + 1, end_pos - start_pos - 1);
                std::istringstream iss(names_str);
                std::string name;
                while (std::getline(iss, name, ',')) {
                    name.erase(0, name.find_first_not_of(" \t'\""));
                    name.erase(name.find_last_not_of(" \t'\"") + 1);
                    if (!name.empty()) {
                        source_names.push_back(name);
                    }
                }
            }
        }
        else if (line.find("prior_source1 :") != std::string::npos) {
            std::size_t start_pos = line.find("[");
            std::size_t end_pos = line.find("]");
            if (start_pos != std::string::npos && end_pos != std::string::npos) {
                std::string values_str = line.substr(start_pos + 1, end_pos - start_pos - 1);
                std::vector<float> values = extractFloatArray(values_str);
                if (values.size() >= 3) {
                    prior_source1_concentration = values[0];
                    prior_source1_std = values[1];
                    prior_source1_decay_constant = values[2];
                }
            }
        }
        else if (line.find("real_source1_boundary :") != std::string::npos) {
            std::size_t start_pos = line.find("[");
            std::size_t end_pos = line.find("]");
            if (start_pos != std::string::npos && end_pos != std::string::npos) {
                std::string values_str = line.substr(start_pos + 1, end_pos - start_pos - 1);
                std::vector<float> values = extractFloatArray(values_str);
                if (values.size() >= 2) {
                    real_source1_boundary_min = values[0];
                    real_source1_boundary_max = values[1];
                }
            }
        }
        
        // Parse number of receptors
        else if (line.find("nreceptor :") != std::string::npos) {
            num_receptors = safeStoi(parseValue(line, "nreceptor"));
            std::cout << "[DEBUG] Found " << num_receptors << " receptors" << std::endl;
        }
        
        // Start reading receptor positions
        else if (line.find("receptor_position:") != std::string::npos) {
            reading_receptor_positions = true;
            receptor_data = "";
        }
        
        // End reading receptor positions
        else if (reading_receptor_positions && line.find("]") != std::string::npos && 
                 line.find("[") == std::string::npos) {
            reading_receptor_positions = false;
            receptors = parseReceptorPositions(receptor_data);
            std::cout << "[DEBUG] Parsed " << receptors.size() << " receptor positions" << std::endl;
        }
        
        // Collect receptor position data
        else if (reading_receptor_positions) {
            receptor_data += line + "\n";
        }
        
        // Parse Source_1 emission data (but not Prior_Source_1)
        else if (line.find("Source_1:") != std::string::npos && line.find("Prior_Source_1:") == std::string::npos) {
            // Read multiple lines for source emission data
            std::string source_data = line;
            std::string next_line;
            while (std::getline(file, next_line)) {
                source_data += "\n" + next_line;
                if (next_line.find("'Co-60']") != std::string::npos) {
                    break;
                }
            }
            source_emission = parseSourceEmission(source_data);
            std::cout << "[DEBUG] Parsed source emission with " << source_emission.num_time_steps << " time steps" << std::endl;
        }
        
        // Parse Prior_Source_1 data
        else if (line.find("Prior_Source_1:") != std::string::npos) {
            // Read multiple lines for prior source data
            std::string prior_data = line;
            std::string next_line;
            while (std::getline(file, next_line)) {
                prior_data += "\n" + next_line;
                if (next_line.find("'Co-60']") != std::string::npos) {
                    break;
                }
            }
            prior_source = parsePriorSource(prior_data);
            std::cout << "[DEBUG] Parsed prior source with " << prior_source.num_time_steps << " time steps" << std::endl;
        }
    }
    
    file.close();
    return true;
}

std::vector<Receptor> parseReceptorPositions(const std::string& receptor_data) {
    std::vector<Receptor> receptors;
    
    // Find all receptor position arrays [lat, lon, alt]
    std::regex receptor_regex(R"(\[\s*([0-9.-]+)\s*,\s*([0-9.-]+)\s*,\s*([0-9.-]+)\s*\])");
    std::sregex_iterator iter(receptor_data.begin(), receptor_data.end(), receptor_regex);
    std::sregex_iterator end;
    
    for (; iter != end; ++iter) {
        std::smatch match = *iter;
        Receptor r;
        r.lat = std::stof(match[1].str());
        r.lon = std::stof(match[2].str());
        r.alt = std::stof(match[3].str());
        receptors.push_back(r);
        std::cout << "[DEBUG] Receptor: lat=" << r.lat << ", lon=" << r.lon << ", alt=" << r.alt << std::endl;
    }
    
    return receptors;
}

SourceEmission parseSourceEmission(const std::string& source_data) {
    SourceEmission emission;
    emission.num_time_steps = 0;
    
    std::cout << "[DEBUG] Parsing source data: " << source_data.substr(0, 100) << "..." << std::endl;
    
    // Find the concentration array which should be after the position array
    // Look for the pattern "], [" to find where position array ends and concentration array starts
    std::size_t pos_end = source_data.find("], [");
    if (pos_end != std::string::npos) {
        // Find the start of concentration array (skip "], [")
        std::size_t conc_start = pos_end + 4; 
        
        // Find the end of concentration array by looking for the end pattern
        std::size_t conc_end = source_data.find("  ], 0.0e-0, 0.0e-0, 'Co-60'");
        if (conc_end == std::string::npos) {
            conc_end = source_data.find("], 0.0e-0, 0.0e-0, 'Co-60'");
        }
        
        if (conc_end != std::string::npos && conc_end > conc_start) {
            std::string array_data = source_data.substr(conc_start, conc_end - conc_start);
            std::cout << "[DEBUG] Extracted array data length: " << array_data.length() << std::endl;
            std::cout << "[DEBUG] First 300 chars: " << array_data.substr(0, 300) << std::endl;
            
            emission.time_series = extractFloatArray(array_data);
            emission.num_time_steps = emission.time_series.size();
            
            std::cout << "[DEBUG] Source emission values (" << emission.num_time_steps << " total):" << std::endl;
            for (int i = 0; i < emission.num_time_steps && i < 5; i++) {
                std::cout << "[DEBUG]   Time step " << i << ": " << emission.time_series[i] << std::endl;
            }
            if (emission.num_time_steps > 5) {
                std::cout << "[DEBUG]   ... and " << (emission.num_time_steps - 5) << " more values" << std::endl;
            }
        } else {
            std::cout << "[DEBUG] Could not find end of concentration array" << std::endl;
        }
    } else {
        std::cout << "[DEBUG] Could not find position array end marker '], ['" << std::endl;
    }
    
    return emission;
}

PriorSource parsePriorSource(const std::string& prior_data) {
    PriorSource prior;
    prior.num_time_steps = 0;
    
    // Extract decay constant and uncertainty
    std::regex decay_regex(R"(Prior_Source_1:\s*\[\s*([0-9.e-]+)\s*,\s*([0-9.e-]+))");
    std::smatch match;
    if (std::regex_search(prior_data, match, decay_regex)) {
        prior.decay_constant = std::stof(match[1].str());
        prior.uncertainty = std::stof(match[2].str());
    }
    
    // Extract position [lon, lat, alt] and std
    std::regex pos_regex(R"(\[\[\s*([0-9.-]+)\s*,\s*([0-9.-]+)\s*,\s*([0-9.-]+)\s*\]\s*,\s*\[\s*([0-9.-]+)\s*\]\])");
    if (std::regex_search(prior_data, match, pos_regex)) {
        prior.position[0] = std::stof(match[1].str()); // lon
        prior.position[1] = std::stof(match[2].str()); // lat  
        prior.position[2] = std::stof(match[3].str()); // alt
        prior.position_std = std::stof(match[4].str());
    }
    
    // Extract prior values array
    std::cout << "[DEBUG] Parsing prior data: " << prior_data.substr(0, 150) << "..." << std::endl;
    
    // Look for the pattern "[[" followed by the actual values array  
    std::size_t values_start = prior_data.find("[[", prior_data.find("Prior_Source_1:"));
    if (values_start != std::string::npos) {
        values_start = prior_data.find("[[", values_start + 1); // Find second [[
        if (values_start != std::string::npos) {
            // Look for the end pattern more accurately
            std::size_t values_end = prior_data.find("  ],[0.1]], 'Co-60'");
            if (values_end == std::string::npos) {
                values_end = prior_data.find("],[0.1]], 'Co-60'");
            }
            
            if (values_end != std::string::npos) {
                std::string values_data = prior_data.substr(values_start + 2, values_end - values_start - 2);
                std::cout << "[DEBUG] Prior values data length: " << values_data.length() << std::endl;
                std::cout << "[DEBUG] First 300 chars: " << values_data.substr(0, 300) << std::endl;
                
                prior.prior_values = extractFloatArray(values_data);
                prior.num_time_steps = prior.prior_values.size();
                
                std::cout << "[DEBUG] Prior values (" << prior.num_time_steps << " total):" << std::endl;
                for (int i = 0; i < prior.num_time_steps && i < 5; i++) {
                    std::cout << "[DEBUG]   Time step " << i << ": " << prior.prior_values[i] << std::endl;
                }
                if (prior.num_time_steps > 5) {
                    std::cout << "[DEBUG]   ... and " << (prior.num_time_steps - 5) << " more values" << std::endl;
                }
            } else {
                std::cout << "[DEBUG] Could not find end of prior values array" << std::endl;
            }
        } else {
            std::cout << "[DEBUG] Could not find second [[ for prior values" << std::endl;
        }
    } else {
        std::cout << "[DEBUG] Could not find first [[ for prior values" << std::endl;
    }
    
    // Extract value std
    std::regex std_regex(R"(\],\[\s*([0-9.-]+)\s*\]\])");
    if (std::regex_search(prior_data, match, std_regex)) {
        prior.value_std = std::stof(match[1].str());
    }
    
    // Extract nuclide name
    std::regex nuclide_regex(R"('([^']+)'\s*\])");
    if (std::regex_search(prior_data, match, nuclide_regex)) {
        prior.nuclide_name = match[1].str();
    }
    
    std::cout << "[DEBUG] Prior source: decay=" << prior.decay_constant 
              << ", pos=[" << prior.position[0] << "," << prior.position[1] << "," << prior.position[2] << "]"
              << ", nuclide=" << prior.nuclide_name << std::endl;
    
    return prior;
}

std::vector<float> extractFloatArray(const std::string& str) {
    std::vector<float> values;
    
    std::cout << "[DEBUG] extractFloatArray input: " << str.substr(0, 500) << "..." << std::endl;
    
    // Remove comments and split by comma or newline
    std::istringstream iss(str);
    std::string line;
    
    // Process line by line to handle multi-line arrays
    while (std::getline(iss, line)) {
        std::istringstream line_stream(line);
        std::string token;
        
        while (std::getline(line_stream, token, ',')) {
            // Remove comments (everything after #)
            std::size_t comment_pos = token.find('#');
            if (comment_pos != std::string::npos) {
                token = token.substr(0, comment_pos);
            }
            
            // Remove whitespace and brackets
            token.erase(0, token.find_first_not_of(" \t\n\r[]"));
            token.erase(token.find_last_not_of(" \t\n\r[]") + 1);
            
            // Convert to float if not empty
            if (!token.empty()) {
                try {
                    float value = std::stof(token);
                    values.push_back(value);
                    std::cout << "[DEBUG] Extracted value: " << value << std::endl;
                } catch (const std::exception& e) {
                    std::cout << "[DEBUG] Failed to parse: '" << token << "'" << std::endl;
                }
            }
        }
    }
    
    std::cout << "[DEBUG] extractFloatArray extracted " << values.size() << " values" << std::endl;
    return values;
}