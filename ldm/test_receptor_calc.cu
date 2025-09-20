#include "src/include/ldm_eki.cuh"
#include "src/include/ldm_eki_logger.cuh"

int main() {
    // Load EKI configuration
    EKIConfig* ekiConfig = EKIConfig::getInstance();
    std::string eki_config_file = "../eki/config/input_data";
    
    if (!ekiConfig->loadFromFile(eki_config_file)) {
        std::cerr << "[ERROR] Failed to load EKI configuration" << std::endl;
        return 1;
    }
    
    std::cout << "[INFO] Testing receptor concentration calculation..." << std::endl;
    
    // Test the prepareObservationData function
    prepareObservationData(ekiConfig);
    
    std::cout << "[INFO] Test completed." << std::endl;
    return 0;
}