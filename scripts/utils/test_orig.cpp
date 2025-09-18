#include "ldm.cuh"
#include <iostream>

int main() {
    std::cout << "Testing original weather data reading..." << std::endl;
    
    LDM ldm;
    
    // Test loading height data
    std::cout << "\n=== Testing Height Data Loading (Original) ===" << std::endl;
    ldm.loadFlexHeightData();
    
    // Test loading FLEX data  
    std::cout << "\n=== Testing FLEX Data Loading (Original) ===" << std::endl;
    ldm.initializeFlexGFSData();
    
    std::cout << "\nOriginal weather data reading test completed!" << std::endl;
    return 0;
}