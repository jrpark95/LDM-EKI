#include <cuda_runtime.h>
#include <iostream>

void printCudaError(cudaError_t error, const char* operation) {
    std::cout << operation << " failed with error: " << cudaGetErrorString(error) 
              << " (code " << error << ")" << std::endl;
}

int main() {
    std::cout << "=== CUDA Diagnosis ===" << std::endl;
    
    // 1. Get device count
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        printCudaError(err, "cudaGetDeviceCount");
        return 1;
    }
    std::cout << "Device count: " << deviceCount << std::endl;
    
    // 2. Try to get current device
    int currentDevice;
    err = cudaGetDevice(&currentDevice);
    if (err != cudaSuccess) {
        printCudaError(err, "cudaGetDevice");
    } else {
        std::cout << "Current device: " << currentDevice << std::endl;
    }
    
    // 3. Try to set device 0
    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        printCudaError(err, "cudaSetDevice(0)");
        
        // Try other devices
        for (int i = 1; i < deviceCount && i < 8; i++) {
            err = cudaSetDevice(i);
            if (err == cudaSuccess) {
                std::cout << "Successfully set device " << i << std::endl;
                break;
            } else {
                printCudaError(err, ("cudaSetDevice(" + std::to_string(i) + ")").c_str());
            }
        }
        
        if (err != cudaSuccess) {
            std::cout << "Failed to set any device" << std::endl;
            return 1;
        }
    }
    
    // 4. Get device properties
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        printCudaError(err, "cudaGetDeviceProperties");
    } else {
        std::cout << "Device 0: " << prop.name << std::endl;
        std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Total memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    }
    
    // 5. Try minimal memory allocation
    void* d_ptr = nullptr;
    err = cudaMalloc(&d_ptr, 4);
    if (err != cudaSuccess) {
        printCudaError(err, "cudaMalloc(4 bytes)");
        return 1;
    }
    std::cout << "Successfully allocated 4 bytes at " << d_ptr << std::endl;
    
    // 6. Try to free
    err = cudaFree(d_ptr);
    if (err != cudaSuccess) {
        printCudaError(err, "cudaFree");
        return 1;
    }
    std::cout << "Successfully freed memory" << std::endl;
    
    std::cout << "=== CUDA Test PASSED ===" << std::endl;
    return 0;
}
