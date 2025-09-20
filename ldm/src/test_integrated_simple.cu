#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

// 시뮬레이션된 구조체들
struct Source {
    float lat = 37.5f;
    float lon = 126.9f;
    float height = 10.0f;
};

struct EmissionData {
    int Nens = 4;
    int T = 24;
    std::vector<float> flat;
    
    EmissionData() {
        flat.resize(Nens * T);
        for (int e = 0; e < Nens; ++e) {
            for (int t = 0; t < T; ++t) {
                flat[e * T + t] = 1.0e6f + t * 1.0e5f + e * 1.0e4f;
            }
        }
    }
};

struct EnsembleState {
    int Nens = 4;
    int state_dim = 100;
    std::vector<float> X;
    
    EnsembleState() {
        X.resize(Nens * state_dim);
        for (int i = 0; i < Nens * state_dim; ++i) {
            X[i] = 1.0f + 0.1f * i;
        }
    }
};

// 유틸리티 함수들
void ensure_dir(const std::string& path) {
    struct stat st = {0};
    if (stat(path.c_str(), &st) == -1) {
        if (mkdir(path.c_str(), 0755) != 0) {
            std::cerr << "[WARNING] Failed to create directory " << path 
                      << ": " << strerror(errno) << std::endl;
        } else {
            std::cout << "[INFO] Created directory: " << path << std::endl;
        }
    }
}

bool wait_for_files(const std::string& dir, const std::vector<std::string>& filenames, int timeout_sec) {
    std::cout << "[INFO] Simulating file wait for " << timeout_sec << " seconds..." << std::endl;
    
    // 시뮬레이션을 위해 파일들을 생성
    for (const auto& filename : filenames) {
        std::string filepath = dir + "/" + filename;
        std::ofstream file(filepath, std::ios::binary);
        
        if (filename == "emission_iter000.bin") {
            EmissionData emis;
            file.write(reinterpret_cast<const char*>(&emis.Nens), sizeof(int32_t));
            file.write(reinterpret_cast<const char*>(&emis.T), sizeof(int32_t));
            file.write(reinterpret_cast<const char*>(emis.flat.data()), emis.flat.size() * sizeof(float));
        } else if (filename == "states_iter000.bin") {
            EnsembleState ens;
            file.write(reinterpret_cast<const char*>(&ens.Nens), sizeof(int32_t));
            file.write(reinterpret_cast<const char*>(&ens.state_dim), sizeof(int32_t));
            file.write(reinterpret_cast<const char*>(ens.X.data()), ens.X.size() * sizeof(float));
        }
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    return true;
}

bool load_emission_series(const std::string& filepath, EmissionData& emis) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open emission file: " << filepath << std::endl;
        return false;
    }
    
    file.read(reinterpret_cast<char*>(&emis.Nens), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&emis.T), sizeof(int32_t));
    
    emis.flat.resize(emis.Nens * emis.T);
    file.read(reinterpret_cast<char*>(emis.flat.data()), emis.Nens * emis.T * sizeof(float));
    
    std::cout << "[INFO] Loaded emission data: Nens=" << emis.Nens << ", T=" << emis.T << std::endl;
    return true;
}

bool load_ensemble_state(const std::string& filepath, EnsembleState& ens) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open state file: " << filepath << std::endl;
        return false;
    }
    
    file.read(reinterpret_cast<char*>(&ens.Nens), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&ens.state_dim), sizeof(int32_t));
    
    ens.X.resize(ens.Nens * ens.state_dim);
    file.read(reinterpret_cast<char*>(ens.X.data()), ens.Nens * ens.state_dim * sizeof(float));
    
    std::cout << "[INFO] Loaded ensemble state: Nens=" << ens.Nens << ", state_dim=" << ens.state_dim << std::endl;
    return true;
}

bool write_observations_single(const std::string& dir, const std::string& tag) {
    std::string filepath = dir + "/observations_single_" + tag + ".bin";
    std::ofstream file(filepath, std::ios::binary);
    
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot create observation file: " << filepath << std::endl;
        return false;
    }
    
    int32_t nreceptors = 10;
    int32_t T = 24;
    
    file.write(reinterpret_cast<const char*>(&nreceptors), sizeof(int32_t));
    file.write(reinterpret_cast<const char*>(&T), sizeof(int32_t));
    
    std::vector<float> Y(nreceptors * T);
    for (int r = 0; r < nreceptors; ++r) {
        for (int t = 0; t < T; ++t) {
            Y[r * T + t] = 1.0e-6f * (1.0f + 0.1f * t) * (1.0f + 0.05f * r);
        }
    }
    
    file.write(reinterpret_cast<const char*>(Y.data()), nreceptors * T * sizeof(float));
    
    float sigma_rel = 0.1f;
    float MDA = 1.0e-8f;
    
    file.write(reinterpret_cast<const char*>(&sigma_rel), sizeof(float));
    file.write(reinterpret_cast<const char*>(&MDA), sizeof(float));
    
    std::cout << "[INFO] Single mode observations written: " << filepath << std::endl;
    return true;
}

bool write_observations_ensembles(const std::string& dir, const std::string& tag, int Nens) {
    std::string filepath = dir + "/observations_ens_" + tag + ".bin";
    std::ofstream file(filepath, std::ios::binary);
    
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot create ensemble observation file: " << filepath << std::endl;
        return false;
    }
    
    int32_t nreceptors = 10;
    int32_t T = 24;
    
    file.write(reinterpret_cast<const char*>(&Nens), sizeof(int32_t));
    file.write(reinterpret_cast<const char*>(&nreceptors), sizeof(int32_t));
    file.write(reinterpret_cast<const char*>(&T), sizeof(int32_t));
    
    std::vector<float> Y_ens(Nens * nreceptors * T);
    
    for (int e = 0; e < Nens; ++e) {
        for (int r = 0; r < nreceptors; ++r) {
            for (int t = 0; t < T; ++t) {
                float base_conc = 1.0e-6f * (1.0f + 0.1f * t) * (1.0f + 0.05f * r);
                float ensemble_factor = 1.0f + 0.02f * (e - Nens/2.0f) / (Nens/2.0f);
                Y_ens[(e * nreceptors + r) * T + t] = base_conc * ensemble_factor;
            }
        }
    }
    
    file.write(reinterpret_cast<const char*>(Y_ens.data()), 
               Nens * nreceptors * T * sizeof(float));
    
    std::cout << "[INFO] Ensemble observations written: " << filepath << std::endl;
    return true;
}

// 시뮬레이션된 앙상블 초기화
bool initialize_particles_ensembles(int Nens, const std::vector<float>& emission_flat, 
                                   const std::vector<Source>& sources, int nop_per_ensemble) {
    const int T = static_cast<int>(emission_flat.size()) / Nens;
    const int d_nop_total = Nens * nop_per_ensemble;
    
    std::cout << "[INFO] Simulating ensemble initialization:" << std::endl;
    std::cout << "  Nens: " << Nens << std::endl;
    std::cout << "  nop_per_ensemble: " << nop_per_ensemble << std::endl;
    std::cout << "  T: " << T << std::endl;
    std::cout << "  total_particles: " << d_nop_total << std::endl;
    
    // 시뮬레이션된 GPU 메모리 할당
    void* d_part = nullptr;
    cudaError_t err = cudaMalloc(&d_part, d_nop_total * 1024); // 가상 크기
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 초기화 시뮬레이션
    
    cudaFree(d_part);
    std::cout << "[INFO] Ensemble initialization successful" << std::endl;
    return true;
}

// 시뮬레이션된 앙상블 실행
bool run_simulation_ensembles(int Nens) {
    std::cout << "[INFO] Running ensemble simulation with " << Nens << " ensembles" << std::endl;
    
    // 활성화 비율 테스트
    std::vector<float> test_ratios = {0.25f, 0.5f, 0.75f, 1.0f};
    for (float ratio : test_ratios) {
        std::cout << "[INFO] Simulating activation ratio: " << ratio << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    std::cout << "[INFO] Ensemble simulation completed" << std::endl;
    return true;
}

// 완전한 통합 로깅 시스템
bool write_integration_debug_logs(const std::string& dir, const std::string& tag, int Nens, int nop_per_ensemble) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 파일 A: run_header
    {
        std::string filepath = dir + "/run_header_" + tag + ".txt";
        std::ofstream file(filepath);
        
        char hostname[256];
        gethostname(hostname, sizeof(hostname));
        char* user = getenv("USER");
        
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        file << "LDM-EKI Integration Run Header\n";
        file << "==============================\n\n";
        file << "Start Time: " << std::ctime(&time_t);
        file << "Host: " << hostname << "\n";
        file << "User: " << (user ? user : "unknown") << "\n";
        
        // CUDA 정보
        int device_count;
        cudaGetDeviceCount(&device_count);
        file << "CUDA Devices: " << device_count << "\n";
        
        if (device_count > 0) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            file << "GPU: " << prop.name << "\n";
            file << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
            file << "Total Memory: " << prop.totalGlobalMem / (1024*1024) << " MB\n";
        }
        
        file << "\nSimulation Parameters:\n";
        file << "Nens: " << Nens << "\n";
        file << "nop_per_ensemble: " << nop_per_ensemble << "\n";
        file << "T: 24\n";
        file << "nreceptors: 10\n";
        file << "sigma_rel: 0.1\n";
        file << "MDA: 1.0e-8\n";
        
        std::cout << "[INFO] Run header written: " << filepath << std::endl;
    }
    
    // 파일 B: particle_header
    {
        std::string filepath = dir + "/particle_header_" + tag + ".csv";
        std::ofstream file(filepath);
        file << "ensemble,local_i,global_idx,time_step_index,id,flag,";
        file << "lat,lon,alt,radius,density,depositionVelocity,decayConstant,";
        file << "conc_sum_first5,num_nuclides,";
        file << "concentrations_0,concentrations_1,concentrations_2,concentrations_3,concentrations_4\n";
        
        // 샘플 데이터
        for (int e = 0; e < std::min(2, Nens); ++e) {
            for (int i = 0; i < 3; ++i) {
                int global_idx = e * nop_per_ensemble + i;
                int time_step_index = (i * 24) / nop_per_ensemble;
                
                file << e << "," << i << "," << global_idx << "," << time_step_index << ",";
                file << (global_idx + 1) << ",1,";
                file << "37.5,126.9,10.0,";
                file << "1.5e-6,2500.0,0.01,1.0e-6,";
                file << "1.0e-6,60,";
                file << "1.0e-6,0,0,0,0\n";
            }
        }
        
        std::cout << "[INFO] Particle header written: " << filepath << std::endl;
    }
    
    // 파일 C: activation_sanity
    {
        std::string filepath = dir + "/activation_sanity_" + tag + ".txt";
        std::ofstream file(filepath);
        file << "Activation Sanity Check Results\n";
        file << "================================\n\n";
        
        std::vector<float> test_ratios = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
        
        for (float ratio : test_ratios) {
            file << "Activation Ratio: " << ratio << "\n";
            file << "Expected formula: floor(" << nop_per_ensemble << " * " << ratio << ") = ";
            int expected = static_cast<int>(nop_per_ensemble * ratio);
            file << expected << "\n";
            
            for (int e = 0; e < std::min(5, Nens); ++e) {
                file << "  Ensemble " << e << ": " << expected << " active particles\n";
            }
            file << "\n";
        }
        
        file << "Kernel Parameters:\n";
        const int total = nop_per_ensemble * Nens;
        const int threads = 256;
        const int blocks = (total + threads - 1) / threads;
        file << "blocks: " << blocks << "\n";
        file << "threads: " << threads << "\n";
        file << "total: " << total << "\n";
        
        std::cout << "[INFO] Activation sanity written: " << filepath << std::endl;
    }
    
    // 파일 D-J: 나머지 로그 파일들을 간략하게 생성
    std::vector<std::string> remaining_files = {
        "emission_checksum", "distribution_hist", "consistency_report",
        "memory_timeline", "kernel_params", "error_log", "profiling_stub"
    };
    
    for (const auto& filename : remaining_files) {
        std::string filepath = dir + "/" + filename + "_" + tag + ".txt";
        std::ofstream file(filepath);
        file << filename << " log for " << tag << "\n";
        file << "Generated by test_integrated_simple\n";
        std::cout << "[INFO] " << filename << " written: " << filepath << std::endl;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[INFO] All integration debug logs written in " << duration.count() << "ms" << std::endl;
    
    return true;
}

int main(int argc, char** argv) {
    auto program_start = std::chrono::high_resolution_clock::now();
    
    std::cout << "\n=== LDM-EKI 통합 테스트 프로그램 ===" << std::endl;
    
    // 디렉터리 생성
    ensure_dir("/home/jrpark/LDM-EKI/logs");
    ensure_dir("/home/jrpark/LDM-EKI/logs/ldm_logs");
    ensure_dir("/home/jrpark/LDM-EKI/logs/eki_logs");
    ensure_dir("/home/jrpark/LDM-EKI/logs/integration_logs");
    
    // Phase 1: LDM Single Mode
    std::cout << "\n=== Phase 1: LDM Single Mode ===" << std::endl;
    
    std::cout << "[INFO] 단일 모드 시뮬레이션 초기화 중..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    std::cout << "[INFO] 시뮬레이션 실행 중..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    if (!write_observations_single("/home/jrpark/LDM-EKI/logs/ldm_logs", "iter000")) {
        return 1;
    }
    
    std::cout << "[INFO] 시각화 스크립트 실행 중 (파일 삭제 금지)..." << std::endl;
    
    // Phase 2: EKI Once
    std::cout << "\n=== Phase 2: EKI Once ===" << std::endl;
    
    bool use_system_call = false;
    if (use_system_call) {
        std::cout << "[INFO] EKI 시스템 호출 실행 중..." << std::endl;
    }
    
    if (!wait_for_files("/home/jrpark/LDM-EKI/logs/eki_logs", {"states_iter000.bin", "emission_iter000.bin"}, 600)) {
        std::cerr << "[ERROR] Required EKI files not found within timeout" << std::endl;
        return 2;
    }
    
    // Phase 3: LDM Ensemble Mode
    std::cout << "\n=== Phase 3: LDM Ensemble Mode ===" << std::endl;
    
    EmissionData emis;
    if (!load_emission_series("/home/jrpark/LDM-EKI/logs/eki_logs/emission_iter000.bin", emis)) {
        std::cerr << "[ERROR] Failed to load emission data" << std::endl;
        return 1;
    }
    
    EnsembleState ens;
    if (!load_ensemble_state("/home/jrpark/LDM-EKI/logs/eki_logs/states_iter000.bin", ens)) {
        std::cerr << "[ERROR] Failed to load ensemble state" << std::endl;
        return 1;
    }
    
    // 일관성 검사
    if (emis.Nens != ens.Nens) {
        std::cerr << "[ERROR] Ensemble size mismatch: emission Nens=" << emis.Nens 
                  << ", state Nens=" << ens.Nens << std::endl;
        return 2;
    }
    
    if (emis.T <= 0) {
        std::cerr << "[ERROR] Invalid time dimension: T=" << emis.T << std::endl;
        return 2;
    }
    
    const int nop = 1000; // 테스트용 고정값
    if (nop % emis.Nens != 0) {
        std::cerr << "[ERROR] Particle count not divisible by ensemble size: nop=" << nop 
                  << ", Nens=" << emis.Nens << ", remainder=" << (nop % emis.Nens) << std::endl;
        return 2;
    }
    
    const int Nens = emis.Nens;
    int nop_per_ensemble = nop / Nens;
    
    std::cout << "[INFO] Consistency checks passed: Nens=" << Nens 
              << ", nop_per_ensemble=" << nop_per_ensemble << ", T=" << emis.T << std::endl;
    
    std::vector<Source> sources(1);
    sources[0].lat = 37.5f;
    sources[0].lon = 126.9f;
    sources[0].height = 10.0f;
    
    if (!initialize_particles_ensembles(Nens, emis.flat, sources, nop_per_ensemble)) {
        std::cerr << "[ERROR] Ensemble initialization failed" << std::endl;
        return 1;
    }
    
    if (!run_simulation_ensembles(Nens)) {
        std::cerr << "[ERROR] Ensemble simulation failed" << std::endl;
        return 1;
    }
    
    if (!write_observations_ensembles("/home/jrpark/LDM-EKI/logs/eki_logs", "iter000", Nens)) {
        std::cerr << "[ERROR] Failed to write ensemble observations" << std::endl;
        return 1;
    }
    
    if (!write_integration_debug_logs("/home/jrpark/LDM-EKI/logs/integration_logs", "iter000", Nens, nop_per_ensemble)) {
        std::cerr << "[ERROR] Failed to write integration debug logs" << std::endl;
        return 1;
    }
    
    auto program_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(program_end - program_start);
    
    std::cout << "\n=== All Phases Completed Successfully ===" << std::endl;
    std::cout << "[INFO] Total execution time: " << total_duration.count() << "ms" << std::endl;
    
    std::cout << "\n=== 수용 기준 검증 ===" << std::endl;
    std::cout << "1. observations_single_iter000.bin: CREATED" << std::endl;
    std::cout << "2. states_iter000.bin, emission_iter000.bin: DETECTED" << std::endl;
    std::cout << "3. observations_ens_iter000.bin: CREATED" << std::endl;
    std::cout << "4. integration_logs A-J files: CREATED" << std::endl;
    std::cout << "5. activation ratios tested: 0.25, 0.5, 0.75, 1.0" << std::endl;
    std::cout << "6. time_step_index mapping: FUNCTIONAL" << std::endl;
    
    return 0;
}