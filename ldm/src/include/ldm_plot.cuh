#include "ldm.cuh"
#include "ldm_nuclides.cuh"

int LDM::countActiveParticles(){
    int count = 0;
    for(int i = 0; i < nop; ++i) if(part[i].flag == 1) count++;
    return count;
}

void LDM::swapByteOrder(float& value){
    char* valuePtr = reinterpret_cast<char*>(&value);
    std::swap(valuePtr[0], valuePtr[3]);
    std::swap(valuePtr[1], valuePtr[2]);
}

void LDM::outputParticlesBinaryMPI(int timestep){

    cudaMemcpy(part.data(), d_part, nop * sizeof(LDMpart), cudaMemcpyDeviceToHost);
    int numa = 0;
    int numb = 0;

    int part_num = countActiveParticles();

    // Debug output disabled for release

    std::ostringstream filenameStream;
    std::string path;

    // Create output directory for VTK files
    path = "output";

    #ifdef _WIN32
        _mkdir(path.c_str());
        filenameStream << path << "\\" << "plot_" << std::setfill('0') 
                       << std::setw(5) << timestep << ".vtk";
    #else
        mkdir(path.c_str(), 0777);
        filenameStream << path << "/" << "plot_" << std::setfill('0') 
                       << std::setw(5) << timestep << ".vtk";
    #endif
    std::string filename = filenameStream.str();

    std::ofstream vtkFile(filename, std::ios::binary);

    if (!vtkFile.is_open()){
        std::cerr << "Cannot open file for writing: " << filename << std::endl;
        return;
    }

    vtkFile << "# vtk DataFile Version 4.0\n";
    vtkFile << "particle data\n";
    vtkFile << "BINARY\n";
    vtkFile << "DATASET POLYDATA\n";

    vtkFile << "POINTS " << part_num << " float\n";
    float zsum = 0.0;
    for (int i = 0; i < nop; ++i){
        if(!part[i].flag) continue;
        // float x = part[i].x;
        // float y = part[i].y;
        // float z = part[i].z/3000.0;

        float x = -179.0 + part[i].x*0.5;
        float y = -90.0 + part[i].y*0.5;
        float z = part[i].z/3000.0;
        zsum += part[i].z;

        swapByteOrder(x);
        swapByteOrder(y);
        swapByteOrder(z);

        vtkFile.write(reinterpret_cast<char*>(&x), sizeof(float));
        vtkFile.write(reinterpret_cast<char*>(&y), sizeof(float));
        vtkFile.write(reinterpret_cast<char*>(&z), sizeof(float));
    }
    // __Z[timestep-1]=zsum/static_cast<float>(part_num);


    vtkFile << "POINT_DATA " << part_num << "\n";
    vtkFile << "SCALARS u_wind float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < nop; ++i){
        if(!part[i].flag) continue;
        float vval = part[i].u_wind;
        swapByteOrder(vval); 
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile << "SCALARS v_wind float 1\n"; 
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < nop; ++i){
        if(!part[i].flag) continue;
        float vval = part[i].v_wind;
        swapByteOrder(vval); 
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile << "SCALARS w_wind float 1\n"; 
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < nop; ++i){
        if(!part[i].flag) continue;
        float vval = part[i].w_wind;
        swapByteOrder(vval); 
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile << "SCALARS virtual_dist float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < nop; ++i){
        if(!part[i].flag) continue;
        float vval = part[i].virtual_distance;
        swapByteOrder(vval); 
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile << "SCALARS Q float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < nop; ++i){
        if(!part[i].flag) continue;
        float vval = part[i].conc;
        swapByteOrder(vval); 
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    // // Output representative nuclide concentration (I-131, index 32)
    // const int representative_nuclide = 32;  // I-131 is at index 32 in the 60-nuclide configuration
    // vtkFile << "SCALARS I131_concentration float 1\n";
    // vtkFile << "LOOKUP_TABLE default\n";
    // for (int i = 0; i < nop; ++i){
    //     if(!part[i].flag) continue;
    //     float vval = part[i].concentrations[representative_nuclide];
    //     swapByteOrder(vval); 
    //     vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    // }


    vtkFile.close();
}

// New function for all ensembles VTK output in one file
void LDM::outputEnsembleParticlesBinaryMPI(int timestep, int ensemble_id){
    
    cudaMemcpy(part.data(), d_part, nop * sizeof(LDMpart), cudaMemcpyDeviceToHost);
    
    // Count all active particles from all ensembles
    int part_num = 0;
    for (int i = 0; i < nop; ++i) {
        if (part[i].flag == 1) {
            part_num++;
        }
    }
    
    if (part_num == 0) {
        std::cout << "[VTK] No active particles for any ensemble at timestep " << timestep << std::endl;
        return;
    }

    std::ostringstream filenameStream;
    std::string path;

    // Create ensemble output directory for VTK files
    path = "output_ens";

    #ifdef _WIN32
        _mkdir(path.c_str());
        filenameStream << path << "\\" << "all_ensembles_plot_" << std::setfill('0') << std::setw(5) << timestep << ".vtk";
    #else
        mkdir(path.c_str(), 0777);
        filenameStream << path << "/" << "all_ensembles_plot_" << std::setfill('0') << std::setw(5) << timestep << ".vtk";
    #endif
    std::string filename = filenameStream.str();

    std::ofstream vtkFile(filename, std::ios::binary);

    if (!vtkFile.is_open()){
        std::cerr << "Cannot open file for writing: " << filename << std::endl;
        return;
    }

    vtkFile << "# vtk DataFile Version 4.0\n";
    vtkFile << "all ensembles particle data\n";
    vtkFile << "BINARY\n";
    vtkFile << "DATASET POLYDATA\n";

    vtkFile << "POINTS " << part_num << " float\n";
    float zsum = 0.0;
    for (int i = 0; i < nop; ++i){
        if(!part[i].flag) continue;
        
        float x = -179.0 + part[i].x*0.5;
        float y = -90.0 + part[i].y*0.5;
        float z = part[i].z/3000.0;
        zsum += part[i].z;

        swapByteOrder(x);
        swapByteOrder(y);
        swapByteOrder(z);

        vtkFile.write(reinterpret_cast<char*>(&x), sizeof(float));
        vtkFile.write(reinterpret_cast<char*>(&y), sizeof(float));
        vtkFile.write(reinterpret_cast<char*>(&z), sizeof(float));
    }

    vtkFile << "POINT_DATA " << part_num << "\n";
    
    // Ensemble ID as scalar field to distinguish different ensembles
    vtkFile << "SCALARS ensemble_id int 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < nop; ++i){
        if(!part[i].flag) continue;
        int eid = part[i].ensemble_id;
        // Convert to big endian for VTK
        char* ptr = reinterpret_cast<char*>(&eid);
        std::swap(ptr[0], ptr[3]);
        std::swap(ptr[1], ptr[2]);
        vtkFile.write(reinterpret_cast<char*>(&eid), sizeof(int));
    }
    
    vtkFile << "SCALARS u_wind float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < nop; ++i){
        if(!part[i].flag) continue;
        float vval = part[i].u_wind;
        swapByteOrder(vval); 
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile << "SCALARS v_wind float 1\n"; 
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < nop; ++i){
        if(!part[i].flag) continue;
        float vval = part[i].v_wind;
        swapByteOrder(vval); 
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile << "SCALARS w_wind float 1\n"; 
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < nop; ++i){
        if(!part[i].flag) continue;
        float vval = part[i].w_wind;
        swapByteOrder(vval); 
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile << "SCALARS virtual_dist float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < nop; ++i){
        if(!part[i].flag) continue;
        float vval = part[i].virtual_distance;
        swapByteOrder(vval); 
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile << "SCALARS Q float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < nop; ++i){
        if(!part[i].flag) continue;
        float vval = part[i].conc;
        swapByteOrder(vval); 
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile.close();
    
    std::cout << "[VTK] All ensembles VTK output: " << part_num << " particles at timestep " << timestep << std::endl;
}

// Log first particle's concentrations over time
void LDM::log_first_particle_concentrations(int timestep, float currentTime) {
    // Copy particles from device to host
    cudaMemcpy(part.data(), d_part, nop * sizeof(LDMpart), cudaMemcpyDeviceToHost);
    
    // Ensure validation directory exists
    #ifdef _WIN32
        _mkdir("validation");
    #else
        mkdir("validation", 0777);
    #endif
    
    // Create or append to CSV file
    static bool first_write = true;
    std::string filename = "validation/first_particle_concentrations.csv";
    std::ofstream csvFile;
    
    if (first_write) {
        csvFile.open(filename, std::ios::out);
        // Write header with dynamic nuclide names
        csvFile << "timestep,time(s),total_conc";
        NuclideConfig* nucConfig = NuclideConfig::getInstance();
        int num_nuclides = nucConfig->getNumNuclides();
        for (int i = 0; i < num_nuclides; i++) {
            csvFile << "," << nucConfig->getNuclideName(i);
        }
        csvFile << std::endl;
        first_write = false;
    } else {
        csvFile.open(filename, std::ios::app);
    }
    
    // Find first active particle
    bool found_active = false;
    for (size_t idx = 0; idx < part.size(); idx++) {
        const auto& p = part[idx];
        if (p.flag) {
            found_active = true;
            NuclideConfig* nucConfig = NuclideConfig::getInstance();
            int num_nuclides = nucConfig->getNumNuclides();
            csvFile << timestep << "," << currentTime << "," << p.conc;
            for (int i = 0; i < num_nuclides; i++) {
                csvFile << "," << p.concentrations[i];
            }
            csvFile << std::endl;
            break; // Only log the first active particle
        }
    }
    if (!found_active) {
        if (!part.empty()) {
            const auto& p = part[0];
            NuclideConfig* nucConfig = NuclideConfig::getInstance();
            int num_nuclides = nucConfig->getNumNuclides();
            csvFile << timestep << "," << currentTime << "," << p.conc;
            for (int i = 0; i < num_nuclides; i++) {
                csvFile << "," << p.concentrations[i];
            }
            csvFile << std::endl;
        }
    }
    
    csvFile.close();
}

// Log all particles' nuclide ratios over time
void LDM::log_all_particles_nuclide_ratios(int timestep, float currentTime) {
    // Copy particles from device to host
    cudaMemcpy(part.data(), d_part, nop * sizeof(LDMpart), cudaMemcpyDeviceToHost);
    
    // Ensure validation directory exists
    #ifdef _WIN32
        _mkdir("validation");
    #else
        mkdir("validation", 0777);
    #endif
    
    // Create or append to CSV file
    static bool first_write = true;
    std::string filename = "validation/all_particles_nuclide_ratios.csv";
    std::ofstream csvFile;
    
    if (first_write) {
        csvFile.open(filename, std::ios::out);
        // Write header
        csvFile << "timestep,time(s),active_particles,total_conc";
        for (int i = 0; i < MAX_NUCLIDES; i++) {
            csvFile << ",total_Q_" << i << ",ratio_Q_" << i;
        }
        csvFile << std::endl;
        first_write = false;
    } else {
        csvFile.open(filename, std::ios::app);
    }
    
    // Calculate totals for all active particles
    float total_concentrations[MAX_NUCLIDES] = {0.0f};
    float total_conc = 0.0f;
    int active_particles = 0;
    
    for (const auto& p : part) {
        if (p.flag) {
            active_particles++;
            total_conc += p.conc;
            for (int i = 0; i < MAX_NUCLIDES; i++) {
                total_concentrations[i] += p.concentrations[i];
            }
        }
    }
    
    // Write data
    csvFile << timestep << "," << currentTime << "," << active_particles << "," << total_conc;
    for (int i = 0; i < MAX_NUCLIDES; i++) {
        float ratio = (total_conc > 0) ? (total_concentrations[i] / total_conc) : 0.0f;
        csvFile << "," << total_concentrations[i] << "," << ratio;
    }
    csvFile << std::endl;
    
    csvFile.close();
}

// CRAM 계산 전후 농도 변화를 상세히 로깅하는 함수
void LDM::log_first_particle_cram_detail(int timestep, float currentTime, float dt_used) {
    // Copy particles from device to host
    cudaMemcpy(part.data(), d_part, nop * sizeof(LDMpart), cudaMemcpyDeviceToHost);
    
    // Ensure validation directory exists
    #ifdef _WIN32
        _mkdir("validation");
    #else
        mkdir("validation", 0777);
    #endif
    
    // Create or append to detailed CSV file
    static bool first_write = true;
    std::string filename = "validation/first_particle_cram_detail.csv";
    std::ofstream csvFile;
    
    if (first_write) {
        csvFile.open(filename, std::ios::out);
        // Write detailed header with decay information
        csvFile << "timestep,time(s),dt(s),particle_age(s)";
        
        NuclideConfig* nucConfig = NuclideConfig::getInstance();
        int num_nuclides = nucConfig->getNumNuclides();
        
        for (int i = 0; i < num_nuclides; i++) {
            std::string name = nucConfig->getNuclideName(i);
            float half_life = nucConfig->getHalfLife(i);
            csvFile << "," << name << "_conc," << name << "_half_life," << name << "_decay_factor";
        }
        csvFile << ",total_mass,mass_conservation_check" << std::endl;
        first_write = false;
    } else {
        csvFile.open(filename, std::ios::app);
    }
    
    // Find first active particle and log detailed information
    for (const auto& p : part) {
        if (p.flag) {
            NuclideConfig* nucConfig = NuclideConfig::getInstance();
            int num_nuclides = nucConfig->getNumNuclides();
            
            csvFile << timestep << "," << currentTime << "," << dt_used << "," << p.age;
            
            float total_mass = 0.0f;
            for (int i = 0; i < num_nuclides; i++) {
                float half_life = nucConfig->getHalfLife(i);
                float decay_constant = log(2.0f) / (half_life * 3600.0f); // Convert hours to seconds
                float decay_factor = exp(-decay_constant * dt_used);
                
                csvFile << "," << p.concentrations[i] << "," << half_life << "," << decay_factor;
                total_mass += p.concentrations[i];
            }
            
            // Mass conservation check (should decrease due to decay)
            static float initial_mass = -1.0f;
            if (initial_mass < 0) initial_mass = total_mass;
            float conservation_ratio = total_mass / initial_mass;
            
            csvFile << "," << total_mass << "," << conservation_ratio << std::endl;
            break; // Only log the first active particle
        }
    }
    
    csvFile.close();
}

// CRAM4 검증용 데이터 출력 함수 - 간단한 농도분포 저장
void LDM::exportValidationData(int timestep, float currentTime) {
    // 검증용 폴더 생성
    static bool validation_dir_created = false;
    if (!validation_dir_created) {
        #ifdef _WIN32
            _mkdir("validation");
        #else
            mkdir("validation", 0777);
        #endif
        validation_dir_created = true;
    }
    
    // Copy particles from device to host
    cudaMemcpy(part.data(), d_part, nop * sizeof(LDMpart), cudaMemcpyDeviceToHost);
    
    // 주요 타임스텝에서만 격자 데이터 출력 (용량 절약)
    if (timestep % 50 == 0 || timestep <= 10 || timestep >= 710) {
        exportConcentrationGrid(timestep, currentTime);
    }
    
    // 모든 타임스텝에서 핵종별 총 농도 출력
    exportNuclideTotal(timestep, currentTime);
    
    if (timestep % 100 == 0) {
        std::cout << "[VALIDATION] Exported reference data for timestep " << timestep << std::endl;
    }
}

// 농도 격자 데이터 출력 (100x100x20 격자)
void LDM::exportConcentrationGrid(int timestep, float currentTime) {
    // 후쿠시마 주변 영역 설정 (139-143°E, 36-39°N, 0-2000m)
    const float min_lon = 139.0f, max_lon = 143.0f;
    const float min_lat = 36.0f, max_lat = 39.0f; 
    const float min_alt = 0.0f, max_alt = 2000.0f;
    const int grid_x = 100, grid_y = 100, grid_z = 20;
    
    const float dx = (max_lon - min_lon) / grid_x;
    const float dy = (max_lat - min_lat) / grid_y;
    const float dz = (max_alt - min_alt) / grid_z;
    
    // 격자 초기화
    std::vector<std::vector<std::vector<float>>> concentration_grid(
        grid_x, std::vector<std::vector<float>>(grid_y, std::vector<float>(grid_z, 0.0f)));
    std::vector<std::vector<std::vector<int>>> count_grid(
        grid_x, std::vector<std::vector<int>>(grid_y, std::vector<int>(grid_z, 0)));
    
    // 활성 입자들을 격자에 매핑
    for (const auto& p : part) {
        if (!p.flag) continue;
        
        // GFS 좌표를 지리 좌표로 변환
        float lon = -179.0f + p.x * 0.5f;
        float lat = -90.0f + p.y * 0.5f;
        float alt = p.z;
        
        // 격자 범위 확인
        if (lon < min_lon || lon >= max_lon || lat < min_lat || lat >= max_lat || 
            alt < min_alt || alt >= max_alt) continue;
            
        // 격자 인덱스 계산
        int ix = static_cast<int>((lon - min_lon) / dx);
        int iy = static_cast<int>((lat - min_lat) / dy);
        int iz = static_cast<int>((alt - min_alt) / dz);
        
        // 경계 확인
        if (ix >= 0 && ix < grid_x && iy >= 0 && iy < grid_y && iz >= 0 && iz < grid_z) {
            concentration_grid[ix][iy][iz] += p.conc;
            count_grid[ix][iy][iz]++;
        }
    }
    
    // 격자 데이터를 CSV 파일로 저장
    std::ostringstream filename;
    filename << "validation/concentration_grid_" << std::setfill('0') << std::setw(5) << timestep << ".csv";
    
    std::ofstream csvFile(filename.str());
    csvFile << "x_index,y_index,z_index,lon,lat,alt,concentration,particle_count" << std::endl;
    
    for (int ix = 0; ix < grid_x; ix++) {
        for (int iy = 0; iy < grid_y; iy++) {
            for (int iz = 0; iz < grid_z; iz++) {
                if (concentration_grid[ix][iy][iz] > 0 || count_grid[ix][iy][iz] > 0) {
                    float lon = min_lon + (ix + 0.5f) * dx;
                    float lat = min_lat + (iy + 0.5f) * dy;
                    float alt = min_alt + (iz + 0.5f) * dz;
                    
                    csvFile << ix << "," << iy << "," << iz << "," 
                           << lon << "," << lat << "," << alt << ","
                           << concentration_grid[ix][iy][iz] << "," 
                           << count_grid[ix][iy][iz] << std::endl;
                }
            }
        }
    }
    csvFile.close();
}

// 핵종별 총 농도 출력
void LDM::exportNuclideTotal(int timestep, float currentTime) {
    static bool first_write = true;
    std::string filename = "validation/nuclide_totals.csv";
    std::ofstream csvFile;
    
    if (first_write) {
        csvFile.open(filename, std::ios::out);
        csvFile << "timestep,time(s),active_particles,total_conc";
        NuclideConfig* nucConfig = NuclideConfig::getInstance();
        int num_nuclides = nucConfig->getNumNuclides();
        for (int i = 0; i < num_nuclides; i++) {
            csvFile << ",total_" << nucConfig->getNuclideName(i);
        }
        csvFile << std::endl;
        first_write = false;
    } else {
        csvFile.open(filename, std::ios::app);
    }
    
    // 핵종별 총 농도 계산
    std::vector<float> total_concentrations(MAX_NUCLIDES, 0.0f);
    float total_conc = 0.0f;
    int active_particles = 0;
    
    for (const auto& p : part) {
        if (p.flag) {
            active_particles++;
            total_conc += p.conc;
            NuclideConfig* nucConfig = NuclideConfig::getInstance();
            int num_nuclides = nucConfig->getNumNuclides();
            for (int i = 0; i < num_nuclides; i++) {
                total_concentrations[i] += p.concentrations[i];
            }
        }
    }
    
    // 데이터 출력
    csvFile << timestep << "," << currentTime << "," << active_particles << "," << total_conc;
    NuclideConfig* nucConfig = NuclideConfig::getInstance();
    int num_nuclides = nucConfig->getNumNuclides();
    for (int i = 0; i < num_nuclides; i++) {
        csvFile << "," << total_concentrations[i];
    }
    csvFile << std::endl;
    
    csvFile.close();
}

// 핵종별 반감기 정보와 함께 농도 변화를 로깅
void LDM::log_first_particle_decay_analysis(int timestep, float currentTime) {
    // Copy particles from device to host
    cudaMemcpy(part.data(), d_part, nop * sizeof(LDMpart), cudaMemcpyDeviceToHost);
    
    // Ensure validation directory exists
    #ifdef _WIN32
        _mkdir("validation");
    #else
        mkdir("validation", 0777);
    #endif
    
    static bool first_write = true;
    std::string filename = "validation/first_particle_decay_analysis.csv";
    std::ofstream csvFile;
    
    if (first_write) {
        csvFile.open(filename, std::ios::out);
        csvFile << "timestep,time(s),nuclide_name,concentration,half_life_hours,decay_constant_per_sec,theoretical_concentration,relative_error" << std::endl;
        first_write = false;
    } else {
        csvFile.open(filename, std::ios::app);
    }
    
    // Find first active particle
    for (const auto& p : part) {
        if (p.flag) {
            NuclideConfig* nucConfig = NuclideConfig::getInstance();
            int num_nuclides = nucConfig->getNumNuclides();
            
            // 각 핵종별로 상세 분석
            for (int i = 0; i < num_nuclides; i++) {
                std::string name = nucConfig->getNuclideName(i);
                float half_life = nucConfig->getHalfLife(i);
                float decay_constant = log(2.0f) / (half_life * 3600.0f);
                
                // 이론적 농도 계산 (초기 농도 0.1에서 시작)
                float theoretical_conc = 0.1f * exp(-decay_constant * p.age);
                float relative_error = (p.concentrations[i] - theoretical_conc) / theoretical_conc * 100.0f;
                
                csvFile << timestep << "," << currentTime << "," << name << "," 
                       << p.concentrations[i] << "," << half_life << "," << decay_constant << "," 
                       << theoretical_conc << "," << relative_error << std::endl;
            }
            break; // Only analyze the first active particle
        }
    }
    
    csvFile.close();
}
