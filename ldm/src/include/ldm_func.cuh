
#include "ldm.cuh"

// External CRAM matrix declarations
extern float* d_A_matrix_global;
extern float* d_exp_matrix_global;

LDM::LDM() 
    : device_meteorological_data_pres(nullptr), 
      device_meteorological_data_unis(nullptr), 
      device_meteorological_data_etas(nullptr){
        // Initialize CRAM A matrix system
        std::string cram_path = "./cram/A60.csv";
        if (!initialize_cram_system(cram_path.c_str())) {
            std::cerr << "[WARNING] Failed to initialize CRAM A matrix from " << cram_path << std::endl;
            std::cerr << "[WARNING] Multi-nuclide decay calculations will be disabled" << std::endl;
        }
    }

LDM::~LDM(){

        if (device_meteorological_data_pres){
            cudaFree(device_meteorological_data_pres);
        }
        if (device_meteorological_data_unis){
            cudaFree(device_meteorological_data_unis);
        }
        if (device_meteorological_data_etas){
            cudaFree(device_meteorological_data_etas);
        }
        if (d_part){
            cudaFree(d_part);
        }
        
    }

void LDM::startTimer(){
        
        timerStart = std::chrono::high_resolution_clock::now();
    }

void LDM::stopTimer(){

        timerEnd = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(timerEnd - timerStart);
        std::cout << "Elapsed time: " << duration.count()/1.0e6 << " seconds" << std::endl;
    }

void LDM::allocateGPUMemory(){
        if (part.empty()) {
            std::cerr << "[ERROR] No particles to copy to device (part vector is empty)" << std::endl;
            return;
        }
        
        size_t total_size = part.size() * sizeof(LDMpart);

        cudaError_t err = cudaMalloc((void**)&d_part, total_size);
        if (err != cudaSuccess){
            std::cerr << "[ERROR] Failed to allocate device memory for particles: " << cudaGetErrorString(err) << std::endl;
            size_t free_mem, total_mem;
            cudaMemGetInfo(&free_mem, &total_mem);
            std::cerr << "[ERROR] GPU memory info - Free: " << free_mem/(1024*1024) << " MB, Total: " << total_mem/(1024*1024) << " MB" << std::endl;
            std::cerr << "[ERROR] Requested: " << total_size/(1024*1024) << " MB" << std::endl;
            exit(EXIT_FAILURE);
        }

        err = cudaMemcpy(d_part, part.data(), total_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess){
            std::cerr << "[ERROR] Failed to copy particle data from host to device: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_part);
            exit(EXIT_FAILURE);
        }
        
        // Clear any previous CUDA errors before checking
        cudaGetLastError(); // Clear previous errors
        err = cudaGetLastError(); // Should now be cudaSuccess
        if (err != cudaSuccess) {
            std::cerr << "[ERROR] CUDA error still present after particle memory copy: " << cudaGetErrorString(err) << std::endl;
            std::cerr << "[ERROR] This indicates a problem with earlier CUDA operations" << std::endl;
            std::cerr << "[ERROR] This may cause NaN values in meteorological data interpolation" << std::endl;
        } else {
            std::cout << "[DEBUG] Particle memory copy successful, no CUDA errors detected" << std::endl;
        }
    }

void LDM::runSimulation(){
    
    cudaError_t err = cudaGetLastError();

    std::future<void> gfs_future;
    bool gfs_ready = false;

    int ded;
    
    float currentTime = 0.0f;
    int threadsPerBlock = 256;
    int blocks = (part.size() + threadsPerBlock - 1) / threadsPerBlock;
    int timestep = 0;
    float activationRatio = 0.0;
    float t0 = 0.0;
    float totalElapsedTime = 0.0;

    GridConfig grid_config = loadGridConfig();
    float start_lat = grid_config.start_lat;
    float start_lon = grid_config.start_lon;
    float end_lat = grid_config.end_lat;
    float end_lon = grid_config.end_lon;
    float lat_step = grid_config.lat_step;
    float lon_step = grid_config.lon_step;
    
    int lat_num = static_cast<int>(round((end_lat - start_lat) / lat_step) + 1);
    int lon_num = static_cast<int>(round((end_lon - start_lon) / lon_step) + 1);

    err = cudaMemcpyToSymbol(d_start_lat, &start_lat, sizeof(float));
    if (err != cudaSuccess) printf("Error copying start_lat to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_start_lon, &start_lon, sizeof(float));
    if (err != cudaSuccess) printf("Error copying start_lon to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_lat_step, &lat_step, sizeof(float));
    if (err != cudaSuccess) printf("Error copying lat_step to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_lon_step, &lon_step, sizeof(float));
    if (err != cudaSuccess) printf("Error copying lon_step to symbol: %s\n", cudaGetErrorString(err));

    Mesh mesh(start_lat, start_lon, lat_step, lon_step, lat_num, lon_num);

    size_t meshSize = mesh.lat_count * mesh.lon_count * sizeof(float);

    std::cout << mesh.lon_count << mesh.lat_count << std::endl;

    float* d_dryDep = nullptr;
    float* d_wetDep = nullptr;

    cudaMalloc((void**)&d_dryDep, meshSize);
    cudaMalloc((void**)&d_wetDep, meshSize);

    cudaMemset(d_dryDep, 0, meshSize);
    cudaMemset(d_wetDep, 0, meshSize);

    FlexPres* flexpresdata = new FlexPres[(dimX_GFS + 1) * dimY_GFS * dimZ_GFS];
    FlexUnis* flexunisdata = new FlexUnis[(dimX_GFS + 1) * dimY_GFS];

    log_first_particle_concentrations(0, 0.0f);

    while(currentTime < time_end){


        stepStart = std::chrono::high_resolution_clock::now();

        currentTime += dt;

        // EKI-based time-dependent particle emission
        static float last_emission_time = -1.0f;
        static size_t last_particle_count = part.size();
        
        if (currentTime - last_emission_time >= 900.0f) {  // Every 15 minutes (900 seconds)
            releaseParticlesForCurrentTime(currentTime);
            last_emission_time = currentTime;
            
            // Update GPU memory if new particles were added
            if (part.size() > last_particle_count) {
                updateGPUParticleMemory();
                last_particle_count = part.size();
                std::cout << "[DEBUG] GPU memory updated. Total particles: " << part.size() << std::endl;
            }
        }
        
        // Update particle concentrations based on EKI time series
        updateParticleConcentrationsFromEKI(currentTime);

        activationRatio = (currentTime) / time_end;
        t0 = (currentTime - static_cast<int>((currentTime-1e-5)/time_interval)*time_interval) / time_interval;

        // Recalculate blocks in case new particles were added
        blocks = (part.size() + threadsPerBlock - 1) / threadsPerBlock;

        update_particle_flags<<<blocks, threadsPerBlock>>>
            (d_part, activationRatio);
        cudaDeviceSynchronize();

        NuclideConfig* nucConfig = NuclideConfig::getInstance();
        move_part_by_wind_mpi<<<blocks, threadsPerBlock>>>
        (d_part, t0, mpiRank, d_dryDep, d_wetDep, mesh.lon_count, mesh.lat_count,
            device_meteorological_flex_unis0,
            device_meteorological_flex_pres0,
            device_meteorological_flex_unis1,
            device_meteorological_flex_pres1);
        cudaDeviceSynchronize();

        timestep++; 

        // Debug: Copy and print first particle position every 5 timesteps  
        if(timestep % 5 == 0) {  // Every 5 timesteps for tracking
            LDMpart first_particle;
            cudaError_t err = cudaMemcpy(&first_particle, d_part, sizeof(LDMpart), cudaMemcpyDeviceToHost);
            if (err == cudaSuccess) {
                // Convert from GFS grid coordinates to geographic coordinates
                float lon = first_particle.x * 0.5f - 179.0f;
                float lat = first_particle.y * 0.5f - 90.0f;
                float z = first_particle.z;
                
                // Debug output disabled for release
                // printf("[CRAM2] Timestep %d: Particle 0: lon=%.6f, lat=%.6f, z=%.2f | Wind: u=%.6f, v=%.6f, w=%.6f m/s\n", 
                //        timestep, lon, lat, z, first_particle.u_wind, first_particle.v_wind, first_particle.w_wind);
                // fflush(stdout); // Force output
            }
        }

        if(timestep % freq_output==0){
            printf("-------------------------------------------------\n");
            printf("Time : %f\tsec\n", currentTime);
            printf("Time steps : \t%d of \t%d\n", timestep, (int)(time_end/dt));
            
            // Debug GFS loading condition at key timepoints
            if(timestep == 1080) {  // Exactly at 10800 seconds
                printf("[DEBUG] GFS condition check: currentTime=%.1f, time_interval=%d, left=%d, gfs_idx=%d, condition=%s\n", 
                       currentTime, time_interval, static_cast<int>(currentTime/time_interval), gfs_idx,
                       (static_cast<int>(currentTime/time_interval) > gfs_idx) ? "TRUE" : "FALSE");
            }

            //particle_output_ASCII(timestep);
            outputParticlesBinaryMPI(timestep);
            
            // Log concentration data for analysis
            log_first_particle_concentrations(timestep, currentTime);
            log_all_particles_nuclide_ratios(timestep, currentTime);
            log_first_particle_cram_detail(timestep, currentTime, dt);
            log_first_particle_decay_analysis(timestep, currentTime);
            
            // Log particle data for visualization
            logParticlePositionsForVisualization(timestep, currentTime);
            logParticleCountData(timestep, currentTime);
            
            // Export validation reference data
            exportValidationData(timestep, currentTime);
        }
        stepEnd = std::chrono::high_resolution_clock::now();
        auto duration0 = std::chrono::duration_cast<std::chrono::microseconds>(stepEnd - stepStart);
        totalElapsedTime += static_cast<double>(duration0.count()/1.0e6);

        // Check GFS loading condition every timestep 
        int left_val = static_cast<int>(currentTime/time_interval);
        if(timestep >= 1079 && timestep <= 1081) {
            printf("[DEBUG] Step %d: currentTime=%.1f, left=%d, gfs_idx=%d\n", timestep, currentTime, left_val, gfs_idx);
        }
        
        if(left_val > gfs_idx) {
            printf("[INFO] Condition met: currentTime=%.1f, time_interval=%d, left=%d, gfs_idx=%d\n", 
                   currentTime, time_interval, left_val, gfs_idx);
            loadFlexGFSData();
        }

    }

    std::cout << "Elapsed data time: " << totalElapsedTime << " seconds" << std::endl;

    
}
