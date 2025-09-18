#pragma once
#ifndef LDM_CLASS_DECLARED
#error "Include the header that declares class LDM before including ldm_init.cuh"
#endif
#include <chrono>

void LDM::loadSimulationConfiguration(){

    if (!g_config.loadConfig("data/input/setting.txt")) {
        std::cerr << "Failed to load configuration file" << std::endl;
        exit(1);
    }

    FILE* sourceFile;

    time_end = g_config.getFloat("Time_end(s)", 64800.0f);
    dt = g_config.getFloat("dt(s)", 10.0f);
    freq_output = g_config.getInt("Plot_output_freq", 10);
    nop = g_config.getInt("Total_number_of_particle", 10000);
    isRural = g_config.getInt("Rural/Urban", 1);
    isPG = g_config.getInt("Pasquill-Gifford/Briggs-McElroy-Pooler", 1);
    isGFS = g_config.getInt("Data", 1);
    
    // Load physics model settings
    g_turb_switch = g_config.getInt("turbulence_model", 0);
    g_drydep = g_config.getInt("dry_deposition_model", 0);
    g_wetdep = g_config.getInt("wet_deposition_model", 0);
    g_raddecay = g_config.getInt("radioactive_decay_model", 1);
    
    // Print physics model status
    std::cout << "Physics models: TURB=" << g_turb_switch 
              << ", DRYDEP=" << g_drydep 
              << ", WETDEP=" << g_wetdep 
              << ", RADDECAY=" << g_raddecay << std::endl;
    
    // Clean output directory before simulation
    cleanOutputDirectory();
    
    //loadRadionuclideData();

    std::vector<std::string> species_names = g_config.getStringArray("species_names");
    std::vector<float> decay_constants = g_config.getFloatArray("decay_constants");
    std::vector<float> deposition_velocities = g_config.getFloatArray("deposition_velocities");
    std::vector<float> particle_sizes = g_config.getFloatArray("particle_sizes");
    std::vector<float> particle_densities = g_config.getFloatArray("particle_densities");
    std::vector<float> size_standard_deviations = g_config.getFloatArray("size_standard_deviations");
    
    for (int i = 0; i < 4 && i < species_names.size(); i++) {
        g_mpi.species[i] = species_names[i];
        g_mpi.decayConstants[i] = (i < decay_constants.size()) ? decay_constants[i] : 1.00e-6f;
        g_mpi.depositionVelocities[i] = (i < deposition_velocities.size()) ? deposition_velocities[i] : 0.01f;
        g_mpi.particleSizes[i] = (i < particle_sizes.size()) ? particle_sizes[i] : 0.6f;
        g_mpi.particleDensities[i] = (i < particle_densities.size()) ? particle_densities[i] : 2500.0f;
        g_mpi.sizeStandardDeviations[i] = (i < size_standard_deviations.size()) ? size_standard_deviations[i] : 0.01f;
    }
    
    // Initialize CRAM system with dynamic dt
    if (initialize_cram_system("cram/A60.csv")) {
        // Compute exp(-A*dt) matrix using the dt from configuration
    } else {
        std::cerr << "Warning: CRAM system initialization failed, using traditional decay" << std::endl;
    }

    std::string source_file_path = g_config.getString("input_base_path", "./data/input/") + "source.txt";
    sourceFile = fopen(source_file_path.c_str(), "r");

    if (!sourceFile){
        std::cerr << "Failed to open source.txt" << std::endl;
        exit(1);
    }

    char buffer[256];
    while (fgets(buffer, sizeof(buffer), sourceFile)){
        if (buffer[0] == '#') continue;
    
        // SOURCE coordinates
        if (strstr(buffer, "[SOURCE]")) {
            while (fgets(buffer, sizeof(buffer), sourceFile) && !strstr(buffer, "[SOURCE_TERM]")) {
                if (buffer[0] == '#') continue;
    
                Source src;
                sscanf(buffer, "%f %f %f", &src.lon, &src.lat, &src.height);
                sources.push_back(src);
            }
            sources.pop_back();
        }
    
        // SOURCE_TERM values
        if (strstr(buffer, "[SOURCE_TERM]")){
            while (fgets(buffer, sizeof(buffer), sourceFile) && !strstr(buffer, "[RELEASE_CASES]")) {
                if (buffer[0] == '#') continue;
    
                int srcnum;
                float decay, depvel;
                sscanf(buffer, "%d %f %f", &srcnum, &decay, &depvel);
                decayConstants.push_back(decay);
                drydepositionVelocity.push_back(depvel);
            }
            decayConstants.pop_back();
            drydepositionVelocity.pop_back();
        }
    
        // RELEASE_CASES
        if (strstr(buffer, "[RELEASE_CASES]")){
            while (fgets(buffer, sizeof(buffer), sourceFile)) {
                if (buffer[0] == '#') continue;
                Concentration conc;
                sscanf(buffer, "%d %d %lf", &conc.location, &conc.sourceterm, &conc.value);
                concentrations.push_back(conc);
            }
        }
    }
    
    fclose(sourceFile);

    //nop = floor(nop/(sources.size()*decayConstants.size()))*sources.size()*decayConstants.size();

    cudaError_t err;

    err = cudaMemcpyToSymbol(d_time_end, &time_end, sizeof(float));
    if (err != cudaSuccess) printf("Error copying to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_dt, &dt, sizeof(float));
    if (err != cudaSuccess) printf("Error copying to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_freq_output, &freq_output, sizeof(int));
    if (err != cudaSuccess) printf("Error copying to symbol: %s\n", cudaGetErrorString(err));
    
    // Copy physics model settings to device constant memory
    printf("[DEBUG] Physics model settings: TURB=%d, DRYDEP=%d, WETDEP=%d, RADDECAY=%d\n", 
           g_turb_switch, g_drydep, g_wetdep, g_raddecay);
    
    err = cudaMemcpyToSymbol(d_turb_switch, &g_turb_switch, sizeof(int));
    if (err != cudaSuccess) printf("Error copying turb_switch to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_drydep, &g_drydep, sizeof(int));
    if (err != cudaSuccess) printf("Error copying drydep to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_wetdep, &g_wetdep, sizeof(int));
    if (err != cudaSuccess) printf("Error copying wetdep to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_raddecay, &g_raddecay, sizeof(int));
    if (err != cudaSuccess) printf("Error copying raddecay to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_nop, &nop, sizeof(int));
    if (err != cudaSuccess) printf("Error copying to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_isRural, &isRural, sizeof(bool));
    if (err != cudaSuccess) printf("Error copying to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_isPG, &isPG, sizeof(bool));
    if (err != cudaSuccess) printf("Error copying to symbol: %s\n", cudaGetErrorString(err));

}

void LDM::initializeParticles(){
    if (concentrations.empty()) {
        std::cerr << "[ERROR] No concentrations data loaded for particle initialization" << std::endl;
        return;
    }
    
    if (sources.empty()) {
        std::cerr << "[ERROR] No sources data loaded for particle initialization" << std::endl;
        return;
    }
    
    int partPerConc = nop / concentrations.size();

    std::random_device rd;
    auto now = std::chrono::high_resolution_clock::now();
    auto seed = now.time_since_epoch().count() + mpiRank * 1000;
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(g_mpi.particleSizes[mpiRank], g_mpi.sizeStandardDeviations[mpiRank]);

    int particle_count = 0;
    for (const auto& conc : concentrations) {
        if (conc.location - 1 >= sources.size()) {
            std::cerr << "[ERROR] Invalid source location index: " << conc.location << " (max: " << sources.size() << ")" << std::endl;
            continue;
        }
        
        for (int i = 0; i < partPerConc; ++i) {
            float x = (sources[conc.location - 1].lon + 179.0) / 0.5;
            float y = (sources[conc.location - 1].lat + 90) / 0.5;
            float z = sources[conc.location - 1].height;

            float random_radius = dist(gen);

            part.push_back(LDMpart(x, y, z, 
                                   g_mpi.decayConstants[mpiRank], 
                                   conc.value, 
                                   g_mpi.depositionVelocities[mpiRank], 
                                   random_radius, 
                                   g_mpi.particleDensities[mpiRank],
                                   i + 1));
            
            // Initialize multi-nuclide concentrations for the newly created particle
            LDMpart& current_particle = part.back();
            
            // Get nuclide configuration
            NuclideConfig* nucConfig = NuclideConfig::getInstance();
            int num_nuclides = nucConfig->getNumNuclides();
            
            
            // Set up multi-nuclide concentrations - use individual initial ratios from config
            for(int nuc = 0; nuc < MAX_NUCLIDES; nuc++) {
                if (nuc < num_nuclides) {
                    // Use individual initial ratio for each nuclide from config file
                    float initial_ratio = nucConfig->getInitialRatio(nuc);
                    current_particle.concentrations[nuc] = conc.value * initial_ratio;
                    
                    // Debug: Check for NaN in initialization
                    if (particle_count < 3 && nuc < 5) {
                        std::cout << "[DEBUG INIT] Particle " << particle_count 
                                  << " Nuclide " << nuc 
                                  << " conc.value=" << conc.value 
                                  << " initial_ratio=" << initial_ratio 
                                  << " final_conc=" << current_particle.concentrations[nuc];
                        if (std::isnan(current_particle.concentrations[nuc])) {
                            std::cout << " **NaN DETECTED**";
                        }
                        std::cout << std::endl;
                    }
                } else {
                    current_particle.concentrations[nuc] = 0.0f;
                }
            }
            
            particle_count++;
        }
    }

    std::sort(part.begin(), part.end(), [](const LDMpart& a, const LDMpart& b) {
        return a.timeidx < b.timeidx;
    });
}

void LDM::calculateAverageSettlingVelocity(){
            
            // settling rho, temp
            float prho = 2500.0;
            float radi = 6.0e-7*1.0e+6; // m to um
            float dsig = 3.0e-1;
    
            float xdummy = sqrt(2.0f)*logf(dsig);
            float delta = 6.0/static_cast<float>(NI);
            float d01 = radi*pow(dsig,-3.0);
            float d02, x01, x02, dmean, kn, alpha, cun, dc;
    
            float fract[NI] = {0, };
            float schmi[NI] = {0, };
            float vsh[NI] = {0, };
    
    
            for(int i=1; i<NI+1; i++){
                d02 = d01;
                d01 = radi*pow(dsig, -3.0 + delta*static_cast<float>(i));
                x01 = logf(d01/radi)/xdummy;
                x02 = logf(d02/radi)/xdummy;

    
                fract[i-1] = 0.5*(std::erf(x01)-std::erf(x02));
                dmean = 1.0e-6*exp(0.5*log(d01*d02));
                kn = 2.0f*_lam/dmean;


                if(-1.1/kn <= log10(_eps)*log(10.0))
                    alpha = 1.257;
                else
                    alpha = 1.257+0.4*exp(-1.1/kn);
    
                cun = 1.0 + alpha*kn;
                dc = _kb*_Tr*cun/(3.0*PI*_myl*dmean);
                //schmi = pow(_nyl/dc,-2.0/3.0);
                vsh[i-1] = _ga*prho*dmean*dmean*cun/(18.0*_myl);

            }

            for(int i=1; i<NI+1; i++){
                vsetaver -= fract[i-1]*vsh[i-1];
                cunningham += fract[i-1]*cun;
            }

            cudaError_t err = cudaMemcpyToSymbol(d_vsetaver, &vsetaver, sizeof(float));
            if (err != cudaSuccess) printf("Error copying vsetaver to symbol: %s\n", cudaGetErrorString(err));
            err = cudaMemcpyToSymbol(d_cunningham, &cunningham, sizeof(float));
            if (err != cudaSuccess) printf("Error copying cunningham to symbol: %s\n", cudaGetErrorString(err));


}

void LDM::calculateSettlingVelocity(){
    float prho = g_mpi.particleDensities[mpiRank];
    float radi = g_mpi.particleSizes[mpiRank];
    float dsig = g_mpi.sizeStandardDeviations[mpiRank];

    if (radi == 0.0f) {
        vsetaver = 1.0f;
        cunningham = -1.0f;
        cudaError_t err = cudaMemcpyToSymbol(d_vsetaver, &vsetaver, sizeof(float));
        if (err != cudaSuccess) printf("Error copying vsetaver to symbol: %s\n", cudaGetErrorString(err));
        err = cudaMemcpyToSymbol(d_cunningham, &cunningham, sizeof(float));
        if (err != cudaSuccess) printf("Error copying cunningham to symbol: %s\n", cudaGetErrorString(err));
        return;
    }

    float xdummy = sqrt(2.0f)*logf(dsig);
    float delta = 6.0/static_cast<float>(NI);
    float d01 = radi*pow(dsig,-3.0);
    float d02, x01, x02, dmean, kn, alpha, cun, dc;

    float fract[NI] = {0, };
    float schmi[NI] = {0, };
    float vsh[NI] = {0, };

    vsetaver = 0.0f;
    cunningham = 0.0f;

    for(int i=1; i<NI+1; i++){
        d02 = d01;
        d01 = radi*pow(dsig, -3.0 + delta*static_cast<float>(i));
        x01 = logf(d01/radi)/xdummy;
        x02 = logf(d02/radi)/xdummy;


        fract[i-1] = 0.5*(std::erf(x01)-std::erf(x02));
        dmean = 1.0e-6*exp(0.5*log(d01*d02));
        kn = 2.0f*_lam/dmean;


        if(-1.1/kn <= log10(_eps)*log(10.0))
            alpha = 1.257;
        else
            alpha = 1.257+0.4*exp(-1.1/kn);

        cun = 1.0 + alpha*kn;
        dc = _kb*_Tr*cun/(3.0*PI*_myl*dmean);
        //schmi = pow(_nyl/dc,-2.0/3.0);
        vsh[i-1] = _ga*prho*dmean*dmean*cun/(18.0*_myl);

    }

    for(int i=1; i<NI+1; i++){
        vsetaver -= fract[i-1]*vsh[i-1];
        cunningham += fract[i-1]*cun;
    }

    cudaError_t err = cudaMemcpyToSymbol(d_vsetaver, &vsetaver, sizeof(float));
    if (err != cudaSuccess) printf("Error copying vsetaver to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_cunningham, &cunningham, sizeof(float));
    if (err != cudaSuccess) printf("Error copying cunningham to symbol: %s\n", cudaGetErrorString(err));
}

void LDM::cleanOutputDirectory() {
    std::cout << "Cleaning output directory..." << std::endl;
    
    // Remove all files in output directory
    #ifdef _WIN32
        system("del /Q output\\*.* 2>nul");
    #else
        system("rm -f output/*.vtk 2>/dev/null");
        system("rm -f output/*.csv 2>/dev/null");
        system("rm -f output/*.txt 2>/dev/null");
    #endif
    
    std::cout << "Output directory cleaned." << std::endl;
}