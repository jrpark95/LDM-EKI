#include "ldm.cuh"


void LDM::initializeFlexGFSData(){

    std::cout << "[DEBUG] Starting read_meteorological_flex_gfs_init3 function..." << std::endl;
    
    // First, detect how many timestep files are available
    num_timesteps_available = 0;
    char test_filename[256];
    while (true) {
        sprintf(test_filename, "./data/input/flexprewind/%d.txt", num_timesteps_available);
        std::ifstream test_file(test_filename);
        if (!test_file.is_open()) {
            break;
        }
        test_file.close();
        num_timesteps_available++;
        if (num_timesteps_available > 100) { // Safety limit (more reasonable for 3-hour intervals)
            std::cerr << "[WARNING] Too many meteorological files detected (>100), limiting to 100" << std::endl;
            num_timesteps_available = 100;
            break;
        }
    }
    
    if (num_timesteps_available < 3) {
        std::cerr << "[ERROR] Not enough meteorological files found. Need at least 3, found " << num_timesteps_available << std::endl;
        return;
    }
    
    std::cout << "[INFO] Found " << num_timesteps_available << " meteorological timestep files" << std::endl;
    
    // Allocate arrays for all timesteps
    device_meteorological_flex_unis_all = new FlexUnis*[num_timesteps_available];
    device_meteorological_flex_pres_all = new FlexPres*[num_timesteps_available];
    host_unisA_all = new float4*[num_timesteps_available];
    host_unisB_all = new float4*[num_timesteps_available];
    host_presA_all = new float4*[num_timesteps_available];
    host_presB_all = new float4*[num_timesteps_available];
    d_unisArrayA_all = new cudaArray*[num_timesteps_available];
    d_unisArrayB_all = new cudaArray*[num_timesteps_available];
    d_presArrayA_all = new cudaArray*[num_timesteps_available];
    d_presArrayB_all = new cudaArray*[num_timesteps_available];
    
    // Initialize all pointers to nullptr
    for (int t = 0; t < num_timesteps_available; t++) {
        device_meteorological_flex_unis_all[t] = nullptr;
        device_meteorological_flex_pres_all[t] = nullptr;
        host_unisA_all[t] = nullptr;
        host_unisB_all[t] = nullptr;
        host_presA_all[t] = nullptr;
        host_presB_all[t] = nullptr;
        d_unisArrayA_all[t] = nullptr;
        d_unisArrayB_all[t] = nullptr;
        d_presArrayA_all[t] = nullptr;
        d_presArrayB_all[t] = nullptr;
    }

    flex_hgt.resize(dimZ_GFS);
    std::cout << "[DEBUG] flex_hgt vector resized to " << dimZ_GFS << " elements" << std::endl;

    const char* filename = "./data/input/flexprewind/0.txt";
    int recordMarker;

    size_t pres_data_size = (dimX_GFS + 1) * dimY_GFS * dimZ_GFS;
    size_t unis_data_size = (dimX_GFS + 1) * dimY_GFS;
    std::cout << "[DEBUG] Allocating memory for FlexPres data: " << pres_data_size << " elements" << std::endl;
    std::cout << "[DEBUG] Allocating memory for FlexUnis data: " << unis_data_size << " elements" << std::endl;

    FlexPres* flexpresdata = new FlexPres[pres_data_size];
    FlexUnis* flexunisdata = new FlexUnis[unis_data_size];

    if (!flexpresdata || !flexunisdata) {
        std::cerr << "[ERROR] Failed to allocate memory for meteorological data" << std::endl;
        if (flexpresdata) delete[] flexpresdata;
        if (flexunisdata) delete[] flexunisdata;
        return;
    }

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "[ERROR] Cannot open file: " << filename << std::endl;
        delete[] flexpresdata;
        delete[] flexunisdata;
        return;
    }
    std::cout << "[DEBUG] Successfully opened file: " << filename << std::endl;

    // Read HMIX data
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].HMIX), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            // Sample debug output for verification
            if ((i == 100 && j == 50) || (i == 200 && j == 150) || (i == 360 && j == 180)) {
                std::cout << "[VERIFY] HMIX[" << i << "," << j << "] = " << flexunisdata[index].HMIX << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].TROP), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].USTR), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            // if(i==50) std::cout << "flexunisdata.USTR[" << j << "] = " << flexunisdata.USTR[index] << std::endl;
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].WSTR), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].OBKL), sizeof(float));
            //std::cout << "OBKL = " << flexunisdata[index].OBKL << std::endl;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].LPREC), sizeof(float));
            //if(flexunisdata[index].LPREC>0.0f)std::cout << "LPREC = " << flexunisdata[index].LPREC << std::endl;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].CPREC), sizeof(float));
            //std::cout << "CPREC = " << flexunisdata[index].CPREC << std::endl;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].TCC), sizeof(float));
            //std::cout << "TCC = " << flexunisdata[index].TCC << std::endl;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            int intBuffer;
            file.read(reinterpret_cast<char*>(&intBuffer), sizeof(int));
            flexunisdata[index].CLDH = static_cast<float>(intBuffer);
            //file.read(reinterpret_cast<char*>(&flexunisdata[index].CLDH), sizeof(int));
            //std::cout << "CLDH = " << flexunisdata[index].CLDH << std::endl;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].RHO), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                //if(i==50&&j==100) std::cout << "flexpresdata.RHO[" << k << "] = " << flexpresdata[index].RHO << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].TT), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                //if(i==50&&j==100) std::cout << "flexpresdata.TT[" << k << "] = " << flexpresdata[index].TT << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].UU), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // Debug disabled
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].VV), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // Debug disabled
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].WW), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // Debug disabled
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].VDEP), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            //if(i==50&&j==100) 
            //if(i==50&&j==100) std::cout << "VDEP[" << i << "," << j << "] = " << flexunisdata[index].VDEP << std::endl;
            //if(i<10&&j<10) std::cout << "VDEP[" << i << "," << j << "] = " << flexunisdata[index].VDEP << std::endl;
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                int intBuffer;
                file.read(reinterpret_cast<char*>(&intBuffer), sizeof(int));
                flexpresdata[index].CLDS = static_cast<float>(intBuffer);
                // file.read(reinterpret_cast<char*>(&flexpresdata[index].CLDS), sizeof(int));
                //std::cout << "CLDS = " << static_cast<int>(flexpresdata[index].CLDS) << std::endl;
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // Debug disabled
            }
        }
    }



    // for (int k = 0; k < 1; ++k) {
    //     for (int i = 0; i < dimX_GFS+1; ++i) {
    //         for (int j = 0; j < dimY_GFS; ++j) {
    //             int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
    //             file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
    //             file.read(reinterpret_cast<char*>(&flexpresdata[index].VDEP), sizeof(float));
    //             file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
    //             if(i==50&&j==100) std::cout << "flexpresdata.VDEP[" << k << "] = " << flexpresdata[index].VDEP << std::endl;
    //         }
    //     }
    // }

    // Debug height data before DRHO calculation
    std::cout << "[DRHO_DEBUG] Height check: flex_hgt[0]=" << flex_hgt[0] << ", flex_hgt[1]=" << flex_hgt[1] 
              << ", diff=" << (flex_hgt[1] - flex_hgt[0]) << std::endl;
    
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            float rho0 = flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].RHO;
            float rho1 = flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + 1].RHO;
            float hgt_diff = flex_hgt[1] - flex_hgt[0];
            
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].DRHO = (rho1 - rho0) / hgt_diff;
            
            // Debug first few points
            if (i < 2 && j < 2) {
                std::cout << "[DRHO_CALC] [" << i << "," << j << "]: rho0=" << rho0 << ", rho1=" << rho1 
                          << ", hgt_diff=" << hgt_diff << ", DRHO=" << flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].DRHO << std::endl;
            }

            //if(i==100 && j==50) printf("drho(%d) = %f\n", 0, flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].DRHO);
            
            for (int k = 1; k < dimZ_GFS-2; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k;
                flexpresdata[index].DRHO = 
                (flexpresdata[index+1].RHO - flexpresdata[index-1].RHO) / (flex_hgt[k+1]-flex_hgt[k-1]);
                //if(i==100 && j==50) printf("drho(%d) = %f\n", k, flexpresdata[index].DRHO);
            }

            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO = 
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-3].DRHO;

            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-1].DRHO = 
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO;

            //if(i==100 && j==50) printf("drho(%d) = %f\n", dimZ_GFS-2, flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO);

        }
    }
    
    // Debug simplified - just check if data loaded
    std::cout << "[INFO] Wind data loaded - UU[0,0,0]=" << flexpresdata[0].UU 
              << ", VV[0,0,0]=" << flexpresdata[0].VV 
              << ", WW[0,0,0]=" << flexpresdata[0].WW << std::endl;
    
    // Check file read status
    if (file.fail() || file.bad()) {
        std::cerr << "[ERROR] File read error detected. fail=" << file.fail() << ", bad=" << file.bad() << std::endl;
    } else {
        std::cout << "[DEBUG] File read completed successfully" << std::endl;
    }



    file.close();

    cudaMalloc((void**)&device_meteorological_flex_pres0, 
    (dimX_GFS+1) * dimY_GFS * dimZ_GFS * sizeof(FlexPres));
    cudaMalloc((void**)&device_meteorological_flex_unis0, 
    (dimX_GFS+1) * dimY_GFS * sizeof(FlexUnis));
    
    cudaError_t copy_err = cudaMemcpy(device_meteorological_flex_pres0, 
        flexpresdata, (dimX_GFS+1) * dimY_GFS * dimZ_GFS * sizeof(FlexPres), 
        cudaMemcpyHostToDevice);
    if (copy_err != cudaSuccess) {
        std::cerr << "[ERROR] Failed to copy meteorological pres data to GPU: " << cudaGetErrorString(copy_err) << std::endl;
    } else {
        std::cout << "[DEBUG] Meteorological pres data copied successfully" << std::endl;
    }
    
    copy_err = cudaMemcpy(device_meteorological_flex_unis0, 
        flexunisdata, (dimX_GFS+1) * dimY_GFS * sizeof(FlexUnis), 
        cudaMemcpyHostToDevice);
    if (copy_err != cudaSuccess) {
        std::cerr << "[ERROR] Failed to copy meteorological unis data to GPU: " << cudaGetErrorString(copy_err) << std::endl;
    } else {
        std::cout << "[DEBUG] Meteorological unis data copied successfully" << std::endl;
    }


    int size2D = (dimX_GFS + 1) * dimY_GFS;
    int size3D = (dimX_GFS + 1) * dimY_GFS * dimZ_GFS;

    this->host_unisA0 = new float4[size2D]; // HMIX, USTR, WSTR, OBKL
    this->host_unisB0 = new float4[size2D]; // VDEP, LPREC, CPREC, TCC
    this->host_presA0 = new float4[size3D]; // DRHO, RHO, TT, QV
    this->host_presB0 = new float4[size3D]; // UU, VV, WW, 0.0 

    for(int i = 0; i < size2D; i++){
        host_unisA0[i] = make_float4(
            flexunisdata[i].HMIX,
            flexunisdata[i].USTR,
            flexunisdata[i].WSTR,
            flexunisdata[i].OBKL
        );
        host_unisB0[i] = make_float4(
            flexunisdata[i].VDEP,
            flexunisdata[i].LPREC,
            flexunisdata[i].CPREC,
            flexunisdata[i].TCC
        );
    }

    for(int i = 0; i < size3D; i++){
        host_presA0[i] = make_float4(
            flexpresdata[i].DRHO,
            flexpresdata[i].RHO,
            flexpresdata[i].TT,
            flexpresdata[i].QV
        );
        host_presB0[i] = make_float4(
            flexpresdata[i].UU,
            flexpresdata[i].VV,
            flexpresdata[i].WW,
            0.0f        
        );
    }

    size_t width  = dimX_GFS + 1; 
    size_t height = dimY_GFS;

    cudaMallocArray(&d_unisArrayA0, &channelDesc2D, width, dimY_GFS);
    cudaMallocArray(&d_unisArrayB0, &channelDesc2D, dimX_GFS + 1, dimY_GFS);

    cudaMemcpy2DToArray(
        d_unisArrayA0,
        0, 0,
        host_unisA0,
        width*sizeof(float4),
        width*sizeof(float4),
        height,
        cudaMemcpyHostToDevice
    );
    
    cudaMemcpy2DToArray(
        d_unisArrayB0,
        0, 0,
        host_unisB0,
        width*sizeof(float4),
        width*sizeof(float4),
        height,
        cudaMemcpyHostToDevice
    );

    cudaExtent extent = make_cudaExtent(dimX_GFS+1, dimY_GFS, dimZ_GFS);
    cudaMalloc3DArray(&d_presArrayA0, &channelDesc3D, extent);
    cudaMalloc3DArray(&d_presArrayB0, &channelDesc3D, extent);

    cudaMemcpy3DParms copyParamsA0 = {0};
    copyParamsA0.srcPtr = make_cudaPitchedPtr(
        (void*)host_presA0, 
        (dimX_GFS+1)*sizeof(float4), // pitch in bytes
        (dimX_GFS+1),                // width in elements
        dimY_GFS                     // height
    );
    copyParamsA0.dstArray = d_presArrayA0;
    copyParamsA0.extent   = extent;
    copyParamsA0.kind     = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParamsA0);

    cudaMemcpy3DParms copyParamsB0 = {0};
    copyParamsB0.srcPtr = make_cudaPitchedPtr(
        (void*)host_presB0, 
        (dimX_GFS+1)*sizeof(float4), // pitch in bytes
        (dimX_GFS+1),                // width in elements
        dimY_GFS                     // height
    );
    copyParamsB0.dstArray = d_presArrayB0;
    copyParamsB0.extent   = extent;
    copyParamsB0.kind     = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParamsB0);

    
    filename = "./data/input/flexprewind/1.txt";

    file.open(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].HMIX), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].TROP), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].USTR), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            // if(i==50) std::cout << "flexunisdata.USTR[" << j << "] = " << flexunisdata.USTR[index] << std::endl;
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].WSTR), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].OBKL), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].LPREC), sizeof(float));
            //if(flexunisdata[index].LPREC>0.0)printf("lsp = %e\n", flexunisdata[index].LPREC);
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].CPREC), sizeof(float));
            //if(flexunisdata[index].CPREC>0.0)printf("convprec = %e\n", flexunisdata[index].CPREC);
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].TCC), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            int intBuffer;
            file.read(reinterpret_cast<char*>(&intBuffer), sizeof(int));
            flexunisdata[index].CLDH = static_cast<float>(intBuffer);
            //file.read(reinterpret_cast<char*>(&flexunisdata[index].CLDH), sizeof(int));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].RHO), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.RHO[" << k << "] = " << flexpresdata[index].RHO << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].TT), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.TT[" << k << "] = " << flexpresdata[index].TT << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].UU), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.UU[" << k << "] = " << flexpresdata[index].UU << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].VV), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.VV[" << k << "] = " << flexpresdata[index].VV << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].WW), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.WW[" << k << "] = " << flexpresdata[index].WW << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].VDEP), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            //if(i==50&&j==100) std::cout << "VDEP[" << i << "," << j << "] = " << flexunisdata[index].VDEP << std::endl;
        }
    }


    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                int intBuffer;
                file.read(reinterpret_cast<char*>(&intBuffer), sizeof(int));
                flexpresdata[index].CLDS = static_cast<float>(intBuffer);
                // file.read(reinterpret_cast<char*>(&flexpresdata[index].CLDS), sizeof(int));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.WW[" << k << "] = " << flexpresdata[index].WW << std::endl;
            }
        }
    }

    // for (int j = 0; j < dimY_GFS; ++j) {
    //     int index = 647 * dimY_GFS + j; 
    //     std::cout << "CLDH(" << 647 << ", " << j << ") = " << flexunisdata[index].CLDH << std::endl;
    //     index = 647 * dimY_GFS * dimZ_GFS + j * dimZ_GFS + 5; 
    //     std::cout << "CLDS(" << 647 << ", " << j << ") = " << flexpresdata[index].CLDS << std::endl;
    // }


    // for (int i = 0; i < dimX_GFS+1; ++i) {
    //     for (int j = 0; j < dimY_GFS; ++j) {
    //         for (int k = 0; k < dimZ_GFS; ++k) {
    //             int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
    //             file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
    //             file.read(reinterpret_cast<char*>(&flexpresdata[index].VDEP), sizeof(float));
    //             file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
    //             if(i==50&&j==100) std::cout << "flexpresdata.VDEP[" << k << "] = " << flexpresdata[index].VDEP << std::endl;
    //         }
    //     }
    // }


    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].DRHO = 
            (flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + 1].RHO - flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].RHO) /
            (flex_hgt[1]-flex_hgt[0]);

            //if(i==100 && j==50) printf("drho(%d) = %f\n", 0, flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].DRHO);
            
            for (int k = 1; k < dimZ_GFS-2; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k;
                flexpresdata[index].DRHO = 
                (flexpresdata[index+1].RHO - flexpresdata[index-1].RHO) / (flex_hgt[k+1]-flex_hgt[k-1]);
                //if(i==100 && j==50) printf("drho(%d) = %f\n", k, flexpresdata[index].DRHO);
            }

            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO = 
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-3].DRHO;

            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-1].DRHO = 
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO;

            //if(i==100 && j==50) printf("drho(%d) = %f\n", dimZ_GFS-2, flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO);

        }
    }

    file.close();

    cudaMalloc((void**)&device_meteorological_flex_pres1, 
    (dimX_GFS+1) * dimY_GFS * dimZ_GFS * sizeof(FlexPres));
    cudaMalloc((void**)&device_meteorological_flex_unis1, 
    (dimX_GFS+1) * dimY_GFS * sizeof(FlexUnis));
    
    cudaMemcpy(device_meteorological_flex_pres1, 
        flexpresdata, (dimX_GFS+1) * dimY_GFS * dimZ_GFS * sizeof(FlexPres), 
        cudaMemcpyHostToDevice);
    cudaMemcpy(device_meteorological_flex_unis1, 
        flexunisdata, (dimX_GFS+1) * dimY_GFS * sizeof(FlexUnis), 
        cudaMemcpyHostToDevice);

    
    this->host_unisA1 = new float4[size2D]; // HMIX, USTR, WSTR, OBKL
    this->host_unisB1 = new float4[size2D]; // VDEP, LPREC, CPREC, TCC
    this->host_presA1 = new float4[size3D]; // DRHO, RHO, TT, QV
    this->host_presB1 = new float4[size3D]; // UU, VV, WW, 0.0 

    for(int i = 0; i < size2D; i++){
        host_unisA1[i] = make_float4(
            flexunisdata[i].HMIX,
            flexunisdata[i].USTR,
            flexunisdata[i].WSTR,
            flexunisdata[i].OBKL
        );
        host_unisB1[i] = make_float4(
            flexunisdata[i].VDEP,
            flexunisdata[i].LPREC,
            flexunisdata[i].CPREC,
            flexunisdata[i].TCC
        );
    }

    for(int i = 0; i < size3D; i++){
        host_presA1[i] = make_float4(
            flexpresdata[i].DRHO,
            flexpresdata[i].RHO,
            flexpresdata[i].TT,
            flexpresdata[i].QV
        );
        host_presB1[i] = make_float4(
            flexpresdata[i].UU,
            flexpresdata[i].VV,
            flexpresdata[i].WW,
            0.0f        
        );
    }

    cudaMallocArray(&d_unisArrayA1, &channelDesc2D, width, dimY_GFS);
    cudaMallocArray(&d_unisArrayB1, &channelDesc2D, dimX_GFS + 1, dimY_GFS);

    cudaMemcpy2DToArray(
        d_unisArrayA1,
        0, 0,
        host_unisA1,
        width*sizeof(float4),
        width*sizeof(float4),
        height,
        cudaMemcpyHostToDevice
    );
    
    cudaMemcpy2DToArray(
        d_unisArrayB1,
        0, 0,
        host_unisB1,
        width*sizeof(float4),
        width*sizeof(float4),
        height,
        cudaMemcpyHostToDevice
    );

    cudaMalloc3DArray(&d_presArrayA1, &channelDesc3D, extent);
    cudaMalloc3DArray(&d_presArrayB1, &channelDesc3D, extent);

    cudaMemcpy3DParms copyParamsA1 = {0};
    copyParamsA1.srcPtr = make_cudaPitchedPtr(
        (void*)host_presA1, 
        (dimX_GFS+1)*sizeof(float4), // pitch in bytes
        (dimX_GFS+1),                // width in elements
        dimY_GFS                     // height
    );
    copyParamsA1.dstArray = d_presArrayA1;
    copyParamsA1.extent   = extent;
    copyParamsA1.kind     = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParamsA1);

    cudaMemcpy3DParms copyParamsB1 = {0};
    copyParamsB1.srcPtr = make_cudaPitchedPtr(
        (void*)host_presB1, 
        (dimX_GFS+1)*sizeof(float4), // pitch in bytes
        (dimX_GFS+1),                // width in elements
        dimY_GFS                     // height
    );
    copyParamsB1.dstArray = d_presArrayB1;
    copyParamsB1.extent   = extent;
    copyParamsB1.kind     = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParamsB1);


    
    
    filename = "./data/input/flexprewind/2.txt";

    file.open(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].HMIX), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].TROP), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].USTR), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            // if(i==50) std::cout << "flexunisdata.USTR[" << j << "] = " << flexunisdata.USTR[index] << std::endl;
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].WSTR), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].OBKL), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].LPREC), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].CPREC), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].TCC), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            int intBuffer;
            file.read(reinterpret_cast<char*>(&intBuffer), sizeof(int));
            flexunisdata[index].CLDH = static_cast<float>(intBuffer);
            //file.read(reinterpret_cast<char*>(&flexunisdata[index].CLDH), sizeof(int));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].RHO), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.RHO[" << k << "] = " << flexpresdata[index].RHO << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].TT), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.TT[" << k << "] = " << flexpresdata[index].TT << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].UU), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.UU[" << k << "] = " << flexpresdata[index].UU << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].VV), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.VV[" << k << "] = " << flexpresdata[index].VV << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].WW), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.WW[" << k << "] = " << flexpresdata[index].WW << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].VDEP), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            //if(i==50&&j==100) std::cout << "VDEP[" << i << "," << j << "] = " << flexunisdata[index].VDEP << std::endl;
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                int intBuffer;
                file.read(reinterpret_cast<char*>(&intBuffer), sizeof(int));
                flexpresdata[index].CLDS = static_cast<float>(intBuffer);
                //file.read(reinterpret_cast<char*>(&flexpresdata[index].CLDS), sizeof(int));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.WW[" << k << "] = " << flexpresdata[index].WW << std::endl;
            }
        }
    }



    // for (int i = 0; i < dimX_GFS+1; ++i) {
    //     for (int j = 0; j < dimY_GFS; ++j) {
    //         for (int k = 0; k < dimZ_GFS; ++k) {
    //             int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
    //             file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
    //             file.read(reinterpret_cast<char*>(&flexpresdata[index].VDEP), sizeof(float));
    //             file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
    //             if(i==50&&j==100) std::cout << "flexpresdata.VDEP[" << k << "] = " << flexpresdata[index].VDEP << std::endl;
    //         }
    //     }
    // }


    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].DRHO = 
            (flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + 1].RHO - flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].RHO) /
            (flex_hgt[1]-flex_hgt[0]);

            //if(i==100 && j==50) printf("drho(%d) = %f\n", 0, flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].DRHO);
            
            for (int k = 1; k < dimZ_GFS-2; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k;
                flexpresdata[index].DRHO = 
                (flexpresdata[index+1].RHO - flexpresdata[index-1].RHO) / (flex_hgt[k+1]-flex_hgt[k-1]);
                //if(i==100 && j==50) printf("drho(%d) = %f\n", k, flexpresdata[index].DRHO);
            }

            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO = 
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-3].DRHO;

            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-1].DRHO = 
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO;

            //if(i==100 && j==50) printf("drho(%d) = %f\n", dimZ_GFS-2, flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO);

        }
    }

    file.close();

    cudaMalloc((void**)&device_meteorological_flex_pres2, 
    (dimX_GFS+1) * dimY_GFS * dimZ_GFS * sizeof(FlexPres));
    cudaMalloc((void**)&device_meteorological_flex_unis2, 
    (dimX_GFS+1) * dimY_GFS * sizeof(FlexUnis));
    
    cudaMemcpy(device_meteorological_flex_pres2, 
        flexpresdata, (dimX_GFS+1) * dimY_GFS * dimZ_GFS * sizeof(FlexPres), 
        cudaMemcpyHostToDevice);
    cudaMemcpy(device_meteorological_flex_unis2, 
        flexunisdata, (dimX_GFS+1) * dimY_GFS * sizeof(FlexUnis), 
        cudaMemcpyHostToDevice);


    this->host_unisA2 = new float4[size2D]; // HMIX, USTR, WSTR, OBKL
    this->host_unisB2 = new float4[size2D]; // VDEP, LPREC, CPREC, TCC
    this->host_presA2 = new float4[size3D]; // DRHO, RHO, TT, QV
    this->host_presB2 = new float4[size3D]; // UU, VV, WW, 0.0 

    for(int i = 0; i < size2D; i++){
        host_unisA2[i] = make_float4(
            flexunisdata[i].HMIX,
            flexunisdata[i].USTR,
            flexunisdata[i].WSTR,
            flexunisdata[i].OBKL
        );
        host_unisB2[i] = make_float4(
            flexunisdata[i].VDEP,
            flexunisdata[i].LPREC,
            flexunisdata[i].CPREC,
            flexunisdata[i].TCC
        );
    }

    for(int i = 0; i < size3D; i++){
        host_presA2[i] = make_float4(
            flexpresdata[i].DRHO,
            flexpresdata[i].RHO,
            flexpresdata[i].TT,
            flexpresdata[i].QV
        );
        host_presB2[i] = make_float4(
            flexpresdata[i].UU,
            flexpresdata[i].VV,
            flexpresdata[i].WW,
            0.0f        
        );
    }

    cudaMallocArray(&d_unisArrayA2, &channelDesc2D, width, dimY_GFS);
    cudaMallocArray(&d_unisArrayB2, &channelDesc2D, dimX_GFS + 1, dimY_GFS);

    cudaMemcpy2DToArray(
        d_unisArrayA2,
        0, 0,
        host_unisA2,
        width*sizeof(float4),
        width*sizeof(float4),
        height,
        cudaMemcpyHostToDevice
    );
    
    cudaMemcpy2DToArray(
        d_unisArrayB2,
        0, 0,
        host_unisB2,
        width*sizeof(float4),
        width*sizeof(float4),
        height,
        cudaMemcpyHostToDevice
    );

    cudaMalloc3DArray(&d_presArrayA2, &channelDesc3D, extent);
    cudaMalloc3DArray(&d_presArrayB2, &channelDesc3D, extent);

    cudaMemcpy3DParms copyParamsA2 = {0};
    copyParamsA2.srcPtr = make_cudaPitchedPtr(
        (void*)host_presA2, 
        (dimX_GFS+1)*sizeof(float4), // pitch in bytes
        (dimX_GFS+1),                // width in elements
        dimY_GFS                     // height
    );
    copyParamsA2.dstArray = d_presArrayA2;
    copyParamsA2.extent   = extent;
    copyParamsA2.kind     = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParamsA2);

    cudaMemcpy3DParms copyParamsB2 = {0};
    copyParamsB2.srcPtr = make_cudaPitchedPtr(
        (void*)host_presB2, 
        (dimX_GFS+1)*sizeof(float4), // pitch in bytes
        (dimX_GFS+1),                // width in elements
        dimY_GFS                     // height
    );
    copyParamsB2.dstArray = d_presArrayB2;
    copyParamsB2.extent   = extent;
    copyParamsB2.kind     = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParamsB2);

        
    delete[] flexunisdata;
    delete[] flexpresdata;

    // Now load all remaining timesteps into memory
    loadAllTimestepsToMemory();

}

void LDM::loadAllTimestepsToMemory() {
    std::cout << "[INFO] Loading all " << num_timesteps_available << " timesteps to GPU memory..." << std::endl;
    
    size_t pres_data_size = (dimX_GFS + 1) * dimY_GFS * dimZ_GFS;
    size_t unis_data_size = (dimX_GFS + 1) * dimY_GFS;
    int size2D = (dimX_GFS + 1) * dimY_GFS;
    int size3D = (dimX_GFS + 1) * dimY_GFS * dimZ_GFS;
    
    size_t width = dimX_GFS + 1;
    size_t height = dimY_GFS;
    cudaExtent extent = make_cudaExtent(dimX_GFS+1, dimY_GFS, dimZ_GFS);
    
    for (int t = 0; t < num_timesteps_available; t++) {
        std::cout << "[INFO] Loading timestep " << t << "..." << std::endl;
        
        // Allocate temporary host memory
        FlexPres* flexpresdata = new FlexPres[pres_data_size];
        FlexUnis* flexunisdata = new FlexUnis[unis_data_size];
        
        // Load data from file
        if (!loadSingleTimestepFromFile(t, flexpresdata, flexunisdata)) {
            std::cerr << "[ERROR] Failed to load timestep " << t << std::endl;
            delete[] flexpresdata;
            delete[] flexunisdata;
            continue;
        }
        
        // Allocate GPU memory for this timestep
        cudaError_t err = cudaMalloc((void**)&device_meteorological_flex_pres_all[t], pres_data_size * sizeof(FlexPres));
        if (err != cudaSuccess) {
            std::cerr << "[ERROR] Failed to allocate GPU memory for timestep " << t << ": " << cudaGetErrorString(err) << std::endl;
            delete[] flexpresdata;
            delete[] flexunisdata;
            continue;
        }
        
        err = cudaMalloc((void**)&device_meteorological_flex_unis_all[t], unis_data_size * sizeof(FlexUnis));
        if (err != cudaSuccess) {
            std::cerr << "[ERROR] Failed to allocate GPU memory for timestep " << t << ": " << cudaGetErrorString(err) << std::endl;
            cudaFree(device_meteorological_flex_pres_all[t]);
            device_meteorological_flex_pres_all[t] = nullptr;
            delete[] flexpresdata;
            delete[] flexunisdata;
            continue;
        }
        
        // Copy to GPU
        err = cudaMemcpy(device_meteorological_flex_pres_all[t], flexpresdata, pres_data_size * sizeof(FlexPres), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "[ERROR] Failed to copy pres data to GPU for timestep " << t << ": " << cudaGetErrorString(err) << std::endl;
        }
        
        err = cudaMemcpy(device_meteorological_flex_unis_all[t], flexunisdata, unis_data_size * sizeof(FlexUnis), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "[ERROR] Failed to copy unis data to GPU for timestep " << t << ": " << cudaGetErrorString(err) << std::endl;
        }
        
        // Prepare texture arrays
        host_unisA_all[t] = new float4[size2D];
        host_unisB_all[t] = new float4[size2D];
        host_presA_all[t] = new float4[size3D];
        host_presB_all[t] = new float4[size3D];
        
        for(int i = 0; i < size2D; i++){
            host_unisA_all[t][i] = make_float4(
                flexunisdata[i].HMIX, flexunisdata[i].USTR, flexunisdata[i].WSTR, flexunisdata[i].OBKL);
            host_unisB_all[t][i] = make_float4(
                flexunisdata[i].VDEP, flexunisdata[i].LPREC, flexunisdata[i].CPREC, flexunisdata[i].TCC);
        }
        
        for(int i = 0; i < size3D; i++){
            host_presA_all[t][i] = make_float4(
                flexpresdata[i].DRHO, flexpresdata[i].RHO, flexpresdata[i].TT, flexpresdata[i].QV);
            host_presB_all[t][i] = make_float4(
                flexpresdata[i].UU, flexpresdata[i].VV, flexpresdata[i].WW, 0.0f);
        }
        
        // Create texture arrays
        cudaMallocArray(&d_unisArrayA_all[t], &channelDesc2D, width, height);
        cudaMallocArray(&d_unisArrayB_all[t], &channelDesc2D, width, height);
        cudaMalloc3DArray(&d_presArrayA_all[t], &channelDesc3D, extent);
        cudaMalloc3DArray(&d_presArrayB_all[t], &channelDesc3D, extent);
        
        // Copy to texture arrays
        cudaMemcpy2DToArray(d_unisArrayA_all[t], 0, 0, host_unisA_all[t], width*sizeof(float4), width*sizeof(float4), height, cudaMemcpyHostToDevice);
        cudaMemcpy2DToArray(d_unisArrayB_all[t], 0, 0, host_unisB_all[t], width*sizeof(float4), width*sizeof(float4), height, cudaMemcpyHostToDevice);
        
        cudaMemcpy3DParms copyParamsA = {0};
        copyParamsA.srcPtr = make_cudaPitchedPtr((void*)host_presA_all[t], width*sizeof(float4), width, height);
        copyParamsA.dstArray = d_presArrayA_all[t];
        copyParamsA.extent = extent;
        copyParamsA.kind = cudaMemcpyHostToDevice;
        cudaMemcpy3D(&copyParamsA);
        
        cudaMemcpy3DParms copyParamsB = {0};
        copyParamsB.srcPtr = make_cudaPitchedPtr((void*)host_presB_all[t], width*sizeof(float4), width, height);
        copyParamsB.dstArray = d_presArrayB_all[t];
        copyParamsB.extent = extent;
        copyParamsB.kind = cudaMemcpyHostToDevice;
        cudaMemcpy3D(&copyParamsB);
        
        // Clean up temporary host memory
        delete[] flexpresdata;
        delete[] flexunisdata;
    }
    
    std::cout << "[INFO] Successfully loaded " << num_timesteps_available << " timesteps to GPU memory" << std::endl;
}

bool LDM::loadSingleTimestepFromFile(int timestep, FlexPres* flexpresdata, FlexUnis* flexunisdata) {
    char filename[256];
    sprintf(filename, "./data/input/flexprewind/%d.txt", timestep);
    
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "[ERROR] Cannot open file: " << filename << std::endl;
        return false;
    }
    
    int recordMarker;
    
    // Load HMIX
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].HMIX), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }
    
    // Load TROP
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].TROP), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }
    
    // Load USTR, WSTR, OBKL, LPREC, CPREC, TCC, CLDH
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].USTR), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }
    
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].WSTR), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }
    
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].OBKL), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }
    
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].LPREC), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }
    
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].CPREC), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }
    
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].TCC), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }
    
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            int intBuffer;
            file.read(reinterpret_cast<char*>(&intBuffer), sizeof(int));
            flexunisdata[index].CLDH = static_cast<float>(intBuffer);
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }
    
    // Load 3D pressure level data (RHO, TT, UU, VV, WW)
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].RHO), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            }
        }
    }
    
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].TT), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            }
        }
    }
    
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].UU), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            }
        }
    }
    
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].VV), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            }
        }
    }
    
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].WW), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            }
        }
    }
    
    // Load VDEP
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].VDEP), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }
    
    // Load CLDS
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                int intBuffer;
                file.read(reinterpret_cast<char*>(&intBuffer), sizeof(int));
                flexpresdata[index].CLDS = static_cast<float>(intBuffer);
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            }
        }
    }
    
    // Calculate DRHO
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].DRHO = 
            (flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + 1].RHO - flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].RHO) /
            (flex_hgt[1]-flex_hgt[0]);
            
            for (int k = 1; k < dimZ_GFS-2; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k;
                flexpresdata[index].DRHO = 
                (flexpresdata[index+1].RHO - flexpresdata[index-1].RHO) / (flex_hgt[k+1]-flex_hgt[k-1]);
            }
            
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO = 
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-3].DRHO;
            
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-1].DRHO = 
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO;
        }
    }
    
    file.close();
    return true;
}

FlexUnis* LDM::getMeteorologicalDataUnis(int timestep) {
    if (timestep < 0 || timestep >= num_timesteps_available) {
        std::cerr << "[ERROR] Invalid timestep " << timestep << " requested (available: 0-" << (num_timesteps_available-1) << ")" << std::endl;
        return nullptr;
    }
    return device_meteorological_flex_unis_all[timestep];
}

FlexPres* LDM::getMeteorologicalDataPres(int timestep) {
    if (timestep < 0 || timestep >= num_timesteps_available) {
        std::cerr << "[ERROR] Invalid timestep " << timestep << " requested (available: 0-" << (num_timesteps_available-1) << ")" << std::endl;
        return nullptr;
    }
    return device_meteorological_flex_pres_all[timestep];
}

void LDM::loadFlexGFSData(){

    gfs_idx ++;

    char filename[256];
    sprintf(filename, "./data/input/flexprewind/%d.txt", gfs_idx+1);
    std::cout << "[DEBUG] Loading next flexprewind file: " << filename << std::endl;
    int recordMarker;

    // FlexUnis flexunisdata;
    // FlexPres flexpresdata;

    FlexPres* flexpresdata = new FlexPres[(dimX_GFS + 1) * dimY_GFS * dimZ_GFS];
    FlexUnis* flexunisdata = new FlexUnis[(dimX_GFS + 1) * dimY_GFS];

    cudaMemcpy(device_meteorological_flex_pres0, 
        device_meteorological_flex_pres1, (dimX_GFS+1) * dimY_GFS * dimZ_GFS * sizeof(FlexPres), 
        cudaMemcpyDeviceToDevice);
    cudaMemcpy(device_meteorological_flex_unis0, 
        device_meteorological_flex_unis1, (dimX_GFS+1) * dimY_GFS * sizeof(FlexUnis), 
        cudaMemcpyDeviceToDevice);

    size_t width  = dimX_GFS + 1; 
    size_t height = dimY_GFS;

    // flexunisdata.HMIX.resize((dimX_GFS + 1) * dimY_GFS);
    // flexunisdata.TROP.resize((dimX_GFS + 1) * dimY_GFS);
    // flexunisdata.USTR.resize((dimX_GFS + 1) * dimY_GFS);
    // flexunisdata.WSTR.resize((dimX_GFS + 1) * dimY_GFS);
    // flexunisdata.OBKL.resize((dimX_GFS + 1) * dimY_GFS);

    // flexpresdata.RHO.resize((dimX_GFS + 1) * dimY_GFS * dimZ_GFS);
    // flexpresdata.TT.resize((dimX_GFS + 1) * dimY_GFS * dimZ_GFS);
    // flexpresdata.UU.resize((dimX_GFS + 1) * dimY_GFS * dimZ_GFS);
    // flexpresdata.VV.resize((dimX_GFS + 1) * dimY_GFS * dimZ_GFS);
    // flexpresdata.WW.resize((dimX_GFS + 1) * dimY_GFS * dimZ_GFS);

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }
    else{
        //std::cout << "open file: " << filename << std::endl;
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].HMIX), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].TROP), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].USTR), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            // if(i==50) std::cout << "flexunisdata.USTR[" << j << "] = " << flexunisdata.USTR[index] << std::endl;
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].WSTR), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].OBKL), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].LPREC), sizeof(float));
            //std::cout << "LPREC = " << flexunisdata[index].LPREC << std::endl;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].CPREC), sizeof(float));
            //std::cout << "cprec = " << flexunisdata[index].CPREC << std::endl;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].TCC), sizeof(float));
            //std::cout << "TCC = " << flexunisdata[index].TCC << std::endl;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            int intBuffer;
            file.read(reinterpret_cast<char*>(&intBuffer), sizeof(int));
            flexunisdata[index].CLDH = static_cast<float>(intBuffer);
            //file.read(reinterpret_cast<char*>(&flexunisdata[index].CLDH), sizeof(int));
            //std::cout << "CLDH = " << flexunisdata[index].CLDH << std::endl;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }
    
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].RHO), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                //if(i==50&&j==100) std::cout << "flexpresdata.RHO[" << k << "] = " << flexpresdata[index].RHO << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].TT), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                //if(i==50&&j==100) std::cout << "flexpresdata.TT[" << k << "] = " << flexpresdata[index].TT << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].UU), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // Debug disabled
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].VV), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // Debug disabled
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].WW), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // Debug disabled
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].VDEP), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            //if(i==50&&j==100) std::cout << "VDEP[" << i << "," << j << "] = " << flexunisdata[index].VDEP << std::endl;
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                int intBuffer;
                file.read(reinterpret_cast<char*>(&intBuffer), sizeof(int));
                flexpresdata[index].CLDS = static_cast<float>(intBuffer);
                //file.read(reinterpret_cast<char*>(&flexpresdata[index].CLDS), sizeof(int));
                //std::cout << "CLDS = " << static_cast<int>(flexpresdata[index].CLDS) << std::endl;
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.WW[" << k << "] = " << flexpresdata[index].WW << std::endl;
            }
        }
    }



    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].DRHO = 
            (flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + 1].RHO - flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].RHO) /
            (flex_hgt[1]-flex_hgt[0]);

            //if(i==100 && j==50) printf("drho(%d) = %f\n", 0, flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].DRHO);
            
            for (int k = 1; k < dimZ_GFS-2; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k;
                flexpresdata[index].DRHO = 
                (flexpresdata[index+1].RHO - flexpresdata[index-1].RHO) / (flex_hgt[k+1]-flex_hgt[k-1]);
                //if(i==100 && j==50) printf("drho(%d) = %f\n", k, flexpresdata[index].DRHO);
            }

            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO = 
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-3].DRHO;

            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-1].DRHO = 
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO;

            //if(i==100 && j==50) printf("drho(%d) = %f\n", dimZ_GFS-2, flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO);

        }
    }

    file.close();
    
    cudaMemcpy(device_meteorological_flex_pres1, 
        flexpresdata, (dimX_GFS+1) * dimY_GFS * dimZ_GFS * sizeof(FlexPres), 
        cudaMemcpyHostToDevice);
    cudaMemcpy(device_meteorological_flex_unis1, 
        flexunisdata, (dimX_GFS+1) * dimY_GFS * sizeof(FlexUnis), 
        cudaMemcpyHostToDevice);


    char filename_hgt[256];
    sprintf(filename_hgt, "./data/input/flexprewind/hgt_%d.txt", gfs_idx);
    //std::cout << filename_hgt << std::endl;

    int recordMarker_hgt;

    std::ifstream file_hgt(filename_hgt, std::ios::binary);
    if (!file_hgt) {
        std::cerr << "Cannot open file: " << filename_hgt << std::endl;
        return;
    }

    for (int index = 0; index < dimZ_GFS; ++index) {
        file_hgt.read(reinterpret_cast<char*>(&recordMarker_hgt), sizeof(int));
        file_hgt.read(reinterpret_cast<char*>(&flex_hgt[index]), sizeof(float));
        file_hgt.read(reinterpret_cast<char*>(&recordMarker_hgt), sizeof(int));
        // std::cout << "flex_hgt[" << index << "] = " << flex_hgt[index] << std::endl;
    }

    file_hgt.close();
    
    cudaError_t err = cudaMemcpyToSymbol(d_flex_hgt, flex_hgt.data(), sizeof(float) * (dimZ_GFS));
    if (err != cudaSuccess) std::cerr << "Failed to copy data to constant memory: " << cudaGetErrorString(err) << std::endl;
        

}

void LDM::loadFlexHeightData(){ 

    std::cout << "[DEBUG] Starting read_meteorological_flex_hgt function..." << std::endl;
    
    flex_hgt.resize(dimZ_GFS);
    std::cout << "[DEBUG] flex_hgt vector resized to " << dimZ_GFS << " elements" << std::endl;

    const char* filename = "./data/input/flexprewind/hgt_0.txt";
    int recordMarker;

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "[ERROR] Cannot open file: " << filename << std::endl;
        return;
    }
    std::cout << "[DEBUG] Successfully opened file: " << filename << std::endl;

    for (int index = 0; index < dimZ_GFS; ++index) {
        file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        if (file.fail()) {
            std::cerr << "[ERROR] Failed to read record marker at index " << index << std::endl;
            return;
        }
        
        file.read(reinterpret_cast<char*>(&flex_hgt[index]), sizeof(float));
        if (file.fail()) {
            std::cerr << "[ERROR] Failed to read flex_hgt data at index " << index << std::endl;
            return;
        }
        
        // Check for NaN in height data
        if (std::isnan(flex_hgt[index])) {
            std::cout << "[HGT_NAN] flex_hgt[" << index << "] is NaN!" << std::endl;
        }
        
        file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        if (file.fail()) {
            std::cerr << "[ERROR] Failed to read closing record marker at index " << index << std::endl;
            return;
        }
        
        // std::cout << "flex_hgt[" << index << "] = " << flex_hgt[index] << std::endl;
    }

    file.close();
    std::cout << "[DEBUG] Successfully read " << dimZ_GFS << " height levels from file" << std::endl;
    
    std::cout << "[DEBUG] Attempting to copy " << dimZ_GFS << " float values to d_flex_hgt constant memory" << std::endl;
    cudaError_t err = cudaMemcpyToSymbol(d_flex_hgt, flex_hgt.data(), sizeof(float) * (dimZ_GFS));
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] Failed to copy flex_hgt to GPU constant memory: " << cudaGetErrorString(err) << std::endl;
        std::cerr << "[ERROR] Size attempted: " << sizeof(float) * (dimZ_GFS) << " bytes (" << dimZ_GFS << " floats)" << std::endl;
    } else {
        std::cout << "[DEBUG] Successfully copied flex_hgt to GPU constant memory" << std::endl;
    }
        

}