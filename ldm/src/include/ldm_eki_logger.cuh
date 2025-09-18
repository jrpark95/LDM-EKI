#ifndef LDM_EKI_LOGGER_CUH
#define LDM_EKI_LOGGER_CUH

#include "ldm_eki.cuh"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <ctime>
#include <algorithm>
#include <cmath>
#include <iomanip>

void writeEKIConfigLog(EKIConfig* ekiConfig);

#endif // LDM_EKI_LOGGER_CUH