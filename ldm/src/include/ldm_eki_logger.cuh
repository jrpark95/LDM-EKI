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
#include <numeric>

void writeEKIConfigLog(EKIConfig* ekiConfig);

// EKI integration functions
void printSystemInfo();
void printEKIConfiguration(EKIConfig* ekiConfig);
void printSimulationStatus(const std::string& status);
void cleanLogDirectory();
void prepareObservationData(EKIConfig* ekiConfig);
void runVisualization();
void runEKIEstimation();
void logLDMtoEKIMatrix(const std::vector<float>& data, int num_receptors, int time_intervals);
bool loadEKIEnsembleStates(std::vector<std::vector<float>>& ensemble_matrix, int& time_intervals, int& ensemble_size, int iteration);
void logLDMEnsembleReception(const std::vector<std::vector<float>>& ensemble_matrix, int time_intervals, int ensemble_size, int iteration);

#endif // LDM_EKI_LOGGER_CUH