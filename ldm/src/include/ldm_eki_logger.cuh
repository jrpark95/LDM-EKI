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
#include <cstdlib>
#include <numeric>

// Configuration logging
void writeEKIConfigLog(EKIConfig* ekiConfig);

// Debug and status printing
void printSystemInfo();
void printEKIConfiguration(EKIConfig* ekiConfig);
void printSimulationStatus(const std::string& status);
void cleanLogDirectory();

// Data export and external program execution
void prepareObservationData(EKIConfig* ekiConfig);
void runVisualization();
void runEKIEstimation();

// Matrix logging functions
void logLDMtoEKIMatrix(const std::vector<float>& data, int num_receptors, int time_intervals);

// Ensemble handling functions
bool loadEKIEnsembleStates(std::vector<std::vector<float>>& ensemble_matrix, int& time_intervals, int& ensemble_size, int iteration);
void logLDMEnsembleReception(const std::vector<std::vector<float>>& ensemble_matrix, int time_intervals, int ensemble_size, int iteration);

#endif // LDM_EKI_LOGGER_CUH