#pragma once
#ifndef LDM_ENSEMBLE_INIT_CUH
#define LDM_ENSEMBLE_INIT_CUH

#include <vector>
#include <random>
#include <chrono>
#include <cassert>
#include <cuda_runtime.h>

// Forward declarations to avoid circular dependency
class LDM;

// Debug logging control
#ifdef LDM_DEBUG_ENS
#define DEBUG_LOG(fmt, ...) printf("[DEBUG_ENS] " fmt "\n", ##__VA_ARGS__)
#else
#define DEBUG_LOG(fmt, ...)
#endif

// Fast deterministic RNG utilities
class EnsembleRNG {
public:
    // XORShift128+ implementation for fast CPU random generation
    struct XORShift128Plus {
        uint64_t s[2];
        
        explicit XORShift128Plus(uint64_t seed) {
            // SplitMix64 for seeding
            auto splitmix64 = [](uint64_t& z) -> uint64_t {
                z += 0x9e3779b97f4a7c15;
                z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
                z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
                return z ^ (z >> 31);
            };
            
            uint64_t z = seed;
            s[0] = splitmix64(z);
            s[1] = splitmix64(z);
        }
        
        uint64_t next() {
            uint64_t s1 = s[0];
            const uint64_t s0 = s[1];
            s[0] = s0;
            s1 ^= s1 << 23;
            s[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
            return s[1] + s0;
        }
        
        float uniform() {
            return (next() >> 11) * 0x1.0p-53f;
        }
        
        float normal(float mean = 0.0f, float stddev = 1.0f) {
            static thread_local bool has_spare = false;
            static thread_local float spare;
            
            if (has_spare) {
                has_spare = false;
                return spare * stddev + mean;
            }
            
            has_spare = true;
            float u = uniform();
            float v = uniform();
            float mag = stddev * sqrtf(-2.0f * logf(u));
            spare = mag * cosf(2.0f * M_PI * v);
            return mag * sinf(2.0f * M_PI * v) + mean;
        }
    };
    
    static uint64_t getEnsembleSeed(uint64_t base_seed, int ensemble_idx) {
        return base_seed + static_cast<uint64_t>(ensemble_idx) * 10007ULL;
    }
};

// Forward declarations need to be complete for kernel use
class LDM;

#endif // LDM_ENSEMBLE_INIT_CUH