#include <iostream>
#include <vector>
#include <cmath>

// Simple test: 2x2 system
// A = [[-0.1, 0.0], [0.1, -0.2]]  // parent -> daughter with dt=1
// Expected: exp(A*dt) should show parent decay and daughter growth

int main() {
    const int n = 2;
    float dt = 1.0f;
    
    // A matrix: parent decays at 0.1/s, daughter receives from parent and decays at 0.2/s
    float A[4] = {
        -0.1f,  0.0f,    // parent row: decay only
         0.1f, -0.2f     // daughter row: gain from parent, decay self
    };
    
    std::cout << "A matrix:" << std::endl;
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            std::cout << A[i*n + j] << " ";
        }
        std::cout << std::endl;
    }
    
    // Initial concentrations: parent=1, daughter=0
    float x[2] = {1.0f, 0.0f};
    
    // Manual matrix exponential approximation: exp(A*dt) ≈ I + A*dt + (A*dt)²/2 + ...
    // For small dt, exp(A*dt) ≈ I + A*dt
    float T[4] = {
        1.0f + A[0]*dt, A[1]*dt,
        A[2]*dt, 1.0f + A[3]*dt
    };
    
    std::cout << "\nApprox exp(A*dt) matrix:" << std::endl;
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            std::cout << T[i*n + j] << " ";
        }
        std::cout << std::endl;
    }
    
    // Apply: y = T * x
    float y[2] = {0.0f, 0.0f};
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            y[i] += T[i*n + j] * x[j];
        }
    }
    
    std::cout << "\nInitial: parent=" << x[0] << ", daughter=" << x[1] << std::endl;
    std::cout << "After 1 step: parent=" << y[0] << ", daughter=" << y[1] << std::endl;
    
    // Expected: parent should decrease, daughter should increase
    if (y[0] < x[0] && y[1] > x[1]) {
        std::cout << "✅ Matrix direction is CORRECT: parent decreased, daughter increased" << std::endl;
    } else {
        std::cout << "❌ Matrix direction is WRONG: need to check transpose" << std::endl;
        
        // Try transpose
        std::cout << "\nTrying transpose..." << std::endl;
        float y_t[2] = {0.0f, 0.0f};
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                y_t[i] += T[j*n + i] * x[j];  // Transpose: T[j,i] instead of T[i,j]
            }
        }
        std::cout << "With transpose: parent=" << y_t[0] << ", daughter=" << y_t[1] << std::endl;
    }
    
    return 0;
}