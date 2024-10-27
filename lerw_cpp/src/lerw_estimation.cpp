#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>  
#include "utils.h"
#include "lerw_2d.h"
#include "lerw_3d.h"
#include "random_utils.h"

int main() {
    // Parameters
    std::vector<double> R_values = {10, 20, 40, 80, 160};
    int num_trials = 1000;

    // Stores for lengths
    std::vector<double> avg_lengths_2D(R_values.size(), 0.0);
    std::vector<double> avg_lengths_3D(R_values.size(), 0.0);

    // Simulation for 2D
    std::cout << "Simulating LERW in 2D..." << std::endl;
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < R_values.size(); ++i) {
        double total_length = 0.0;
        double R = R_values[i];
        
        // Need thread-local RNG to avoid race conditions
        std::random_device local_rd;
        std::mt19937 local_rng(local_rd());
        
        for (int j = 0; j < num_trials; ++j) {
            std::vector<Vec2D> path = simulate_LERW_2D(R, local_rng);
            total_length += path.size();
        }
        avg_lengths_2D[i] = total_length / num_trials;
        
        #pragma omp critical
        {
            std::cout << "R = " << R << ", Avg Length = " << avg_lengths_2D[i] << std::endl;
        }
    }

    // Simulation for 3D
    std::cout << "\nSimulating LERW in 3D..." << std::endl;
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < R_values.size(); ++i) {
        double total_length = 0.0;
        double R = R_values[i];
        
        // Thread-local RNG
        std::random_device local_rd;
        std::mt19937 local_rng(local_rd());
        
        for (int j = 0; j < num_trials; ++j) {
            std::vector<Vec3D> path = simulate_LERW_3D(R, local_rng);
            total_length += path.size();
        }
        avg_lengths_3D[i] = total_length / num_trials;
        
        #pragma omp critical
        {
            std::cout << "R = " << R << ", Avg Length = " << avg_lengths_3D[i] << std::endl;
        }
    }

    // Estimating the exponent D
    std::cout << "\nEstimating the exponent D..." << std::endl;
    double sum_logR = 0.0, sum_logL_2D = 0.0, sum_logL_3D = 0.0;
    double sum_logR2 = 0.0, sum_logR_logL_2D = 0.0, sum_logR_logL_3D = 0.0;
    int n = R_values.size();

    for (int i = 0; i < n; ++i) {
        double logR = log(R_values[i]);
        double logL_2D = log(avg_lengths_2D[i]);
        double logL_3D = log(avg_lengths_3D[i]);

        sum_logR += logR;
        sum_logL_2D += logL_2D;
        sum_logL_3D += logL_3D;
        sum_logR2 += logR * logR;
        sum_logR_logL_2D += logR * logL_2D;
        sum_logR_logL_3D += logR * logL_3D;
    }

    double D_2D = (n * sum_logR_logL_2D - sum_logR * sum_logL_2D) / (n * sum_logR2 - sum_logR * sum_logR);
    double D_3D = (n * sum_logR_logL_3D - sum_logR * sum_logL_3D) / (n * sum_logR2 - sum_logR * sum_logR);

    std::cout << "Estimated exponent D in 2D: " << D_2D << std::endl;
    std::cout << "Estimated exponent D in 3D: " << D_3D << std::endl;

    return 0;
}
