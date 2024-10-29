// src/lerw_estimation.cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <string>
#include <chrono>
#include "utils.h"
#include "lerw_2d.h"
#include "lerw_3d.h"
#include "random_utils.h"

struct SimulationParams {
    bool run_2d = false;
    std::vector<double> R_values = {40, 80, 160, 320, 500, 1000};
    int num_trials = 1000;
    int num_threads = -1;  // -1 means use all available threads
};

void print_usage() {
    std::cout << "Usage: lerw_simulation [options]\n"
              << "Options:\n"
              << "  --2d              Enable 2D simulation (default: 3D only)\n"
              << "  --trials N        Number of trials per R value (default: 1000)\n"
              << "  --threads N       Number of threads to use (default: all available)\n"
              << "  --r-values R1,R2,R3,...  Comma-separated list of R values\n"
              << "  --help            Show this help message\n";
}

SimulationParams parse_args(int argc, char* argv[]) {
    SimulationParams params;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--2d") {
            params.run_2d = true;
        } else if (arg == "--trials" && i + 1 < argc) {
            params.num_trials = std::stoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            params.num_threads = std::stoi(argv[++i]);
        } else if (arg == "--r-values" && i + 1 < argc) {
            std::string r_list = argv[++i];
            params.R_values.clear();
            size_t pos = 0;
            while ((pos = r_list.find(',')) != std::string::npos) {
                params.R_values.push_back(std::stod(r_list.substr(0, pos)));
                r_list.erase(0, pos + 1);
            }
            if (!r_list.empty()) {
                params.R_values.push_back(std::stod(r_list));
            }
        } else if (arg == "--help") {
            print_usage();
            exit(0);
        }
    }
    
    return params;
}

template<typename Vec, typename SimFunc>
double estimate_dimension(const SimulationParams& params, 
                         SimFunc simulate_func,
                         const std::string& dimension_label) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<double> avg_lengths(params.R_values.size(), 0.0);
    const int num_threads = (params.num_threads > 0) ? 
                           params.num_threads : 
                           omp_get_max_threads();
    
    // Pre-create RNGs for each thread
    std::vector<std::mt19937> thread_rngs(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        thread_rngs[i].seed(std::random_device{}());
    }

    omp_set_num_threads(num_threads);
    
    std::cout << dimension_label << " simulation starting with "
              << omp_get_max_threads() << " threads\n";
    
    // Main simulation loop
    #pragma omp parallel
    {
        #pragma omp single
        {
            std::cout << "Actual threads being used in parallel region: "
                      << omp_get_num_threads() << "\n";
        }
    }
    
    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < params.R_values.size(); ++i) {
        const double R = params.R_values[i];
        const int thread_id = omp_get_thread_num();
        
        double total_length = 0.0;
        std::vector<double> thread_lengths(params.num_trials);
        
        #pragma omp simd reduction(+:total_length)
        for (int j = 0; j < params.num_trials; ++j) {
            auto path = simulate_func(R, thread_rngs[thread_id]);
            thread_lengths[j] = path.size();
            total_length += thread_lengths[j];
        }
        
        avg_lengths[i] = total_length / params.num_trials;
        
        #pragma omp critical
        {
            std::cout << dimension_label << " R = " << R 
                     << ", Avg Length = " << avg_lengths[i] 
                     << " (Thread " << thread_id << ")" << std::endl;
        }
    }

    // Rest of the function remains the same...
    // Calculate dimension using linear regression
    double sum_logR = 0, sum_logL = 0, sum_logR2 = 0, sum_logR_logL = 0;
    const int n = params.R_values.size();
    
    #pragma omp simd reduction(+:sum_logR,sum_logL,sum_logR2,sum_logR_logL)
    for (int i = 0; i < n; ++i) {
        double logR = std::log(params.R_values[i]);
        double logL = std::log(avg_lengths[i]);
        sum_logR += logR;
        sum_logL += logL;
        sum_logR2 += logR * logR;
        sum_logR_logL += logR * logL;
    }
    
    double D = (n * sum_logR_logL - sum_logR * sum_logL) / 
               (n * sum_logR2 - sum_logR * sum_logR);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    std::cout << dimension_label << " Simulation completed in " 
              << duration.count() << " seconds using "
              << num_threads << " threads\n";
    std::cout << "Estimated dimension D: " << D << "\n\n";
    
    return D;
}

int main(int argc, char* argv[]) {
    SimulationParams params = parse_args(argc, argv);
    
    std::cout << "LERW Simulation Parameters:\n"
              << "Dimensions: " << (params.run_2d ? "2D and 3D" : "3D only") << "\n"
              << "Number of trials per R: " << params.num_trials << "\n"
              << "Number of threads: " << (params.num_threads > 0 ? 
                                         std::to_string(params.num_threads) : "auto") << "\n"
              << "R values: ";
    for (size_t i = 0; i < params.R_values.size(); ++i) {
        std::cout << params.R_values[i];
        if (i < params.R_values.size() - 1) std::cout << ", ";
    }
    std::cout << "\n\n";

    if (params.run_2d) {
        estimate_dimension<Vec2D>(params, simulate_LERW_2D, "2D");
    }
    
    estimate_dimension<Vec3D>(params, simulate_LERW_3D, "3D");

    return 0;
}