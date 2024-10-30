// src/lerw_estimation.cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <string>
#include <chrono>
#include <random>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include "utils.h"
#include "lerw_2d.h"
#include "lerw_3d.h"
#include "random_utils.h"

struct SimulationParams {
    bool run_2d = false;
    std::vector<double> R_values = {256, 512, 1024, 2048, 4096};
    int num_trials = 1000;
    int num_threads = -1;  // -1 means use all available threads
    int num_bootstrap = 1000;  // Number of bootstrap samples for CI
    double confidence_level = 0.95;  // Confidence level for intervals
};

struct DimensionEstimate {
    double D;              // Point estimate
    double ci_lower;       // Lower bound of CI
    double ci_upper;       // Upper bound of CI
    double r_squared;      // R² value for fit quality
    std::vector<double> residuals;
    std::chrono::seconds computation_time;
};

class LinearRegression {
private:
    double slope;
    double intercept;
    double r_squared;
    std::vector<double> residuals;

public:
    LinearRegression(const std::vector<double>& x, const std::vector<double>& y) {
        const int n = x.size();
        double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
        
        for (int i = 0; i < n; i++) {
            sum_x += x[i];
            sum_y += y[i];
            sum_xy += x[i] * y[i];
            sum_xx += x[i] * x[i];
        }
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
        intercept = (sum_y - slope * sum_x) / n;
        
        // Calculate R² and residuals
        double mean_y = sum_y / n;
        double ss_tot = 0, ss_res = 0;
        residuals.resize(n);
        
        for (int i = 0; i < n; i++) {
            double predicted = slope * x[i] + intercept;
            residuals[i] = y[i] - predicted;
            ss_res += residuals[i] * residuals[i];
            double diff_from_mean = y[i] - mean_y;
            ss_tot += diff_from_mean * diff_from_mean;
        }
        
        r_squared = 1 - (ss_res / ss_tot);
    }
    
    double get_slope() const { return slope; }
    double get_r_squared() const { return r_squared; }
    const std::vector<double>& get_residuals() const { return residuals; }
};

void print_usage() {
    std::cout << "Usage: lerw_simulation [options]\n"
              << "Options:\n"
              << "  --2d              Enable 2D simulation (default: 3D only)\n"
              << "  --trials N        Number of trials per R value (default: 1000)\n"
              << "  --threads N       Number of threads to use (default: all available)\n"
              << "  --bootstrap N     Number of bootstrap samples (default: 1000)\n"
              << "  --confidence C    Confidence level (default: 0.95)\n"
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
        } else if (arg == "--bootstrap" && i + 1 < argc) {
            params.num_bootstrap = std::stoi(argv[++i]);
        } else if (arg == "--confidence" && i + 1 < argc) {
            params.confidence_level = std::stod(argv[++i]);
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
DimensionEstimate estimate_dimension(
    const SimulationParams& params,
    SimFunc simulate_func,
    const std::string& dimension_label
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Store all trial lengths for each R value
    std::vector<std::vector<size_t>> all_lengths(params.R_values.size());
    for (auto& lengths : all_lengths) {
        lengths.reserve(params.num_trials);
    }
    
    const int num_threads = (params.num_threads > 0) ? 
                           params.num_threads : 
                           omp_get_max_threads();
    
    // Pre-create RNGs for each thread
    std::vector<std::mt19937> thread_rngs(num_threads);
    for (int i = 0; i < num_threads; i++) {
        thread_rngs[i].seed(std::random_device{}());
    }

    omp_set_num_threads(num_threads);
    
    std::cout << dimension_label << " simulation starting with "
              << num_threads << " threads\n";
    
    // Main simulation loop
    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < params.R_values.size(); i++) {
        const double R = params.R_values[i];
        const int thread_id = omp_get_thread_num();
        std::vector<size_t> thread_lengths;
        thread_lengths.reserve(params.num_trials);
        
        for (int j = 0; j < params.num_trials; j++) {
            auto path = simulate_func(R, thread_rngs[thread_id]);
            thread_lengths.push_back(path.size());
        }
        
        #pragma omp critical
        {
            all_lengths[i] = std::move(thread_lengths);
            std::cout << dimension_label << " R = " << R 
                     << ", Completed " << params.num_trials << " trials"
                     << " (Thread " << thread_id << ")" << std::endl;
        }
    }

    // Prepare data for regression
    std::vector<double> log_R(params.R_values.size());
    std::vector<double> log_L(params.R_values.size());
    
    for (size_t i = 0; i < params.R_values.size(); i++) {
        log_R[i] = std::log(params.R_values[i]);
        double mean_length = std::accumulate(all_lengths[i].begin(),
                                           all_lengths[i].end(), 0.0) /
                            all_lengths[i].size();
        log_L[i] = std::log(mean_length);
    }
    
    // Calculate point estimate
    LinearRegression initial_fit(log_R, log_L);
    double D_point = initial_fit.get_slope();
    
    // Bootstrap resampling for confidence interval
    std::vector<double> bootstrap_estimates(params.num_bootstrap);
    std::mt19937 bootstrap_rng(std::random_device{}());
    
    #pragma omp parallel for
    for (int b = 0; b < params.num_bootstrap; b++) {
        std::vector<double> resampled_log_L(params.R_values.size());
        
        for (size_t i = 0; i < params.R_values.size(); i++) {
            std::vector<double> resampled_lengths;
            resampled_lengths.reserve(all_lengths[i].size());
            
            std::uniform_int_distribution<size_t> dist(0, all_lengths[i].size() - 1);
            for (size_t j = 0; j < all_lengths[i].size(); j++) {
                size_t idx = dist(bootstrap_rng);
                resampled_lengths.push_back(all_lengths[i][idx]);
            }
            
            double mean_length = std::accumulate(resampled_lengths.begin(),
                                               resampled_lengths.end(), 0.0) /
                               resampled_lengths.size();
            resampled_log_L[i] = std::log(mean_length);
        }
        
        LinearRegression bootstrap_fit(log_R, resampled_log_L);
        bootstrap_estimates[b] = bootstrap_fit.get_slope();
    }
    
    // Calculate confidence interval
    std::sort(bootstrap_estimates.begin(), bootstrap_estimates.end());
    int lower_idx = (params.num_bootstrap * (1 - params.confidence_level) / 2);
    int upper_idx = params.num_bootstrap - lower_idx - 1;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    // Print detailed results
    std::cout << "\n" << dimension_label << " Results:\n"
              << "Estimated dimension D: " << D_point 
              << " (" << (params.confidence_level * 100) << "% CI: "
              << bootstrap_estimates[lower_idx] << " - "
              << bootstrap_estimates[upper_idx] << ")\n"
              << "R² value: " << initial_fit.get_r_squared() << "\n"
              << "Computation time: " << duration.count() << " seconds\n\n"
              << "Detailed statistics for each R value:\n";
              
    for (size_t i = 0; i < params.R_values.size(); i++) {
        double mean = std::accumulate(all_lengths[i].begin(),
                                    all_lengths[i].end(), 0.0) /
                     all_lengths[i].size();
        double variance = 0;
        for (const auto& len : all_lengths[i]) {
            variance += (len - mean) * (len - mean);
        }
        variance /= (all_lengths[i].size() - 1);
        
        std::cout << "R = " << std::setw(6) << params.R_values[i]
                  << ": mean length = " << std::setw(10) << mean
                  << ", std dev = " << std::setw(10) << std::sqrt(variance) << "\n";
    }
    
    return DimensionEstimate{
        D_point,
        bootstrap_estimates[lower_idx],
        bootstrap_estimates[upper_idx],
        initial_fit.get_r_squared(),
        initial_fit.get_residuals(),
        duration
    };
}

int main(int argc, char* argv[]) {
    SimulationParams params = parse_args(argc, argv);
    
    std::cout << "LERW Simulation Parameters:\n"
              << "Dimensions: " << (params.run_2d ? "2D and 3D" : "3D only") << "\n"
              << "Number of trials per R: " << params.num_trials << "\n"
              << "Number of bootstrap samples: " << params.num_bootstrap << "\n"
              << "Confidence level: " << (params.confidence_level * 100) << "%\n"
              << "Number of threads: " << (params.num_threads > 0 ? 
                                         std::to_string(params.num_threads) : "auto") << "\n"
              << "R values: ";
    for (size_t i = 0; i < params.R_values.size(); i++) {
        std::cout << params.R_values[i];
        if (i < params.R_values.size() - 1) std::cout << ", ";
    }
    std::cout << "\n\n";

    if (params.run_2d) {
        auto results_2d = estimate_dimension<Vec2D>(params, simulate_LERW_2D, "2D");
        std::cout << "\nFinal 2D Results Summary:\n"
                  << "D = " << results_2d.D << " ± "
                  << (results_2d.ci_upper - results_2d.ci_lower) / 2
                  << " (R² = " << results_2d.r_squared << ")\n";
    }
    
    auto results_3d = estimate_dimension<Vec3D>(params, simulate_LERW_3D, "3D");
    std::cout << "\nFinal 3D Results Summary:\n"
              << "D = " << results_3d.D << " ± "
              << (results_3d.ci_upper - results_3d.ci_lower) / 2
              << " (R² = " << results_3d.r_squared << ")\n";

    return 0;
}