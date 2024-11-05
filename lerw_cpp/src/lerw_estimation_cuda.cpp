#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <vector>
#include <chrono>
#include "utils.h"

// CUDA error checking helper
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Structure for GPU random state
struct CudaRNG {
    curandState* states;
    int num_states;

    CudaRNG(int n) : num_states(n) {
        CHECK_CUDA(cudaMalloc(&states, n * sizeof(curandState)));
    }

    ~CudaRNG() {
        cudaFree(states);
    }
};

// CUDA kernel for random walk simulation in 3D
__global__ void simulate_LERW_3D_kernel(
    curandState* rng_states,
    size_t* lengths,
    double R,
    int num_trials
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_trials) return;

    curandState local_state = rng_states[tid];
    
    // Allocate local memory for the walk
    const int max_steps = static_cast<int>(R * R * 10); // Heuristic maximum length
    int* visited = new int[max_steps * 3];
    int path_length = 0;
    
    // Starting point
    int x = 0, y = 0, z = 0;
    
    while (x*x + y*y + z*z <= R*R && path_length < max_steps) {
        // Store current position
        visited[path_length * 3] = x;
        visited[path_length * 3 + 1] = y;
        visited[path_length * 3 + 2] = z;
        path_length++;
        
        // Random step
        float direction = curand_uniform(&local_state);
        if (direction < 1.0f/6.0f) x++;
        else if (direction < 2.0f/6.0f) x--;
        else if (direction < 3.0f/6.0f) y++;
        else if (direction < 4.0f/6.0f) y--;
        else if (direction < 5.0f/6.0f) z++;
        else z--;
        
        // Loop erasure
        for (int i = 0; i < path_length - 1; i++) {
            if (visited[i * 3] == x && visited[i * 3 + 1] == y && visited[i * 3 + 2] == z) {
                path_length = i + 1;
                break;
            }
        }
    }
    
    lengths[tid] = path_length;
    delete[] visited;
    rng_states[tid] = local_state;
}

// CUDA kernel for bootstrap resampling
__global__ void bootstrap_kernel(
    curandState* rng_states,
    const size_t* all_lengths,
    double* bootstrap_slopes,
    const double* log_R,
    int num_R_values,
    int samples_per_R,
    int num_bootstrap
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_bootstrap) return;
    
    curandState local_state = rng_states[tid];
    
    // Temporary storage for resampled means
    double* resampled_log_L = new double[num_R_values];
    
    // Perform resampling for each R value
    for (int i = 0; i < num_R_values; i++) {
        double sum = 0.0;
        const size_t* lengths_for_R = &all_lengths[i * samples_per_R];
        
        for (int j = 0; j < samples_per_R; j++) {
            int idx = static_cast<int>(curand_uniform(&local_state) * samples_per_R);
            sum += lengths_for_R[idx];
        }
        
        resampled_log_L[i] = log(sum / samples_per_R);
    }
    
    // Calculate slope using linear regression
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    for (int i = 0; i < num_R_values; i++) {
        sum_x += log_R[i];
        sum_y += resampled_log_L[i];
        sum_xy += log_R[i] * resampled_log_L[i];
        sum_xx += log_R[i] * log_R[i];
    }
    
    bootstrap_slopes[tid] = (num_R_values * sum_xy - sum_x * sum_y) / 
                           (num_R_values * sum_xx - sum_x * sum_x);
    
    delete[] resampled_log_L;
    rng_states[tid] = local_state;
}

SimulationResults estimate_dimension_cuda(const SimulationParams& params) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Initialize CUDA RNG
    const int block_size = 256;
    const int num_blocks = (params.num_trials + block_size - 1) / block_size;
    CudaRNG rng(params.num_trials);
    
    // Initialize RNG states
    init_rng_kernel<<<num_blocks, block_size>>>(rng.states, params.num_trials);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Allocate device memory for results
    thrust::device_vector<size_t> d_lengths(params.num_trials * params.R_values.size());
    thrust::host_vector<size_t> h_lengths(params.num_trials * params.R_values.size());
    
    // Run simulations for each R value
    for (size_t i = 0; i < params.R_values.size(); i++) {
        const double R = params.R_values[i];
        simulate_LERW_3D_kernel<<<num_blocks, block_size>>>(
            rng.states,
            thrust::raw_pointer_cast(d_lengths.data() + i * params.num_trials),
            R,
            params.num_trials
        );
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    
    // Copy results back to host
    thrust::copy(d_lengths.begin(), d_lengths.end(), h_lengths.begin());
    
    // Calculate log values for regression
    std::vector<double> log_R(params.R_values.size());
    thrust::host_vector<double> log_L(params.R_values.size());
    
    for (size_t i = 0; i < params.R_values.size(); i++) {
        log_R[i] = std::log(params.R_values[i]);
        double mean_length = thrust::reduce(
            thrust::host,
            h_lengths.begin() + i * params.num_trials,
            h_lengths.begin() + (i + 1) * params.num_trials,
            0.0
        ) / params.num_trials;
        log_L[i] = std::log(mean_length);
    }
    
    // Copy log_R to device for bootstrap calculations
    thrust::device_vector<double> d_log_R = log_R;
    
    // Perform bootstrap analysis
    thrust::device_vector<double> d_bootstrap_slopes(params.num_bootstrap);
    const int bootstrap_blocks = (params.num_bootstrap + block_size - 1) / block_size;
    
    bootstrap_kernel<<<bootstrap_blocks, block_size>>>(
        rng.states,
        thrust::raw_pointer_cast(d_lengths.data()),
        thrust::raw_pointer_cast(d_bootstrap_slopes.data()),
        thrust::raw_pointer_cast(d_log_R.data()),
        params.R_values.size(),
        params.num_trials,
        params.num_bootstrap
    );
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Copy bootstrap results to host and calculate confidence intervals
    thrust::host_vector<double> h_bootstrap_slopes = d_bootstrap_slopes;
    thrust::sort(h_bootstrap_slopes.begin(), h_bootstrap_slopes.end());
    
    int lower_idx = (params.num_bootstrap * (1 - params.confidence_level) / 2);
    int upper_idx = params.num_bootstrap - lower_idx - 1;
    
    // Calculate initial fit for R-squared
    LinearRegression initial_fit(log_R, thrust::host_vector<double>(log_L));
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    // Prepare results
    SimulationResults results;
    results.lengths.resize(params.R_values.size());
    for (size_t i = 0; i < params.R_values.size(); i++) {
        results.lengths[i].assign(
            h_lengths.begin() + i * params.num_trials,
            h_lengths.begin() + (i + 1) * params.num_trials
        );
    }
    results.R_values = params.R_values;
    results.dimension_label = "3D (CUDA)";
    results.estimate = DimensionEstimate{
        initial_fit.get_slope(),
        h_bootstrap_slopes[lower_idx],
        h_bootstrap_slopes[upper_idx],
        initial_fit.get_r_squared(),
        initial_fit.get_residuals(),
        duration
    };
    
    // Print results summary
    std::cout << "\n3D (CUDA) Results:\n"
              << "Estimated dimension D: " << results.estimate.D 
              << " (" << (params.confidence_level * 100) << "% CI: "
              << results.estimate.ci_lower << " - "
              << results.estimate.ci_upper << ")\n"
              << "R² value: " << results.estimate.r_squared << "\n"
              << "Computation time: " << duration.count() << " seconds\n";
    
    write_results_csv(results, params);
    
    return results;
}

// Additional helper kernel for RNG initialization
__global__ void init_rng_kernel(curandState* states, int num_states) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_states) {
        curand_init(clock64(), tid, 0, &states[tid]);
    }
}