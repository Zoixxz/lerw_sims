import numpy as np
from scipy import stats  
import time
import multiprocessing
# from lerw_3d import *
# from lerw_2d import * # Optional import for when include_2d is True
from lerw_nn import simulate_nn
import time
from functools import wraps
from lerw_lr_1d import simulate_lr_1d
from lerw_hr import simulate_hr_L, simulate_hr_M
from lerw_lr_3d import simulate_lr_3d

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        print(f"\nExecution time of {func.__name__}: {execution_time:.2f} seconds")
        return result
    return wrapper

# def worker_3D(R):
#     return lambda: len(simulate_LERW_3D(R))

# def worker_2D(R):
#     path = simulate_LERW_2D(R)
#     return len(path)

def estimate_exponent(R_values, avg_lengths):
    log_R = np.log(R_values)
    log_L = np.log(avg_lengths)
    D, C = np.polyfit(log_R, log_L, 1)
    return D

def simulate_parallel(L, num_trials, dim):
    lerw = LERW_NN(L=L, dim=dim)

    with multiprocessing.Pool() as pool:
        lengths = pool.map(lerw.get_path_len, range(num_trials))
        total_length = sum(lengths)
        avg_length = total_length / num_trials
    return avg_length

def plot_results(R_values, avg_lengths, dimension, D):
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        # plt.title(f"LERW in {dimension}D")
        plt.title('Hierarchical LERW')
        plt.loglog(R_values, avg_lengths, 'o-', label='Data')
        plt.loglog(R_values, 
                  np.exp(np.polyval([D, np.log(avg_lengths[0]) - D * np.log(R_values[0])], 
                  np.log(R_values))), '--', 
                  label=f'Fit: D={D:.6f}')
        plt.xlabel('R')
        plt.ylabel('Average Length L')
        plt.legend()
        plt.show()
    except ImportError:
        print("Matplotlib not available for plotting")

def estimate_exponent_with_errors(R_values, avg_lengths, n_bootstrap=1000):
    """
    Estimate the scaling exponent D and its uncertainty using both linear regression
    and bootstrap resampling.
    
    Parameters:
    -----------
    R_values : list
        The R values used in the simulation
    avg_lengths : list
        The corresponding average lengths
    n_bootstrap : int
        Number of bootstrap samples for error estimation
    
    Returns:
    --------
    dict
        Contains the exponent D, its standard error, and bootstrap confidence intervals
    """
    log_R = np.log(R_values)
    log_L = np.log(avg_lengths)
    
    stat_results = stats.linregress(log_R, log_L)
    
    # Bootstrap estimation
    bootstrap_slopes = []
    bootstrap_intercepts = []
    n_points = len(R_values)
    
    for _ in range(n_bootstrap):
        # Resample indices with replacement
        indices = np.random.randint(0, n_points, size=n_points)
        boot_log_R = log_R[indices]
        boot_log_L = log_L[indices]
        
        # Fit on bootstrapped sample
        boot_slope, boot_intercept = np.polyfit(boot_log_R, boot_log_L, 1)
        bootstrap_slopes.append(boot_slope)
        bootstrap_intercepts.append(boot_intercept)
    
    # Calculate bootstrap confidence intervals
    bootstrap_slopes = np.array(bootstrap_slopes)
    bootstrap_intercepts = np.array(bootstrap_intercepts)
    confidence_intervals_slope = np.percentile(bootstrap_slopes, [2.5, 97.5])
    confidence_intervals_intercept = np.percentile(bootstrap_intercepts, [2.5, 97.5])
    
    return {
        'D': stat_results.slope,
        'a': stat_results.intercept,
        'std_error_slope': stat_results.stderr,
        'std_error_intercept': stat_results.intercept_stderr,
        'bootstrap_ci_slope': confidence_intervals_slope,
        'bootstrap_ci_intercept': confidence_intervals_intercept,
        'r_squared': stat_results.rvalue**2,
        'bootstrap_std_slope': np.std(bootstrap_slopes),
        'bootstrap_std_intercept': np.std(bootstrap_intercepts)
    }

def plot_results_with_errors(R_values, avg_lengths, dimension, results):
    """
    Plot results including error estimates
    """
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.title(f"Hierarchical LERW")
        
        # Data points
        plt.loglog(R_values, avg_lengths, 'o', label='Data')
        
        # Best fit line
        log_R = np.log(R_values)
        fit_line = np.exp(np.polyval(
            [results['D'], np.log(avg_lengths[0]) - results['D'] * np.log(R_values[0])],
            np.log(R_values)))

        plt.loglog(R_values, fit_line, '--',
                  label=f'Fit: D={results["D"]:.6f}±{results["std_error_slope"]:.6f}')
        
        # Add bootstrap CI information to plot
        plt.text(0.05, 0.95, 
                f'95% Bootstrap CI: [{results["bootstrap_ci_slope"][0]:.6f}, {results["bootstrap_ci_slope"][1]:.6f}]\n' +
                f'R² = {results["r_squared"]:.6f}',
                transform=plt.gca().transAxes,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.xlabel('R')
        plt.ylabel('Average Length L')
        plt.legend(loc='lower right')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for plotting")

def run_simulation(R_values, num_trials=1000, plot=False, type='nn'):
    """
    Updated run_simulation function with error estimation
    """
    results = {
        '3D': {'avg_lengths': [], 'stats': None}
    }
    # if include_2d:
    #     results['2D'] = {'avg_lengths': [], 'stats': None}
    
    print('\nSimulating LERW...')
    for R in R_values:
        if type == 'gpu':
            from lerw_gpu import simulate_lerw_gpu
            avg_length = simulate_lerw_gpu(L=R, num_trials=num_trials)
        elif type == 'nn':
            avg_length = simulate_nn(L=R, num_trials=num_trials)
        elif type == 'lr1':
            avg_length = simulate_lr_1d(L=R, alpha=1, num_trials=num_trials)
        elif type == 'hr':
            avg_length = simulate_hr_L(L=R, alpha=0.5, num_trials=num_trials)
        elif type == 'lr3':
            avg_length = simulate_lr_3d(L=R, alpha=0.5, num_trials=num_trials)
        results['3D']['avg_lengths'].append(avg_length)
        print(f"R = {R}, Avg Length = {avg_length}")

    # 3D Simulation
    # print("\nSimulating LERW...")
    # for R in R_values:
        # globals()['R'] = R
        # avg_length = simulate_parallel(R, num_trials, dim=3)
        # avg_length = simulate_lerw_gpu(L=R, num_trials=num_trials)
        # results['3D']['avg_lengths'].append(avg_length)
        # print(f"R = {R}, Avg Length = {avg_length}")
    
    # Optional 2D Simulation
    # if include_2d:
    #     print("\nSimulating LERW in 2D...")
    #     for R in R_values:
    #         globals()['R'] = R
    #         avg_length = simulate_parallel(R, num_trials, '2D')
    #         results['2D']['avg_lengths'].append(avg_length)
    #         print(f"R = {R}, Avg Length = {avg_length}")
    
    # Calculate exponents with error estimates
    print("\nEstimating the exponent D with error analysis...")
    results['3D']['stats'] = estimate_exponent_with_errors(
        np.array(R_values), 
        np.array(results['3D']['avg_lengths'])
    )
    print(f"3D Results:")
    print(f"D (slope) = {results['3D']['stats']['D']:.6f} ± {results['3D']['stats']['std_error_slope']:.6f}")
    print(f"a (intercept) = {results['3D']['stats']['a']:.6f} ± {results['3D']['stats']['std_error_intercept']:.6f}")
    print(f"D 95% Bootstrap CI: [{results['3D']['stats']['bootstrap_ci_slope'][0]:.6f}, "
          f"{results['3D']['stats']['bootstrap_ci_slope'][1]:.6f}]")
    print(f"a 95% Bootstrap CI: [{results['3D']['stats']['bootstrap_ci_intercept'][0]:.6f}, "
          f"{results['3D']['stats']['bootstrap_ci_intercept'][1]:.6f}]")
    print(f"R² = {results['3D']['stats']['r_squared']:.6f}")
    
    # if include_2d:
    #     results['2D']['stats'] = estimate_exponent_with_errors(
    #         np.array(R_values), 
    #         np.array(results['2D']['avg_lengths'])
    #     )
    #     print(f"R² = {results['2D']['stats']['r_squared']:.6f}")
    
    # Plotting with error information
    if plot:
        plot_results_with_errors(R_values, results['3D']['avg_lengths'], '3', results['3D']['stats'])
        # if include_2d:
        #     plot_results_with_errors(R_values, results['2D']['avg_lengths'], '2', results['2D']['stats'])
    
    return results

@timer
def main():
    # Example usage
    # R_values = [pow(2, i) for i in range(5, 10)]
    # R_values = [50, 100, 150, 200, 250]
    R_values = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

    # Run simulation with only 3D
    results = run_simulation(R_values, num_trials=10000, plot=True, type='lr3')
    # results = run_simulation(R_values, num_trials=1000, include_2d=True)

if __name__ == "__main__":
    main()
