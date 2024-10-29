import numpy as np
from scipy import stats  
import random
import math
from collections import defaultdict
import time
import multiprocessing
from lerw_3d import *
from lerw_2d import * # Optional import for when include_2d is True
import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        print(f"\nExecution time of {func.__name__}: {execution_time:.2f} seconds")
        return result
    return wrapper

def worker_3D(_):
    path = simulate_LERW_3D(R)
    return len(path)

def worker_2D(_):
    path = simulate_LERW_2D(R)
    return len(path)

def estimate_exponent(R_values, avg_lengths):
    log_R = np.log(R_values)
    log_L = np.log(avg_lengths)
    D, C = np.polyfit(log_R, log_L, 1)
    return D

def simulate_parallel(R, num_trials, worker_func):
    with multiprocessing.Pool() as pool:
        lengths = pool.map(worker_func, range(num_trials))
        total_length = sum(lengths)
        avg_length = total_length / num_trials
    return avg_length

def plot_results(R_values, avg_lengths, dimension, D):
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.title(f"LERW in {dimension}D")
        plt.loglog(R_values, avg_lengths, 'o-', label='Data')
        plt.loglog(R_values, 
                  np.exp(np.polyval([D, np.log(avg_lengths[0]) - D * np.log(R_values[0])], 
                  np.log(R_values))), '--', 
                  label=f'Fit: D={D:.3f}')
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
    R_values : array-like
        The R values used in the simulation
    avg_lengths : array-like
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
    
    # Regular linear regression with stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_R, log_L)
    
    # Bootstrap estimation
    bootstrap_slopes = []
    n_points = len(R_values)
    
    for _ in range(n_bootstrap):
        # Resample indices with replacement
        indices = np.random.randint(0, n_points, size=n_points)
        boot_log_R = log_R[indices]
        boot_log_L = log_L[indices]
        
        # Fit on bootstrapped sample
        boot_slope, _ = np.polyfit(boot_log_R, boot_log_L, 1)
        bootstrap_slopes.append(boot_slope)
    
    # Calculate bootstrap confidence intervals
    bootstrap_slopes = np.array(bootstrap_slopes)
    confidence_intervals = np.percentile(bootstrap_slopes, [2.5, 97.5])
    
    return {
        'D': slope,
        'std_error': std_err,
        'bootstrap_ci': confidence_intervals,
        'r_squared': r_value**2,
        'bootstrap_std': np.std(bootstrap_slopes)
    }

def plot_results_with_errors(R_values, avg_lengths, dimension, results):
    """
    Plot results including error estimates
    """
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.title(f"LERW in {dimension}D")
        
        # Data points
        plt.loglog(R_values, avg_lengths, 'o', label='Data')
        
        # Best fit line
        log_R = np.log(R_values)
        fit_line = np.exp(np.polyval(
            [results['D'], np.log(avg_lengths[0]) - results['D'] * np.log(R_values[0])],
            np.log(R_values)))
        plt.loglog(R_values, fit_line, '--',
                  label=f'Fit: D={results["D"]:.3f}±{results["std_error"]:.3f}')
        
        # Add bootstrap CI information to plot
        plt.text(0.05, 0.95, 
                f'95% Bootstrap CI: [{results["bootstrap_ci"][0]:.3f}, {results["bootstrap_ci"][1]:.3f}]\n' +
                f'R² = {results["r_squared"]:.3f}',
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.xlabel('R')
        plt.ylabel('Average Length L')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for plotting")

def run_simulation(R_values, num_trials=1000, include_2d=False, plot=False):
    """
    Updated run_simulation function with error estimation
    """
    results = {
        '3D': {'avg_lengths': [], 'stats': None}
    }
    if include_2d:
        results['2D'] = {'avg_lengths': [], 'stats': None}
    
    # 3D Simulation
    print("\nSimulating LERW in 3D...")
    for R in R_values:
        globals()['R'] = R
        avg_length = simulate_parallel(R, num_trials, worker_3D)
        results['3D']['avg_lengths'].append(avg_length)
        print(f"R = {R}, Avg Length = {avg_length}")
    
    # Optional 2D Simulation
    if include_2d:
        print("\nSimulating LERW in 2D...")
        for R in R_values:
            globals()['R'] = R
            avg_length = simulate_parallel(R, num_trials, worker_2D)
            results['2D']['avg_lengths'].append(avg_length)
            print(f"R = {R}, Avg Length = {avg_length}")
    
    # Calculate exponents with error estimates
    print("\nEstimating the exponent D with error analysis...")
    results['3D']['stats'] = estimate_exponent_with_errors(
        np.array(R_values), 
        np.array(results['3D']['avg_lengths'])
    )
    print(f"3D Results:")
    print(f"D = {results['3D']['stats']['D']:.3f} ± {results['3D']['stats']['std_error']:.3f}")
    print(f"95% Bootstrap CI: [{results['3D']['stats']['bootstrap_ci'][0]:.3f}, "
          f"{results['3D']['stats']['bootstrap_ci'][1]:.3f}]")
    print(f"R² = {results['3D']['stats']['r_squared']:.3f}")
    
    if include_2d:
        results['2D']['stats'] = estimate_exponent_with_errors(
            np.array(R_values), 
            np.array(results['2D']['avg_lengths'])
        )
        print(f"R² = {results['2D']['stats']['r_squared']:.3f}")
    
    # Plotting with error information
    if plot:
        plot_results_with_errors(R_values, results['3D']['avg_lengths'], '3', results['3D']['stats'])
        if include_2d:
            plot_results_with_errors(R_values, results['2D']['avg_lengths'], '2', results['2D']['stats'])
    
    return results

@timer
def main():
    # Example usage
    R_values = [40, 80, 160, 320, 500, 1000]
    
    # Run simulation with only 3D
    results = run_simulation(R_values, num_trials=1000, include_2d=False, plot=False)
    # results = run_simulation(R_values, num_trials=1000, include_2d=True)

if __name__ == "__main__":
    main()
