import numpy as np
import random
import math
from collections import defaultdict
import time
import multiprocessing
from lerw_3d import *
from lerw_2d import * # Optional import for when include_2d is True

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

def run_simulation(R_values, num_trials=1000, include_2d=False):
    """
    Run LERW simulation primarily for 3D, with optional 2D simulation.
    
    Parameters:
    -----------
    R_values : list
        List of R values to simulate
    num_trials : int, optional
        Number of trials per R value (default: 1000)
    include_2d : bool, optional
        Whether to include 2D simulation (default: False)
    
    Returns:
    --------
    dict
        Dictionary containing simulation results and exponents
    """
    results = {
        '3D': {'avg_lengths': [], 'exponent': None}
    }
    if include_2d:
        results['2D'] = {'avg_lengths': [], 'exponent': None}

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

    # Calculate exponents
    print("\nEstimating the exponent D...")
    results['3D']['exponent'] = estimate_exponent(R_values, results['3D']['avg_lengths'])
    print(f"Estimated exponent D in 3D: {results['3D']['exponent']}")
    
    if include_2d:
        results['2D']['exponent'] = estimate_exponent(R_values, results['2D']['avg_lengths'])
        print(f"Estimated exponent D in 2D: {results['2D']['exponent']}")

    # Plotting
    plot_results(R_values, results['3D']['avg_lengths'], '3', results['3D']['exponent'])
    if include_2d:
        plot_results(R_values, results['2D']['avg_lengths'], '2', results['2D']['exponent'])

    return results

def main():
    # Example usage
    R_values = list(range(0, 241, 20))[1:]
    
    # Run simulation with only 3D
    results = run_simulation(R_values, num_trials=500, include_2d=False)
    
    # If you want to include 2D simulation, uncomment the following line:
    # results = run_simulation(R_values, num_trials=1000, include_2d=True)

if __name__ == "__main__":
    main()
