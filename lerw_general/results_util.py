import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import warnings

import matplotlib.pyplot as plt

def basic_2d_plot(x_vals, y_vals):
    plt.figure(figsize=(12, 7))
    plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='b', label='Data')

    # Adding labels and title
    plt.xlabel('alpha')
    plt.ylabel('D')
    plt.title('D-alpha graph 3D')

    # Setting axis limits and ticks
    plt.xlim(0, 3)
    plt.ylim(0, 2)
    plt.xticks([i * 0.1 for i in range(0, 32)])  # From 0 to 3 with 0.1 steps
    plt.yticks([i * 0.05 for i in range(0, 41)])  # From 0 to 1.3 with 0.05 steps

    # Optional: Add a grid and legend
    plt.grid(True)
    plt.legend()

    # Save and display the plot
    plt.savefig('d-alpha-fig_temp.png')
    plt.show()
    plt.close()


def estimate_exponent_with_errors(L_values, avg_lengths, n_bootstrap=1000):
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
    log_R = np.log(L_values)
    log_L = np.log(avg_lengths)
    
    stat_results = stats.linregress(log_R, log_L)
    
    # Bootstrap estimation
    bootstrap_slopes = []
    bootstrap_intercepts = []
    n_points = len(L_values)
    
    for _ in range(n_bootstrap):
        # Resample indices with replacement
        indices = np.random.randint(0, n_points, size=n_points)
        boot_log_R = log_R[indices]
        boot_log_L = log_L[indices]
        
        # Fit on bootstrapped sample
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
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

def plot_results_with_errors(L_values, avg_lengths, results, alpha=None):
    """
    Plot results including error estimates
    """
        
    plt.figure(figsize=(10, 6))
    plt.title(f"Long Range LERW")
    
    # Data points
    plt.loglog(L_values, avg_lengths, 'o', label='Data')
    
    # Best fit line
    log_R = np.log(L_values)
    fit_line = np.exp(np.polyval(
        [results['D'], np.log(avg_lengths[0]) - results['D'] * np.log(L_values[0])],
        np.log(L_values)))

    plt.loglog(L_values, fit_line, '--',
                label=f'Fit: D={results["D"]:.6f}±{results["std_error_slope"]:.6f}')
    
    # Add bootstrap CI information to plot
    plt.text(0.05, 0.95, 
            f'95% Bootstrap CI: [{results["bootstrap_ci_slope"][0]:.6f}, {results["bootstrap_ci_slope"][1]:.6f}]\n' +
            f'R² = {results["r_squared"]:.6f}\n' +
            f'alpha = {alpha}',
            transform=plt.gca().transAxes,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel('R')
    plt.ylabel('Average Length L')
    plt.legend(loc='lower right')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(f'temp_fig{alpha}.png')
    # plt.show()
    plt.close()

if __name__ == '__main__':
    f = open('/home/epsilon/Code/Research/Long_Range_Simulations/lerw_sims/lerw_long_range/results_3d_test.txt', 'r')

    len_data = {}

    for line in f:
        info = list(line.split(' '))
        alpha = int(info[0])
        L = 2**int(info[1])
        avg_len = float(info[2])

        if alpha not in len_data:
            len_data[alpha] = {}

        len_data[alpha][L] = avg_len

    D_vals = []
    alpha_vals = list(len_data.keys())
    for alpha in alpha_vals:
        L_vals = list(len_data[alpha].keys())
        avg_lengths = list(len_data[alpha].values())
        results = estimate_exponent_with_errors(L_vals, avg_lengths, n_bootstrap=1000)
        plot_results_with_errors(L_vals, avg_lengths, results, alpha/10.0)
        D_vals.append(results['D'])

    alpha_vals = [a/10.0 for a in alpha_vals]
    basic_2d_plot(alpha_vals, D_vals)
