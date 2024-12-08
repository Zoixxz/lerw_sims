import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import warnings

def basic_2d_plot(x_vals, y_vals, color, label):
    plt.plot(x_vals, y_vals, marker='o', linestyle='-', color=color, label=label)

    # Adding labels and title
    plt.xlabel('alpha/d')
    plt.ylabel('D/d')
    plt.title('D-alpha graph')

    # Setting axis limits and ticks
    plt.xlim(0, 3)
    plt.ylim(0, 1.3)
    plt.xticks([i * 0.1 for i in range(0, 32)])  # From 0 to 3 with 0.1 steps
    plt.yticks([i * 0.05 for i in range(0, 27)])  # From 0 to 1.3 with 0.05 steps

    # Add grid and legend
    plt.grid(True)
    plt.legend()

def estimate_exponent_with_errors(L_values, avg_lengths, n_bootstrap=1000):
    log_R = np.log(L_values)
    log_L = np.log(avg_lengths)
    stat_results = stats.linregress(log_R, log_L)

    bootstrap_slopes = []
    n_points = len(L_values)

    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n_points, size=n_points)
        boot_log_R = log_R[indices]
        boot_log_L = log_L[indices]
        boot_slope, _ = np.polyfit(boot_log_R, boot_log_L, 1)
        bootstrap_slopes.append(boot_slope)

    bootstrap_slopes = np.array(bootstrap_slopes)
    confidence_intervals_slope = np.percentile(bootstrap_slopes, [2.5, 97.5])

    return {
        'D': stat_results.slope,
        'bootstrap_ci_slope': confidence_intervals_slope,
        'std_error_slope': stat_results.stderr,
    }

if __name__ == '__main__':
    files = [
        '/home/epsilon/Code/Research/Long_Range_Simulations/lerw_sims/lerw_long_range/results_1d_1.txt',
        '/home/epsilon/Code/Research/Long_Range_Simulations/lerw_sims/lerw_long_range/results_2d_1.txt',
        '/home/epsilon/Code/Research/Long_Range_Simulations/lerw_sims/lerw_long_range/results_3d_1.txt'
    ]

    plt.figure(figsize=(12, 7))

    for i, file_path in enumerate(files):
        len_data = {}
        with open(file_path, 'r') as f:
            for line in f:
                info = list(line.split(' '))
                alpha = int(info[0])
                L = 2 ** int(info[1])
                avg_len = float(info[2])

                if alpha not in len_data:
                    len_data[alpha] = {}
                len_data[alpha][L] = avg_len

        D_vals = []
        alpha_vals = list(len_data.keys())
        for alpha in alpha_vals:
            L_vals = list(len_data[alpha].keys())
            avg_lengths = list(len_data[alpha].values())
            results = estimate_exponent_with_errors(L_vals, avg_lengths)
            D_vals.append(results['D'])

        alpha_vals = [a / 10.0 for a in alpha_vals]
        
        if i == 0:
            color = 'b'
        elif i == 1:
            color = 'r'
        else:
            color = 'g'

        label = f'{i + 1}D Simulation'
        
        for j in range(len(alpha_vals)):
            alpha_vals[j] /= (i + 1)
        for j in range(len(D_vals)):
            D_vals[j] /= (i + 1)

        basic_2d_plot(alpha_vals, D_vals, color=color, label=label)

    # Save and display the combined plot
    plt.savefig('d-alpha-fig_combined.png')
    plt.show()
