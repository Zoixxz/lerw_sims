# disclaimer: plotting parts are GPT generated
import numpy as np
from typing import List, Tuple

from scipy.stats import norm

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec


def visualize_samples(samples: List[tuple[int, ...]], alpha: float, d: int, 
                     max_display_dims: int = 3) -> None:
    """
    Create comprehensive visualization of MHS samples.
    
    Args:
        samples: List of d-dimensional samples from MHS
        alpha: The alpha parameter used in sampling
        d: Dimension of the samples
        max_display_dims: Maximum number of dimensions to display (default 3)
    """
    # Convert samples to numpy array for easier manipulation
    samples_array = np.array(samples)
    
    # Set up the figure layout
    display_dims = min(d, max_display_dims)
    fig = plt.figure(figsize=(15, 5 * (1 + display_dims // 2)))
    gs = GridSpec(2 + display_dims // 2, 3)
    
    # 1. Plot trace plots for first few dimensions
    ax_trace = fig.add_subplot(gs[0, :])
    for i in range(display_dims):
        ax_trace.plot(samples_array[:, i], 
                     alpha=0.7, 
                     label=f'Dimension {i+1}')
    ax_trace.set_title('Trace Plot of First Few Dimensions')
    ax_trace.set_xlabel('Sample Index')
    ax_trace.set_ylabel('Value')
    ax_trace.legend()
    
    # 2. Plot marginal distributions
    for i in range(display_dims):
        row = 1 + i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col])
        
        # Convert data to numpy array explicitly before plotting
        data = samples_array[:, i].astype(np.float64)
        
        # Use plt.hist instead of sns.histplot for more reliable behavior
        ax.hist(data, bins='auto', density=True, alpha=0.7)
        
        # Add KDE separately if desired
        try:
            from scipy import stats
            kde = stats.gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            ax.plot(x_range, kde(x_range), 'r-', lw=2)
        except Exception as e:
            print(f"Warning: Could not compute KDE for dimension {i+1}: {e}")
            
        ax.set_title(f'Marginal Distribution - Dimension {i+1}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        
    # 3. If d >= 2, add scatter plot of first two dimensions
    if d >= 2:
        ax_scatter = fig.add_subplot(gs[1, 2])
        scatter = ax_scatter.scatter(samples_array[:, 0],
                                   samples_array[:, 1],
                                   alpha=0.1,
                                   s=1)
        ax_scatter.set_title('2D Scatter Plot (Dim 1 vs Dim 2)')
        ax_scatter.set_xlabel('Dimension 1')
        ax_scatter.set_ylabel('Dimension 2')
        
        # Add density contours using numpy arrays
        try:
            x = samples_array[:, 0]
            y = samples_array[:, 1]
            sns.kdeplot(x=x, y=y,
                       ax=ax_scatter,
                       levels=5,
                       color='red',
                       alpha=0.5)
        except Exception as e:
            print(f"Warning: Could not compute density contours: {e}")
            
    # 4. If d >= 3, add 3D scatter plot
    if d >= 3:
        ax_3d = fig.add_subplot(gs[-1, 2], projection='3d')
        scatter_3d = ax_3d.scatter(samples_array[:, 0],
                                  samples_array[:, 1],
                                  samples_array[:, 2],
                                  alpha=0.1,
                                  s=1)
        ax_3d.set_title('3D Scatter Plot')
        ax_3d.set_xlabel('Dimension 1')
        ax_3d.set_ylabel('Dimension 2')
        ax_3d.set_zlabel('Dimension 3')
    
    plt.tight_layout()
    plt.show()

def plot_acceptance_rate_history(samples: List[Tuple[int, ...]], 
                               window_size: int = 1000) -> None:
    """
    Plot the running acceptance rate over time.
    
    Args:
        samples: List of samples from MHS
        window_size: Size of window for computing running acceptance rate
    """
    samples_array = np.array(samples)
    n_samples = len(samples)
    
    # Compute running acceptance rate
    transitions = np.any(samples_array[1:] != samples_array[:-1], axis=1)
    running_rate = np.convolve(transitions, 
                              np.ones(window_size)/window_size, 
                              mode='valid') * 100
    
    # Create x-axis values with matching length
    x_values = np.arange(len(running_rate)) + window_size - 1
    
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, running_rate)
    plt.title(f'Running Acceptance Rate (Window Size: {window_size})')
    plt.xlabel('Sample Index')
    plt.ylabel('Acceptance Rate (%)')
    plt.grid(True)
    plt.show()

def compute_autocorrelation(samples: List[Tuple[int, ...]], 
                          max_lag: int = 100,
                          dimension: int = 0) -> np.ndarray:
    """
    Compute autocorrelation function for a specific dimension of the chain.
    
    Args:
        samples: List of d-dimensional samples from MHS
        max_lag: Maximum lag to compute autocorrelation for
        dimension: Which dimension to compute autocorrelation for
    
    Returns:
        Array of autocorrelation values for lags 0 to max_lag
    """
    # Extract the specified dimension
    x = np.array([sample[dimension] for sample in samples])
    
    # Compute mean and variance
    mean = np.mean(x)
    var = np.var(x)
    
    # Initialize autocorrelation array
    acf = np.zeros(max_lag + 1)
    n = len(x)
    
    # Compute autocorrelation for each lag
    for lag in range(max_lag + 1):
        # Compute autocovariance
        autocovariance = np.mean((x[:-lag] - mean) * (x[lag:] - mean)) if lag > 0 else var
        acf[lag] = autocovariance / var
        
    return acf

def estimate_mixing_time(acf: np.ndarray, threshold: float = 0.05) -> int:
    """
    Estimate mixing time based on when autocorrelation falls below threshold.
    
    Args:
        acf: Array of autocorrelation values
        threshold: Threshold below which chain is considered mixed
        
    Returns:
        Estimated mixing time (lag where autocorrelation falls below threshold)
    """
    # Find first lag where autocorrelation falls below threshold
    mixing_time = np.where(np.abs(acf) < threshold)[0]
    
    if len(mixing_time) == 0:
        return len(acf)  # If threshold never reached, return max lag
    
    return mixing_time[0]

def plot_autocorrelation(samples: List[Tuple[int, ...]], 
                        max_lag: int = 100,
                        dimensions: List[int] = None,
                        threshold: float = 0.05) -> None:
    """
    Plot autocorrelation function and estimate mixing time.
    
    Args:
        samples: List of d-dimensional samples from MHS
        max_lag: Maximum lag to compute autocorrelation for
        dimensions: List of dimensions to analyze (default: first 3 dimensions)
        threshold: Threshold for mixing time estimation
    """
    if dimensions is None:
        dimensions = list(range(min(3, len(samples[0]))))
    
    plt.figure(figsize=(12, 6))
    
    # Compute confidence bands for pure noise (95% confidence interval)
    n = len(samples)
    conf_interval = norm.ppf(0.975) / np.sqrt(n)
    
    for dim in dimensions:
        # Compute autocorrelation
        acf = compute_autocorrelation(samples, max_lag, dim)
        lags = np.arange(len(acf))
        
        # Plot autocorrelation
        plt.plot(lags, acf, label=f'Dimension {dim+1}', alpha=0.7)
        
        # Estimate mixing time
        mixing_time = estimate_mixing_time(acf, threshold)
        
        # Only plot mixing time point if it's within our lag range
        if mixing_time < len(acf):
            plt.plot(mixing_time, acf[mixing_time], 'o', 
                    label=f'Mixing time D{dim+1}: {mixing_time}')
        else:
            print(f"Warning: Dimension {dim+1} did not mix within {max_lag} lags")
    
    # Add confidence bands
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axhline(y=conf_interval, color='r', linestyle=':', alpha=0.3, 
                label='95% Confidence Bands')
    plt.axhline(y=-conf_interval, color='r', linestyle=':', alpha=0.3)
    plt.axhline(y=threshold, color='g', linestyle=':', alpha=0.3,
                label=f'Threshold ({threshold})')
    plt.axhline(y=-threshold, color='g', linestyle=':', alpha=0.3)
    
    plt.title('Autocorrelation Function by Dimension')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
def compute_integrated_autocorrelation_time(samples: List[Tuple[int, ...]],
                                          dimension: int = 0,
                                          max_lag: int = None) -> float:
    """
    Compute integrated autocorrelation time for a specific dimension.
    
    Args:
        samples: List of d-dimensional samples from MHS
        dimension: Which dimension to analyze
        max_lag: Maximum lag to consider (default: automated selection)
        
    Returns:
        Integrated autocorrelation time
    """
    # Extract the specified dimension
    x = np.array([sample[dimension] for sample in samples])
    
    if max_lag is None:
        # Automated selection of max_lag (using Sokal's adaptive truncation criterion)
        max_lag = int(np.min([len(x) / 3, 1000]))
    
    # Compute autocorrelation
    acf = compute_autocorrelation(samples, max_lag, dimension)
    
    # Compute integrated autocorrelation time
    # Using the formula: τ_int = 1 + 2∑(ρ(k)) from k=1 to max_lag
    tau_int = 1 + 2 * np.sum(acf[1:])
    
    return tau_int
