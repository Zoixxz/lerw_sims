from typing import List, Union, Tuple, Optional, Sequence
from numpy.typing import NDArray
import numpy as np
from scipy import stats

# FIXME: add typehints

def rank_normalize(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    0.375 and 0.25 from [Blom, G. (1958). Statistical Estimates and Transformed Beta-Variables. Wiley; New
    York. MR0095553. 12]
    """
    ranks = stats.rankdata(x, method='average')
    return stats.norm.ppf((ranks - 0.375) / (len(ranks) + 0.25))

def compute_acf(x: NDArray[np.float64], nlags: int = 40) -> NDArray[np.float64]:
    """
    autocorrelation function for input array
    nlags : int
        Number of lags to compute
    Returns:
    array
        Array of autocorrelations
    """
    n = len(x)
    if n < 2:
        return np.array([])
        
    x = np.ravel(x)
    x = x - np.mean(x)
    # autocovariance
    acf = np.correlate(x, x, 'full')[n-1:] / np.sum(x**2)

    return acf[:min(len(acf), nlags+1)]

def fold_values(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """mean or median? Vehtari et. al. use median (why? is it becaue of stability under outliers?)"""
    m_x = np.median(x)  # alternatively why not use np.mean()?
    return np.abs(x - m_x)

def equalize_chain_lengths(chains: Union[List[NDArray[np.float64]],
                                         NDArray[np.float64]]) -> NDArray[np.float64]:
    """
    make chains have the same length by truncating to shortest chain
    """
    if isinstance(chains, list):
        min_length = min(len(chain) for chain in chains)
        # truncate 
        chains = [chain[:min_length] for chain in chains]
        chains = np.array(chains)
    return chains

def bulk_tail_ess(chains: Union[List[NDArray[np.float64]],
                                         NDArray[np.float64]]) -> Tuple[float, float]:

    """
    Calculate bulk and tail ESS 
    return 
    (bulk_ess, tail_ess)
    """
    # equal chain lengths convert convert to np array
    chains = equalize_chain_lengths(chains)
    chains = np.array(chains)
    
    if len(chains.shape) == 2:
        n_chains, n_samples = chains.shape
        n_dims = 1
        chains = chains.reshape(n_chains, n_samples, 1)
    else:
        n_chains, n_samples, n_dims = chains.shape
    
    # results arrays 
    bulk_ess_dims = np.zeros(n_dims)
    tail_ess_dims = np.zeros(n_dims)
    
    for dim in range(n_dims):
        chains_dim = chains[:, :, dim]
        
        normalized_chains = np.array([rank_normalize(chain) for chain in chains_dim])
        
        bulk_acf = np.array([compute_acf(chain) for chain in normalized_chains])
        
        # positive autocorrelations up to first negative value
        positive_acf_sums = []
        for acf in bulk_acf:
            positive_sum = 0
            for rho in acf[1:]:  # Skip lag 0
                if rho <= 0:
                    break
                positive_sum += rho
            positive_acf_sums.append(positive_sum)
        
        # bulk ESS using mean of positive autocorrelations
        mean_positive_sum = np.mean(positive_acf_sums)
        bulk_ess_dims[dim] = chains_dim.size / (1 + 2 * mean_positive_sum)
        
        # tail ESS using quantile-based approach
        q_low = np.percentile(chains_dim, 25, axis=1)
        q_high = np.percentile(chains_dim, 75, axis=1)
        quantile_diffs = q_high - q_low
        mean_diff = np.mean(quantile_diffs)
        var_diff = np.var(quantile_diffs)
        
        # zero variance
        if var_diff < 1e-10:  # Using small threshold instead of exact zero
            tail_ess_dims[dim] = chains_dim.size  # variance zero then all chains agree perfectly
        else:
            tail_ess_dims[dim] = chains_dim.size * (mean_diff ** 2) / var_diff
    
    return np.min(bulk_ess_dims), np.min(tail_ess_dims)

def improved_rhat(chains: Union[List[NDArray[np.float64]], NDArray[np.float64]],
                  is_bounded: bool = False) -> float:
    """
    improved R-hat statistic
    """
    chains = np.array(equalize_chain_lengths(chains))
    
    # if different input shapes
    if len(chains.shape) == 2:
        n_chains, n_samples = chains.shape
        n_dims = 1
        chains = chains.reshape(n_chains, n_samples, 1)
    else:
        n_chains, n_samples, n_dims = chains.shape
    
    if n_samples < 4:
        raise ValueError("Chains must have at least 4 samples each")
    
    rhat_values = []

    for dim in range(n_dims):
        chains_dim = chains[:, :, dim]
        
        # Split chains
        split_chains = np.vstack([chains_dim[:, :n_samples//2], 
                                chains_dim[:, n_samples//2:n_samples//2*2]])
        
        # Rank normalize
        normalized_chains = np.array([rank_normalize(chain) for chain in split_chains])
        
        # Fold if bounded
        if is_bounded:
            normalized_chains = np.array([fold_values(chain) for chain in normalized_chains])
        
        # Calculate between and within-chain variances
        chain_means = np.mean(normalized_chains, axis=1)
        chain_vars = np.var(normalized_chains, axis=1, ddof=1)
        
        # Overall variance
        W = np.mean(chain_vars)  # within-chain variance
        B = np.var(chain_means, ddof=1) * normalized_chains.shape[1]  # between-chain variance
        
        # Calculate variance estimate and R-hat
        var_estimate = (normalized_chains.shape[1] - 1) / normalized_chains.shape[1] * W + B / normalized_chains.shape[1]
        rhat_values.append(np.sqrt(var_estimate / W))
    
    return np.max(rhat_values)  # Return maximum R-hat across dimensions