from MH_sampler_v1 import MHS
from MH_sampler_v1 import visualize_samples, plot_acceptance_rate_history
from MH_sampler_v1 import plot_autocorrelation, compute_integrated_autocorrelation_time
from MH_sampler_v1 import improved_rhat, bulk_tail_ess
import numpy as np

if __name__ == "__main__":
    # Parameters
    a = 2
    d = 3
    n_chains = 4  # Number of chains, need to be multiple of 4
    n_samples = int(1e4)
    n_warmup = int(1e4)
    
    # Run multiple chains
    chains = []
    for _ in range(n_chains):
        mhs = MHS(alpha=a, d=d)
        chain_samples = mhs.sample_gen(N=n_samples, discard_N=n_warmup)
        # Convert samples to array for easier handling
        chain_array = np.array(chain_samples)
        chains.append(chain_array)
    
    # Stack chains for visualization (using first chain)
    samples = chains[0]  # Use first chain for original visualizations
    
    # Compute diagnostics using all chains
    rhat = improved_rhat(chains)
    print(f"R-hat statistic: {rhat:.3f}")
    
    bulk_ess, tail_ess = bulk_tail_ess(chains)
    print(f"Bulk ESS: {bulk_ess:.1f}")
    print(f"Tail ESS: {tail_ess:.1f}")
    
    # Original visualizations (using first chain)
    visualize_samples(samples, alpha=a, d=d)
    plot_acceptance_rate_history(samples)

    # Autocorrelation analysis
    max_lag = int(np.min([len(samples) / 3, 1000]))
    plot_autocorrelation(samples, max_lag=max_lag)
    
    # Print integrated autocorrelation times for each dimension
    for dim in range(d):
        tau_int = compute_integrated_autocorrelation_time(samples, dimension=dim)
        print(f"Integrated autocorrelation time for dimension {dim+1}: {tau_int:.2f}")