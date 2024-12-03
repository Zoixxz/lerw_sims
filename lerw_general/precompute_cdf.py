from scipy.special import zeta
import numpy as np
import array

def lr_1d(alpha):
    DIM = 1
    Z = 1.0 / zeta(DIM + alpha, 1)
    exponent = DIM + alpha

    p = 0.999999
    cdf = array.array('f')
    cumulative = 0.0
    r = 1
    
    while True:
        P_r = Z / (r ** exponent)
        cumulative += P_r
        cdf.append(cumulative)

        if r % 5000000 == 0:
            print(cumulative, r)

        if cumulative >= p:
            np.save(f'cdf_lr_1d_a_{alpha}_6f.npy', np.array(cdf))
            return
        
        r += 1

if __name__ == '__main__':
    lr_1d(alpha=0.5)