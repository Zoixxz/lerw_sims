# long range correlation, everything is 1D for now

from scipy.special import zeta
from scipy.constants import *
import numpy as np
import random
import multiprocessing
from functools import partial

DIM = 1 # simulating on 1D

def generate_random_r(Z, alpha):
    p = random.random()

    cdf = 0
    r = 1
    while True:
        P_r = Z / pow(r, DIM + alpha)
        cdf += P_r

        if cdf >= p:
            return r
        
        r += 1

def simulate(L, alpha, Z):
        path = []
        visited = {}

        curr_pos = 0
        path.append(curr_pos)

        # store index in path and winding number
        curr_w = 0
        visited[curr_pos] = (0, curr_w)

        while True:
            r = generate_random_r(Z, alpha)
            dir = random.choice([-1, 1])

            next_pos = curr_pos + dir * r
            if next_pos >= L or next_pos < 0:
                curr_w += (next_pos // L)
                next_pos %= L

            if next_pos in visited:
                if curr_w != visited[next_pos][1]:
                    # noncontractible loop, process complete
                    print(f'Completed a walk successfuly with L = {L}, with path length {len(path)}')
                    return len(path)
                else:
                    # contractible loop, erase
                    index_in_path = visited[next_pos][0]
                    for pos in path[index_in_path + 1:]:
                        del visited[pos]
                    path = path[:index_in_path + 1]
            else:
                visited[next_pos] = (len(path), curr_w)
                path.append(next_pos)
                if len(path) > 10000:
                    print('path discarded')
                    return -1
            
            curr_pos = next_pos

def simulate_lr_1d(L, alpha, num_trials):
    # find Z = normalization constant
    cdf = zeta(DIM + alpha, 1)
    Z = 1.0 / cdf

    with multiprocessing.Pool() as pool:
        lengths = pool.starmap(simulate, [(L, alpha, Z) for _ in range(num_trials)])

    lengths = np.array(lengths)
    valid_lengths = lengths[lengths > 0]
    total_length = np.sum(valid_lengths)
    avg_length = total_length / num_trials
    
    print(f'\nAverage path length for L={L}: {avg_length}')

    return avg_length

if __name__ == '__main__':
    simulate_lr_1d(L=8, alpha=4, num_trials=1000)
