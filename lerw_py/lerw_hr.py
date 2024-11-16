# long range hierarchical correlation, everything is 1D for now

from scipy.special import zeta
from scipy.constants import *
import numpy as np
import random
import multiprocessing
import itertools

DIM = 1 # simulating on 1D

def generate_random_i(L, alpha, Z):
    p = random.random()

    cdf = 0
    i = 1
    while True:
        P_i = Z / pow(L, i * (DIM + alpha))
        cdf += P_i

        if cdf >= p:
            return i
        
        i += 1

def get_next_pos(M, curr_pos, L, alpha, Z):
    # get i
    i = generate_random_i(L, alpha, Z)
    if i >= M:
        return -1

    next_pos = []
    for j in range(i - 1):
        next_pos.append(random.randint(0, L - 1))

    if len(curr_pos) >= i:
        while True:
            to_add = random.randint(0, L - 1)
            if to_add != curr_pos[i - 1]:
                next_pos.append(to_add)
                break
        for j in range(i, len(curr_pos)):
            next_pos.append(curr_pos[j])
    else:
        while True:
            to_add = random.randint(0, L - 1)
            if to_add != 0:
                next_pos.append(to_add)
                break

    for j in range(len(next_pos) - 1, 0, -1):
        if next_pos[j] == 0:
            del next_pos[j]

    # print(f'i: {i}, next_pos: {next_pos}')
    return next_pos


def simulate(M, L, alpha, Z):
        path = []
        visited = {}

        curr_pos = tuple([0])
        path.append(curr_pos)

        # store index in path and winding number
        visited[curr_pos] = 0

        while True:
            next_pos = get_next_pos(M, curr_pos, L, alpha, Z)
            if next_pos == -1:
                print(f'Completed a walk successfuly with L = {L}, with path length {len(path)}')
                return len(path)
            
            next_pos = tuple(next_pos)
            
            if next_pos in visited:
                # print('loop')
                # erase loop
                index_in_path = visited[next_pos]
                # print('ind', index_in_path)
                for pos in path[index_in_path + 1:]:
                    # print('deleting ', pos)
                    del visited[pos]
                path = path[:index_in_path + 1]
                # print(path)
            else:
                visited[next_pos] = len(path)
                path.append(next_pos)

            curr_pos = next_pos


def simulate_hr_L(L, alpha, num_trials):
    # find Z = normalization constant
    M = 4
    Z = pow(L, DIM + alpha) - 1

    args = itertools.repeat((M, L, alpha, Z), num_trials)

    total_length = 0
    with multiprocessing.Pool() as pool:
        for length in pool.starmap(simulate, args):
            if length > 0:
                total_length += length
    
    avg_length = total_length / num_trials
    
    print(f'\nAverage path length for L={L}: {avg_length}')
    return avg_length

def simulate_hr_M(M, alpha, num_trials):
    # find Z = normalization constant
    L = 2
    Z = pow(L, DIM + alpha) - 1

    # simulate(M=7, L=V, alpha=alpha, Z=Z)

    with multiprocessing.Pool() as pool:
        lengths = pool.starmap(simulate, [(M, L, alpha, Z) for _ in range(num_trials)])

    lengths = np.array(lengths)
    valid_lengths = lengths[lengths > 0]
    total_length = np.sum(valid_lengths)
    avg_length = total_length / num_trials
    
    print(f'\nAverage path length for M={M}: {avg_length}')
    return avg_length

if __name__ == '__main__':
    simulate_hr_L(L=25, alpha=0.5, num_trials=100)
