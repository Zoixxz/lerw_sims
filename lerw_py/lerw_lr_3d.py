# long range correlation, on 3D

from scipy.constants import *
import numpy as np
import random
import multiprocessing
import math
import itertools
from tqdm import tqdm

DIM = 3 # simulating on 1D

def wrong_form(x):
    while x % 4 == 0:
        x //= 4
    
    return x % 8 == 7

def generate_random_r_squared(Z, alpha):
    p = random.random()

    cdf = 0
    r = 1
    while True:
        # print('here', p)
        if wrong_form(r):
            r += 1
            continue

        P_r = Z / pow(r, DIM + alpha)
        cdf += P_r

        if cdf >= p:
            return r
        
        r += 1

def get_next_pos(r_squared, curr_pos):
    r = math.isqrt(r_squared)
    possible = []
    for a in range(-r, r + 1):
        for b in range(-r, r + 1):
            c_squared = r_squared - pow(a, 2) - pow(b, 2)
            if c_squared >= 0 and math.isqrt(c_squared)**2 == c_squared:
                possible.append((a, b, int(math.isqrt(c_squared))))
    jump = random.choice(possible)
    next_pos = []
    for i in range(3):
        next_pos.append(curr_pos[i] + jump[i])

    return next_pos

def simulate(L, alpha, Z):
        path = []
        visited = {}

        curr_pos = (0, 0, 0)
        path.append(curr_pos)

        # store index in path and winding number
        curr_w = [0, 0, 0]
        visited[curr_pos] = (0, curr_w.copy())

        while True:
            r_squared = generate_random_r_squared(Z, alpha)
            next_pos = get_next_pos(r_squared, curr_pos)

            for i in range(DIM):
                if next_pos[i] >= L or next_pos[i] < 0:
                    curr_w[i] += (next_pos[i] // L)
                    next_pos[i] %= L

            next_pos = tuple(next_pos)

            if next_pos in visited:
                equal = True
                for i in range(DIM):
                    if curr_w[i] != visited[next_pos][1][i]:
                        equal = False
                        break
                
                if equal:
                    # contractible loop
                    index_in_path = visited[next_pos][0]
                    for pos in path[index_in_path + 1:]:
                        del visited[pos]
                    path = path[:index_in_path + 1]
                    # print('deleted contractible loop')
                else:
                    # noncontractible loop
                    print(f'Completed a walk successfuly with L = {L}, with path length {len(path)}')
                    return len(path)
            else:
                visited[next_pos] = (len(path), curr_w.copy())
                path.append(next_pos)
                # if len(path) >= L**2:
                #     print('path discarded')
                #     return -1
            
            curr_pos = next_pos

def simulate_lr_3d(L, alpha, num_trials):
    # find Z = normalization constant
    # Z = 1.0 / 1.08188 # a = 1
    Z = 1.0 / 1.12551597161377 # a = 0.5

    args = itertools.repeat((L, alpha, Z), num_trials)

    total_length = 0
    with multiprocessing.Pool() as pool:
        for length in pool.starmap(simulate, args):
            if length > 0:
                total_length += length

    avg_length = total_length / num_trials
    
    print(f'\nAverage path length for L={L}: {avg_length}')
    return avg_length

if __name__ == '__main__':
    simulate_lr_3d(L=25, alpha=0.5, num_trials=10000)
