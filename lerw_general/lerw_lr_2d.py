from scipy.constants import *
import random
import multiprocessing
import itertools
import math
from sympy import isprime
from results_util import *
import time


def is_prime(n):
    count = 0
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            count += 1
            break
    
    return count == 0

def sum_of_two_squares(n):
    if n == 1:
        return True
    
    while n % 2 == 0:
        n /= 2

    for p in range(3, int(math.sqrt(n)) + 1, 4):
        if isprime(p):
            if n % p == 0:
                count = 0
                while n % p == 0:
                    n /= p
                    count += 1
                if count % 2 != 0:
                    return False
                
    if n > 1 and n % 4 == 3:
        return False
    
    return True

DIM = 2 # simulating on 2D
ALPHA_START = 1
ALPHA_END = 1
ALPHA_JUMP = 1

Z_VALS = {}

print('Calculating valid nums...')
valid_nums = [i for i in range(1, 1000001) if sum_of_two_squares(i)]
print('Done.')

print('Constructing Z_vals...')
for a in range(ALPHA_START, ALPHA_END + 1, ALPHA_JUMP):
    zeta_sum = 0
    exponent = (DIM + a/10) / 2
    for num in valid_nums:
        zeta_sum += 1 / (num ** exponent)
    Z_VALS[a] = 1 / zeta_sum
print('Done.')

def generate_random_r_squared(alpha):
    p = random.random()

    exponent = (DIM + alpha/10) / 2
    cdf = 0
    r_squared = 1
    while True:
        if not sum_of_two_squares(r_squared):
            r_squared += 1
            continue

        P_r = Z_VALS[alpha] / (r_squared ** exponent)
        cdf += P_r

        if cdf >= p:
            return r_squared
        
        r_squared += 1

possible_pos = {}

def get_next_pos(curr_pos, r_squared):
    if r_squared in possible_pos:
        possible = possible_pos[r_squared]
    else:
        r = math.isqrt(r_squared)
        possible = []
        for a in range(r + 1):
            b_squared = r_squared - a**2
            b = math.isqrt(b_squared)
            if b**2 == b_squared:
                possible.append((a, b))
                if a != 0:
                    possible.append((-a, b))
                if b != 0:
                    possible.append((a, -b))
                if a != 0 and b != 0:
                    possible.append((-a, -b))

        possible_pos[r_squared] = possible
        
    jump = random.choice(possible)
    return (curr_pos[0] + jump[0], curr_pos[1] + jump[1])


def simulate(args):
        L, alpha = args

        path = []
        visited = {}

        curr_pos = (0, 0)
        path.append(curr_pos)

        visited[curr_pos] = 0

        while True:
            r_squared = generate_random_r_squared(alpha)
            next_pos = get_next_pos(curr_pos, r_squared)

            if next_pos in visited:
                # loop formed
                index_in_path = visited[next_pos]
                for pos in path[index_in_path + 1:]:
                    del visited[pos]
                path = path[:index_in_path + 1]
            else:
                visited[next_pos] = len(path)
                path.append(next_pos)

            curr_pos = next_pos

            dist_squared = next_pos[0]**2 + next_pos[1]**2
            if dist_squared >= L*L:
                pathLen = len(path)
                print(f'Completed a walk successfully with L = {L}, with path length {pathLen}')
                return pathLen


def simulate_lr_2d(L, alpha, num_trials):
    start = time.time()

    args = itertools.repeat((L, alpha), num_trials)

    with multiprocessing.Pool() as pool:
        total_length = sum(
            length for length in pool.imap_unordered(simulate, args) if length > 0
        )

    avg_length = total_length / num_trials
    
    end = time.time()

    print(f'Average path length for L={L}: {avg_length}, time elapsed: {(end - start):.2f} seconds')

    return avg_length

if __name__ == '__main__':
    alpha_vals = [i for i in range(ALPHA_START, ALPHA_END + 1, ALPHA_JUMP)]
    D_vals = []
    L_vals = [2**i for i in range(20, 25)]
    num_trials = int(1e2)

    for alpha in alpha_vals:
        print(f'\nProcessing alpha = {alpha}...')
        avg_lengths = [simulate_lr_2d(L=L, alpha=alpha, num_trials=num_trials) for L in L_vals]
        results = estimate_exponent_with_errors(L_vals, avg_lengths, n_bootstrap=1000)
        plot_results_with_errors(L_vals, avg_lengths, results, alpha)
        D_vals.append(results['D'])
        print('\n')

    basic_2d_plot(alpha_vals, D_vals)
