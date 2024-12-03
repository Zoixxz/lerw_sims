from scipy.constants import *
import random
import multiprocessing
import itertools
import math
from sympy import isprime
from results_util import *
import time

sum_squares_yes = {1,}
sum_squares_no = {3,}

def sum_of_two_squares(n):
    og_n = n

    if og_n in sum_squares_yes:
        return True
    elif og_n in sum_squares_no:
        return False

    while n % 2 == 0:
        n = n // 2

    for p in range(3, int(math.sqrt(n)) + 1, 4):
        if isprime(p):
            if n % p == 0:
                count = 0
                while n % p == 0:
                    n //= p
                    count += 1
                if count % 2 != 0:
                    sum_squares_no.add(og_n)
                    return False
    if n > 1 and n % 4 == 3:
        sum_squares_no.add(og_n)
        return False
    
    sum_squares_yes.add(og_n)
    return True
    

DIM = 2 # simulating on 2D

Z_VALS = {}

print('Calculating valid nums...', end=' ')
valid_nums = [i for i in range(1, 500001) if sum_of_two_squares(i)]
print('Done.')

print('Constructing Z_vals...', end=' ')
for a in range(5, 26, 1):
    zeta_sum = 0
    for num in valid_nums:
        zeta_sum += 1.0 / (num ** (DIM + a/10))
    Z_VALS[a/10] = 1.0 / zeta_sum
print('Done.')

def generate_random_r_squared(alpha):
    p = random.random()

    cdf = 0
    r_squared = 1
    while True:
        if not sum_of_two_squares(r_squared):
            r_squared += 1
            continue

        P_r = Z_VALS[alpha] / (r_squared ** (DIM + alpha))
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
                possible.append((-a, b))
                possible.append((a, -b))
                possible.append((-a, -b))
        possible_pos[r_squared] = possible
        
    jump = random.choice(possible)
    next_pos = []
    for i in range(DIM):
        next_pos.append(curr_pos[i] + jump[i])
    return tuple(next_pos)

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

            dist = math.sqrt(next_pos[0]**2 + next_pos[1]**2)
            if dist >= L:
                pathLen = len(path)
                # print(f'Completed a walk successfully with L = {L}, with path length {pathLen}')
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

    alpha_vals = [a/10 for a in range(5, 26, 1)]
    D_vals = []
    L_vals = [2**i for i in range(7, 12)]
    num_trials = int(1e4)

    for alpha in alpha_vals:
        print(f'\nProcessing alpha = {alpha}...')
        avg_lengths = [simulate_lr_2d(L=L, alpha=alpha, num_trials=num_trials) for L in L_vals]
        results = estimate_exponent_with_errors(L_vals, avg_lengths, n_bootstrap=1000)
        plot_results_with_errors(L_vals, avg_lengths, results, alpha)
        D_vals.append(results['D'])
        print('\n')

    basic_2d_plot(alpha_vals, D_vals)
