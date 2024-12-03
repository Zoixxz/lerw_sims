# long range correlation, everything is 1D for now

from scipy.stats import zipf
from scipy.constants import *
import numpy as np
import random
import multiprocessing
import itertools
import bisect

DIM = 1 # simulating on 1D

def generate_random_r(alpha):
    # p = random.uniform(0, 0.999999)
    # print(p)

    # cdf = 0
    # r = 1
    # while True:
    #     P_r = Z / pow(r, DIM + alpha)
    #     cdf += P_r

    #     if cdf >= p:
    #         print(r)
    #         break
    #         # return r
        
    #     r += 1

    return zipf.rvs(DIM + alpha)

# def simulate(L, alpha):
#         path = []
#         visited = {}

#         curr_pos = 0
#         path.append(curr_pos)

#         # store index in path and winding number
#         curr_w = 0
#         visited[curr_pos] = (0, curr_w)

#         while True:
#             r = generate_random_r(alpha)
#             dir = random.choice([-1, 1])

#             next_pos = curr_pos + dir * r
#             if next_pos >= L or next_pos < 0:
#                 curr_w += (next_pos // L)
#                 next_pos %= L

#             if next_pos in visited:
#                 if curr_w != visited[next_pos][1]:
#                     # noncontractible loop, process complete
#                     print(f'Completed a walk successfuly with L = {L}, with path length {len(path)}')
#                     return len(path)
#                 else:
#                     # contractible loop, erase
#                     index_in_path = visited[next_pos][0]
#                     for pos in path[index_in_path + 1:]:
#                         del visited[pos]
#                     path = path[:index_in_path + 1]
#             else:
#                 visited[next_pos] = (len(path), curr_w)
#                 path.append(next_pos)
#                 # if len(path) > 10000:
#                 #     print('path discarded')
#                 #     return -1
            
#             curr_pos = next_pos

def simulate(args):
        L, alpha = args

        path = []
        visited = {}

        curr_pos = 0
        path.append(curr_pos)

        # store index in path and winding number
        visited[curr_pos] = 0

        while True:
            r = generate_random_r(alpha)
            dir = random.choice([-1, 1])

            next_pos = curr_pos + dir * r

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

            if abs(next_pos) >= L:
                pathLen = len(path)
                # print(f'Completed a walk successfully with L = {L}, with path length {pathLen}')
                del path
                del visited
                return pathLen


def simulate_lr_1d(L, alpha, num_trials):
    args = itertools.repeat((L, alpha), num_trials)

    with multiprocessing.Pool() as pool:
        total_length = sum(
            length for length in pool.imap_unordered(simulate, args) if length > 0
        )

    avg_length = total_length / num_trials
    
    print(f'\nAverage path length for L={L}: {avg_length}')

    return avg_length

if __name__ == '__main__':
    simulate_lr_1d(L=pow(2, 13), alpha=0.5, num_trials=1000)
