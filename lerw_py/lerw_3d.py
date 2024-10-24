import numpy as np
import random
import math
from collections import defaultdict
import time
import multiprocessing

def simulate_LERW_3D(R_max):
    visited = {}
    path = []
    current = (0, 0, 0)
    path.append(current)
    visited[current] = 0

    directions = [(1, 0, 0), (-1, 0, 0),
                  (0, 1, 0), (0, -1, 0),
                  (0, 0, 1), (0, 0, -1)]

    while True:
        # Choose a random direction
        dir = random.choice(directions)
        next_pos = (current[0] + dir[0], current[1] + dir[1], current[2] + dir[2])

        # Check if next position is already visited
        if next_pos in visited:
            # Loop detected; erase the loop
            loop_start = visited[next_pos]
            for pos in path[loop_start+1:]:
                del visited[pos]
            path = path[:loop_start+1]
        else:
            # No loop; proceed normally
            visited[next_pos] = len(path)
            path.append(next_pos)

        current = next_pos

        # Check if the Euclidean distance exceeds R_max
        distance = math.sqrt(current[0]**2 + current[1]**2 + current[2]**2)
        if distance >= R_max:
            break

    return path
