import random
import math

# file contains function for the simulation of a single LERW path in 3d (on Z^3)

def simulate_LERW_3D(L_max):
    visited = {}
    path = []
    current = (0, 0, 0)
    path.append(current)
    visited[current] = 0

    directions = [(1, 0, 0), (-1, 0, 0),
                  (0, 1, 0), (0, -1, 0),
                  (0, 0, 1), (0, 0, -1)]

    while True:
        # random direction
        dir = random.choice(directions)
        next_pos = (current[0] + dir[0], current[1] + dir[1], current[2] + dir[2])

        if next_pos in visited:
            # Loop detected: erase the loop
            loop_start = visited[next_pos]
            for pos in path[loop_start+1:]: 
                del visited[pos]
            path = path[:loop_start+1]
        else:
            # No loop: proceed normally
            visited[next_pos] = len(path)
            path.append(next_pos)

        current = next_pos

        distance = math.sqrt(current[0]**2 + current[1]**2 + current[2]**2)  
        if distance >= L_max:
            break

    return path
