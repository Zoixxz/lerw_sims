import numpy as np
import matplotlib.pyplot as plt
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import time
import multiprocessing


class AESRandom:
    def __init__(self, seed):
        self.key = seed
        self.cipher = AES.new(self.key, AES.MODE_ECB)

    def random_bytes(self):
        return self.cipher.encrypt(get_random_bytes(16))

    def randint(self, a, b):
        random_int = int.from_bytes(self.random_bytes(), byteorder='big')
        return a + (random_int % (b - a + 1))

DIRECTIONS = {
    'FCC': {
        2: [
            (1, 0),
            (-1, 0),
            (0.5, 0.866),
            (-0.5, 0.866),
            (1/2, -0.866),
            (-1/2, -0.866)
        ],
        3: [
            (1, 1, 0),
            (1, -1, 0),
            (-1, 1, 0),
            (-1, -1, 0),
            (1, 0, 1),
            (1, 0, -1),
            (-1, 0, 1),
            (-1, 0, -1),
            (0, 1, 1),
            (0, 1, -1),
            (0, -1, 1),
            (0, -1, -1)
        ]
    },
    'CUBIC': {
        2: [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1)
        ],
        3: [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1)
        ]
    }
}

rng = AESRandom(get_random_bytes(32))

class LERW_NN:
    def __init__(self, L, dim=3, lattice='FCC', plot=False):
        self.L = L
        self.directions = DIRECTIONS[lattice][dim]
        self.dim = dim
        self.plot = plot
        self.lattice = lattice

        if self.plot:
            # Initialize plotting
            self.fig = plt.figure()
            if self.dim == 2:
                self.ax = self.fig.add_subplot(111)
            elif self.dim == 3:
                self.ax = self.fig.add_subplot(111, projection='3d')
            plt.ion()
            plt.show()

    def _is_wrapped(self, pos1, pos2):
        for i in range(len(pos1)):
            if abs(pos1[i] - pos2[i]) > 1:
                return True
        return False
    
    def get_path_len(self, _):
        return len(self.simulate())

    def simulate(self):
        path = []
        visited = {}

        curr_pos = tuple(np.zeros(self.dim))
        path.append(curr_pos)

        # store position in path and winding number
        curr_w = np.zeros(self.dim, dtype=int)
        visited[curr_pos] = (0, curr_w.copy())

        if self.plot:
        # Initialize plot data
            if self.dim == 2:
                self.ax.clear()
                self.ax.set_aspect('equal')
                self.ax.set_xlim(0, self.L)
                self.ax.set_ylim(0, self.L)
                self.ax.set_title('Loop-Erased Random Walk (2D)')
            elif self.dim == 3:
                self.ax.clear()
                self.ax.set_xlim(0, self.L)
                self.ax.set_ylim(0, self.L)
                self.ax.set_zlim(0, self.L)
                self.ax.set_title('Loop-Erased Random Walk (3D)')

        while True:
            dir_idx = rng.randint(0, len(self.directions) - 1)
            dir = self.directions[dir_idx]
            next_pos = np.array(curr_pos) + dir

            for idx, coord in enumerate(next_pos):
                if coord >= self.L:
                    next_pos[idx] -= self.L
                    curr_w[idx] += 1
                    # print('winded')
                elif coord < 0:
                    next_pos[idx] += self.L
                    curr_w[idx] -= 1
                    # print('winded')

            next_pos = tuple(next_pos)

            # if loop
            if next_pos in visited:
                if not np.array_equal(curr_w, visited[next_pos][1]):
                    # noncontractible: return loop
                    path = path[visited[next_pos][0]:]
                    print(f'Completed a walk successfuly with L = {self.L}, with path length {len(path)}')
                    return path
                else:
                    #contractible: erase loop
                    loop_start = visited[next_pos][0]
                    for pos in path[loop_start + 1:]:
                        del visited[pos]
                    path = path[:loop_start + 1]
                    # print('contractible loop deleted')

                    if self.plot:
                        if self.dim == 2:
                            self.ax.clear()
                            self.ax.set_aspect('equal')
                            self.ax.set_xlim(0, self.L)
                            self.ax.set_ylim(0, self.L)
                        elif self.dim == 3:
                            self.ax.clear()
                            self.ax.set_xlim(0, self.L)
                            self.ax.set_ylim(0, self.L)
                            self.ax.set_zlim(0, self.L)

                        # Re-plot the path without the deleted loop
                        for i in range(len(path) - 1):
                            prev_pos = path[i]
                            curr_pos_plot = path[i + 1]
                            if not self._is_wrapped(prev_pos, curr_pos_plot):
                                if self.dim == 2:
                                    x_vals = [prev_pos[0], curr_pos_plot[0]]
                                    y_vals = [prev_pos[1], curr_pos_plot[1]]
                                    self.ax.plot(x_vals, y_vals, color='blue')
                                elif self.dim == 3:
                                    x_vals = [prev_pos[0], curr_pos_plot[0]]
                                    y_vals = [prev_pos[1], curr_pos_plot[1]]
                                    z_vals = [prev_pos[2], curr_pos_plot[2]]
                                    self.ax.plot(x_vals, y_vals, z_vals, color='blue')
                        plt.draw()
                        plt.pause(0.001)
            else:
                # no loop
                visited[next_pos] = (len(path), curr_w.copy())
                path.append(next_pos)
                # if len(path) % 10 == 0:
                    # print(f'current path length {len(path)}')

                if self.plot:
                    if len(path) > 1:
                        prev_pos = path[-2]
                        curr_pos = path[-1]
                        if not self._is_wrapped(prev_pos, curr_pos):
                            # Plot line from prev_pos to curr_pos
                            if self.dim == 2:
                                x_vals = [prev_pos[0], curr_pos[0]]
                                y_vals = [prev_pos[1], curr_pos[1]]
                                self.ax.plot(x_vals, y_vals, color='blue')
                            elif self.dim == 3:
                                x_vals = [prev_pos[0], curr_pos[0]]
                                y_vals = [prev_pos[1], curr_pos[1]]
                                z_vals = [prev_pos[2], curr_pos[2]]
                                self.ax.plot(x_vals, y_vals, z_vals, color='blue')
                        else:
                            # Wrapping occurred, do not plot line
                            pass
                        plt.draw()
                        plt.pause(0.001)

            curr_pos = next_pos

def simulate_nn(L, num_trials, dim=3):
    lerw = LERW_NN(L=L, dim=dim, lattice='FCC')

    with multiprocessing.Pool() as pool:
        lengths = pool.map(lerw.get_path_len, range(num_trials))
        total_length = sum(lengths)
        avg_length = total_length / num_trials
    
    print(f'\nAverage path length for L={L}, dim={dim}, lattice={lerw.lattice}: {avg_length}')
    return avg_length

if __name__ == '__main__':
    lerw = LERW_NN(100, 3, 'FCC', plot=True)

    start = time.time()
    total = 0
    for i in range(1):
        total += lerw.get_path_len(5)
    end = time.time()

    print((end - start) / 1)
    avg = total / 1
    print(avg)

    # print(lerw.get_path_len(5))