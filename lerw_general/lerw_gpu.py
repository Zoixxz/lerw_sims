import numpy as np
from numba import cuda, int32
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from tqdm import tqdm  # Import tqdm for the progress bar
import math

# Constants
DIM = 3  # everything 3d for now
# MAX_PATH_LENGTH = 7053950  # adjust based on largest L value
# HASH_SIZE = 7053950
MAX_WINDING = 10  # Maximum absolute value of winding numbers (-10 to 10)
PRIME1 = 11369  # Prime numbers for hashing
PRIME2 = 14407
PRIME3 = 16333

# Directions for FCC and CUBIC lattices
DIRECTIONS_DICT = {
    'FCC': {
        2: np.array([
            (1, 0),
            (-1, 0),
            (1, 1),
            (-1, 1),
            (1, -1),
            (-1, -1)
        ], dtype=np.int32),
        3: np.array([
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
        ], dtype=np.int32)
    },
    'CUBIC': {
        2: np.array([
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1)
        ], dtype=np.int32),
        3: np.array([
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1)
        ], dtype=np.int32)
    }
}

@cuda.jit(device=True)
def pack_winding_numbers(winding):
    """
    Packs the winding numbers into a single int32.
    Assumes winding numbers are in the range [-MAX_WINDING, MAX_WINDING].
    Uses 5 bits per winding number.
    """
    # Shift winding numbers to be non-negative
    w0 = winding[0] + MAX_WINDING  # Now in [0, 20]
    w1 = winding[1] + MAX_WINDING
    w2 = winding[2] + MAX_WINDING
    # Pack into a single int32
    packed = (w0 << 10) | (w1 << 5) | w2  # 15 bits used
    return packed

@cuda.jit(device=True)
def unpack_winding_numbers(packed, winding):
    """
    Unpacks the winding numbers from a single int32.
    """
    w0 = (packed >> 10) & 0x1F  # Extract bits 10-14
    w1 = (packed >> 5) & 0x1F   # Extract bits 5-9
    w2 = packed & 0x1F          # Extract bits 0-4
    # Shift back to original range [-MAX_WINDING, MAX_WINDING]
    winding[0] = w0 - MAX_WINDING
    winding[1] = w1 - MAX_WINDING
    winding[2] = w2 - MAX_WINDING

@cuda.jit(device=True)
def position_hash(pos, hash_size):
    """
    Computes a hash value for a position using modular hashing.
    """
    h = int32(0)
    h = (h + pos[0] * PRIME1) % hash_size
    h = (h + pos[1] * PRIME2) % hash_size
    h = (h + pos[2] * PRIME3) % hash_size
    return h

@cuda.jit(device=True)
def hash_insert(hash_table, key, index, packed_winding, position, hash_size):
    """
    Inserts a key with associated index, packed winding number, and position into the hash table using linear probing.
    Each entry has [key, index, packed_winding, pos[0], pos[1], pos[2]]
    """
    h1 = key % hash_size
    Entry_Size = 6
    for i in range(hash_size):
        idx = (h1 + i + i*i) % hash_size  # Quadratic probing
        base = idx * Entry_Size
        if hash_table[base] == -1:
            # Insert key and associated data
            hash_table[base] = key
            hash_table[base + 1] = index
            hash_table[base + 2] = packed_winding
            hash_table[base + 3] = position[0]
            hash_table[base + 4] = position[1]
            hash_table[base + 5] = position[2]
            return
        elif hash_table[base] == key:
            # Check if positions match
            if (hash_table[base + 3] == position[0] and
                hash_table[base + 4] == position[1] and
                hash_table[base + 5] == position[2]):
                # Update existing entry
                hash_table[base + 1] = index
                hash_table[base + 2] = packed_winding
                return
    # Hash table full
    return

@cuda.jit(device=True)
def hash_lookup(hash_table, key, position, stored_packed_winding, hash_size):
    """
    Looks up a key and position in the hash table and retrieves the stored packed winding number.
    Returns the index in the path or -1 if not found.
    """
    h1 = key % hash_size
    Entry_Size = 6
    for i in range(hash_size):
        idx = (h1 + i + i*i) % hash_size  # Quadratic probing
        base = idx * Entry_Size
        if hash_table[base] == key:
            # Key matches, check positions
            if (hash_table[base + 3] == position[0] and
                hash_table[base + 4] == position[1] and
                hash_table[base + 5] == position[2]):
                # Position matches
                stored_packed_winding[0] = hash_table[base + 2]
                return hash_table[base + 1]
        elif hash_table[base] == -1:
            # Empty slot, key not found
            return -1
        # Else, continue probing
    # Key not found
    return -1

@cuda.jit(device=True)
def hash_remove(hash_table, key, idx, hash_size):
    """
    Removes a key and position from the hash table.
    """
    h1 = key % hash_size
    Entry_Size = 6
    for i in range(hash_size):
        idx = (h1 + i + i*i) % hash_size  # Quadratic probing
        base = idx * Entry_Size
        if hash_table[base] == key and hash_table[base + 1] == idx: # match key and index, for current use case
            hash_table[base] = -1
            hash_table[base + 1] = -1
            hash_table[base + 2] = 0
            hash_table[base + 3] = -1
            hash_table[base + 4] = -1
            hash_table[base + 5] = -1
            return
        elif hash_table[base] == -1:
            # Empty slot, key not found
            return
        # Else, continue probing
    # Key not found
    return

@cuda.jit
def lerw_kernel(L, num_directions, directions, num_simulations, paths_lengths, rng_states, hash_tables, key_stacks, max_path_length, hash_size):
    """
    CUDA kernel function to simulate LERWs in parallel without storing the full path.
    """
    tid = cuda.grid(1)
    if tid >= num_simulations:
        return

    # Use the per-thread hash table and key stack from global memory
    hash_table = hash_tables[tid]
    key_stack = key_stacks[tid]

    # Initialize current position and winding number as local arrays
    current_position = cuda.local.array(3, dtype=int32)
    current_winding = cuda.local.array(3, dtype=int32)

    # Initialize to zero
    for i in range(DIM):
        current_position[i] = 0
        current_winding[i] = 0

    path_length = 0  # Stack pointer

    Entry_Size = 6

    # Initialize hash table entries to -1 (empty) and 0
    for i in range(hash_size):
        base = i * Entry_Size
        hash_table[base] = -1        # Key
        hash_table[base + 1] = -1    # Index
        hash_table[base + 2] = 0     # Packed winding
        hash_table[base + 3] = -1    # pos[0]
        hash_table[base + 4] = -1    # pos[1]
        hash_table[base + 5] = -1    # pos[2]

    # Insert initial position into hash table
    pos_key = position_hash(current_position, hash_size)
    packed_winding = pack_winding_numbers(current_winding)
    hash_insert(hash_table, pos_key, path_length, packed_winding, current_position, hash_size)
    key_stack[path_length] = pos_key  # Store key in stack
    path_length += 1  # Move stack pointer

    while path_length < max_path_length:
        # Choose a random direction
        rand_float = xoroshiro128p_uniform_float32(rng_states, tid)
        dir_idx = int(rand_float * num_directions)
        if dir_idx >= num_directions:
            dir_idx = num_directions - 1  # Ensure index is within bounds
        dir = directions[dir_idx]

        # Update current position and winding number, simulate torus lattice
        for i in range(DIM):
            current_position[i] += dir[i]
            if current_position[i] >= L:
                current_position[i] -= L
                current_winding[i] += 1
            elif current_position[i] < 0:
                current_position[i] += L
                current_winding[i] -= 1

        # Compute position key
        pos_key = position_hash(current_position, hash_size)

        # Look up position in hash table
        stored_packed_winding = cuda.local.array(1, dtype=int32)
        index_in_path = hash_lookup(hash_table, pos_key, current_position, stored_packed_winding, hash_size)

        if index_in_path == -1:
            # Position not visited before
            packed_winding = pack_winding_numbers(current_winding)
            hash_insert(hash_table, pos_key, path_length, packed_winding, current_position, hash_size)
            key_stack[path_length] = pos_key  # Store key in stack
            path_length += 1  # Move stack pointer
        else:
            # Position visited before
            stored_winding = cuda.local.array(3, dtype=int32)
            unpack_winding_numbers(stored_packed_winding[0], stored_winding)
            equal = True
            for d in range(DIM):
                if stored_winding[d] != current_winding[d]:
                    equal = False
                    break
            if equal:
                # Contractible loop, erase loop
                # Remove entries from hash table between index_in_path + 1 and path_length
                for i in range(index_in_path + 1, path_length):
                    pos_key_remove = key_stack[i]
                    hash_remove(hash_table, pos_key_remove, i, hash_size)
                path_length = index_in_path + 1
                # print('contractible loop removed')
            else:
                # Non-contractible loop, simulation complete
                # print('SUCCESS')
                paths_lengths[tid] = path_length
                return

    # If MAX_PATH_LENGTH reached without completion
    # Indicate failure by setting path length to -1
    print('FAILURE: MAX PATH LENGTH HIT')
    paths_lengths[tid] = -1
    return

def kernel_host(L, max_path_length, hash_size, lattice='FCC', num_simulations=1024, batch_size=32):
    """
    Simulates Loop-Erased Random Walks (LERWs) using CUDA without storing the full path.

    Parameters:
    - L (int): Size of the lattice.
    - lattice (str): Type of lattice ('FCC' or 'CUBIC').
    - num_simulations (int): Total number of simulations to run.
    - batch_size (int): Number of simulations to process per batch.

    Returns:
    - all_paths_lengths (np.ndarray): Array of all path lengths.
    """
    # Get directions
    directions = DIRECTIONS_DICT[lattice][DIM]
    num_directions = directions.shape[0]

    # Initialize result container
    results_lengths = []

    # Calculate number of batches
    num_batches = (num_simulations + batch_size - 1) // batch_size

    # Initialize tqdm progress bar
    with tqdm(total=num_batches, desc="Simulations", unit="batch") as pbar:
        for batch_idx in range(num_batches):
            current_batch_size = min(batch_size, num_simulations - batch_idx * batch_size)

            # Allocate arrays for the current batch
            paths_lengths = np.zeros(current_batch_size, dtype=np.int32)
            Entry_Size = 6
            hash_tables = np.empty((current_batch_size, hash_size * Entry_Size), dtype=np.int32)
            key_stacks = np.empty((current_batch_size, max_path_length), dtype=np.int32)

            # Transfer data to device
            d_paths_lengths = cuda.to_device(paths_lengths)
            d_directions = cuda.to_device(directions)
            d_hash_tables = cuda.to_device(hash_tables)
            d_key_stacks = cuda.to_device(key_stacks)

            # Initialize random states for the batch with unique seeds
            rng_states = create_xoroshiro128p_states(current_batch_size, seed=np.random.randint(1, 1e9))

            # Launch kernel
            threads_per_block = 256
            blocks_per_grid = (current_batch_size + threads_per_block - 1) // threads_per_block

            lerw_kernel[blocks_per_grid, threads_per_block](
                L, num_directions, d_directions, current_batch_size,
                d_paths_lengths, rng_states, d_hash_tables, d_key_stacks, max_path_length, hash_size
            )

            # Copy results back to host
            batch_paths_lengths = d_paths_lengths.copy_to_host()

            # Collect results
            results_lengths.append(batch_paths_lengths)

            # Update progress bar
            pbar.update(1)

    # Combine results from all batches
    return np.concatenate(results_lengths)

def simulate_lerw_gpu(L, lattice='FCC', num_trials=1000):
    MAX_PATH_LENGTH = int(pow(2, (math.log2(L) + 3) * 1.69))
    HASH_SIZE = int((1.69) * MAX_PATH_LENGTH)

    print(MAX_PATH_LENGTH, HASH_SIZE)

    path_lengths = kernel_host(L, max_path_length=MAX_PATH_LENGTH, hash_size=HASH_SIZE, lattice=lattice, num_simulations=num_trials, batch_size=50)

    # Filter out failed simulations
    valid_lengths = path_lengths[path_lengths >= 0]

    if len(valid_lengths) > 0:
        average_length = np.mean(valid_lengths)
        print(f'\nAverage path length for L={L}, dim={DIM}, lattice={lattice}: {average_length}')
    else:
        print('No successful simulations.')
    return average_length

if __name__ == '__main__':
    simulate_lerw_gpu(L=1024)
