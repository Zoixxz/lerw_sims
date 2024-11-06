# Loop-Erased Random Walk (LERW) Simulation

This project provides implementations of Loop-Erased Random Walk simulations in both 2D and 3D, with some parallel processing.

## Building the C++ Project
### Prerequisites
- CMake (version 3.10 or higher)
- C++ compiler with C++17 support
- OpenMP support
### Build Instructions
#### Linux/macOS
```bash
# Create and enter build directory inside /lerw_cpp
mkdir build
cd build

# Configure CMake
cmake ..

# Build the project
make
```

### Running the Simulation

The compiled program has options to customize a run without rebuilding:

```bash
./lerw_simulation [options]

Options:
  --2d              Enable 2D simulation (default: 3D only)
  --trials N        Number of trials per R value 
  --threads N       Number of threads to use
  --bootstrap N     Number of bootstrap samples
  --confidence C    Confidence level 
  --output-prefix P Output filename prefix (default: lerw_results)
  --r-values R1,R2,R3,...  Comma-separated list of R values
  --help           Show help message
```

Example usage:
```bash
# Run with custom parameters
./lerw_simulation --2d --trials 100 --threads 4 --r-values 5000,10000,20000

# Run with default parameters
./lerw_simulation
```
