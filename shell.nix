{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    # Python environment
    (python3.withPackages (ps: with ps; [
      numpy
      matplotlib
      # Additional standard library modules like random, math, collections, 
      # multiprocessing are included in Python by default
    ]))

    # C++ development tools
    gcc
    gdb
    cmake
    gnumake

    # Development tools
    clang-tools # For C++ linting and formatting
    python3Packages.black # Python formatter
    python3Packages.pylint # Python linter
  ];

  shellHook = ''
    echo "Python and C++ development environment loaded"
    echo "Python version: $(python --version)"
    echo "GCC version: $(gcc --version | head -n 1)"
    
    # Set up any environment variables if needed
    export PYTHONPATH="$PYTHONPATH:$(pwd)"
    
    # Create a basic CMakeLists.txt if it doesn't exist
    if [ ! -f CMakeLists.txt ]; then
      echo "Creating basic CMakeLists.txt..."
      cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.10)
project(LERW)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(lerw_cpp main.cpp)
target_compile_options(lerw_cpp PRIVATE -Wall -Wextra)
EOF
    fi
  '';

  # Environment variables
  CPLUS_INCLUDE_PATH = "${pkgs.gcc}/include/c++/${pkgs.gcc.version}";
  LD_LIBRARY_PATH = "${pkgs.gcc}/lib";
}
