cmake_minimum_required(VERSION 3.31)

# Enable CUDA language support more explicitly
enable_language(CUDA)

# Use C++ 17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA library paths
find_package(CUDAToolkit REQUIRED)  # Modern way to find CUDA

# Include directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# # Create CUDA library
add_library(cuda_kernel1 SHARED sgemm1.cu)
target_link_libraries(cuda_kernel1 PRIVATE CUDA::cudart)
set_target_properties(cuda_kernel1 PROPERTIES 
    CUDA_ARCHITECTURES "86"
    CUDA_SEPARABLE_COMPILATION ON)

# Create executable
add_executable(main main.cpp)
target_link_libraries(main PRIVATE CUDA::cudart cuda_kernel1)
