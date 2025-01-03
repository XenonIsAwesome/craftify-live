cmake_minimum_required(VERSION 3.18)

# Set CUDA architecture for NVIDIA RTX 2060 Super
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe")
set(CMAKE_GENERATOR_TOOLSET "cuda=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6")
set(CMAKE_CUDA_ARCHITECTURES 75)

# Project name
project(CraftifyLive LANGUAGES CXX CUDA)

# Find OpenCV
find_package(OpenCV REQUIRED)

# add kernels to include path
include_directories("src/kernels src/executers")

# GLOB recurse to find all .h .cpp .cu .cuh in src/
file(GLOB_RECURSE SOURCES "src/*.h" "src/*.cpp" "src/*.cu" "src/*.cuh")

# Add executable target
add_executable(CraftifyLive ${SOURCES})

# Set target properties for CUDA
set_target_properties(CraftifyLive PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        )

# Link libraries if necessary
target_link_libraries(CraftifyLive PUBLIC ${OpenCV_LIBS}
                                   PRIVATE cuda)
