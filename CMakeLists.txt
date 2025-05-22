cmake_minimum_required(VERSION 3.10)
project(triangle_mult_cuda CUDA CXX)

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_80 --extended-lambda --expt-relaxed-constexpr")

add_library(triangle_mult_cuda SHARED
    triangle_mult.cpp
    triangle_mult_kernel.cu
)

target_link_libraries(triangle_mult_cuda PRIVATE "${TORCH_LIBRARIES}")
set_target_properties(triangle_mult_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)