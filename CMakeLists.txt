cmake_minimum_required(VERSION 3.10)
project(triangle_mult_cuda CUDA CXX)

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

# CUDA 11以降のアーキテクチャ設定（BFloat16サポート用）
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80")

add_library(triangle_mult_cuda SHARED
    triangle_mult.cpp
    triangle_mult_kernel.cu
)

target_link_libraries(triangle_mult_cuda PRIVATE "${TORCH_LIBRARIES}")
set_target_properties(triangle_mult_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)