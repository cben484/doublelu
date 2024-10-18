cmake_minimum_required(VERSION 3.25)

project(DOUBLELU_ VERSION 0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

find_path(CUDA_GEMM_INCLUDE_DIRS utils_check_device.cuh HINTS ${CMAKE_SOURCE_DIR}/include)

add_executable(Singlelu Singlelu.cu)
target_include_directories(Singlelu PUBLIC ${CUDA_GEMM_INCLUDE_DIRS})
target_link_libraries(Singlelu ${CUDA_LIBRARIES} ${CUDA_cusolver_LIBRARY} ${CUDA_curand_LIBRARY})
set_target_properties(Singlelu PROPERTIES CUDA_ARCHITECTURES native)
