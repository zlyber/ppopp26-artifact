cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(miss_ops)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}  -std=c++17 -O0 -Xcompiler -fPIC -g -w ${CUDA_GEN_CODE}")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets")

# Automatically detect target GPU architecture
set(CMAKE_CUDA_ARCHITECTURES native)


find_package(Threads REQUIRED)
find_package(CUDA REQUIRED)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

include_directories(
    ${ROOT_DIR}/PLONK/blst/include
    ${ROOT_DIR}/PLONK/utils/mont
    ${ROOT_DIR}/PLONK/utils/zkp
    ${ROOT_DIR}/PLONK/utils/zkp/cuda/zksnark_ntt/parameters
    ${CUDA_INCLUDE_DIRS}
)



file(GLOB_RECURSE plonk_utils_cpp "${ROOT_DIR}/PLONK/utils/*.cpp")
file(GLOB_RECURSE plonk_utils_cuda "${ROOT_DIR}/PLONK/utils/*.cu")
file(GLOB_RECURSE caffe_cpp "${ROOT_DIR}/caffe/*.cpp")
file(GLOB_RECURSE caffe_utils_cpp "${ROOT_DIR}/caffe/utils/*.cpp")
file(GLOB test "${ROOT_DIR}/miss_ops.cu")


set(CPU_SRCS  ${plonk_utils_cpp} ${caffe_cpp} ${caffe_utils_cpp})
set(CUDA_SRCS ${plonk_utils_cuda} ${test})

add_subdirectory(${ROOT_DIR}/PLONK/blst ${CMAKE_BINARY_DIR}/blst_build)

cuda_add_executable(${PROJECT_NAME} ${CPU_SRCS} ${CUDA_SRCS})

target_include_directories(${PROJECT_NAME} PRIVATE ${ROOT_DIR}/PLONK/blst/include)

target_link_libraries(${PROJECT_NAME} blst ${CUDA_LIBRARIES} Threads::Threads)