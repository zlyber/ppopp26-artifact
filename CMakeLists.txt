cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ZPrize)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}  -std=c++17 -O0 -Xcompiler -fPIC -g -w ${CUDA_GEN_CODE}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_90,code=sm_90")
find_package(Threads REQUIRED)
find_package(CUDA REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}/PLONK/blst/include
    ${CMAKE_SOURCE_DIR}/PLONK/utils/mont
    ${CMAKE_SOURCE_DIR}/PLONK/utils/zkp
    ${CMAKE_SOURCE_DIR}/PLONK/utils/zkp/cuda/zksnark_ntt/parameters
    ${CMAKE_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
)
# file(GLOB plonk_utils_h "PLONK/utils/*.h")
# file(GLOB plonk_utils_hpp "PLONK/utils/*.hpp")
# file(GLOB caffe_hpp "caffe/*.hpp")
# file(GLOB_RECURSE plonk_src_h "PLONK/src/*.h","PLONK/src/*.hpp")
# file(GLOB_RECURSE plonk_utils_cuh "PLONK/utils/*.cuh")

file(GLOB_RECURSE plonk_utils_cpp "PLONK/utils/*.cpp")
file(GLOB_RECURSE plonk_src_cu "PLONK/src/*.cu")
file(GLOB_RECURSE plonk_utils_cuda "PLONK/utils/*.cu")
file(GLOB_RECURSE caffe_cpp "caffe/*.cpp")
file(GLOB_RECURSE caffe_utils_cpp "caffe/utils/*.cpp")
file(GLOB main "main.cu")


set(CPU_SRCS  ${plonk_utils_cpp} ${caffe_cpp} ${caffe_utils_cpp} )
set(CUDA_SRCS ${plonk_src_cu} ${plonk_utils_cuda} ${main})



# 添加blst子目录
add_subdirectory(PLONK/blst)

# 生成可执行文件
cuda_add_executable(${PROJECT_NAME} ${CPU_SRCS} ${CUDA_SRCS})

# 设置blst库的包含路径
target_include_directories(${PROJECT_NAME} PRIVATE PLONK/blst/include)

# 链接blst库
target_link_libraries(${PROJECT_NAME} blst ${CUDA_LIBRARIES} Threads::Threads)