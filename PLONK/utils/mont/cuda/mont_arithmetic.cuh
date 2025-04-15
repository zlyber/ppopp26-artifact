#pragma once
#include <assert.h>
#include <stdexcept>
#include <limits>
#include <cuda.h>
#include <cuda_runtime.h>
#include "curve_def.cuh"
#include "../../../../caffe/interface.hpp"

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace cuda{

#define BLOCK_SIZE (512)
#define MAX_NUM_BLOCKS (BLOCK_SIZE)

#define BIN_KERNEL(name, op)                           \
  template <typename T>                                \
  __launch_bounds__(1024) __global__ void mont_##name##_mod_kernel(  \
      const int64_t N, T* c, const T* a, const T* b) { \
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i < N) {                                       \
      c[i] = a[i] op b[i];                             \
    }                                                  \
  }                                                    \
  template <typename T>                                \
  __global__ void mont_##name##_mod_kernel_(           \
      const int64_t N, T* self, const T* other) {      \
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i < N) {                                       \
      self[i] op## = other[i];                         \
    }                                                  \
  }                                                    

#define SCALAR_KERNEL(name, op)                             \
  template <typename T>                                     \
  __global__ void mont_##name##_scalar_mod_kernel(          \
      const int64_t N, T* c, const T* a, const T* scalar) { \
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;      \
    if (i < N) {                                            \
      c[i] = a[i] op scalar[0];                             \
    }                                                       \
  }                                                         \
  template <typename T>                                     \
  __global__ void mont_##name##_scalar_mod_kernel_(         \
      const int64_t N, T* self, const T* scalar) {          \
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;      \
    if (i < N) {                                            \
      self[i] op## = scalar[0];                             \
    }                                                       \
  }

#define BIN_OP_TEMPLATE(name)                                                  \
  template <typename T>                                                        \
  static void name##_template(T* c, T* a, T* b, int64_t numel, cudaStream_t stream) {  \
      int64_t N = numel / num_uint64(a);                                       \
      int64_t grid = (N + 128 - 1) / 128;          \
      mont_##name##_mod_kernel<<<grid, block_work_size(), 0, stream>>>(           \
          N, c, a, b);                                                         \
      CUDA_CHECK(cudaGetLastError());                                          \
  }                                                                            \
  template <typename T>                                                        \
  static void name##_template_(T* self, T* other, int64_t numel, cudaStream_t stream) {  \
      int64_t N = numel / num_uint64(self);                                    \
      int64_t grid = (N + 128 - 1) / 128;          \
      mont_##name##_mod_kernel_<<<grid, 128, 0, stream>>>(          \
          N, self, other);                                                     \
      CUDA_CHECK(cudaGetLastError());                                          \
  }                                                                            \                                                 

#define SCALAR_OP_TEMPLATE(name)                                           \
  template <typename T>                                                    \
  static void name##_scalar_template(                                      \
      T* c, T* a, T* b, int64_t numel, cudaStream_t stream) {              \
          int64_t N = numel / num_uint64(a);                               \
          int64_t grid = (N + block_work_size() - 1) / block_work_size();  \
          mont_##name##_scalar_mod_kernel<<<                               \
              grid,                                                        \
              block_work_size(),                                           \
              0,                                                           \
              stream>>>(N, c, a, b);                                       \
          CUDA_CHECK(cudaGetLastError());                                  \
  }                                                                        \
  template <typename T>                                                    \
  static void name##_scalar_template_(                                     \
      T* self, T* other, int64_t numel, cudaStream_t stream) {             \
          int64_t N = numel / num_uint64(other);                           \
          int64_t grid = (N + block_work_size() - 1) / block_work_size();  \
          mont_##name##_scalar_mod_kernel_<<<                              \
              grid,                                                        \
              block_work_size(),                                           \
              0,                                                           \
              stream>>>(N, self, other);                                   \
          CUDA_CHECK(cudaGetLastError());                                  \
  }                                                                        \


#define BIN_OP(name)                                                       \
  SyncedMemory name##_mod_cuda(SyncedMemory a, SyncedMemory b, cudaStream_t stream = (cudaStream_t)0); \
  void name##_mod_cuda_(SyncedMemory self, SyncedMemory other, cudaStream_t stream = (cudaStream_t)0); \

#define SCALAR_OP(name)                                                    \
  SyncedMemory name##_mod_scalar_cuda(SyncedMemory a, SyncedMemory b, cudaStream_t stream = (cudaStream_t)0);     \
  void name##_mod_scalar_cuda_(SyncedMemory self, SyncedMemory other, cudaStream_t stream = (cudaStream_t)0);      \

  BIN_KERNEL(add, +);
  BIN_KERNEL(sub, -);
  BIN_KERNEL(mul, *);
  BIN_KERNEL(div, /);
  SCALAR_KERNEL(add, +);
  SCALAR_KERNEL(sub, -);
  SCALAR_KERNEL(mul, *);
  SCALAR_KERNEL(div, /);

  BIN_OP_TEMPLATE(add);
  BIN_OP_TEMPLATE(sub);
  BIN_OP_TEMPLATE(mul);
  BIN_OP_TEMPLATE(div);
  SCALAR_OP_TEMPLATE(add);
  SCALAR_OP_TEMPLATE(sub);
  SCALAR_OP_TEMPLATE(mul);
  SCALAR_OP_TEMPLATE(div); 

  BIN_OP(add);
  BIN_OP(sub);
  BIN_OP(mul);
  BIN_OP(div);
  SCALAR_OP(add);
  SCALAR_OP(sub);
  SCALAR_OP(mul);
  SCALAR_OP(div); 

  SyncedMemory to_mont_cuda(SyncedMemory input);

  SyncedMemory to_base_cuda(SyncedMemory input, cudaStream_t stream);

  SyncedMemory inv_mod_cuda(SyncedMemory input);

  SyncedMemory neg_mod_cuda(SyncedMemory input);

  SyncedMemory exp_mod_cuda(SyncedMemory input, int64_t exp);

  SyncedMemory pad_poly_cuda(SyncedMemory input, int64_t N, cudaStream_t stream = (cudaStream_t)0);

  void pad_poly_cuda_(SyncedMemory input, SyncedMemory output, int64_t N, cudaStream_t stream = (cudaStream_t)0); 

  SyncedMemory repeat_to_poly_cuda(SyncedMemory input, int64_t N, cudaStream_t stream = (cudaStream_t)0);
  void repeat_to_poly_cuda_(SyncedMemory input, SyncedMemory output, int64_t N, cudaStream_t stream = (cudaStream_t)0);

  SyncedMemory poly_eval_cuda(SyncedMemory x, int64_t N, cudaStream_t stream = (cudaStream_t)0);
  void poly_eval_cuda_(SyncedMemory x, SyncedMemory y, int64_t N, cudaStream_t stream);

  SyncedMemory poly_reduce_cuda(SyncedMemory x, SyncedMemory coeff);

  SyncedMemory poly_div_cuda(SyncedMemory divid_poly, SyncedMemory c);

  SyncedMemory accumulate_mul_poly_cuda(SyncedMemory product_poly, cudaStream_t stream = (cudaStream_t)0);
  void accumulate_mul_poly_cuda_(SyncedMemory product_poly, SyncedMemory output, cudaStream_t stream = (cudaStream_t)0);
}