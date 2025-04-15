#pragma once

#include <stdint.h>
#include <string.h>

#include "../common.hpp"


namespace caffe {

inline void caffe_memset(const size_t N, const int alpha, void* X) {
  memset(X, alpha, N);  // NOLINT(caffe/alt_fn)
}

template <typename Dtype>
void caffe_copy(const int N, const Dtype *X, Dtype *Y);

void caffe_gpu_memcpy(const size_t N, const void* X, void* Y);

inline void caffe_gpu_memset(const size_t N, const int alpha, void* X) {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaMemset(X, alpha, N));  // NOLINT(caffe/alt_fn)
#else
  NO_GPU;
#endif
}

inline void caffe_gpu_memset_async(const size_t N, const int alpha, void* X, cudaStream_t stream = (cudaStream_t)0) {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaMemsetAsync(X, alpha, N, stream));  // NOLINT(caffe/alt_fn)
#else
  NO_GPU;
#endif
}

void caffe_gpu_memcpy_async(const size_t N, const void* X, void* Y, cudaStream_t stream = (cudaStream_t)0);
}

