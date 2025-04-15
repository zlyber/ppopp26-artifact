#include "math_functions.hpp"

namespace caffe {

void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}

void caffe_gpu_memcpy_async(const size_t N, const void* X, void* Y, cudaStream_t stream) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpyAsync(Y, X, N, cudaMemcpyDefault, stream));  // NOLINT(caffe/alt_fn)
  }
}


template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
  }
}

}

