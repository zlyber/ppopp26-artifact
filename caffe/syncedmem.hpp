#pragma once

#include <cstdlib>
#include "common.hpp"
#include "utils/math_functions.hpp"
namespace caffe {

// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
// The improvement in performance seems negligible in the single GPU case,
// but might be more significant for parallel training. Most importantly,
// it improved stability for large models on many GPUs.
inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda) {
#ifndef CPU_ONLY
    CUDA_CHECK(cudaMallocHost(ptr, size));
    *use_cuda = true;
    return;
#endif
  *ptr = malloc(size);
  *use_cuda = false;
}

inline void CaffeFreeHost(void* ptr, bool use_cuda) {
#ifndef CPU_ONLY
  if (use_cuda) {
    CUDA_CHECK(cudaFreeHost(ptr));
    return;
  }
#endif
// #ifdef USE_MKL
//   mkl_free(ptr);
// #else
  free(ptr);
// #endif
}


/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *
 * TODO(dox): more thorough description.
 */
class SyncedMemory {
 public:
  SyncedMemory();
  explicit SyncedMemory(size_t size);
  ~SyncedMemory();
  const void* cpu_data();
  void set_cpu_data(void* data);
  const void* gpu_data();
  void set_gpu_data(void* data);
  void* mutable_cpu_data();
  void* mutable_gpu_data();
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  SyncedHead head() const { return head_; }
  size_t size() const { return size_; }

#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif

 private:
  void check_device();
  void to_cpu();
  void to_gpu();
  void* cpu_ptr_;
  void* gpu_ptr_;
  size_t size_;
  SyncedHead head_;
  bool own_cpu_data_;
  bool cpu_malloc_use_cuda_;
  bool own_gpu_data_;
  int device_;
public:
  SyncedMemory(const SyncedMemory&) = delete;
  SyncedMemory& operator=(const SyncedMemory&) = delete;
};  // class SyncedMemory

}  // namespace caffe
