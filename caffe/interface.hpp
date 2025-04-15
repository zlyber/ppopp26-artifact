#pragma once
#include "syncedmem.hpp"
#include <memory>
using namespace caffe;


class SyncedMemory {
 public:
  std::shared_ptr<SyncedMemorykernel> kernel;
  bool type_flag;
  SyncedMemory();
  explicit SyncedMemory(size_t size);
  explicit SyncedMemory(size_t size, bool flag);
  // ~SyncedMemory();
  const void* cpu_data();
  void set_cpu_data(void* data);
  const void* gpu_data();
  void set_gpu_data(void* data);
  void* mutable_cpu_data();
  void* mutable_cpu_data_async(cudaStream_t stream);
  void* mutable_gpu_data();
  void* mutable_gpu_data_async(cudaStream_t stream);
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  SyncedHead head();
  size_t size();
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
};  // class SyncedMemory