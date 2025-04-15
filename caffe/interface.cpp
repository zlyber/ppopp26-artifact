#include "interface.hpp"


SyncedMemory::SyncedMemory()
:kernel(nullptr), type_flag(true) {}

SyncedMemory::SyncedMemory(size_t size)
: kernel(std::make_shared<SyncedMemorykernel>(size)), type_flag(true) {}

SyncedMemory::SyncedMemory(size_t size, bool flag)
: kernel(std::make_shared<SyncedMemorykernel>(size)), type_flag(flag) {}

// SyncedMemory::~SyncedMemory() {
//   kernel->~SyncedMemorykernel();        
// }


const void* SyncedMemory::cpu_data() {
  if(kernel.use_count()){
    return kernel->cpu_data();
  }
}

void SyncedMemory::set_cpu_data(void* data) {
  if(kernel.use_count()){
    kernel->set_cpu_data(data);
  }
}

const void* SyncedMemory::gpu_data() {
  if(kernel.use_count()){
    return kernel->gpu_data();
  }
}

void SyncedMemory::set_gpu_data(void* data) {
  if(kernel.use_count()){
    kernel->set_gpu_data(data);
  }
}

void* SyncedMemory::mutable_cpu_data() {
  if(kernel.use_count()){
    return kernel->mutable_cpu_data();
  }
}

void* SyncedMemory::mutable_cpu_data_async(cudaStream_t stream) {
  if(kernel.use_count()){
    return kernel->mutable_cpu_data_async(stream);
  }
}

void* SyncedMemory::mutable_gpu_data() {
  if(kernel.use_count()){
    return kernel->mutable_gpu_data();
  }
}

void* SyncedMemory::mutable_gpu_data_async(cudaStream_t stream) {
  if(kernel.use_count()){
    return kernel->mutable_gpu_data_async(stream);
  }
}

#ifndef CPU_ONLY
void SyncedMemory::async_gpu_push(const cudaStream_t& stream) {
  if(kernel.use_count()){
    kernel->async_gpu_push(stream);
  }
}
#endif

SyncedMemory::SyncedHead SyncedMemory::head(){
    auto head = kernel->head();
    if (head == SyncedMemorykernel::HEAD_AT_CPU)
        return SyncedMemory::SyncedHead::HEAD_AT_CPU;
    else if (head == SyncedMemorykernel::HEAD_AT_GPU)
        return SyncedMemory::SyncedHead::HEAD_AT_GPU;
    else if (head == SyncedMemorykernel::SYNCED)
        return SyncedMemory::SyncedHead::SYNCED;
    else
      return SyncedMemory::SyncedHead::UNINITIALIZED;
}

size_t SyncedMemory::size() {
    return kernel->size();
}