#include <stdint.h>
#include <vector>
#include <iostream>
#include "caffe/syncedmem.hpp"
#include "PLONK/src/structure.cuh"

int main(){
    uint64_t host_data[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    caffe::SyncedMemory mem(10 * sizeof(uint64_t));
    void* gpu_data = mem.mutable_gpu_data();
    caffe::caffe_gpu_memcpy(10 * sizeof(uint64_t), host_data, gpu_data);

    void* cpu_data = mem.mutable_cpu_data();
    for(int i = 0;i < 10; i++){
        printf("%ld  ", (static_cast<uint64_t*>(cpu_data))[i]);
    }
    printf("\n");
    return 0;
}

