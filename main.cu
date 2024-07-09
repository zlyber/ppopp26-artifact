#include <stdint.h>
#include <vector>
#include <iostream>
#include "caffe/syncedmem.hpp"
#include "PLONK/src/structure.cuh"
#include <string.h>  // for memcpy

int main(){
    uint64_t host_data[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    uint64_t host_data2[5] = {11, 11, 11, 11,11};

    caffe::SyncedMemory mem(15 * sizeof(uint64_t));
    void* gpu_data = mem.mutable_gpu_data();
    caffe::caffe_gpu_memcpy(10 * sizeof(uint64_t), host_data, gpu_data);
    caffe::caffe_gpu_memcpy(5 * sizeof(uint64_t), host_data2, gpu_data+8*10);

    caffe::SyncedMemory mem2(12 * sizeof(uint64_t));
    void* gpu_data2 = mem2.mutable_gpu_data();
    caffe::caffe_gpu_memcpy(12 * sizeof(uint64_t),gpu_data , gpu_data2);
    void* cpu_data = mem.mutable_cpu_data();
    void* cpu_data2 = mem2.mutable_cpu_data();
    // std::cout<<static_cast<uint64_t*>(gpu_data)[0]<<"\n";
    int a = mem.size();
    std::cout<< a<<"\n";
    // for(int i = 0;i < 10; i++){
    //    static_cast<uint64_t*>(gpu_data)[i]=host_data[i];
    // 
    for(int i = 0;i < 15; i++){
       std::cout<<static_cast<uint64_t*>(cpu_data2)[i]<<" ";
    }
    printf("\n");
    return 0;
}
