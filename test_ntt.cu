#include <iostream>
#include <fstream>
#include "PLONK/utils/function.cuh"
#include "caffe/interface.hpp"

void read_file(const char* filename, void* data){
   // 打开文件
   std::ifstream file(filename, std::ios::binary);
   if (!file) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
   }
   file.seekg(0, std::ios::end);
   size_t fileSize = file.tellg();
   file.seekg(0, std::ios::beg);

   file.read(reinterpret_cast<char*>(data), fileSize);
   if (!file) {
    std::cerr << "Error reading file: " << filename << std::endl;
   }
   file.close();
}  

int main(){
    uint32_t N = 1<<22;
    uint32_t LDE_size = 1<<25;
    int chunk_num = 2;
    SyncedMemory w_l(N*4*sizeof(uint64_t));
    
    void* w_l_ = w_l.mutable_cpu_data();
    const char* w_l_f = "/home/zhiyuan/w_l_poly-15.bin";
    read_file(w_l_f, w_l_);

    SyncedMemory w_l_d(LDE_size*4*sizeof(uint64_t));
    void* w_l_d_ = w_l_d.mutable_gpu_data();
    // Intt INTT(32);
    SyncedMemory out(LDE_size*4*sizeof(uint64_t)); 
    void* out_ = out.mutable_cpu_data();
    // for(int i =0; i < N/chunk_num;i++){
    //     cudaMemcpy2D(w_l_chunk_ + i*8*sizeof(uint64_t), 2*4*sizeof(uint64_t)/chunk_num, w_l_, N*4*sizeof(uint64_t)/chunk_num, 4*sizeof(uint64_t), 2, cudaMemcpyHostToDevice);

    //     cudaMemcpy(out_ + i*4* sizeof(uint64_t), w_l_chunk_ + N*4*sizeof(uint64_t)/chunk_num + i*4*sizeof(uint64_t), 8*sizeof(uint64_t), cudaMemcpyDeviceToHost);
    // }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Time elapsed: " << milliseconds << " ms" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}