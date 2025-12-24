#include <iostream>
#include <cstdint>
#include "caffe/interface.hpp"
#include "PLONK/utils/function.cuh"
#include <chrono>
#include <thread>
#include <atomic>
#include <cmath>
#include <mutex>

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


void our_lde(SyncedMemory input, int lg_N, int lambda, int lg_i, int lg_j, int lg_k, int step1_thread, int step2_thread, Ntt_coset ntt)
{   
    uint32_t N = 1<<lg_N;
    uint32_t coset_size = N<<lambda;
    uint32_t I = 1<<lg_i;
    uint32_t J = 1<<lg_j;
    uint32_t k = 1<<lg_k;
    
    SyncedMemory in(input.size());
    SyncedMemory step1_in(input.size());
    SyncedMemory step1_out(input.size()<<lambda);
    SyncedMemory step2_in(input.size()<<lambda);
    SyncedMemory step2_out(input.size()<<lambda);
    SyncedMemory d_in(I*4*sizeof(uint64_t)*k/2);
    SyncedMemory d_in_2(I*4*sizeof(uint64_t)*k/2);
    SyncedMemory d_out(d_in.size()<<lambda);
    SyncedMemory d_out_2(d_in.size()<<lambda);
    void* in_ = input.mutable_cpu_data();
    void* d_in_ = d_in.mutable_gpu_data();
    void* d_in_2_ = d_in_2.mutable_gpu_data();
    void* d_out_ = d_out.mutable_gpu_data();
    void* d_out_2_ = d_out_2.mutable_gpu_data();
    void* inp_ = in.mutable_cpu_data();
    void* step1_out_ = step1_out.mutable_cpu_data();
    void* step2_in_=step2_in.mutable_cpu_data();
    void* step2_out_ = step2_out.mutable_cpu_data();

    int stage1 = 0;
    int iterations1 = 8;
    int stage2 = 8;
    int iterations2 = 8;

    uint32_t* step1_map = new uint32_t[coset_size];
    uint32_t* step2_map = new uint32_t[coset_size];
    
    calculate_map1(step1_map, lg_N);

    uint32_t block_size2 = 1<<9;
    uint32_t num_threads2 = 1 << (lg_N - 1);
    uint32_t num_block2 = (num_threads2 + block_size2 - 1) / block_size2;

    for(int i=0;i<num_block2;i++){
        for(int j=0;j<block_size2;j++){
            uint32_t tid = j + i*block_size2;
            calculate_indices(tid, stage1, 12, 12, 10, step2_map);
        }
    }
    //warm_up
    cudaMemcpy(d_in_, inp_ + d_in.size(), d_in.size(), cudaMemcpyHostToDevice);
    ntt.forward1(d_in, lg_N, 0, 0);
    ntt.forward2(d_in, lg_N, 0, 0);

    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaEvent_t even1, even2, even3, warmupstart, warmupend;
    cudaEventCreate(&even1);
    cudaEventCreate(&even2);
    cudaEventCreate(&even3);

    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);

    bit_rev_step1_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(inp_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(in_), step1_map, step1_thread, lg_N, 0);
    cudaDeviceSynchronize();
    for(int i = 0; i < J*2/k;i+=2){
        cudaMemcpyAsync(d_in_, inp_ + d_in.size()*i, d_in.size(), cudaMemcpyHostToDevice, stream1);
        cudaEventRecord(even1, stream1);
        cudaStreamWaitEvent(stream2, even1);
        ntt.init_and_forward(d_in, d_out, lg_N+lambda, i, lambda, 0, false, stream1);
        cudaMemcpyAsync(d_in_2_, inp_ + d_in.size()*(i+1), d_in_2.size(), cudaMemcpyHostToDevice, stream2);
        cudaEventRecord(even2, stream2);
        ntt.init_and_forward(d_in_2, d_out, lg_N+lambda, i+1, lambda, 0, false, stream2);
        cudaStreamWaitEvent(stream1, even2);
        cudaMemcpyAsync(step1_out_ + d_out.size()*i, d_out_, d_out.size(), cudaMemcpyDeviceToHost, stream1);
        cudaMemcpyAsync(step1_out_ + d_out_2.size()*(i+1), d_out_2_, d_out_2.size(), cudaMemcpyDeviceToHost, stream2);
        cudaStreamSynchronize(stream1);
        bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(step2_in_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(step1_out_ + i*d_out.size()), step2_map, lg_i+lg_k+lambda, step2_thread, i*I*k);
    }

    cudaDeviceSynchronize();

    for(int i = 0; i < J*2/k;i+=2){
        cudaMemcpyAsync(d_out_, step2_in_ + d_out.size()*i, d_out.size(), cudaMemcpyHostToDevice, stream1);
        cudaEventRecord(even1, stream1);
        cudaStreamWaitEvent(stream2, even1);
        ntt.forward2(d_out, lg_N, 8, i, stream1);
        cudaMemcpyAsync(d_out_2_, step2_in_ + d_out.size()*(i+1), d_out_2.size(), cudaMemcpyHostToDevice, stream2);
        cudaEventRecord(even2, stream2);
        ntt.forward2(d_out_2, lg_N, 8, i+1, stream2);
        cudaStreamWaitEvent(stream1, even2);
        cudaMemcpyAsync(step2_out_ + d_out.size()*i, d_out_, d_out.size(), cudaMemcpyDeviceToHost, stream1);
        cudaEventRecord(even3, stream1);
        cudaStreamWaitEvent(stream2, even3);
        cudaMemcpyAsync(step2_out_ + d_out_2.size()*(i+1), d_out_2_, d_out_2.size(), cudaMemcpyDeviceToHost, stream2);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "our-lde execution time: " << milliseconds<< " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(even1);
    cudaEventDestroy(even2);
    cudaEventDestroy(even3);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);

    delete step1_map;
    delete step2_map;
}

int main() {
    int lg_N = 22;
    int lg_i = 12;
    int lg_j = 10;
    int lambda = 3;
    uint64_t N = 1<<lg_N;

    int step1_thread = 16;

    Ntt_coset ntt(32, lg_N+lambda);


    SyncedMemory input(N*32);
    void* inp_ = input.mutable_cpu_data();
    const char* file_path = "../../input.bin";

    for (int lg_k = 1; lg_k <= 4; ++lg_k) {
        int step2_thread;

        if (lg_k == 1 || lg_k == 2) {
            step2_thread = 2;
        } else if (lg_k == 3) {
            step2_thread = 4;
        } else {
            step2_thread = 8;
        }

        std::cout << "=== lg_k = " << lg_k
                  << ", step1_thread = " << step1_thread
                  << ", step2_thread = " << step2_thread
                  << " ===" << std::endl;

        // repeat 3 times for each settings
        for (int rep = 0; rep < 3; ++rep) {
            read_file(file_path, inp_);

            std::cout << "[our_lde] run " << (rep + 1) << "/3" << std::endl;
            our_lde(input, lg_N, lambda, lg_i, lg_j, lg_k,
                    step1_thread, step2_thread, ntt);
        }
    }
    return 0;
}