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

template <typename fr>
void transpose_block(fr* input, fr* output, size_t start_row, size_t end_row, size_t rows, size_t cols) {
    for (size_t i = start_row; i < end_row; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

template <typename fr>
void transpose_block_chunk(fr* input, fr* output, size_t start_row, size_t end_row, size_t rows, size_t cols, size_t idx) {
    for (size_t i = start_row; i < end_row; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            output[idx+ j * rows + i] = input[i * cols + j];
        }
    }
}

template <typename fr>
void transpose_matrix(fr* input, fr* output, size_t rows, size_t cols, size_t num_threads) {
    size_t rows_per_thread = rows / num_threads;
    std::vector<std::thread> threads;

    for (size_t i = 0; i < num_threads; ++i) {
        size_t start_row = i * rows_per_thread;
        size_t end_row = (i == num_threads - 1) ? rows : (i + 1) * rows_per_thread;
        threads.push_back(std::thread(transpose_block<fr>, input, output, start_row, end_row, rows, cols));
    }

    for (auto& t : threads) {
        t.join();
    }
}

void naive_ntt(SyncedMemory input, int lg_N, int lg_i, int lg_j, int lg_k, int num_threads, Ntt ntt)
{   
    uint32_t N = 1<<lg_N;
    uint32_t I = 1<<lg_i;
    uint32_t J = 1<<lg_j;
    uint32_t k = 1<<lg_k;

    SyncedMemory d_in(I*4*sizeof(uint64_t)*k);
    SyncedMemory d_in_2(J*4*sizeof(uint64_t)*k);
    SyncedMemory out(I*4*sizeof(uint64_t)*k);
    void* in_ = input.mutable_cpu_data();
    void* out_ = out.mutable_cpu_data();
    void* d_in_ = d_in.mutable_gpu_data();
    void* d_in_2_ = d_in_2.mutable_gpu_data();
    
    uint32_t* step1_map = new uint32_t[N];

    calculate_map1(step1_map, lg_N);
    // warmup
    int stage = 0;
    cudaMemcpy(d_in_2_, in_, d_in_2.size(), cudaMemcpyHostToDevice);
    ntt.forward2_kcolumn(d_in_2, lg_N, lg_j, lg_k,&stage);
    cudaMemcpy(in_, d_in_2_, d_in_2.size(), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    for(int i =0; i < J/k;i++){
        int stage = 0;
        cudaMemcpy2D(d_in_, k*4*sizeof(uint64_t), in_ + i*k*4*sizeof(uint64_t), J*4*sizeof(uint64_t), k*4*sizeof(uint64_t), I, cudaMemcpyHostToDevice);
        ntt.forward1_kcolumn(d_in, lg_N, lg_i, lg_k, &stage);
        cudaMemcpy2D(in_+ i*k*4*sizeof(uint64_t), J*4*sizeof(uint64_t), d_in_, k*4*sizeof(uint64_t), k*4*sizeof(uint64_t), I, cudaMemcpyDeviceToHost);
    }
    
    for(int i =0; i < I/k;i++){
        int stage = 0;
        cudaMemcpy(d_in_2_, in_ + i*d_in_2.size(), d_in_2.size(), cudaMemcpyHostToDevice);
        ntt.forward2_kcolumn(d_in_2, lg_N, lg_j, lg_k, &stage);
        cudaMemcpy(in_ + i*d_in_2.size(), d_in_2_, d_in_2.size(), cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();
    transpose_matrix(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(in_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(in_), I, J, num_threads);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "naive-ntt execution time: " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


void our_ntt(SyncedMemory input, int lg_N, int lg_i, int lg_j, int lg_k, int step1_thread, int step2_thread, Ntt ntt)
{   
    uint32_t N = 1<<lg_N;
    uint32_t I = 1<<lg_i;
    uint32_t J = 1<<lg_j;
    uint32_t k = 1<<lg_k;
    
    SyncedMemory in(input.size());
    SyncedMemory step1_out(input.size());
    SyncedMemory step2_in(input.size());
    SyncedMemory step2_out(input.size());
    SyncedMemory d_in(I*4*sizeof(uint64_t)*k/2);
    SyncedMemory d_in_2(I*4*sizeof(uint64_t)*k/2);
    void* in_ = input.mutable_cpu_data();
    void* d_in_ = d_in.mutable_gpu_data();
    void* d_in_2_ = d_in_2.mutable_gpu_data();
    void* inp_ = in.mutable_cpu_data();
    void* step1_out_ = step1_out.mutable_cpu_data();
    void* step2_in_=step2_in.mutable_cpu_data();
    void* step2_out_ = step2_out.mutable_cpu_data();

    int stage1 = 0;
    int iterations1 = 8;
    int stage2 = 8;
    int iterations2 = 8;

    uint32_t* step1_map = new uint32_t[N];
    uint32_t* step2_map = new uint32_t[N];
    
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

    ntt.forward1(d_in, lg_N, 0, 0, stream3);
    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);
    bit_rev_step1_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(inp_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(in_), step1_map, step1_thread, lg_N, 0);

    for(int i = 0; i < J*2/k;i+=2){
        int stage = 0;
        cudaMemcpyAsync(d_in_, inp_ + d_in.size()*i, d_in.size(), cudaMemcpyHostToDevice, stream1);
        cudaEventRecord(even1, stream1);
        cudaStreamWaitEvent(stream2, even1);
        ntt.forward1(d_in, lg_N, 0, i, stream1);
        cudaMemcpyAsync(d_in_2_, inp_ + d_in.size()*(i+1), d_in_2.size(), cudaMemcpyHostToDevice, stream2);
        cudaEventRecord(even2, stream2);
        ntt.forward1(d_in_2, lg_N, 0, i+1, stream2);
        cudaStreamWaitEvent(stream1, even2);
        cudaMemcpyAsync(step1_out_ + d_in.size()*i, d_in_, d_in.size(), cudaMemcpyDeviceToHost, stream1);
        cudaMemcpyAsync(step1_out_ + d_in_2.size()*(i+1), d_in_2_, d_in_2.size(), cudaMemcpyDeviceToHost, stream2);
        bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(step2_in_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(step1_out_ + i*d_in.size()), step2_map, lg_i+lg_k-1, step2_thread, i*I*k/2);
    }

    cudaDeviceSynchronize();

    for(int i = 0; i < J*2/k;i+=2){
        bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(step2_in_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(step1_out_+ (i+1)*d_in.size()), step2_map, lg_i+lg_k -1, step2_thread, (i+1)*I*k/2);
        cudaMemcpyAsync(d_in_, step2_in_ + d_in.size()*i, d_in.size(), cudaMemcpyHostToDevice, stream1);
        cudaEventRecord(even1, stream1);
        cudaStreamWaitEvent(stream2, even1);
        ntt.forward2(d_in, lg_N, 8, i, stream1);
        cudaMemcpyAsync(d_in_2_, step2_in_ + d_in.size()*(i+1), d_in_2.size(), cudaMemcpyHostToDevice, stream2);
        cudaEventRecord(even2, stream2);
        ntt.forward2(d_in_2, lg_N, 8, i+1, stream2);
        cudaStreamWaitEvent(stream1, even2);
        cudaMemcpyAsync(step2_out_ + d_in.size()*i, d_in_, d_in.size(), cudaMemcpyDeviceToHost, stream1);
        cudaEventRecord(even3, stream1);
        cudaStreamWaitEvent(stream2, even3);
        cudaMemcpyAsync(step2_out_ + d_in_2.size()*(i+1), d_in_2_, d_in_2.size(), cudaMemcpyDeviceToHost, stream2);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "out-ntt execution time: " << milliseconds<< " ms" << std::endl;

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
    uint64_t N = 1<<lg_N;

    int step1_thread = 16;

    Ntt ntt(32);


    SyncedMemory input(N*32);
    void* inp_ = input.mutable_cpu_data();
    const char* file_path = "../../input.bin";

    for (int lg_k = 1; lg_k <= 4; ++lg_k) {
        int step2_thread;

        if (lg_k == 1 || lg_k == 2) {
            step2_thread = 2;
        } else if (lg_k == 3) {
            step2_thread = 4;
        } else { // lg_k == 4
            step2_thread = 8;
        }

        std::cout << "=== lg_k = " << lg_k
                  << ", step1_thread = " << step1_thread
                  << ", step2_thread = " << step2_thread
                  << " ===" << std::endl;

        // repeat 3 times for each settings
        for (int rep = 0; rep < 3; ++rep) {
            read_file(file_path, inp_);

            std::cout << "[naive_ntt] run " << (rep + 1) << "/3" << std::endl;
            naive_ntt(input, lg_N, lg_i, lg_j, lg_k, step1_thread, ntt);
        }

        
        for (int rep = 0; rep < 3; ++rep) {
            read_file(file_path, inp_);

            std::cout << "[our_ntt] run " << (rep + 1) << "/3" << std::endl;
            our_ntt(input, lg_N, lg_i, lg_j, lg_k,
                    step1_thread, step2_thread, ntt);
        }
    }
    return 0;
}