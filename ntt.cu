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

void read_file_chunk(const char* filename, std::vector<SyncedMemory>data, int chunk_num){
    // 打开文件
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
     std::cerr << "Error opening file: " << filename << std::endl;
     return;
    }
    file.seekg(0, std::ios::end);
    size_t filesize = file.tellg();
    file.seekg(0, std::ios::beg);
 
    size_t base_chunk_size = filesize / chunk_num;


    for (size_t i = 0; i < chunk_num; ++i) {

        void* inp_ = data[i].mutable_cpu_data();

        file.read(reinterpret_cast<char*>(inp_), chunk_num);
        if (!file) {
            std::cerr << "Error reading chunk " << i << " from file: " << filename << std::endl;
            break;
        }
    }
    file.close();
}  

template <typename fr>
void transpose_block(fr* input, fr* output, size_t start_row, size_t end_row, size_t rows, size_t cols) {
    for (size_t i = start_row; i < end_row; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            // 转置：第 i 行第 j 列 -> 第 j 行第 i 列
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

template <typename fr>
void transpose_block_chunk(fr* input, fr* output, size_t start_row, size_t end_row, size_t rows, size_t cols, size_t idx) {
    for (size_t i = start_row; i < end_row; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            // 转置：第 i 行第 j 列 -> 第 j 行第 i 列
            output[idx+ j * rows + i] = input[i * cols + j];
        }
    }
}

template <typename fr>
void transpose_matrix(fr* input, fr* output, size_t rows, size_t cols, size_t num_threads) {
    size_t rows_per_thread = rows / num_threads;
    std::vector<std::thread> threads;

    // 启动多个线程来并行转置矩阵
    for (size_t i = 0; i < num_threads; ++i) {
        size_t start_row = i * rows_per_thread;
        size_t end_row = (i == num_threads - 1) ? rows : (i + 1) * rows_per_thread;
        threads.push_back(std::thread(transpose_block<fr>, input, output, start_row, end_row, rows, cols));
    }

    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }
}

template <typename fr>
void transpose_matrix_chunk(fr* input, fr* output, size_t rows, size_t cols, size_t num_threads, size_t idx) {
    size_t rows_per_thread = rows / num_threads;
    std::vector<std::thread> threads;

    // 启动多个线程来并行转置矩阵
    for (size_t i = 0; i < num_threads; ++i) {
        size_t start_row = i * rows_per_thread;
        size_t end_row = (i == num_threads - 1) ? rows : (i + 1) * rows_per_thread;
        threads.push_back(std::thread(transpose_block_chunk<fr>, input, output, start_row, end_row, rows, cols, idx));
    }

    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }
}

void ntt_read_k_column(SyncedMemory input, int lg_N, int lg_i, int lg_j, Ntt ntt)
{   
    uint32_t N = 1<<lg_N;
    uint32_t I = 1<<lg_i;
    uint32_t J = 1<<lg_j;
    int lg_k = 1;
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
    // bit_rev_step1_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(in_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(in_), step1_map, lg_N, 0);
    transpose_matrix(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(in_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(in_), I, J, 32);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void ntt_read_k_column_chunk(std::vector<SyncedMemory> input, int lg_N, int lg_i, int lg_j, int chunk_num, Ntt ntt)
{   
    uint32_t N = 1<<lg_N;
    uint32_t I = 1<<lg_i;
    uint32_t J = 1<<lg_j;
    int lg_k = 6;
    uint32_t k = 1<<lg_k;

    SyncedMemory d_in(I*4*sizeof(uint64_t)*k);
    SyncedMemory d_in_2(J*4*sizeof(uint64_t)*k);
    SyncedMemory out(I*4*sizeof(uint64_t)*k);
    uint64_t* output = new uint64_t[N*4];

    void* in0_ = input[0].mutable_cpu_data();
    void* out_ = out.mutable_cpu_data();
    void* d_in_ = d_in.mutable_gpu_data();
    void* d_in_2_ = d_in_2.mutable_gpu_data();
    
    uint32_t* step1_map = new uint32_t[N];

    calculate_map1(step1_map, lg_N);
    // warmup
    int stage = 0;
    cudaMemcpy(d_in_2_, in0_, d_in_2.size(), cudaMemcpyHostToDevice);
    ntt.forward2_kcolumn(d_in_2, lg_N, lg_j, lg_k,&stage);
    cudaMemcpy(in0_, d_in_2_, d_in_2.size(), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    
    for(int i =0; i < J/k;i++){
        int stage = 0;
        for(int j = 0; j<chunk_num; j++){
            void* in_ = input[j].mutable_cpu_data();
            cudaMemcpy2D(d_in_ + j*k*4*sizeof(uint64_t)*I/chunk_num, k*4*sizeof(uint64_t), in_ + i*k*4*sizeof(uint64_t), J*4*sizeof(uint64_t), k*4*sizeof(uint64_t), I/chunk_num, cudaMemcpyHostToDevice);
        }
        ntt.forward1_kcolumn(d_in, lg_N, lg_i, lg_k, &stage);
        for(int j = 0; j<chunk_num; j++){
            void* in_ = input[j].mutable_cpu_data();
            cudaMemcpy2D(in_+ i*k*4*sizeof(uint64_t), J*4*sizeof(uint64_t), d_in_+ j*k*4*sizeof(uint64_t)*I/chunk_num, k*4*sizeof(uint64_t), k*4*sizeof(uint64_t), I/chunk_num, cudaMemcpyDeviceToHost);
        }
    }
    
    for(int j = 0; j<chunk_num; j++){
        void* in_ = input[j].mutable_cpu_data();
        for(int i =0; i < I/(k*chunk_num);i++){
            int stage = 0;
            cudaMemcpy(d_in_2_, in_ + i*d_in_2.size(), d_in_2.size(), cudaMemcpyHostToDevice);
            ntt.forward2_kcolumn(d_in_2, lg_N, lg_j, lg_k, &stage);
            cudaMemcpy(in_ + i*d_in_2.size(), d_in_2_, d_in_2.size(), cudaMemcpyDeviceToHost);
        }
    }
    cudaDeviceSynchronize();
    // bit_rev_step1_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(in_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(in_), step1_map, lg_N, 0);
    for(int j = 0; j<chunk_num; j++){
        void* in_ = input[j].mutable_cpu_data();
        transpose_matrix_chunk(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(in_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(output), I, J/4, 32, j*N/chunk_num);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void ntt_read_k_column_large(void* input, int lg_N, int lg_i, int lg_j, Ntt ntt)
{   
    uint32_t N = 1<<lg_N;
    uint32_t I = 1<<lg_i;
    uint32_t J = 1<<lg_j;
    int lg_k = 13;
    uint32_t k = 1<<lg_k;

    SyncedMemory d_in(I*4*sizeof(uint64_t)*k);
    SyncedMemory d_in_2(J*4*sizeof(uint64_t)*k);
    SyncedMemory out(I*4*sizeof(uint64_t)*k);

    void* d_in_ = d_in.mutable_gpu_data();
    void* d_in_2_ = d_in_2.mutable_gpu_data();
    // uint64_t* d_in = new uint64_t[I*k*4];
    // uint64_t* d_in_2 = new uint64_t[J*4*k];
    uint32_t* step1_map = new uint32_t[N];

    calculate_map1(step1_map, lg_N);
    // warmup
    int stage = 0;
    cudaMemcpy(d_in_2_, input, d_in_2.size(), cudaMemcpyHostToDevice);
    ntt.forward2_kcolumn(d_in_2, lg_N, lg_j, lg_k,&stage);
    cudaMemcpy(input, d_in_2_, d_in_2.size(), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    for(int i =0; i < J/k;i++){
        int stage = 0;
        cudaMemcpy2D(d_in_, k*4*sizeof(uint64_t), input + i*k*4*sizeof(uint64_t), J*4*sizeof(uint64_t), k*4*sizeof(uint64_t), I, cudaMemcpyHostToDevice);
        ntt.forward1_kcolumn(d_in, lg_N, lg_i, lg_k, &stage);
        cudaMemcpy2D(input + i*k*4*sizeof(uint64_t), J*4*sizeof(uint64_t), d_in_, k*4*sizeof(uint64_t), k*4*sizeof(uint64_t), I, cudaMemcpyDeviceToHost);
    }
    
    for(int i =0; i < I/k;i++){
        int stage = 0;
        cudaMemcpy(d_in_2_, input + i*d_in_2.size(), d_in_2.size(), cudaMemcpyHostToDevice);
        ntt.forward2_kcolumn(d_in_2, lg_N, lg_j, lg_k, &stage);
        cudaMemcpy(input + i*d_in_2.size(), d_in_2_, d_in_2.size(), cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();
    // bit_rev_step1_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(in_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(in_), step1_map, lg_N, 0);
    transpose_matrix(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(input), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(input), I, J, 32);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void our_ntt(SyncedMemory input, int lg_N, int lg_i, int lg_j, Ntt ntt)
{   
    uint32_t N = 1<<lg_N;
    uint32_t I = 1<<lg_i;
    uint32_t J = 1<<lg_j;
    int lg_k = 1;
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

    uint32_t block_size2 = 1<<11;
    uint32_t num_threads2 = 1 << (lg_N - 1);
    uint32_t num_block2 = (num_threads2 + block_size2 - 1) / block_size2;

    for(int i=0;i<num_block2;i++){
        for(int j=0;j<block_size2;j++){
            uint32_t tid = j + i*block_size2;
            calculate_indices(tid, stage1, 14, 14, 12, step2_map);
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
    bit_rev_step1_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(inp_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(in_), step1_map, lg_N, 0);

    

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
        // cudaStreamSynchronize(stream1);
        cudaMemcpyAsync(step1_out_ + d_in_2.size()*(i+1), d_in_2_, d_in_2.size(), cudaMemcpyDeviceToHost, stream2);
        bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(step2_in_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(step1_out_), step2_map, lg_i + lg_k, i);
        // bit_rev_step2(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(step2_in_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(step1_out_), step2_map, lg_i + lg_k, i);
    }

    cudaDeviceSynchronize();

    for(int i = 0; i < J*2/k;i+=2){
        cudaStreamSynchronize(stream2);
        cudaMemcpyAsync(d_in_, step2_in_ + d_in.size()*i, d_in.size(), cudaMemcpyHostToDevice, stream1);
        cudaEventRecord(even1, stream1);
        cudaStreamWaitEvent(stream2, even1);
        ntt.forward2(d_in, lg_N, 8, i, stream1);
        cudaMemcpyAsync(d_in_2_, step2_in_ + d_in.size()*(i+1), d_in_2.size(), cudaMemcpyHostToDevice, stream2);
        cudaEventRecord(even2, stream2);
        ntt.forward2(d_in_2, lg_N, 8, i+1, stream2);
        cudaStreamWaitEvent(stream1, even2);
        cudaMemcpyAsync(step2_out_ + d_in.size()*i, d_in_, d_in.size(), cudaMemcpyDeviceToHost, stream1);
        cudaMemcpyAsync(step2_out_ + d_in_2.size()*(i+1), d_in_2_, d_in_2.size(), cudaMemcpyDeviceToHost, stream2);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Kernel execution time: " << milliseconds<< " ms" << std::endl;

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

void our_ntt_chunk(std::vector<SyncedMemory> input, int lg_N, int lg_i, int lg_j, int chunk_num, Ntt ntt)
{   
    uint32_t N = 1<<lg_N;
    uint32_t I = 1<<lg_i;
    uint32_t J = 1<<lg_j;
    int lg_k = 1;
    uint32_t k = 1<<lg_k;
    
    SyncedMemory d_in(I*4*sizeof(uint64_t)*k/2);
    SyncedMemory d_in_2(I*4*sizeof(uint64_t)*k/2);
    
    void* d_in_ = d_in.mutable_gpu_data();
    void* d_in_2_ = d_in_2.mutable_gpu_data();  
    int stage1 = 0;
    int iterations1 = 8;
    int stage2 = 8;
    int iterations2 = 8;

    uint32_t* step1_map = new uint32_t[N];
    uint32_t* step2_map = new uint32_t[N];
    uint64_t* bitrev = new uint64_t[N*4];
    calculate_map1(step1_map, lg_N);

    uint32_t block_size2 = 1<<11;
    uint32_t num_threads2 = 1 << (lg_N - 1);
    uint32_t num_block2 = (num_threads2 + block_size2 - 1) / block_size2;

    for(int i=0;i<num_block2;i++){
        for(int j=0;j<block_size2;j++){
            uint32_t tid = j + i*block_size2;
            calculate_indices(tid, stage1, 14, 14, 14, step2_map);
        }
    }
    //warm_up
    void* inp_ = input[0].mutable_cpu_data();
    cudaMemcpy(d_in_, inp_, d_in.size(), cudaMemcpyHostToDevice);
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
    cudaEvent_t even1, even2, even3;
    cudaEventCreate(&even1);
    cudaEventCreate(&even2);
    cudaEventCreate(&even3);

    ntt.forward1(d_in, lg_N, 0, 0, stream3);
    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);
    for(int i=0;i<chunk_num;i++){
        void* in_ = input[i].mutable_cpu_data();
        bit_rev_step1_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(bitrev), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(in_), step1_map, lg_N-2, i*N/4);
    }

    for(int i = 0; i < J*2/k;i+=2){
        void* in_ = input[i].mutable_cpu_data();
        int stage = 0;
        cudaMemcpyAsync(d_in_, in_ + d_in.size()*i, d_in.size(), cudaMemcpyHostToDevice, stream1);
        cudaEventRecord(even1, stream1);
        cudaStreamWaitEvent(stream2, even1);
        ntt.forward1(d_in, lg_N, 0, i, stream1);
        cudaMemcpyAsync(d_in_2_, in_ + d_in.size()*(i+1), d_in_2.size(), cudaMemcpyHostToDevice, stream2);
        cudaEventRecord(even2, stream2);
        ntt.forward1(d_in_2, lg_N, 0, i+1, stream2); 
        cudaStreamWaitEvent(stream1, even2);
        cudaMemcpyAsync(in_ + d_in.size()*i, d_in_, d_in.size(), cudaMemcpyDeviceToHost, stream1);
        cudaMemcpyAsync(in_ + d_in_2.size()*(i+1), d_in_2_, d_in_2.size(), cudaMemcpyDeviceToHost, stream2);
        bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(bitrev), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(in_), step1_map, lg_i + lg_k, i*I*k);
    }
    
    cudaDeviceSynchronize();

    for(int i = 0; i < J*2/k;i+=2){
        void* in_ = input[i].mutable_cpu_data();
        cudaStreamSynchronize(stream2);
        cudaMemcpyAsync(d_in_, in_ + d_in.size()*i, d_in.size(), cudaMemcpyHostToDevice, stream1);
        cudaEventRecord(even1, stream1);
        cudaStreamWaitEvent(stream2, even1);
        ntt.forward2(d_in, lg_N, 8, i, stream1);
        cudaMemcpyAsync(d_in_2_, in_ + d_in.size()*(i+1), d_in_2.size(), cudaMemcpyHostToDevice, stream2);
        cudaEventRecord(even2, stream2);
        ntt.forward2(d_in_2, lg_N, 8, i+1, stream2);
        cudaStreamWaitEvent(stream1, even2);
        cudaMemcpyAsync(in_ + d_in.size()*i, d_in_, d_in.size(), cudaMemcpyDeviceToHost, stream1);
        cudaMemcpyAsync(in_ + d_in_2.size()*(i+1), d_in_2_, d_in_2.size(), cudaMemcpyDeviceToHost, stream2);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Kernel execution time: " << milliseconds<< " ms" << std::endl;

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
    delete bitrev;
}

int main() {
    int lg_N = 28;
    int lg_i = 14;
    int lg_j = 14;
    uint64_t N = 1<<lg_N;
    // int lg_LDE = 25;
    // uint64_t coset_size = 1<<lg_LDE;
    // int lg_chunk = 21;
    // uint64_t chunk = 1<<lg_chunk;

    // int stage1 = 0;
    // int iterations1 = 8;
    // int stage2 = 8;
    // int iterations2 = 8;
    // int stage3 = 16;
    // int iterations3 = 9;
    // uint32_t* step1_map = new uint32_t[coset_size];
    // uint32_t* step2_map = new uint32_t[coset_size];
    // uint32_t* step3_map = new uint32_t[coset_size];

    // calculate_map1(step1_map, lg_LDE);

    // uint32_t block_size2 = 128;
    // uint32_t num_threads2 = 1 << (lg_LDE - 1);
    // uint32_t num_block2 = (num_threads2 + block_size2 - 1) / block_size2;

    // uint32_t block_size3 = 256;
    // uint32_t num_threads3 = 1 << (lg_LDE - 1);
    // uint32_t num_block3 = (num_threads3 + block_size3 - 1) / block_size3;

    // for(int i=0;i<num_block2;i++){
    //     for(int j=0;j<block_size2;j++){
    //         uint32_t tid = j + i*block_size2;
    //         calculate_indices(tid, stage1, iterations1, stage2, iterations2, step2_map);
    //     }
    // }

    // for(int i=0;i<num_block3;i++){
    //     for(int j=0;j<block_size3;j++){
    //         uint32_t tid = j + i*block_size3;
    //         calculate_indices(tid, stage1, iterations3, stage3, iterations3, step3_map);
    //     }
    // }

    Ntt ntt(32);
    std::vector<SyncedMemory> input;
    for(int i=0;i<4;i++){
        SyncedMemory in(N*sizeof(uint64_t));
        void* in_ = in.mutable_cpu_data();
        input.push_back(in);
    }

    // SyncedMemory input(N*8);
    // void* inp_ = input.mutable_cpu_data();
    const char* w_l_f = "/home/zhiyuan/w_scalar-28.bin";
    read_file_chunk(w_l_f, input, 4);

    // ntt_read_k_column_chunk(input, lg_N, lg_i, lg_j, 4, ntt);
    our_ntt_chunk(input, lg_N, lg_i, lg_j, 4, ntt);
    return 0;
}