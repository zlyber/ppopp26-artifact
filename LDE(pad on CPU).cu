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
// void rotate_indices(uint32_t& idx0, uint32_t& idx1, int stage, int iterations) {
//     // rotate "iterations" bits in indices
//     uint32_t mask = ((uint32_t)1 << (stage + iterations)) - ((uint32_t)1 << stage);

//     // Rotate idx0
//     uint32_t rotw = idx0 & mask;
//     rotw = (rotw >> 1) | (rotw << (iterations - 1));
//     idx0 = (idx0 & ~mask) | (rotw & mask);

//     // Rotate idx1
//     rotw = idx1 & mask;
//     rotw = (rotw >> 1) | (rotw << (iterations - 1));
//     idx1 = (idx1 & ~mask) | (rotw & mask);
// }

// void calculate_indices(uint32_t tid, int stage1, int iterations1, int stage2, int iterations2, uint32_t* map) {
//     const uint32_t out_ntt_size1 = (1 << (stage1 + iterations1 - 1));
//     const uint32_t inp_ntt_size1 = (uint32_t)1 << stage1;

//     const uint32_t thread_ntt_pos1 = (tid & (out_ntt_size1 - 1)) >> (iterations1 - 1);

//     uint32_t idx1_0 = tid & ~(out_ntt_size1 - 1);
//     idx1_0 += (tid << stage1) & (out_ntt_size1 - 1);
//     idx1_0 = idx1_0 * 2 + thread_ntt_pos1;

//     uint32_t idx1_1 = idx1_0 + inp_ntt_size1;

//     const uint32_t out_ntt_size2 = (1 << (stage2 + iterations2 - 1));
//     const uint32_t inp_ntt_size2 = (uint32_t)1 << stage2;

//     const uint32_t thread_ntt_pos2 = (tid & (out_ntt_size2 - 1)) >> (iterations2 - 1);

//     uint32_t idx2_0 = tid & ~(out_ntt_size2 - 1);
//     idx2_0 += (tid << stage2) & (out_ntt_size2 - 1);
//     idx2_0 = idx2_0 * 2 + thread_ntt_pos2;

//     uint32_t idx2_1 = idx2_0 + inp_ntt_size2;
//     map[idx2_0] = idx1_0;
//     map[idx2_1] = idx1_1;
// }

// // Function to reverse bits for an unsigned integer type
// uint32_t bit_rev_32(uint32_t i, uint32_t nbits) {
//     uint32_t rev = 0;
//     for (uint32_t j = 0; j < nbits; ++j) {
//         rev = (rev << 1) | (i & 1);
//         i >>= 1;
//     }
//     return rev;
// }

// // Overload for long long integer (64-bit)
// uint64_t bit_rev_64(uint64_t i, uint32_t nbits) {
//     uint64_t rev = 0;
//     for (uint32_t j = 0; j < nbits; ++j) {
//         rev = (rev << 1) | (i & 1);
//         i >>= 1;
//     }
//     return rev;
// }

// uint32_t bit_rev(uint32_t i, uint32_t nbits) {
//     if (sizeof(i) == 4 || nbits <= 32) {
//         return bit_rev_32((i), nbits);
//     } else {
//         return bit_rev_64((i), nbits);
//     }
// }

// template <typename fr>
// void bit_rev_permutation(
//     fr* d_out,
//     const fr* d_in,
//     int lg_N) {
//   uint64_t N = 1 << lg_N;
//   for(int i = 0; i< N; i++){
//     uint32_t r = bit_rev(i, lg_N);
//     if (i < r || (d_out != d_in && i == r)) {
//         fr t0 = d_in[i];
//         fr t1 = d_in[r];
//         d_out[r] = t0;
//         d_out[i] = t1;
//     }
//   }
// }


// // Worker function for each thread
// template <typename fr>
// void bit_rev_permutation_worker(
//     fr* d_out,
//     const fr* d_in,
//     int lg_N,
//     uint64_t start_idx,
//     uint64_t end_idx) {

//     for (uint64_t i = start_idx; i < end_idx; ++i) {
//         uint32_t r = bit_rev(i, lg_N);
//         if (i < r || (d_out != d_in && i == r)) {
//             fr t0 = d_in[i];
//             fr t1 = d_in[r];
//             d_out[r] = t0;
//             d_out[i] = t1;
//         }
//     }
// }

// template <typename fr>
// void pad_transpose_worker(
//     fr* d_out,
//     const fr* d_in,
//     uint64_t start_idx,
//     uint64_t end_idx) {
    
//     for (uint64_t i = start_idx; i < end_idx; ++i) {
//         d_out[i << 3] = d_in[i];
//     }
// }

// template <typename fr>
// void bit_rev_step2_worker(
//     fr* d_out,
//     const fr* d_in,
//     const uint32_t* map,
//     int lg_N,
//     uint64_t idx,
//     uint64_t start_idx,
//     uint64_t end_idx) {
    
//     uint64_t N = 1 << lg_N;
//     for (uint64_t i = start_idx; i < end_idx; ++i) {
//         d_out[map[idx + i]] = d_in[i];
//     }
// }

// template <typename fr>
// void bit_rev_step3_worker(
//     fr* d_out,
//     const fr* d_in,
//     const uint32_t* map,
//     int lg_N,
//     uint64_t idx,
//     uint64_t start_idx,
//     uint64_t end_idx) {

//     uint64_t N = 1 << lg_N;
//     for (uint64_t i = start_idx; i < end_idx; ++i) {
//         d_out[map[idx + i]] = d_in[i];
//     }
// }

// // Multithreaded version of bit_rev_permutation
// template <typename fr>
// void bit_rev_permutation_parallel(
//     fr* d_out,
//     const fr* d_in,
//     int lg_N) {

//     uint64_t N = 1 << lg_N;
//     uint32_t num_threads = std::thread::hardware_concurrency();  // 获取硬件并发线程数
//     // num_threads = 1 << (static_cast<int>(std::log2(num_threads)));
//     num_threads = 32;
//     uint32_t chunk_size = N / num_threads;

//     std::vector<std::thread> threads;

//     // 启动多个线程来并行执行比特反转置换
//     for (uint32_t t = 0; t < num_threads; ++t) {
//         uint64_t start_idx = t * chunk_size;
//         uint64_t end_idx = (t == num_threads - 1) ? N : start_idx + chunk_size;
//         threads.push_back(std::thread(bit_rev_permutation_worker<fr>, d_out, d_in, lg_N, start_idx, end_idx));
//     }

//     // 等待所有线程完成
//     for (auto& t : threads) {
//         t.join();
//     }
// }

// template <typename fr>
// void pad_and_transpose(
//     fr* d_out,
//     const fr* d_in,
//     int lg_chunk,
//     uint64_t idx) {

//     uint64_t N = 1 << lg_chunk;
//     uint32_t num_threads = std::thread::hardware_concurrency();  // 获取硬件并发线程数
//     // num_threads = 1 << (static_cast<int>(std::log2(num_threads)));
//     num_threads = 32;
//     uint32_t chunk_size = N / num_threads;

//     std::vector<std::thread> threads;

//     // 启动多个线程来并行执行比特反转置换
//     for (uint32_t t = 0; t < num_threads; ++t) {
//         uint64_t start_idx =  idx + t * chunk_size;
//         uint64_t end_idx = (t == num_threads - 1) ? N + idx : start_idx + chunk_size;
//         threads.push_back(std::thread(pad_transpose_worker<fr>, d_out, d_in, start_idx, end_idx));
//     }

//     // 等待所有线程完成
//     for (auto& t : threads) {
//         t.join();
//     }
// }

// template <typename fr>
// void bit_rev_step2_parallel(
//     fr* d_out,
//     const fr* d_in,
//     uint32_t* map,
//     int lg_N,
//     uint64_t idx) {

//     uint64_t N = 1 << lg_N;
//     uint32_t num_threads = std::thread::hardware_concurrency();  // 获取硬件并发线程数
//     // num_threads = 1 << (static_cast<int>(std::log2(num_threads)));
//     num_threads = 32;
//     uint32_t chunk_size = N / num_threads;

//     std::vector<std::thread> threads;

//     // 启动多个线程来并行执行比特反转置换
//     for (uint32_t t = 0; t < num_threads; ++t) {
//         uint64_t start_idx =  t * chunk_size;
//         uint64_t end_idx = (t == num_threads - 1) ? N : start_idx + chunk_size;
//         threads.push_back(std::thread(bit_rev_step2_worker<fr>, d_out, d_in, map, lg_N, idx, start_idx, end_idx));
//     }

//     // 等待所有线程完成
//     for (auto& t : threads) {
//         t.join();
//     }
// }

// template <typename fr>
// void bit_rev_step3_parallel(
//     fr* d_out,
//     const fr* d_in,
//     const uint32_t* map,
//     int lg_N,
//     uint64_t idx) {

//     uint64_t N = 1 << lg_N;
//     uint32_t num_threads = std::thread::hardware_concurrency();  // 获取硬件并发线程数
//     // num_threads = 1 << (static_cast<int>(std::log2(num_threads)));
//     num_threads = 32;
//     uint32_t chunk_size = N / num_threads;

//     std::vector<std::thread> threads;

//     // 启动多个线程来并行执行比特反转置换
//     for (uint32_t t = 0; t < num_threads; ++t) {
//         uint64_t start_idx = t * chunk_size;
//         uint64_t end_idx = (t == num_threads - 1) ? N : start_idx + chunk_size;
//         threads.push_back(std::thread(bit_rev_step3_worker<fr>, d_out, d_in, map, lg_N, idx, start_idx, end_idx));
//     }

//     // 等待所有线程完成
//     for (auto& t : threads) {
//         t.join();
//     }
// }

int main() {
    int lg_N = 22;
    uint64_t N = 1<<lg_N;
    int lg_LDE = 25;
    uint64_t coset_size = 1<<lg_LDE;
    int lg_chunk = 21;
    uint64_t chunk = 1<<lg_chunk;

    int stage1 = 0;
    int iterations1 = 8;
    int stage2 = 8;
    int iterations2 = 8; 
    int stage3 = 16;
    int iterations3 = 9;
    uint32_t* step1_map = new uint32_t[coset_size];
    uint32_t* step2_map = new uint32_t[coset_size];
    uint32_t* step3_map = new uint32_t[coset_size];

    calculate_map1(step1_map, lg_LDE);

    uint32_t block_size2 = 128;
    uint32_t num_threads2 = 1 << (lg_LDE - 1);
    uint32_t num_block2 = (num_threads2 + block_size2 - 1) / block_size2;

    uint32_t block_size3 = 256;
    uint32_t num_threads3 = 1 << (lg_LDE - 1);
    uint32_t num_block3 = (num_threads3 + block_size3 - 1) / block_size3;

    for(int i=0;i<num_block2;i++){
        for(int j=0;j<block_size2;j++){
            uint32_t tid = j + i*block_size2;
            calculate_indices(tid, stage1, iterations1, stage2, iterations2, step2_map);
        }
    }

    for(int i=0;i<num_block3;i++){
        for(int j=0;j<block_size3;j++){
            uint32_t tid = j + i*block_size3;
            calculate_indices(tid, stage1, iterations3, stage3, iterations3, step3_map);
        }
    }


    SyncedMemory input(N*4*sizeof(uint64_t));
    SyncedMemory step1_in(N*4*sizeof(uint64_t));
    SyncedMemory bit_rev_map(coset_size*4*sizeof(uint64_t));
    SyncedMemory output(coset_size*4*sizeof(uint64_t));

    std::vector<SyncedMemory>cpu_buffer;
    
    for(int j=0;j<16;j++){
        SyncedMemory buffer(N * 16);
        void* buffer_ = buffer.mutable_cpu_data();
        cpu_buffer.push_back(buffer);
    }

    std::vector<SyncedMemory>global_buffer;

    for(int j=0;j<16;j++){
        SyncedMemory buffer(N * 16);
        void* buffer_ = buffer.mutable_cpu_data();
        global_buffer.push_back(buffer);
    }

    void* input_ = input.mutable_cpu_data();
    void* map_ = bit_rev_map.mutable_cpu_data();
    const char* w_l_f = "/home/zhiyuan/w_4_poly-15.bin";
    read_file(w_l_f, input_);

    void* output_ = output.mutable_cpu_data();
    void* step1_in_ = step1_in.mutable_cpu_data();
    void* buffer1_ = global_buffer[0].mutable_gpu_data();
    void* buffer2_ = global_buffer[1].mutable_gpu_data();
    void* buffer3_ = global_buffer[2].mutable_gpu_data();
    void* buffer4_ = global_buffer[3].mutable_gpu_data();
    void* buffer5_ = global_buffer[4].mutable_gpu_data();
    void* buffer6_ = global_buffer[5].mutable_gpu_data();
    void* buffer7_ = global_buffer[6].mutable_gpu_data();
    void* buffer8_ = global_buffer[7].mutable_gpu_data();
    void* buffer9_ = global_buffer[8].mutable_gpu_data();
    void* buffer10_ = global_buffer[9].mutable_gpu_data();
    void* buffer11_ = global_buffer[10].mutable_gpu_data();
    void* buffer12_ = global_buffer[11].mutable_gpu_data();
    void* buffer13_ = global_buffer[12].mutable_gpu_data();
    void* buffer14_ = global_buffer[13].mutable_gpu_data();
    void* buffer15_ = global_buffer[14].mutable_gpu_data();
    void* buffer16_ = global_buffer[15].mutable_gpu_data();

    void* cpu_buffer1_ = cpu_buffer[0].mutable_cpu_data();
    void* cpu_buffer2_ = cpu_buffer[1].mutable_cpu_data();
    void* cpu_buffer3_ = cpu_buffer[2].mutable_cpu_data();
    void* cpu_buffer4_ = cpu_buffer[3].mutable_cpu_data();
    void* cpu_buffer5_ = cpu_buffer[4].mutable_cpu_data();
    void* cpu_buffer6_ = cpu_buffer[5].mutable_cpu_data();
    void* cpu_buffer7_ = cpu_buffer[6].mutable_cpu_data();
    void* cpu_buffer8_ = cpu_buffer[7].mutable_cpu_data();

    void* cpu_buffer9_ = cpu_buffer[8].mutable_cpu_data();
    void* cpu_buffer10_ = cpu_buffer[9].mutable_cpu_data();
    void* cpu_buffer11_ = cpu_buffer[10].mutable_cpu_data();
    void* cpu_buffer12_ = cpu_buffer[11].mutable_cpu_data();
    void* cpu_buffer13_ = cpu_buffer[12].mutable_cpu_data();
    void* cpu_buffer14_ = cpu_buffer[13].mutable_cpu_data();
    void* cpu_buffer15_ = cpu_buffer[14].mutable_cpu_data();
    void* cpu_buffer16_ = cpu_buffer[15].mutable_cpu_data();

    int lg_step0 = 21;
    int lg_step1 = 20;
    int lg_step2 = 21;
    int lg_step3 = 21;

    uint64_t chunk_size = global_buffer[0].size();

    Ntt_coset NTT(32, lg_LDE);

    cudaEvent_t step0, step1, step2, step3;
    cudaEventCreate(&step0);
    cudaEventCreate(&step1);
    cudaEventCreate(&step2);
    cudaEventCreate(&step3);

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    // step0
    // bit_rev_step1_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(step1_in_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(input_), step1_map, lg_step0, 0);
    // cudaDeviceSynchronize();
    // caffe_gpu_memcpy_async(chunk_size, step1_in_, buffer1_, stream1);
    // cudaEventRecord(step0, stream1);
    // bit_rev_step1_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(step1_in_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(input_ + chunk_size), step1_map, lg_step0, 1<<lg_step0);
    // caffe_gpu_memcpy_async(chunk_size, step1_in_ + chunk_size, buffer2_, stream1);
    // cudaStreamWaitEvent(stream2, step0);
    // NTT.init(global_buffer[0], lg_LDE, 0, stream2);
    // cudaEventRecord(step0, stream2);
    // cudaStreamWaitEvent(stream1, step0);
    // caffe_gpu_memcpy_async(chunk_size, buffer1_, step1_in_, stream1);
    // NTT.init(global_buffer[1], lg_LDE, 1, stream2);
    // cudaStreamSynchronize(stream1);

    caffe_gpu_memcpy_async(chunk_size, input_, buffer1_, stream1);
    cudaEventRecord(step0, stream1);
    caffe_gpu_memcpy_async(chunk_size, input_ + chunk_size, buffer2_, stream1);
    cudaStreamWaitEvent(stream2, step0);
    NTT.init_with_bitrev(global_buffer[0], lg_step0, stream2);
    cudaEventRecord(step0, stream1);
    NTT.init_with_bitrev(global_buffer[1], lg_step0, stream2);
    cudaStreamWaitEvent(stream2, step0);
    caffe_gpu_memcpy_async(chunk_size, buffer1_, step1_in_, stream1);
    cudaStreamSynchronize(stream1);
    caffe_gpu_memcpy_async(chunk_size, buffer2_, step1_in_ + chunk_size, stream1);
    pad_and_transpose(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(step1_in_), lg_step1, 0);
    cudaDeviceSynchronize();
    caffe_gpu_memcpy_async(chunk_size, map_ , buffer1_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + chunk_size, buffer2_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 2 * chunk_size, buffer3_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 3 * chunk_size, buffer4_, stream1);
    pad_and_transpose(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(step1_in_), lg_step1, 1<<lg_step1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 4 * chunk_size, buffer5_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 5 * chunk_size, buffer6_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 6 * chunk_size, buffer7_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 7 * chunk_size, buffer8_, stream1);
    cudaEventRecord(step1, stream1);
    pad_and_transpose(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(step1_in_), lg_step1, 2 * 1<<lg_step1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 8 * chunk_size, buffer9_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 9 * chunk_size, buffer10_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 10 * chunk_size, buffer11_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 11 * chunk_size, buffer12_, stream1);
    cudaStreamWaitEvent(stream2, step1);
    NTT.forward1(global_buffer[0], lg_LDE, 0, 0, stream2);
    NTT.forward1(global_buffer[1], lg_LDE, 0, 1, stream2);
    NTT.forward1(global_buffer[2], lg_LDE, 0, 2, stream2);
    NTT.forward1(global_buffer[3], lg_LDE, 0, 3, stream2);
    NTT.forward1(global_buffer[4], lg_LDE, 0, 4, stream2);
    NTT.forward1(global_buffer[5], lg_LDE, 0, 5, stream2);
    NTT.forward1(global_buffer[6], lg_LDE, 0, 6, stream2);
    NTT.forward1(global_buffer[7], lg_LDE, 0, 7, stream2);
    pad_and_transpose(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(step1_in_), lg_step1, 3 * 1<<lg_step1);
    cudaStreamSynchronize(stream1);
    NTT.forward1(global_buffer[8], lg_LDE, 0, 8, stream2);
    NTT.forward1(global_buffer[9], lg_LDE, 0, 9, stream2);
    NTT.forward1(global_buffer[10], lg_LDE, 0, 10, stream2);
    NTT.forward1(global_buffer[11], lg_LDE, 0, 11, stream2);
    caffe_gpu_memcpy_async(chunk_size, map_ + 12 * chunk_size, buffer13_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 13 * chunk_size, buffer14_, stream1);
    cudaStreamSynchronize(stream1);
    NTT.forward1(global_buffer[12], lg_LDE, 0, 12, stream2);
    NTT.forward1(global_buffer[13], lg_LDE, 0, 13, stream2);
    caffe_gpu_memcpy_async(chunk_size, map_ + 14 * chunk_size, buffer15_, stream1);
    cudaStreamSynchronize(stream1);
    NTT.forward1(global_buffer[14], lg_LDE, 0, 14, stream2);
    caffe_gpu_memcpy_async(chunk_size, map_ + 15 * chunk_size, buffer16_, stream1);
    NTT.forward1(global_buffer[15], lg_LDE, 0, 15, stream1);

    caffe_gpu_memcpy_async(chunk_size, buffer1_, cpu_buffer1_, stream2);
    cudaStreamSynchronize(stream2);
    caffe_gpu_memcpy_async(chunk_size, buffer2_, cpu_buffer2_, stream1);
    caffe_gpu_memcpy_async(chunk_size, buffer3_, cpu_buffer3_, stream1);
    caffe_gpu_memcpy_async(chunk_size, buffer4_, cpu_buffer4_, stream1);
    bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer1_), step2_map, lg_step2, 0);
    caffe_gpu_memcpy_async(chunk_size, buffer5_, cpu_buffer5_, stream1);
    caffe_gpu_memcpy_async(chunk_size, buffer6_, cpu_buffer6_, stream1);
    
    bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer2_), step2_map, lg_step2, 1<<lg_step2);
    caffe_gpu_memcpy_async(chunk_size, buffer7_, cpu_buffer7_, stream1);
    caffe_gpu_memcpy_async(chunk_size, buffer8_, cpu_buffer8_, stream1);
    
    bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer3_), step2_map, lg_step2, 2 * 1<<lg_step2);
    caffe_gpu_memcpy_async(chunk_size, buffer9_, cpu_buffer9_, stream1);
    caffe_gpu_memcpy_async(chunk_size, buffer10_, cpu_buffer10_, stream1);
    
    bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer4_), step2_map, lg_step2, 3 * 1<<lg_step2);
    caffe_gpu_memcpy_async(chunk_size, buffer11_, cpu_buffer11_, stream1);
    caffe_gpu_memcpy_async(chunk_size, buffer12_, cpu_buffer12_, stream1);
   
    bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer5_), step2_map, lg_step2, 4 * 1<<lg_step2);
    caffe_gpu_memcpy_async(chunk_size, buffer13_, cpu_buffer13_, stream1);
    caffe_gpu_memcpy_async(chunk_size, buffer14_, cpu_buffer14_, stream1);
    
    bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer6_), step2_map, lg_step2, 5 * 1<<lg_step2);
    caffe_gpu_memcpy_async(chunk_size, buffer15_, cpu_buffer15_, stream1);
    caffe_gpu_memcpy_async(chunk_size, buffer16_, cpu_buffer16_, stream1);
    
    bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer7_), step2_map, lg_step2, 6 * 1<<lg_step2);
    caffe_gpu_memcpy_async(chunk_size, map_, buffer1_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + chunk_size, buffer2_, stream1);
    

    bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer8_), step2_map, lg_step2, 7 * 1<<lg_step2);
    caffe_gpu_memcpy_async(chunk_size, map_ + 2 * chunk_size, buffer3_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 3 * chunk_size, buffer4_, stream1);
    
    bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer9_), step2_map, lg_step2, 8 * 1<<lg_step2);
    caffe_gpu_memcpy_async(chunk_size, map_ + 4 * chunk_size, buffer5_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 5 * chunk_size, buffer6_, stream1);
    
    bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer10_), step2_map, lg_step2, 9 * 1<<lg_step2);
    caffe_gpu_memcpy_async(chunk_size, map_ + 6 * chunk_size, buffer7_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 7 * chunk_size, buffer8_, stream1);
    cudaEventRecord(step2, stream1);
    
    bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer11_), step2_map, lg_step2, 10 * 1<<lg_step2);
    caffe_gpu_memcpy_async(chunk_size, map_ + 8 * chunk_size, buffer9_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 9 * chunk_size, buffer10_, stream1);
    cudaEventRecord(step2, stream1);
    bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer12_), step2_map, lg_step2, 11 * 1<<lg_step2);
    cudaStreamWaitEvent(stream2, step2);
    NTT.forward1(global_buffer[0], lg_LDE, 8, 0, stream2);
    NTT.forward1(global_buffer[1], lg_LDE, 8, 1, stream2);
    NTT.forward1(global_buffer[2], lg_LDE, 8, 2, stream2);
    NTT.forward1(global_buffer[3], lg_LDE, 8, 3, stream2);
    NTT.forward1(global_buffer[4], lg_LDE, 8, 4, stream2);
    NTT.forward1(global_buffer[5], lg_LDE, 8, 5, stream2);
    NTT.forward1(global_buffer[6], lg_LDE, 8, 6, stream2);
    NTT.forward1(global_buffer[7], lg_LDE, 8, 7, stream2);
    
    caffe_gpu_memcpy_async(chunk_size, map_ + 10 * chunk_size, buffer11_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 11 * chunk_size, buffer12_, stream1);
    
    bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer13_), step2_map, lg_step2, 12 * 1<<lg_step2);
    cudaStreamWaitEvent(stream2, step2);
    NTT.forward1(global_buffer[8], lg_LDE, 8, 8, stream2);
    NTT.forward1(global_buffer[9], lg_LDE, 8, 9, stream2);
    
    caffe_gpu_memcpy_async(chunk_size, map_ + 12 * chunk_size, buffer13_, stream1);
    cudaEventRecord(step2, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 13 * chunk_size, buffer14_, stream1);
    
    bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer14_), step2_map, lg_step2, 13 * 1<<lg_step2);
    cudaStreamWaitEvent(stream2, step2);
    NTT.forward1(global_buffer[10], lg_LDE, 8, 10, stream2);
    NTT.forward1(global_buffer[11], lg_LDE, 8, 11, stream2);
    NTT.forward1(global_buffer[12], lg_LDE, 8, 12, stream2);
    
    
    caffe_gpu_memcpy_async(chunk_size, map_ + 14 * chunk_size, buffer15_, stream1);
    cudaEventRecord(step2, stream1);
    bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer15_), step2_map, lg_step2, 14 * 1<<lg_step2);
    bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer16_), step2_map, lg_step2, 15 * 1<<lg_step2);
    cudaStreamWaitEvent(stream2, step2);
    NTT.forward1(global_buffer[13], lg_LDE, 8, 13, stream2);
    NTT.forward1(global_buffer[14], lg_LDE, 8, 14, stream2);
    caffe_gpu_memcpy_async(chunk_size, map_ + 15 * chunk_size, buffer16_, stream1);
    NTT.forward1(global_buffer[15], lg_LDE, 8, 15, stream1);
    

    caffe_gpu_memcpy_async(chunk_size, buffer1_, cpu_buffer1_, stream2);
    cudaStreamSynchronize(stream2);
    caffe_gpu_memcpy_async(chunk_size, buffer2_, cpu_buffer2_, stream1);
    caffe_gpu_memcpy_async(chunk_size, buffer3_, cpu_buffer3_, stream1);
    bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer1_), step3_map, lg_step3, 0);
    
    caffe_gpu_memcpy_async(chunk_size, buffer4_, cpu_buffer4_, stream1);
    caffe_gpu_memcpy_async(chunk_size, buffer5_, cpu_buffer5_, stream1);
    bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer2_), step3_map, lg_step3, 1<<lg_step3);
    
    caffe_gpu_memcpy_async(chunk_size, buffer6_, cpu_buffer6_, stream1);
    caffe_gpu_memcpy_async(chunk_size, buffer7_, cpu_buffer7_, stream1);
    bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer3_), step3_map, lg_step3, 2 * 1<<lg_step3);
    
    caffe_gpu_memcpy_async(chunk_size, buffer8_, cpu_buffer8_, stream1);
    caffe_gpu_memcpy_async(chunk_size, buffer9_, cpu_buffer9_, stream1);
    bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer4_), step3_map, lg_step3, 3 * 1<<lg_step3);
    
    caffe_gpu_memcpy_async(chunk_size, buffer10_, cpu_buffer10_, stream1);
    caffe_gpu_memcpy_async(chunk_size, buffer11_, cpu_buffer11_, stream1);
    bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer5_), step3_map, lg_step3, 4 * 1<<lg_step3);
    
    caffe_gpu_memcpy_async(chunk_size, buffer12_, cpu_buffer12_, stream1);
    caffe_gpu_memcpy_async(chunk_size, buffer13_, cpu_buffer13_, stream1);
    bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer6_), step3_map, lg_step3, 5 * 1<<lg_step3);
    
    caffe_gpu_memcpy_async(chunk_size, buffer14_, cpu_buffer14_, stream1);
    caffe_gpu_memcpy_async(chunk_size, buffer15_, cpu_buffer15_, stream1);
    bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer7_), step3_map, lg_step3, 6 * 1<<lg_step3);
    
    caffe_gpu_memcpy_async(chunk_size, buffer16_, cpu_buffer16_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_, buffer1_, stream1);

    bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer8_), step3_map, lg_step3, 7 * 1<<lg_step3);
    caffe_gpu_memcpy_async(chunk_size, map_ + chunk_size, buffer2_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 2 * chunk_size, buffer3_, stream1);

    bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer9_), step3_map, lg_step3, 8 * 1<<lg_step3);
    caffe_gpu_memcpy_async(chunk_size, map_ + 3 * chunk_size, buffer4_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 4 * chunk_size, buffer5_, stream1);

    bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer10_), step3_map, lg_step3, 9 * 1<<lg_step3);
    caffe_gpu_memcpy_async(chunk_size, map_ + 5 * chunk_size, buffer6_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 6 * chunk_size, buffer7_, stream1);

    bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer11_), step3_map, lg_step3, 10 * 1<<lg_step3);
    caffe_gpu_memcpy_async(chunk_size, map_ + 7 * chunk_size, buffer8_, stream1);
    cudaEventRecord(step3, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 8 * chunk_size, buffer9_, stream1);
    
    bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer12_), step3_map, lg_step3, 11 * 1<<lg_step3);
    cudaStreamWaitEvent(stream2, step3);
    NTT.forward1(global_buffer[0], lg_LDE, 8, 0, stream2);
    NTT.forward1(global_buffer[1], lg_LDE, 8, 1, stream2);
    NTT.forward1(global_buffer[2], lg_LDE, 8, 2, stream2);
    NTT.forward1(global_buffer[3], lg_LDE, 8, 3, stream2);
    NTT.forward1(global_buffer[4], lg_LDE, 8, 4, stream2);
    NTT.forward1(global_buffer[5], lg_LDE, 8, 5, stream2);
    NTT.forward1(global_buffer[6], lg_LDE, 8, 6, stream2);
    NTT.forward1(global_buffer[7], lg_LDE, 8, 7, stream2);
    caffe_gpu_memcpy_async(chunk_size, map_ + 9 * chunk_size, buffer10_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 10 * chunk_size, buffer11_, stream1);
    cudaEventRecord(step3, stream1);
    bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer13_), step3_map, lg_step3, 12 * 1<<lg_step3);
    cudaStreamWaitEvent(stream2, step3);
    NTT.forward1(global_buffer[8], lg_LDE, 8, 8, stream2);
    NTT.forward1(global_buffer[9], lg_LDE, 8, 9, stream2);
    caffe_gpu_memcpy_async(chunk_size, map_ + 11 * chunk_size, buffer12_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 12 * chunk_size, buffer13_, stream1);
    cudaEventRecord(step3, stream1);
    bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer14_), step3_map, lg_step3, 13 * 1<<lg_step3);
    cudaStreamWaitEvent(stream2, step3);
    NTT.forward1(global_buffer[10], lg_LDE, 8, 10, stream2);
    NTT.forward1(global_buffer[11], lg_LDE, 8, 11, stream2);
    NTT.forward1(global_buffer[12], lg_LDE, 8, 12, stream2);
    caffe_gpu_memcpy_async(chunk_size, map_ + 13 * chunk_size, buffer14_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 14 * chunk_size, buffer15_, stream1);
    bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer15_), step3_map, lg_step3, 14 * 1<<lg_step3);
    bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer16_), step3_map, lg_step3, 15 * 1<<lg_step3);
    cudaStreamSynchronize(stream1);
    NTT.forward1(global_buffer[13], lg_LDE, 8, 13, stream2);
    NTT.forward1(global_buffer[14], lg_LDE, 8, 14, stream2);
    caffe_gpu_memcpy_async(chunk_size, map_ + 15 * chunk_size, buffer16_, stream1);
    NTT.forward1(global_buffer[15], lg_LDE, 8, 15, stream1);
    
    for(int i = 0;i<16;i++){
        void* d_ = global_buffer[i].mutable_gpu_data();
        caffe_gpu_memcpy_async(chunk_size, d_, output_ + i * chunk_size, stream2);
    }

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(step0);
    cudaEventDestroy(step1);
    cudaEventDestroy(step2);
    cudaEventDestroy(step3);

    return 0;
}
