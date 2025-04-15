#pragma once
#include "../../caffe/interface.hpp"
#include "mont/cuda/curve_def.cuh"
#include "mont/cpu/mont_arithmetic.h"
#include "mont/cuda/mont_arithmetic.cuh"
#include "zkp/cpu/msmcollect.hpp"
#include "zkp/cuda/zksnark.cuh"
#include <iostream>
#include <fstream>
#include <thread>

SyncedMemory to_mont(SyncedMemory input);

SyncedMemory to_base(SyncedMemory input, cudaStream_t stream = (cudaStream_t)0);

SyncedMemory neg_mod(SyncedMemory input);

SyncedMemory inv_mod(SyncedMemory input);

SyncedMemory add_mod(SyncedMemory input1, SyncedMemory input2, cudaStream_t stream = (cudaStream_t)0);
void add_mod_(SyncedMemory input1, SyncedMemory input2, cudaStream_t stream = (cudaStream_t)0);

SyncedMemory sub_mod(SyncedMemory input1, SyncedMemory input2, cudaStream_t stream = (cudaStream_t)0);
void sub_mod_(SyncedMemory input1, SyncedMemory input2, cudaStream_t stream = (cudaStream_t)0);

SyncedMemory mul_mod(SyncedMemory input1, SyncedMemory input2, cudaStream_t stream = (cudaStream_t)0);
void mul_mod_(SyncedMemory input1, SyncedMemory input2, cudaStream_t stream = (cudaStream_t)0);

SyncedMemory div_mod(SyncedMemory input1, SyncedMemory input2, cudaStream_t stream = (cudaStream_t)0);
void div_mod_(SyncedMemory input1, SyncedMemory input2, cudaStream_t stream = (cudaStream_t)0);

SyncedMemory exp_mod(SyncedMemory input, int exp);

SyncedMemory add_mod_scalar(SyncedMemory input1, SyncedMemory input2, cudaStream_t stream = (cudaStream_t)0);
void add_mod_scalar_(SyncedMemory input1, SyncedMemory input2, cudaStream_t stream = (cudaStream_t)0);

SyncedMemory sub_mod_scalar(SyncedMemory input1, SyncedMemory input2, cudaStream_t stream = (cudaStream_t)0);
void sub_mod_scalar_(SyncedMemory input1, SyncedMemory input2, cudaStream_t stream = (cudaStream_t)0);

SyncedMemory mul_mod_scalar(SyncedMemory input1, SyncedMemory input2, cudaStream_t stream = (cudaStream_t)0);
void mul_mod_scalar_(SyncedMemory input1, SyncedMemory input2, cudaStream_t stream = (cudaStream_t)0);

SyncedMemory div_mod_scalar(SyncedMemory input1, SyncedMemory input2, cudaStream_t stream = (cudaStream_t)0);
void div_mod_scalar_(SyncedMemory input1, SyncedMemory input2, cudaStream_t stream = (cudaStream_t)0);

SyncedMemory gen_sequence(uint64_t N, SyncedMemory x, cudaStream_t stream = (cudaStream_t)0);
void gen_sequence_(uint64_t N, SyncedMemory x, SyncedMemory y, cudaStream_t stream = (cudaStream_t)0);

SyncedMemory repeat_to_poly(SyncedMemory x, uint64_t N, cudaStream_t stream = (cudaStream_t)0);
void repeat_to_poly_(SyncedMemory x, SyncedMemory y, uint64_t N, cudaStream_t stream = (cudaStream_t)0);

SyncedMemory evaluate(SyncedMemory poly, SyncedMemory x);

SyncedMemory poly_div_poly(SyncedMemory divid, SyncedMemory c);

SyncedMemory pad_poly(SyncedMemory x, uint64_t N, cudaStream_t stream = (cudaStream_t)0);
void pad_poly_(SyncedMemory x, SyncedMemory out, uint64_t N, cudaStream_t stream = (cudaStream_t)0);

SyncedMemory repeat_zero(uint64_t N);

SyncedMemory cat(SyncedMemory a, SyncedMemory b);

SyncedMemory slice(SyncedMemory a, uint64_t len, bool forward);

std::vector<SyncedMemory> split_tx_poly(uint64_t n, SyncedMemory t_poly);

SyncedMemory accumulate_mul_poly(SyncedMemory product, cudaStream_t stream = (cudaStream_t)0);

void accumulate_mul_poly_(SyncedMemory product, SyncedMemory output, cudaStream_t stream = (cudaStream_t)0);

SyncedMemory make_tensor(SyncedMemory input, uint64_t pad_len);

void lookup_ratio_step1_(SyncedMemory h_1, SyncedMemory h_2, SyncedMemory h_1_next, 
    SyncedMemory one, SyncedMemory delta, SyncedMemory epsilon, SyncedMemory result, cudaStream_t stream = (cudaStream_t)0);

void lookup_ratio_step2_(SyncedMemory f, SyncedMemory t, SyncedMemory t_next, 
    SyncedMemory one, SyncedMemory delta, SyncedMemory epsilon, SyncedMemory result, cudaStream_t stream = (cudaStream_t)0);

void calculate_indices(uint32_t tid, int stage1, int iterations1, int stage2, int iterations2, uint32_t* map);

class Ntt{
public:
    SyncedMemory Params;

    Ntt(int domain_size, cudaStream_t stream = (cudaStream_t)0);

    SyncedMemory forward(SyncedMemory input, cudaStream_t stream = (cudaStream_t)0);
    void forward_(SyncedMemory input, SyncedMemory output, cudaStream_t stream = (cudaStream_t)0);
    void forward1(SyncedMemory input, int lg_domain_size, int stage, int chunk_id, cudaStream_t stream = (cudaStream_t)0);
    void forward2(SyncedMemory input, int lg_domain_size, int stage, int chunk_id, cudaStream_t stream = (cudaStream_t)0);
    void forward3(SyncedMemory input, int lg_domain_size, int stage, int chunk_id, cudaStream_t stream = (cudaStream_t)0);
    void forward1_kcolumn(SyncedMemory input, int lg_domain_size, int i_size, int k, int* stage, cudaStream_t stream = (cudaStream_t)0);
    void forward1_kcolumn_raw(uint64_t* input, int lg_domain_size, int64_t numel, int i_size, int k, int* stage, cudaStream_t stream = (cudaStream_t)0);
    void forward2_kcolumn(SyncedMemory input, int lg_domain_size, int i_size, int k, int* stage, cudaStream_t stream = (cudaStream_t)0);

    void forward1_internal(SyncedMemory input, SyncedMemory output, int* stage, cudaStream_t stream = (cudaStream_t)0);
    void forward2_internal(SyncedMemory inout, int* stage, cudaStream_t stream = (cudaStream_t)0);
    void forward3_internal(SyncedMemory inout, int* stage, cudaStream_t stream = (cudaStream_t)0);
};

class Intt {
public:
    SyncedMemory Params;

    Intt(int domain_size, cudaStream_t stream = (cudaStream_t)0);

    SyncedMemory forward(SyncedMemory input, cudaStream_t stream = (cudaStream_t)0);
    void forward_(SyncedMemory input, SyncedMemory output, cudaStream_t stream = (cudaStream_t)0);
    void _forward_(SyncedMemory input, cudaStream_t stream = (cudaStream_t)0);
    void forward1(SyncedMemory input, int lg_domain_size, int stage, int chunk_id, cudaStream_t stream = (cudaStream_t)0);
    void forward2(SyncedMemory input, int lg_domain_size, int stage, int chunk_id, cudaStream_t stream = (cudaStream_t)0);
    void forward3(SyncedMemory input, int lg_domain_size, int stage, int chunk_id, cudaStream_t stream = (cudaStream_t)0);
};

class Ntt_coset {
public:
    SyncedMemory Params;
    int lg_domain_size;

    Ntt_coset(int domain_size, int coset_size);

    void init(SyncedMemory input, int lg_domain_size, int chunk_id, bool is_intt = false, cudaStream_t stream = (cudaStream_t)0);
    void init_with_bitrev(SyncedMemory input, int lg_domain_size, bool is_intt = false, cudaStream_t stream = (cudaStream_t)0);
    void forward1(SyncedMemory input, int lg_domain_size, int stage, int chunk_id, cudaStream_t stream = (cudaStream_t)0);
    void forward2(SyncedMemory input, int lg_domain_size, int stage, int chunk_id, cudaStream_t stream = (cudaStream_t)0);
    void forward3(SyncedMemory input, int lg_domain_size, int stage, int chunk_id, cudaStream_t stream = (cudaStream_t)0);
};

class Intt_coset {
public:
    SyncedMemory Params;
    int lg_domain_size;

    Intt_coset(int domain_size, int coset_size, cudaStream_t stream = (cudaStream_t)0);

    void init(SyncedMemory input, int lg_domain_size, int chunk_id, bool is_intt = true, cudaStream_t stream = (cudaStream_t)0);
    void forward1(SyncedMemory input, int lg_domain_size, int stage, int chunk_id, cudaStream_t stream = (cudaStream_t)0);
    void forward2(SyncedMemory input, int lg_domain_size, int stage, int chunk_id, cudaStream_t stream = (cudaStream_t)0);
    void forward3(SyncedMemory input, int lg_domain_size, int stage, int chunk_id, cudaStream_t stream = (cudaStream_t)0);

};

SyncedMemory multi_scalar_mult(SyncedMemory points, SyncedMemory scalars, cudaStream_t stream = (cudaStream_t)0);

bool gt_zkp(SyncedMemory a, SyncedMemory b);

void compress(SyncedMemory output, SyncedMemory t_0, SyncedMemory t_1, SyncedMemory t_2, SyncedMemory t_3, 
                       SyncedMemory challenge, cudaStream_t = (cudaStream_t)0);

void compute_query_table(std::vector<SyncedMemory> f_chunks, std::vector<SyncedMemory> global_buffer, std::vector<SyncedMemory> q_lookup, 
            std::vector<SyncedMemory> w_l_scalar, std::vector<SyncedMemory> w_r_scalar, std::vector<SyncedMemory> w_o_scalar,
            std::vector<SyncedMemory> w_4_scalar, std::vector<SyncedMemory> t_poly, SyncedMemory challenge, int chunk_num, 
            cudaStream_t stream1 = (cudaStream_t)0, cudaStream_t stream2 = (cudaStream_t)0);

void writeToFile(const std::string& filename, uint64_t* array, uint64_t size);

uint32_t bit_rev(uint32_t i, uint32_t nbits);

void calculate_map1(uint32_t* map, int lg_N);

template <typename fr>
void bit_rev_step1_worker(
    fr* d_out,
    fr* d_in,
    uint32_t* map,
    uint64_t idx,
    uint64_t start_idx,
    uint64_t end_idx) {
    
    for (uint64_t i = start_idx; i < end_idx; ++i) {
        d_out[map[idx + i]] = d_in[i];
    }
}

template <typename fr>
void pad_transpose_worker(
    fr* d_out,
    const fr* d_in,
    uint64_t start_idx,
    uint64_t end_idx) {
    
    for (uint64_t i = start_idx; i < end_idx; ++i) {
        d_out[i << 3] = d_in[i];
    }
}

template <typename fr>
void bit_rev_step2_worker(
    fr* d_out,
    fr* d_in,
    uint32_t* map,
    uint64_t idx,
    uint64_t start_idx,
    uint64_t end_idx) {
    
    for (uint64_t i = start_idx; i < end_idx; ++i) {
        d_out[map[idx+i]] = d_in[i];
    }
}

template <typename fr>
void bit_rev_step3_worker(
    fr* d_out,
    fr* d_in,
    uint32_t* map,
    uint64_t idx,
    uint64_t start_idx,
    uint64_t end_idx) {

    for (uint64_t i = start_idx; i < end_idx; ++i) {
        d_out[map[idx + i]] = d_in[i];
    }
}

template <typename fr>
void pad_and_transpose(
    fr* d_out,
    const fr* d_in,
    int lg_chunk,
    uint64_t idx) {

    uint64_t N = 1 << lg_chunk;
    uint32_t num_threads = std::thread::hardware_concurrency();  // 获取硬件并发线程数
    // num_threads = 1 << (static_cast<int>(std::log2(num_threads)));
    num_threads = 32;
    uint32_t chunk_size = N / num_threads;

    std::vector<std::thread> threads;

    for (uint32_t t = 0; t < num_threads; ++t) {
        uint64_t start_idx =  idx + t * chunk_size;
        uint64_t end_idx = (t == num_threads - 1) ? N + idx : start_idx + chunk_size;
        threads.push_back(std::thread(pad_transpose_worker<fr>, d_out, d_in, start_idx, end_idx));
    }

    for (auto& t : threads) {
        t.join();
    }
}

// Multithreaded version of bit_rev_permutation
template <typename fr>
void bit_rev_step1_parallel(
    fr* d_out,
    fr* d_in,
    uint32_t* map,
    int lg_N,
    uint64_t idx) {

    uint64_t N = 1 << lg_N;
    uint32_t num_threads = std::thread::hardware_concurrency();  // 获取硬件并发线程数
    // num_threads = 1 << (static_cast<int>(std::log2(num_threads)));
    // num_threads = 8;
    num_threads = 32;
    uint32_t chunk_size = N / num_threads;

    std::vector<std::thread> threads;

    // 启动多个线程来并行执行比特反转置换
    for (uint32_t t = 0; t < num_threads; ++t) {
        uint64_t start_idx = t * chunk_size;
        uint64_t end_idx = (t == num_threads - 1) ? N : start_idx + chunk_size;
        threads.push_back(std::thread(bit_rev_step1_worker<fr>, d_out, d_in, map, idx, start_idx, end_idx));
    }

    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }
}

template <typename fr>
void bit_rev_step2_parallel(
    fr* d_out,
    fr* d_in,
    uint32_t* map,
    int lg_N,
    uint64_t idx) {

    uint64_t N = 1 << lg_N;
    uint32_t num_threads = std::thread::hardware_concurrency();  // 获取硬件并发线程数
    num_threads = 2;
    uint32_t chunk_size = N / num_threads;

    std::vector<std::thread> threads;

    // 启动多个线程来并行执行比特反转置换
    for (uint32_t t = 0; t < num_threads; ++t) {
        uint64_t start_idx =  t * chunk_size;
        uint64_t end_idx = (t == num_threads - 1) ? N : start_idx + chunk_size;
        threads.push_back(std::thread(bit_rev_step2_worker<fr>, d_out, d_in, map, idx, start_idx, end_idx));
    }

    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }
}

template <typename fr>
void bit_rev_step2(
    fr* d_out,
    fr* d_in,
    uint32_t* map,
    int lg_N,
    uint64_t idx) {

    uint64_t N = 1 << lg_N;

    bit_rev_step2_worker(d_out, d_in, map, idx, 0, N-1);
   
}

template <typename fr>
void bit_rev_step3_parallel(
    fr* d_out,
    fr* d_in,
    uint32_t* map,
    int lg_N,
    uint64_t idx) {

    uint64_t N = 1 << lg_N;
    uint32_t num_threads = std::thread::hardware_concurrency();  // 获取硬件并发线程数
    // num_threads = 1 << (static_cast<int>(std::log2(num_threads)));
    num_threads = 32;
    uint32_t chunk_size = N / num_threads;

    std::vector<std::thread> threads;

    // 启动多个线程来并行执行比特反转置换
    for (uint32_t t = 0; t < num_threads; ++t) {
        uint64_t start_idx = t * chunk_size;
        uint64_t end_idx = (t == num_threads - 1) ? N : start_idx + chunk_size;
        threads.push_back(std::thread(bit_rev_step3_worker<fr>, d_out, d_in, map, idx, start_idx, end_idx));
    }

    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }
}

void COSET_INTT(Intt_coset INTT, 
    int lg_LDE,
    int chunk_num,
    std::vector<SyncedMemory> inout,
    std::vector<SyncedMemory> global_buffer,
    std::vector<SyncedMemory> cpu_buffer,
    uint32_t* step1_map,
    uint32_t* step2_map,
    uint32_t* step3_map,
    SyncedMemory cpu_map,
    cudaStream_t stream1 = (cudaStream_t)0,
    cudaStream_t stream2 = (cudaStream_t)0);

void LDE_forward(Ntt_coset NTT, 
    int lg_LDE,
    SyncedMemory input,
    SyncedMemory output,
    std::vector<SyncedMemory> global_buffer,
    std::vector<SyncedMemory> cpu_buffer,
    uint32_t* step1_map,
    uint32_t* step2_map,
    uint32_t* step3_map,
    SyncedMemory cpu_map,
    SyncedMemory step1_in,
    bool tail,
    cudaStream_t stream1 = (cudaStream_t)0,
    cudaStream_t stream2 = (cudaStream_t)0);

void LDE_forward_chunk(Ntt_coset NTT, 
    int lg_LDE,
    int chunk_num,
    std::vector<SyncedMemory> input,
    SyncedMemory output,
    std::vector<SyncedMemory> global_buffer,
    std::vector<SyncedMemory> cpu_buffer,
    uint32_t* step1_map,
    uint32_t* step2_map,
    uint32_t* step3_map,
    SyncedMemory cpu_map,
    SyncedMemory step1_in,
    bool tail,
    cudaStream_t stream1 = (cudaStream_t)0,
    cudaStream_t stream2 = (cudaStream_t)0);

void LDE_forward_with_commit(
    Ntt_coset NTT, 
    int lg_LDE,
    SyncedMemory input,
    std::vector<SyncedMemory> ck,
    SyncedMemory chunk_msm_workspace, SyncedMemory chunk_msm_out,
    std::vector<SyncedMemory> commits,
    std::vector<SyncedMemory> global_buffer,
    std::vector<SyncedMemory> cpu_buffer,
    uint32_t* step1_map,
    uint32_t* step2_map,
    uint32_t* step3_map,
    SyncedMemory cpu_map,
    SyncedMemory step1_in,
    bool tail,
    cudaStream_t stream1 = (cudaStream_t)0,
    cudaStream_t stream2 = (cudaStream_t)0);
