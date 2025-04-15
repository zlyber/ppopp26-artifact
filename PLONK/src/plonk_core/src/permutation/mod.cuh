#include <iostream>
#include <functional>
#include "../../../structure.cuh"
#include "constants.cuh"
#include "../../../domain.cuh"

std::vector<SyncedMemory> compute_permutation_poly(Radix2EvaluationDomain domain, std::vector<SyncedMemory> global_buffer, 
    std::vector<SyncedMemory> w_l, std::vector<SyncedMemory> w_r, std::vector<SyncedMemory> w_o, std::vector<SyncedMemory> w_4, 
    SyncedMemory beta, SyncedMemory gamma, Permutation sigma_polys, int chunk_num, uint32_t* step1_map, uint32_t* step2_map, uint32_t* step3_map, 
    SyncedMemory ntt_cpu_map, std::vector<SyncedMemory>ntt_cpu_buffer, std::vector<SyncedMemory> h_2_poly_cpu, 
    std::vector<SyncedMemory> h_2_poly_chunks, SyncedMemory w_l_poly, Intt INTT,
    cudaStream_t stream1 = (cudaStream_t)0, cudaStream_t stream2 = (cudaStream_t)0);

std::vector<SyncedMemory> compute_lookup_permutation_poly(uint64_t n, std::vector<SyncedMemory> global_buffer,
    std::vector<SyncedMemory> f, std::vector<SyncedMemory> t, std::vector<SyncedMemory> h_1, std::vector<SyncedMemory> h_2,
    SyncedMemory delta, SyncedMemory epsilon, uint32_t* step1_map, uint32_t* step2_map, uint32_t* step3_map,
    SyncedMemory ntt_cpu_map, std::vector<SyncedMemory>ntt_cpu_buffer, std::vector<SyncedMemory>z_chunks, std::vector<SyncedMemory>z_gpu, 
    int chunk_num, Intt INTT, 
    cudaStream_t stream1 = (cudaStream_t)0, cudaStream_t stream2 = (cudaStream_t)0);