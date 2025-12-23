#include <iostream>
#include <cstdint>
#include <chrono>
#include <thread>
#include "caffe/interface.hpp"
#include "PLONK/utils/function.cuh"

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

SyncedMemory one() {
    SyncedMemory one(4 * sizeof(uint64_t));
    void* one_ = one.mutable_cpu_data();
    uint64_t fr_one[4] = {8589934590UL, 6378425256633387010UL, 11064306276430008309UL, 1739710354780652911UL};
    memcpy(one_,fr_one,one.size());
    return one;
}

SyncedMemory naive_gp(int lg_N, SyncedMemory input){
    uint64_t N = 1<<lg_N;
    SyncedMemory output(input.size());
    cpu::BLS12_381_Fr_G1* input_ = reinterpret_cast<cpu::BLS12_381_Fr_G1*>(input.mutable_cpu_data());
    cpu::BLS12_381_Fr_G1* output_ = reinterpret_cast<cpu::BLS12_381_Fr_G1*>(output.mutable_cpu_data());
    SyncedMemory state = one();
    memcpy((void*)output_, state.mutable_cpu_data(), state.size());
    cpu::BLS12_381_Fr_G1* state_ = reinterpret_cast<cpu::BLS12_381_Fr_G1*>(state.mutable_cpu_data());
    for (size_t i = 0; i < N - 1; ++i) {
        state_[0] *= input_[i];
        memcpy(output_ + i, state.mutable_cpu_data(), state.size());
    }
    return output;
}

SyncedMemory naive_div(int lg_N, SyncedMemory input, SyncedMemory divisor){
    size_t N = 1<<lg_N;

    uint64_t zero[4] = {0,0,0,0};
    
    size_t q_len = N - 1;
    size_t remainder_len = N;
    SyncedMemory quotient(q_len * 4 * sizeof(uint64_t));
    SyncedMemory remainder(input.size());
    cpu::BLS12_381_Fr_G1* quotient_ = reinterpret_cast<cpu::BLS12_381_Fr_G1*>(quotient.mutable_cpu_data());
    memcpy(remainder.mutable_cpu_data(), input.mutable_cpu_data(), input.size());

    for (size_t i = 0; i < q_len; ++i) {
        memcpy(quotient_ + i, zero, 4*sizeof(uint64_t));
    }

    cpu::BLS12_381_Fr_G1* divisor_ = reinterpret_cast<cpu::BLS12_381_Fr_G1*>(divisor.mutable_cpu_data());
    cpu::BLS12_381_Fr_G1 divisor_leading = divisor_[1];
    cpu::BLS12_381_Fr_G1 divisor_leading_inv = 1/divisor_leading;

    cpu::BLS12_381_Fr_G1* remainder_ = reinterpret_cast<cpu::BLS12_381_Fr_G1*>(remainder.mutable_cpu_data());

    //divisor_len = 2
    while (remainder_len >= 2){
        cpu::BLS12_381_Fr_G1 remainder_leading = remainder_[remainder_len - 1];
        cpu::BLS12_381_Fr_G1 cur_q_coeff = remainder_leading * divisor_leading_inv;
        size_t cur_q_degree = remainder_len - 2;
        quotient_[cur_q_degree] = cur_q_coeff;
        for (size_t i = 0; i < 2; ++i){
            cpu::BLS12_381_Fr_G1 div_coeff = divisor_[i];
            cpu::BLS12_381_Fr_G1 temp = cur_q_coeff * div_coeff;
            remainder_[cur_q_degree + i] -= temp;
        }
        while (remainder_[remainder_len - 1].is_zero()){
            remainder_len --;
        }
    }
    return quotient;
}

int main() {
    int lg_N = 22;

    uint64_t N = 1<<lg_N;

    SyncedMemory input(N*32);
    void* inp_ = input.mutable_cpu_data();
    const char* file_path = "../../input.bin";

    // bench grand product
    std::cout << "test grand product" << std::endl;
    for (int rep = 0; rep < 3; ++rep) {
        read_file(file_path, inp_);
        std::cout << "[naive] run " << (rep + 1) << "/3" << std::endl;

        auto start = std::chrono::high_resolution_clock::now();
        SyncedMemory out = naive_gp(lg_N, input);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "[naive_gp] execution time: " << duration_ms << " ms" << std::endl;
    }

    for (int rep = 0; rep < 3; ++rep) {
        read_file(file_path, inp_);
        void* d_inp_ = input.mutable_gpu_data();
        std::cout << "[ours] run " << (rep + 1) << "/3" << std::endl;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);
        SyncedMemory out = accumulate_mul_poly(input);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        std::cout << "[ours_gp] execution time: " << milliseconds << " ms" << std::endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // bench poly evaluation
    SyncedMemory point(4*sizeof(uint64_t)); //random choose
    uint64_t pointt[4] = {123456789UL, 987654321UL, 1122334455UL, 6677889900UL};
    memcpy(point.mutable_cpu_data(), pointt, point.size());

    std::cout << "test poly evaluation" << std::endl;

    for (int rep = 0; rep < 3; ++rep) {
        read_file(file_path, inp_);
        void* d_inp_ = input.mutable_gpu_data();
        void* d_point_ = point.mutable_gpu_data();
        std::cout << "[ours] run " << (rep + 1) << "/3" << std::endl;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);
        SyncedMemory out = evaluate(input, point);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        std::cout << "[ours_eval] execution time: " << milliseconds << " ms" << std::endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // bench poly division
    std::cout << "test poly division" << std::endl;

    SyncedMemory divisor(8*sizeof(uint64_t)); //x-z, z can be random
    uint64_t divisorr[8] = {123456789123456789UL, 987654321987654321UL, 1122334455667788UL, 1122334455667788UL, 8589934590UL, 6378425256633387010UL, 11064306276430008309UL, 1739710354780652911UL};
    memcpy(divisor.mutable_cpu_data(), divisorr, divisor.size());

    for (int rep = 0; rep < 3; ++rep) {
        inp_ = input.mutable_cpu_data();
        read_file(file_path, inp_);
        std::cout << "[naive] run " << (rep + 1) << "/3" << std::endl;

        auto start = std::chrono::high_resolution_clock::now();
        SyncedMemory out = naive_div(lg_N, input, divisor);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "[naive_div] execution time: " << duration_ms << " ms" << std::endl;
    }

    SyncedMemory divisor_gpu(4*sizeof(uint64_t));
    uint64_t divisorr_gpu[4] = {123456789123456789UL, 987654321987654321UL, 1122334455667788UL, 1122334455667788UL};
    memcpy(divisor_gpu.mutable_cpu_data(), divisorr_gpu, divisor_gpu.size());
    void* d_divisor_ = divisor_gpu.mutable_gpu_data();

    for (int rep = 0; rep < 3; ++rep) {
        read_file(file_path, inp_);
        void* d_inp_ = input.mutable_gpu_data();
        std::cout << "[ours] run " << (rep + 1) << "/3" << std::endl;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);
        SyncedMemory out = poly_div_poly(input, divisor_gpu);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        std::cout << "[ours_div] execution time: " << milliseconds << " ms" << std::endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    return 0;
}