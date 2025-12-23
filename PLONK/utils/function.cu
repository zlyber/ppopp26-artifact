#include "function.cuh"
#include <cuda_runtime.h>
#include <fstream>

SyncedMemory to_mont(SyncedMemory input) {
   if(input.head() == SyncedMemory::HEAD_AT_CPU){
        return cpu::to_mont_cpu(input);
   }
   else{
        return cuda::to_mont_cuda(input);
   }
}

SyncedMemory to_base(SyncedMemory input, cudaStream_t stream) {
    if(input.head() == SyncedMemory::HEAD_AT_CPU){
        return cpu::to_base_cpu(input);
    }
    else{
        return cuda::to_base_cuda(input, stream);
    }
}

SyncedMemory neg_mod(SyncedMemory input) {
    if(input.head() == SyncedMemory::HEAD_AT_CPU){
        return cpu::neg_mod_cpu(input);
    }
    else{
        return cuda::neg_mod_cuda(input);
    }
}

SyncedMemory inv_mod(SyncedMemory input) {
    if(input.head() == SyncedMemory::HEAD_AT_CPU){
        return cpu::inv_mod_cpu(input);
    }
    else{
        return cuda::inv_mod_cuda(input);
    }
}

SyncedMemory add_mod(SyncedMemory input1, SyncedMemory input2, cudaStream_t stream) {
    if(input1.head() == SyncedMemory::HEAD_AT_CPU){
        return cpu::add_mod_cpu(input1, input2);
    }
    else{
        return cuda::add_mod_cuda(input1, input2, stream);
    }
}

void add_mod_(SyncedMemory input1, SyncedMemory input2, cudaStream_t stream) {
    if(input1.head() == SyncedMemory::HEAD_AT_CPU){
        cpu::add_mod_cpu_(input1, input2);
    }
    else{
        cuda::add_mod_cuda_(input1, input2, stream);
    }
}

SyncedMemory sub_mod(SyncedMemory input1, SyncedMemory input2, cudaStream_t stream) {
    if(input1.head() == SyncedMemory::HEAD_AT_CPU){
        return cpu::sub_mod_cpu(input1, input2);
    }
    else{
        return cuda::sub_mod_cuda(input1, input2, stream);
    }
}

void sub_mod_(SyncedMemory input1, SyncedMemory input2, cudaStream_t stream) {
    if(input1.head() == SyncedMemory::HEAD_AT_CPU){
        cpu::sub_mod_cpu_(input1, input2);
    }
    else{
        cuda::sub_mod_cuda_(input1, input2, stream);
    }
}

SyncedMemory mul_mod(SyncedMemory input1, SyncedMemory input2, cudaStream_t stream) {
    if(input1.head() == SyncedMemory::HEAD_AT_CPU){
        return cpu::mul_mod_cpu(input1, input2);
    }
    else{
        return cuda::mul_mod_cuda(input1, input2, stream);
    }
}

void mul_mod_(SyncedMemory input1, SyncedMemory input2, cudaStream_t stream) {
    if(input1.head() == SyncedMemory::HEAD_AT_CPU){
        cpu::mul_mod_cpu_(input1, input2);
    }
    else{
        cuda::mul_mod_cuda_(input1, input2, stream);
    }
}

SyncedMemory div_mod(SyncedMemory input1, SyncedMemory input2, cudaStream_t stream) {
    if(input1.head() == SyncedMemory::HEAD_AT_CPU){
        return cpu::div_mod_cpu(input1, input2);
    }
    else{
        return cuda::div_mod_cuda(input1, input2, stream);
    }
}

void div_mod_(SyncedMemory input1, SyncedMemory input2, cudaStream_t stream) {
    if(input1.head() == SyncedMemory::HEAD_AT_CPU){
        cpu::div_mod_cpu_(input1, input2);
    }
    else{
        cuda::div_mod_cuda_(input1, input2, stream);
    }
}

SyncedMemory exp_mod(SyncedMemory input, int exp) {
    if(input.head() == SyncedMemory::HEAD_AT_CPU){
        return cpu::exp_mod_cpu(input, exp);
    }
    else{
        return cuda::exp_mod_cuda(input, exp);
    }
}

SyncedMemory add_mod_scalar(SyncedMemory input1, SyncedMemory input2, cudaStream_t stream){
    return cuda::add_mod_scalar_cuda(input1, input2, stream);
}

void add_mod_scalar_(SyncedMemory input1, SyncedMemory input2, cudaStream_t stream){
    cuda::add_mod_scalar_cuda_(input1, input2, stream);
}

SyncedMemory sub_mod_scalar(SyncedMemory input1, SyncedMemory input2, cudaStream_t stream){
    return cuda::sub_mod_scalar_cuda(input1, input2, stream);
}

void sub_mod_scalar_(SyncedMemory input1, SyncedMemory input2, cudaStream_t stream){
    cuda::sub_mod_scalar_cuda_(input1, input2, stream);
}

SyncedMemory mul_mod_scalar(SyncedMemory input1, SyncedMemory input2, cudaStream_t stream){
    return cuda::mul_mod_scalar_cuda(input1, input2, stream);
}

void mul_mod_scalar_(SyncedMemory input1, SyncedMemory input2, cudaStream_t stream){
    cuda::mul_mod_scalar_cuda_(input1, input2, stream);
}

SyncedMemory div_mod_scalar(SyncedMemory input1, SyncedMemory input2, cudaStream_t stream){
    return cuda::div_mod_scalar_cuda(input1, input2, stream);
}

void div_mod_scalar_(SyncedMemory input1, SyncedMemory input2, cudaStream_t stream){
    cuda::div_mod_scalar_cuda_(input1, input2, stream);
}

SyncedMemory gen_sequence(uint64_t N, SyncedMemory x, cudaStream_t stream){
    return cuda::poly_eval_cuda(x, N, stream);
}

void gen_sequence_(uint64_t N, SyncedMemory x, SyncedMemory y, cudaStream_t stream){
    cuda::poly_eval_cuda_(x, y, N, stream);
}

SyncedMemory repeat_to_poly(SyncedMemory x, uint64_t N, cudaStream_t stream){
    return cuda::repeat_to_poly_cuda(x, N, stream);
}

void repeat_to_poly_(SyncedMemory x, SyncedMemory y, uint64_t N, cudaStream_t stream){
    cuda::repeat_to_poly_cuda_(x, y, N, stream);
}

SyncedMemory evaluate(SyncedMemory poly, SyncedMemory x){
    if (poly.size() == 0){
        SyncedMemory result(4 * sizeof(uint64_t));
        void* res_gpu = result.mutable_gpu_data();
        cudaMemset(res_gpu, 0, 4 * sizeof(uint64_t));
        return result;
    }
    else{
        SyncedMemory y = cuda::poly_eval_cuda(x, poly.size()/ (cuda::fr_LIMBS*sizeof(uint64_t)));
        return cuda::poly_reduce_cuda(y, poly);
    }
}

SyncedMemory poly_div_poly(SyncedMemory divid, SyncedMemory c){
    return cuda::poly_div_cuda(divid,c);
}

SyncedMemory pad_poly(SyncedMemory x, uint64_t N, cudaStream_t stream){
    if(x.head() == SyncedMemory::HEAD_AT_CPU){
        return cpu::pad_poly_cpu(x, N);
    }
    else{
        return cuda::pad_poly_cuda(x, N, stream);
    }
}

void pad_poly_(SyncedMemory x, SyncedMemory out, uint64_t N, cudaStream_t stream){
    if(x.head() == SyncedMemory::HEAD_AT_CPU)
        cpu::pad_poly_cpu_(x, out, N);
    else
        cuda::pad_poly_cuda_(x, out, N, stream);
}

SyncedMemory repeat_zero(uint64_t N){
    SyncedMemory out(N * cpu::fr_LIMBS * sizeof(uint64_t));
    void* out_ = out.mutable_cpu_data();
    memset(out_, 0, out.size());
    return out;
}

SyncedMemory cat(SyncedMemory a, SyncedMemory b){
    SyncedMemory res(a.size() + b.size());
    if(a.head() == SyncedMemory::HEAD_AT_CPU){
        void* res_ = res.mutable_cpu_data();
        void* a_ = a.mutable_cpu_data();
        void* b_ = b.mutable_cpu_data();
        memcpy(res_, a_, a.size());
        memcpy(res_ + a.size(), b_, b.size());
    }
    else{
        void* res_ = res.mutable_gpu_data();
        void* a_ = a.mutable_gpu_data();
        void* b_ = b.mutable_gpu_data();
        caffe_gpu_memcpy(a.size(), a_, res_);
        caffe_gpu_memcpy(b.size(), b_, res_ + a.size());
    }
    return res;
}

SyncedMemory slice(SyncedMemory a, uint64_t len, bool forward){
    if(a.head() == SyncedMemory::HEAD_AT_CPU){
        SyncedMemory res(cpu::fr_LIMBS * len * sizeof(uint64_t));
        void* res_ = res.mutable_cpu_data();
        void* a_ = a.mutable_cpu_data();
        if(forward){
            memcpy(res_, a_, res.size());
        }
        else{
            memcpy(res_, a_ + a.size()-res.size(), res.size());
        }
        return res;
    }
    else{
        SyncedMemory res(cuda::fr_LIMBS * len * sizeof(uint64_t));
        void* res_ = res.mutable_gpu_data();
        void* a_ = a.mutable_gpu_data();
        if(forward){
            caffe_gpu_memcpy(res.size(), a_, res_);
        }
        else{
            caffe_gpu_memcpy(res.size(), a_ + a.size()-res.size(), res_);
        }
        return res;
    }
}

SyncedMemory accumulate_mul_poly(SyncedMemory product, cudaStream_t stream){
    return cuda::accumulate_mul_poly_cuda(product, stream);
}

void accumulate_mul_poly_(SyncedMemory product, SyncedMemory output, cudaStream_t stream){
    cuda::accumulate_mul_poly_cuda_(product, output, stream);
}

SyncedMemory make_tensor(SyncedMemory input, uint64_t pad_len){
    return cuda::make_tensor(input, pad_len);
}

void lookup_ratio_step1_(SyncedMemory h_1, SyncedMemory h_2, SyncedMemory h_1_next, 
    SyncedMemory one, SyncedMemory delta, SyncedMemory epsilon, SyncedMemory result, cudaStream_t stream){
        cuda::lookup_ratio_step1_cuda_(h_1, h_2, h_1_next, one, delta, epsilon, result, stream);
}

void lookup_ratio_step2_(SyncedMemory f, SyncedMemory t, SyncedMemory t_next, 
    SyncedMemory one, SyncedMemory delta, SyncedMemory epsilon, SyncedMemory result, cudaStream_t stream){
        cuda::lookup_ratio_step2_cuda_(f, t, t_next, one, delta, epsilon, result, stream);
}

Ntt::Ntt(int domain_size, cudaStream_t stream): Params(cuda::params_zkp_cuda(domain_size, false, stream)) {}

SyncedMemory Ntt::forward(SyncedMemory input, cudaStream_t stream) {
    return cuda::ntt_zkp_cuda(input, Params, false, stream);
}

void Ntt::forward_(SyncedMemory input, SyncedMemory output, cudaStream_t stream) {
    cuda::ntt_zkp_cuda(input, output, Params, false, stream);
}

void Ntt::forward1(SyncedMemory input, int lg_domain_size, int stage, int chunk_id, cudaStream_t stream) {
    cuda::ntt_zkp_step1_cuda(input, Params, lg_domain_size, false, stage, chunk_id, stream);
}

void Ntt::forward1_kcolumn(SyncedMemory input, int lg_domain_size, int i_size, int k, int* stage, cudaStream_t stream) {
    cuda::ntt_kcolumn_step1_cuda(input, Params, lg_domain_size, i_size, k, false, stage, stream);
}

void Ntt::forward1_kcolumn_raw(uint64_t* input, int lg_domain_size, int64_t numel, int i_size, int k, int* stage, cudaStream_t stream) {
    cuda::ntt_kcolumn_step1_cuda_raw(input, Params, numel, lg_domain_size, i_size, k, false, stage, stream);
}

void Ntt::forward2(SyncedMemory input, int lg_domain_size, int stage, int chunk_id, cudaStream_t stream) {
    cuda::ntt_zkp_step2_cuda(input, Params, lg_domain_size, false, stage, chunk_id, stream);
}

void Ntt::forward2_kcolumn(SyncedMemory input, int lg_domain_size, int j_size, int k, int* stage, cudaStream_t stream) {
    cuda::ntt_kcolumn_step2_cuda(input, Params, lg_domain_size, j_size, k, false, stage, stream);
}

void Ntt::forward3(SyncedMemory input, int lg_domain_size, int stage, int chunk_id, cudaStream_t stream) {
    cuda::ntt_zkp_step3_cuda(input, Params, lg_domain_size, false, stage, chunk_id, stream);
}

void Ntt::forward1_internal(SyncedMemory input, SyncedMemory output, int* stage, cudaStream_t stream) {
    cuda::ntt_zkp_step1_internal_cuda(input, output, Params, false, stage, stream);
}

void Ntt::forward2_internal(SyncedMemory inout, int* stage, cudaStream_t stream) {
    cuda::ntt_zkp_step2_internal_cuda(inout, Params, false, stage, stream);
}

void Ntt::forward3_internal(SyncedMemory inout, int* stage, cudaStream_t stream) {
    cuda::ntt_zkp_step3_internal_cuda(inout, Params, false, stage, stream);
}

Intt::Intt(int domain_size, cudaStream_t stream): Params(cuda::params_zkp_cuda(domain_size, true, stream)) {}

SyncedMemory Intt::forward(SyncedMemory input, cudaStream_t stream) {
    return cuda::ntt_zkp_cuda(input, Params, true, stream);
}

void Intt::forward_(SyncedMemory input, SyncedMemory output, cudaStream_t stream) {
    cuda::ntt_zkp_cuda(input, output, Params, true, stream);
}

void Intt::_forward_(SyncedMemory input, cudaStream_t stream) {
    cuda::ntt_zkp_cuda_(input, Params, true, stream);
}

void Intt::forward1(SyncedMemory input, int lg_domain_size, int stage, int chunk_id, cudaStream_t stream) {
    cuda::ntt_zkp_step1_cuda(input, Params, lg_domain_size, true, stage, chunk_id, stream);
}

void Intt::forward2(SyncedMemory input, int lg_domain_size, int stage, int chunk_id, cudaStream_t stream) {
    cuda::ntt_zkp_step2_cuda(input, Params, lg_domain_size, true, stage, chunk_id, stream);
}

void Intt::forward3(SyncedMemory input, int lg_domain_size, int stage, int chunk_id, cudaStream_t stream) {
    cuda::ntt_zkp_step3_cuda(input, Params, lg_domain_size, true, stage, chunk_id, stream);
}

Ntt_coset::Ntt_coset(int domain_size, int coset_size)
        :lg_domain_size(coset_size), Params(cuda::params_zkp_cuda(domain_size, false)) {}

void Ntt_coset::init(SyncedMemory input, int lg_domain_size, int chunk_id, bool is_intt, cudaStream_t stream) {
    cuda::ntt_zkp_coset_init_cuda(input, Params, lg_domain_size, is_intt, chunk_id, stream);
}

void Ntt_coset::init_and_forward(SyncedMemory input, SyncedMemory output, int lg_domain_size, int chunk_id, int lambda, int stage, bool is_intt, cudaStream_t stream) {
    cuda::ntt_zkp_coset_init_and_step1_cuda(input, output, Params, lg_domain_size, chunk_id, lambda, stage, is_intt, stream);
}

void Ntt_coset::forward1(SyncedMemory input, int lg_domain_size, int stage, int chunk_id, cudaStream_t stream) {
    cuda::ntt_zkp_coset_step1_cuda(input, Params, lg_domain_size, false, stage, chunk_id, stream);
}

void Ntt_coset::forward2(SyncedMemory input, int lg_domain_size, int stage, int chunk_id, cudaStream_t stream) {
    cuda::ntt_zkp_coset_step2_cuda(input, Params, lg_domain_size, false, stage, chunk_id, stream);
}

void Ntt_coset::forward3(SyncedMemory input, SyncedMemory output, int lg_domain_size, int stage, int chunk_id, int lambda, cudaStream_t stream) {
    cuda::ntt_zkp_coset_step3_cuda(input, output, Params, lg_domain_size, false, stage, chunk_id, lambda, stream);
}

Intt_coset::Intt_coset(int domain_size, int coset_size, cudaStream_t stream)
        :lg_domain_size(coset_size), Params(cuda::params_zkp_cuda(domain_size, true, stream)) {}

void Intt_coset::init(SyncedMemory input, int lg_domain_size, int chunk_id, bool is_intt, cudaStream_t stream) {
    cuda::ntt_zkp_coset_init_cuda(input, Params, lg_domain_size, is_intt, chunk_id, stream);
}      

void Intt_coset::forward1(SyncedMemory input, int lg_domain_size, int stage, int chunk_id, cudaStream_t stream) {
    cuda::ntt_zkp_coset_step1_cuda(input, Params, lg_domain_size, true, stage, chunk_id, stream);
}

void Intt_coset::forward2(SyncedMemory input, int lg_domain_size, int stage, int chunk_id, cudaStream_t stream) {
    cuda::ntt_zkp_coset_step2_cuda(input, Params, lg_domain_size, true, stage, chunk_id, stream);
}

void Intt_coset::forward3(SyncedMemory input, SyncedMemory output, int lg_domain_size, int stage, int chunk_id, int lambda, cudaStream_t stream) {
    cuda::ntt_zkp_coset_step3_cuda(input, output, Params, lg_domain_size, true, stage, chunk_id, lambda, stream);
}

SyncedMemory multi_scalar_mult(SyncedMemory points, SyncedMemory scalars, cudaStream_t stream){

    int64_t point_num = points.size()/(sizeof(uint64_t) * cpu::fq_LIMBS);
    int64_t scalar_num = scalars.size()/(sizeof(uint64_t) * cpu::fr_LIMBS);
    int64_t msm_size = std::min(point_num, scalar_num);
    int device;
    cudaError_t err = cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    err = cudaGetDeviceProperties(&deviceProp, device);
    int smcount = deviceProp.multiProcessorCount;

    SyncedMemory step1_res = cuda::msm_zkp_cuda(points, scalars, smcount, stream);
    void* step1_cpu_ = step1_res.mutable_cpu_data_async(stream);
    SyncedMemory step2_res = cpu::msm_collect_cpu(step1_res, msm_size);
    
    return step2_res;
}

void writeToFile(const std::string& filename, uint64_t* array, uint64_t size) {
    std::ofstream outfile(filename);
    if (!outfile) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    // 写入数组到文本文件，每个元素占一行
    for (uint64_t i = 0; i < size; ++i) {
        outfile << array[i] << std::endl;
    }

    if (!outfile) {
        std::cerr << "Error writing to file: " << filename << std::endl;
    }

    outfile.close();
}

void rotate_indices(uint32_t& idx0, uint32_t& idx1, int stage, int iterations) {
    // rotate "iterations" bits in indices
    uint32_t mask = ((uint32_t)1 << (stage + iterations)) - ((uint32_t)1 << stage);

    // Rotate idx0
    uint32_t rotw = idx0 & mask;
    rotw = (rotw >> 1) | (rotw << (iterations - 1));
    idx0 = (idx0 & ~mask) | (rotw & mask);

    // Rotate idx1
    rotw = idx1 & mask;
    rotw = (rotw >> 1) | (rotw << (iterations - 1));
    idx1 = (idx1 & ~mask) | (rotw & mask);
}

void calculate_indices(uint32_t tid, int stage1, int iterations1, int stage2, int iterations2, uint32_t* map) {
    const uint32_t out_ntt_size1 = (1 << (stage1 + iterations1 - 1));
    const uint32_t inp_ntt_size1 = (uint32_t)1 << stage1;

    const uint32_t thread_ntt_pos1 = (tid & (out_ntt_size1 - 1)) >> (iterations1 - 1);

    uint32_t idx1_0 = tid & ~(out_ntt_size1 - 1);
    idx1_0 += (tid << stage1) & (out_ntt_size1 - 1);
    idx1_0 = idx1_0 * 2 + thread_ntt_pos1;

    uint32_t idx1_1 = idx1_0 + inp_ntt_size1;

    const uint32_t out_ntt_size2 = (1 << (stage2 + iterations2 - 1));
    const uint32_t inp_ntt_size2 = (uint32_t)1 << stage2;

    const uint32_t thread_ntt_pos2 = (tid & (out_ntt_size2 - 1)) >> (iterations2 - 1);

    uint32_t idx2_0 = tid & ~(out_ntt_size2 - 1);
    idx2_0 += (tid << stage2) & (out_ntt_size2 - 1);
    idx2_0 = idx2_0 * 2 + thread_ntt_pos2;

    uint32_t idx2_1 = idx2_0 + inp_ntt_size2;
    map[idx1_0] = idx2_0;
    map[idx1_1] = idx2_1;
}

// Function to reverse bits for an unsigned integer type
uint32_t bit_rev_32(uint32_t i, uint32_t nbits) {
    uint32_t rev = 0;
    for (uint32_t j = 0; j < nbits; ++j) {
        rev = (rev << 1) | (i & 1);
        i >>= 1;
    }
    return rev;
}

// Overload for long long integer (64-bit)
uint64_t bit_rev_64(uint64_t i, uint32_t nbits) {
    uint64_t rev = 0;
    for (uint32_t j = 0; j < nbits; ++j) {
        rev = (rev << 1) | (i & 1);
        i >>= 1;
    }
    return rev;
}


uint32_t bit_rev(uint32_t i, uint32_t nbits) {
    if (sizeof(i) == 4 || nbits <= 32) {
        return bit_rev_32((i), nbits);
    } else {
        return bit_rev_64((i), nbits);
    }
}

void calculate_map1(uint32_t* map, int lg_N) {
  uint64_t N = 1 << lg_N;
  for(int i = 0; i< N; i++){
    uint32_t r = bit_rev(i, lg_N);
    map[r] = i;
  }
}