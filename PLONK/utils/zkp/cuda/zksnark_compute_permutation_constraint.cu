#include "zksnark.cuh"

namespace cuda{

template <typename T>
__global__ void compute_quotient_identity_range_check_i(const int64_t N, const T* x,
const T* w_l,  const T* w_r, const T* w_o, const T* w_4, 
const T* z, const T* alpha, const T* beta, const T* gamma, 
const T* k1, const T* k2, const T* k3, T* result)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N){
        result[tid] = (x[tid] * beta[0] + w_l[tid] + gamma[0]) * (x[tid] * beta[0] * k1[0] + w_r[tid] + gamma[0]) * \
                      (x[tid] * beta[0] * k2[0] + w_o[tid] + gamma[0]) * (x[tid] * beta[0] * k3[0] + w_4[tid] + gamma[0]) * z[tid] * alpha[0];
    }
} 

template <typename T>
__global__ void compute_quotient_copy_range_check_i(const int64_t N, 
const T* left_sigma, const T* right_sigma,
const T* out_sigma, const T* fourth_sigma, const T* mod,
const T* w_l,  const T* w_r, const T* w_o, const T* w_4, 
const T* z_next, const T* alpha, const T* beta, const T* gamma, 
T* result)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N){
        result[tid] = result[tid] + \
                      mod[tid] - \
                      (left_sigma[tid] * beta[0] + w_l[tid] + gamma[0]) * \
                      (right_sigma[tid] * beta[0] + w_r[tid] + gamma[0]) * \
                      (out_sigma[tid] * beta[0] + w_o[tid] + gamma[0]) * \
                      (fourth_sigma[tid] * beta[0] + w_4[tid] + gamma[0]) * \
                      z_next[tid] * alpha[0];
    } 
}

template <typename T>
__global__ void compute_quotient_term_check_one_i(const int64_t N, 
const T* z, const T* l1, const T* one, T* result)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N){
        result[tid] = result[tid] + \
                      (z[tid] - one[0]) * l1[tid];
    } 
}

void compute_quotient_identity_range_check_i(SyncedMemory x,
SyncedMemory w_l,  SyncedMemory w_r, SyncedMemory w_o, SyncedMemory w_4, 
SyncedMemory z, SyncedMemory alpha, SyncedMemory beta, SyncedMemory gamma, 
SyncedMemory k1, SyncedMemory k2, SyncedMemory k3, SyncedMemory result, cudaStream_t stream)
{
    int64_t N = x.size()/(fr_LIMBS * sizeof(uint64_t));

    void* x_ = x.mutable_gpu_data_async(stream);
    void* w_l_ = w_l.mutable_gpu_data_async(stream);
    void* w_r_ = w_r.mutable_gpu_data_async(stream);
    void* w_o_ = w_o.mutable_gpu_data_async(stream);
    void* w_4_ = w_4.mutable_gpu_data_async(stream);
    void* z_ = z.mutable_gpu_data_async(stream);
    void* alpha_ = alpha.mutable_gpu_data_async(stream);
    void* beta_ = beta.mutable_gpu_data_async(stream);
    void* gamma_ = gamma.mutable_gpu_data_async(stream);
    void* k1_ = k1.mutable_gpu_data_async(stream);
    void* k2_ = k2.mutable_gpu_data_async(stream);
    void* k3_ = k3.mutable_gpu_data_async(stream);
    void* res_ = result.mutable_gpu_data_async(stream);

    int64_t grid = (N + 128 - 1) / 128;
    compute_quotient_identity_range_check_i<<<grid, 128, 0, stream>>>(N, 
    reinterpret_cast<fr*>(x_), 
    reinterpret_cast<fr*>(w_l_), reinterpret_cast<fr*>(w_r_), reinterpret_cast<fr*>(w_o_), reinterpret_cast<fr*>(w_4_), 
    reinterpret_cast<fr*>(z_), 
    reinterpret_cast<fr*>(alpha_), reinterpret_cast<fr*>(beta_), reinterpret_cast<fr*>(gamma_), 
    reinterpret_cast<fr*>(k1_), reinterpret_cast<fr*>(k2_), reinterpret_cast<fr*>(k3_), 
    reinterpret_cast<fr*>(res_));
}

void compute_quotient_copy_range_check_i(SyncedMemory left_sigma, SyncedMemory right_sigma,
SyncedMemory out_sigma, SyncedMemory fourth_sigma, SyncedMemory mod,
SyncedMemory w_l,  SyncedMemory w_r, SyncedMemory w_o, SyncedMemory w_4, 
SyncedMemory z_next, SyncedMemory alpha, SyncedMemory beta, SyncedMemory gamma, 
SyncedMemory result, cudaStream_t stream)
{
    int64_t N = left_sigma.size()/(fr_LIMBS * sizeof(uint64_t));

    void* left_ = left_sigma.mutable_gpu_data_async(stream);
    void* right_ = right_sigma.mutable_gpu_data_async(stream);
    void* out_ = out_sigma.mutable_gpu_data_async(stream);
    void* fourth_ = fourth_sigma.mutable_gpu_data_async(stream);
    void* w_l_ = w_l.mutable_gpu_data_async(stream);
    void* w_r_ = w_r.mutable_gpu_data_async(stream);
    void* w_o_ = w_o.mutable_gpu_data_async(stream);
    void* w_4_ = w_4.mutable_gpu_data_async(stream);
    void* z_next_ = z_next.mutable_gpu_data_async(stream);
    void* alpha_ = alpha.mutable_gpu_data_async(stream);
    void* beta_ = beta.mutable_gpu_data_async(stream);
    void* gamma_ = gamma.mutable_gpu_data_async(stream);
    void* mod_ = mod.mutable_gpu_data_async(stream);
    void* res_ = result.mutable_gpu_data_async(stream);

    int64_t grid = (N + 256 - 1) / 256;
    compute_quotient_copy_range_check_i<<<grid, 256, 0, stream>>>(N, 
        reinterpret_cast<fr*>(left_), reinterpret_cast<fr*>(right_), reinterpret_cast<fr*>(out_), reinterpret_cast<fr*>(fourth_), 
        reinterpret_cast<fr*>(mod_),
        reinterpret_cast<fr*>(w_l_),  reinterpret_cast<fr*>(w_r_), reinterpret_cast<fr*>(w_o_), reinterpret_cast<fr*>(w_4_), 
        reinterpret_cast<fr*>(z_next_), 
        reinterpret_cast<fr*>(alpha_), reinterpret_cast<fr*>(beta_), reinterpret_cast<fr*>(gamma_), 
        reinterpret_cast<fr*>(res_));
}

void compute_quotient_term_check_one_i(SyncedMemory z, SyncedMemory l1, SyncedMemory one, SyncedMemory result, cudaStream_t stream)
{
    int64_t N = z.size()/(fr_LIMBS * sizeof(uint64_t));

    void* z_ = z.mutable_gpu_data_async(stream);
    void* one_ = one.mutable_gpu_data_async(stream);
    void* l1_ = l1.mutable_gpu_data_async(stream);
    void* res_ = result.mutable_gpu_data_async(stream);

    int64_t grid = (N + 256 - 1) / 256;
    compute_quotient_term_check_one_i<<<grid, 256, 0, stream>>>(N, reinterpret_cast<fr*>(z_), reinterpret_cast<fr*>(l1_), reinterpret_cast<fr*>(one_), reinterpret_cast<fr*>(res_));
}

}