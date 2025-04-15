#include "zksnark.cuh"

namespace cuda{

template <typename T>
__global__ void compute_quotient_i(const int64_t N,
const T* compress_tuple, 
const T* f, const T* table, const T* table_next, const T* h1, const T* h1_next, const T* h2, 
const T* z2, const T* z2_next,
const T* l1,
const T* q_lookup, const T* mod,
const T* delta, const T* epsilon, const T* zeta, const T* one,
const T* lookup_seq, const T* lookup_seq_sq, const T* lookup_seq_cu,  
T* result)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N){
        result[tid] = result[tid] + \
                      q_lookup[tid] * (compress_tuple[tid] - f[tid]) * lookup_seq[0] + \
                      z2[tid] * (one[0] + delta[0]) * (f[tid] + epsilon[0]) * ((table[tid] + epsilon[0] * (one[0] + delta[0])) + table_next[tid] * delta[0]) * lookup_seq_sq[0] + \
                      (mod[tid] - z2_next[tid]) * (h1[tid] + epsilon[0] * (one[0] + delta[0]) + h2[tid] * delta[0]) * (h2[tid] + epsilon[0] * (one[0] + delta[0]) + h1_next[tid] * delta[0]) * lookup_seq_sq[0] + \
                      (z2[tid] - one[0]) * (l1[tid] * lookup_seq_cu[0]);            
    }
} 



void compute_quotient_i(SyncedMemory compress_tuple, 
SyncedMemory f, SyncedMemory table, SyncedMemory table_next, SyncedMemory h1, SyncedMemory h1_next, SyncedMemory h2, 
SyncedMemory z2, SyncedMemory z2_next,
SyncedMemory l1,
SyncedMemory q_lookup, SyncedMemory mod,
SyncedMemory delta, SyncedMemory epsilon, SyncedMemory zeta, SyncedMemory one,
SyncedMemory lookup_seq, SyncedMemory lookup_seq_sq, SyncedMemory lookup_seq_cu,  
SyncedMemory result, cudaStream_t stream)
{
    int64_t N = compress_tuple.size()/(fr_LIMBS * sizeof(uint64_t));

    void* tuple_ = compress_tuple.mutable_gpu_data_async(stream);
    void* f_ = f.mutable_gpu_data_async(stream);
    void* table_ = table.mutable_gpu_data_async(stream);
    void* table_next_ = table_next.mutable_gpu_data_async(stream);
    void* h1_ = h1.mutable_gpu_data_async(stream);
    void* h1_next_ = h1_next.mutable_gpu_data_async(stream);
    void* h2_ = h2.mutable_gpu_data_async(stream);
    void* z2_ = z2.mutable_gpu_data_async(stream);
    void* z2_next_ = z2_next.mutable_gpu_data_async(stream);
    void* l1_ = l1.mutable_gpu_data_async(stream);
    void* q_lookup_ = q_lookup.mutable_gpu_data_async(stream);
    void* mod_ = mod.mutable_gpu_data_async(stream);
    void* res_ = result.mutable_gpu_data_async(stream);
    void* delta_ = delta.mutable_gpu_data_async(stream);
    void* epsilon_ = epsilon.mutable_gpu_data_async(stream);
    void* zeta_ = zeta.mutable_gpu_data_async(stream);
    void* one_ = one.mutable_gpu_data_async(stream);
    void* seq_ = lookup_seq.mutable_gpu_data_async(stream);
    void* sq_ = lookup_seq_sq.mutable_gpu_data_async(stream);
    void* cu_ = lookup_seq_cu.mutable_gpu_data_async(stream);

    int64_t grid = (N + 128 - 1) / 128;
    compute_quotient_i<<<grid, 128, 0, stream>>>(N, 
        reinterpret_cast<fr*>(tuple_), 
        reinterpret_cast<fr*>(f_), reinterpret_cast<fr*>(table_), reinterpret_cast<fr*>(table_next_), reinterpret_cast<fr*>(h1_), reinterpret_cast<fr*>(h1_next_), reinterpret_cast<fr*>(h2_), 
        reinterpret_cast<fr*>(z2_), reinterpret_cast<fr*>(z2_next_),
        reinterpret_cast<fr*>(l1_),
        reinterpret_cast<fr*>(q_lookup_), reinterpret_cast<fr*>(mod_),
        reinterpret_cast<fr*>(delta_), reinterpret_cast<fr*>(epsilon_), reinterpret_cast<fr*>(zeta_), reinterpret_cast<fr*>(one_),
        reinterpret_cast<fr*>(seq_), reinterpret_cast<fr*>(sq_), reinterpret_cast<fr*>(cu_),  
        reinterpret_cast<fr*>(res_));
}


}