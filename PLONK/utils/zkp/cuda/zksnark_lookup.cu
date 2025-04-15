#include "zksnark.cuh"

namespace cuda{

template <typename T>
__global__ void lookup_ratio_step1(const int64_t N, const T* h_1, const T* h_2, const T* h_1_next, 
                                    const T* one, const T* delta, const T* epsilon, T* result){
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < N){
        result[tid] = (((h_2[tid] * delta[0]) + ((delta[0] + one[0]) * epsilon[0] + h_1[tid])) * ((delta[0] + one[0]) * epsilon[0] + h_2[tid] + h_1_next[tid] * delta[0]));
        result[tid] = result[tid].reciprocal();
    }
}


template <typename T>
__global__ void lookup_ratio_step2(const int64_t N, const T* f, const T* t, const T* t_next,
                                    const T* one, const T* delta, const T* epsilon, T* result){
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < N){
        result[tid] = result[tid] * 
                        (((delta[0] + one[0]) * epsilon[0] + t[tid]) + delta[0] * t_next[tid]) *
                        (delta[0] + one[0]) * (epsilon[0] + f[tid]);
    }
}

void lookup_ratio_step1_cuda_(SyncedMemory h_1, SyncedMemory h_2, SyncedMemory h_1_next, 
    SyncedMemory one, SyncedMemory delta, SyncedMemory epsilon, SyncedMemory result, cudaStream_t stream){

        int64_t N = h_1.size()/(fr_LIMBS * sizeof(uint64_t));
        void* h_1_ = h_1.mutable_gpu_data_async(stream);
        void* h_2_ = h_2.mutable_gpu_data_async(stream);
        void* h_1_next_ = h_1_next.mutable_gpu_data_async(stream);
        void* one_ = one.mutable_gpu_data_async(stream);
        void* delta_ = delta.mutable_gpu_data_async(stream);
        void* epsilon_ = epsilon.mutable_gpu_data_async(stream);
        void* result_ = result.mutable_gpu_data_async(stream);

        int64_t grid = (N + 128 - 1) / 128;

        lookup_ratio_step1<<<grid, 128, 0, stream>>>(N, 
            reinterpret_cast<fr*>(h_1_), reinterpret_cast<fr*>(h_2_), reinterpret_cast<fr*>(h_1_next_), 
            reinterpret_cast<fr*>(one_), reinterpret_cast<fr*>(delta_), reinterpret_cast<fr*>(epsilon_), 
            reinterpret_cast<fr*>(result_));
}   


void lookup_ratio_step2_cuda_(SyncedMemory f, SyncedMemory t, SyncedMemory t_next, 
    SyncedMemory one, SyncedMemory delta, SyncedMemory epsilon, SyncedMemory result, cudaStream_t stream){

        int64_t N = f.size()/(fr_LIMBS * sizeof(uint64_t));
        void* f_ = f.mutable_gpu_data_async(stream);
        void* t_ = t.mutable_gpu_data_async(stream);
        void* t_next_ = t_next.mutable_gpu_data_async(stream);
        void* one_ = one.mutable_gpu_data_async(stream);
        void* delta_ = delta.mutable_gpu_data_async(stream);
        void* epsilon_ = epsilon.mutable_gpu_data_async(stream);
        void* result_ = result.mutable_gpu_data_async(stream);

        int64_t grid = (N + block_work_size() - 1) / block_work_size();

        lookup_ratio_step2<<<grid, block_work_size(), 0, stream>>>(N, 
            reinterpret_cast<fr*>(f_), reinterpret_cast<fr*>(t_), reinterpret_cast<fr*>(t_next_), 
            reinterpret_cast<fr*>(one_), reinterpret_cast<fr*>(delta_), reinterpret_cast<fr*>(epsilon_), 
            reinterpret_cast<fr*>(result_));
}   

}