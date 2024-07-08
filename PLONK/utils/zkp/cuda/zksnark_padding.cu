#include "PLONK/utils/zkp/cuda/zksnark.cuh"

namespace cuda{
    template <typename T>
    __global__ void padding(T* input, T* output, int64_t N, int64_t pad_len) {
        int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        if(tid < N){
            output[tid] = input[tid];
        }
        if(tid >= N - pad_len)
        {
            output[tid].zero();
        }
    }

    template<typename T>
    static void padding_template(T* input, T* output, int64_t N, int64_t pad_len) {
        assert((N + pad_len) > 0 && (N + pad_len) <= ::std::numeric_limits<int32_t>::max());
        int64_t grid = ((N + pad_len) + block_work_size() - 1) / block_work_size();
        padding<<<grid, block_work_size(), 0>>>(input, output, N, pad_len);
    }

    SyncedMemory& padding_cuda(SyncedMemory& input, int64_t pad_len){  
        int64_t N = input.size()/(fr_LIMBS * sizeof(uint64_t));
        SyncedMemory output(input.size() + pad_len * fr_LIMBS * sizeof(uint64_t));
        void* out_gpu = output.mutable_gpu_data();      
        void* in_gpu = input.mutable_gpu_data();
        padding_template(static_cast<fr*>(in_gpu), static_cast<fr*>(out_gpu), N, pad_len);
        return output;
    }
}