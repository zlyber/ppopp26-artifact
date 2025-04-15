#include "zksnark.cuh"

namespace cuda{
    // void checkCudaError(cudaError_t err, const char* msg) {
    //     if (err != cudaSuccess) {
    //         std::cerr << "CUDA error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
    //         exit(EXIT_FAILURE);
    //     }
    // }


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

    __global__ void kernel(uint64_t* input, uint64_t* output, uint64_t N) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) {
            output[idx * 4] = input[idx];
        }
    }

    template<typename T>
    static void padding_template(T* input, T* output, int64_t N, int64_t pad_len) {
        assert((N + pad_len) > 0 && (N + pad_len) <= ::std::numeric_limits<int32_t>::max());
        int64_t grid = ((N + pad_len) + block_work_size() - 1) / block_work_size();
        padding<<<grid, block_work_size(), 0>>>(input, output, N, pad_len);
        CUDA_CHECK(cudaGetLastError());
    }

    void make_tensor_template(uint64_t* input, uint64_t* output, uint64_t N, uint64_t pad_len) {
        assert(N > 0 && N <= ::std::numeric_limits<int32_t>::max());
        int64_t grid = (pad_len + block_work_size() - 1) / block_work_size();
        kernel<<<grid, block_work_size(), 0>>>(input, output, N);
        CUDA_CHECK(cudaGetLastError());
    }

    SyncedMemory padding_cuda(SyncedMemory input, uint64_t pad_len){  
        int64_t N = input.size()/(fr_LIMBS * sizeof(uint64_t));
        SyncedMemory output((N + pad_len) * fr_LIMBS * sizeof(uint64_t));
        void* out_gpu = output.mutable_gpu_data();      
        void* in_gpu = input.mutable_gpu_data();
        padding_template(static_cast<fr*>(in_gpu), static_cast<fr*>(out_gpu), N, pad_len);
        return output;
    }

    SyncedMemory make_tensor(SyncedMemory input, uint64_t pad_len){
        uint64_t N = input.size()/sizeof(uint64_t);
        SyncedMemory output(pad_len * fr_LIMBS * sizeof(uint64_t));
        void* out_gpu = output.mutable_gpu_data();     
        void* in_gpu = input.mutable_gpu_data(); 
        caffe_gpu_memset(output.size(), 0, out_gpu);
        make_tensor_template(static_cast<uint64_t*>(in_gpu), static_cast<uint64_t*>(out_gpu), N, pad_len);
        return output;
    }
}