#include "zksnark.cuh"

namespace cuda{
    template <typename T>
    __global__ void compute_query_table_poly(T* q_lookup, T* w_l, T* w_r, T* w_o, T* w_4,
                                            T* t, T* f_0, T* f_1, T* f_2, T* f_3, int64_t N) {
        int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        if(tid < N)
        {
            if(q_lookup[tid].is_zero()){
                    f_0[tid] = *t;
                    f_1[tid].zero();
                    f_2[tid].zero();
                    f_3[tid].zero();
                }
            else{
                f_0[tid] = w_l[tid];
                f_1[tid] = w_r[tid];
                f_2[tid] = w_o[tid];
                f_3[tid] = w_4[tid];
            }
        }
    }


    template <typename T>
    __global__ void compress_poly_cuda(T* f, T* f_0, T* challenge, int64_t N) {
        int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        if(tid < N)
        {
            f[tid] = f[tid] * challenge[0] + f_0[tid];
        }
    }

    template <typename T>
    __global__ void compress_poly_cuda_2(T* f, T* out, T* challenge, int64_t N) {
        int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        if(tid < N)
        {
            out[tid] = f[tid] + f[tid + N] * challenge[0] \
                       + f[tid + 2*N] * challenge[0] * challenge[0] \
                       + f[tid + 3*N] * challenge[0] * challenge[0] * challenge[0];
        }
    }

    template <typename T>
    static void compute_query_table_poly_template(T* padded_q_lookup, T* w_l_scalar,
                                        T* w_r_scalar, T* w_o_scalar, T* w_4_scalar, 
                                        T* t_poly, T* f, int64_t numel, cudaStream_t stream) {
        int64_t N = numel / num_uint64(padded_q_lookup);
        auto f_1 = f + N;
        auto f_2 = f + 2*N;
        auto f_3 = f + 3*N;

        assert(N > 0 && N <= ::std::numeric_limits<int32_t>::max());
        int64_t grid = (N + block_work_size() - 1) / block_work_size();

        compute_query_table_poly<<<grid, block_work_size(), 0, stream>>>(padded_q_lookup, w_l_scalar, w_r_scalar, w_o_scalar, w_4_scalar,
                                                                        t_poly, f, f_1, f_2, f_3, N);
        CUDA_CHECK(cudaGetLastError());
    }

    template <typename T>
    static void compress_poly_template(T* f,
                                    T* f_0, T* f_1, T* f_2,
                                    T* challenge, int64_t numel, cudaStream_t stream) {
        int64_t N = numel/num_uint64(f);
        assert(N > 0 && N <= ::std::numeric_limits<int32_t>::max());
        int64_t grid = (N + block_work_size() - 1) / block_work_size();

        compress_poly_cuda<<<grid, block_work_size(), 0, stream>>>(f, f_2, challenge, N);
        CUDA_CHECK(cudaGetLastError());
        compress_poly_cuda<<<grid, block_work_size(), 0, stream>>>(f, f_1, challenge, N);
        CUDA_CHECK(cudaGetLastError());
        compress_poly_cuda<<<grid, block_work_size(), 0, stream>>>(f, f_0, challenge, N);
        CUDA_CHECK(cudaGetLastError());
    }

    template <typename T>
    static void compress_poly_template_2(T* f, T* out, T* challenge, int64_t numel, cudaStream_t stream) {
        int64_t N = numel / num_uint64(f);
        assert(N > 0 && N <= ::std::numeric_limits<int32_t>::max());
        int64_t grid = (N + block_work_size() - 1) / block_work_size();
        compress_poly_cuda_2<<<grid, block_work_size(), 0, stream>>>(f, out, challenge, N);
        CUDA_CHECK(cudaGetLastError());
    }

    void compute_query_table_cuda(SyncedMemory output, SyncedMemory padded_q_lookup, 
                                    SyncedMemory w_l_scalar, SyncedMemory w_r_scalar, SyncedMemory w_o_scalar, SyncedMemory w_4_scalar,
                                    SyncedMemory t_poly, cudaStream_t stream){
    
        int64_t N = padded_q_lookup.size() * 4/(fr_LIMBS * sizeof(uint64_t));

        void* out_gpu = output.mutable_gpu_data_async(stream);  
        void* lookup_gpu = padded_q_lookup.mutable_gpu_data_async(stream);  
        void* w_l_gpu = w_l_scalar.mutable_gpu_data_async(stream);  
        void* w_r_gpu = w_r_scalar.mutable_gpu_data_async(stream);  
        void* w_o_gpu = w_o_scalar.mutable_gpu_data_async(stream);  
        void* w_4_gpu = w_4_scalar.mutable_gpu_data_async(stream);  
        void* t_gpu = t_poly.mutable_gpu_data_async(stream);  
        int64_t numel = padded_q_lookup.size() / sizeof(uint64_t);
        compute_query_table_poly_template(reinterpret_cast<fr*>(lookup_gpu), reinterpret_cast<fr*>(w_l_gpu),
                                          reinterpret_cast<fr*>(w_r_gpu), reinterpret_cast<fr*>(w_o_gpu), reinterpret_cast<fr*>(w_4_gpu), 
                                          reinterpret_cast<fr*>(t_gpu), reinterpret_cast<fr*>(out_gpu), numel, stream);
        
    }

    void compress_cuda(SyncedMemory output, SyncedMemory f_0, SyncedMemory f_1, 
                               SyncedMemory f_2, SyncedMemory f_3, SyncedMemory challenge, cudaStream_t stream){

        void* out_gpu = output.mutable_gpu_data_async(stream);
        void* f_3_gpu = f_3.mutable_gpu_data_async(stream);  
        void* f_2_gpu = f_2.mutable_gpu_data_async(stream);  
        void* f_1_gpu = f_1.mutable_gpu_data_async(stream);  
        void* f_0_gpu = f_0.mutable_gpu_data_async(stream);  
        void* c_gpu = challenge.mutable_gpu_data_async(stream);  
        cudaMemcpyAsync(
        out_gpu,
        f_3_gpu,
        f_3.size(),
        cudaMemcpyDeviceToDevice,
        stream);
        int64_t numel = f_3.size() / sizeof(uint64_t);
        compress_poly_template(reinterpret_cast<fr*>(out_gpu), reinterpret_cast<fr*>(f_0_gpu),
                            reinterpret_cast<fr*>(f_1_gpu), reinterpret_cast<fr*>(f_2_gpu), reinterpret_cast<fr*>(c_gpu), numel, stream);
    }

    void compress_cuda_2(SyncedMemory concatenated_f, SyncedMemory output, SyncedMemory challenge, int64_t numel, cudaStream_t stream){
        void* out_gpu = output.mutable_gpu_data_async(stream);
        void* f_gpu = concatenated_f.mutable_gpu_data_async(stream);   
        void* c_gpu = challenge.mutable_gpu_data_async(stream);  
        CUDA_CHECK(cudaGetLastError());
        compress_poly_template_2(reinterpret_cast<fr*>(f_gpu), 
                                 reinterpret_cast<fr*>(out_gpu),
                                reinterpret_cast<fr*>(c_gpu), numel,
                                stream);
    }
}//namespace::cuda