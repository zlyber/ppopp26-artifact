#include "mont_arithmetic.cuh"
#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace cuda{

#define BIN_OP_DEF(name)                                                     \
  SyncedMemory name##_mod_cuda(SyncedMemory a, SyncedMemory b, cudaStream_t stream) {             \
    SyncedMemory c(a.size());                                                \
    void* c_gpu = c.mutable_gpu_data_async(stream);                                     \
    void* a_gpu = a.mutable_gpu_data_async(stream);                                     \
    void* b_gpu = b.mutable_gpu_data_async(stream);                                     \
    if(!a.type_flag)                                                         \
      {                                                                      \
        c.type_flag = false;                                                 \
        name##_template(reinterpret_cast<fq*>(c_gpu), reinterpret_cast<fq*>(a_gpu), reinterpret_cast<fq*>(b_gpu), b.size()/sizeof(uint64_t), stream);\
        return c;                                                            \
      }                                                                      \
    name##_template(reinterpret_cast<fr*>(c_gpu), reinterpret_cast<fr*>(a_gpu), reinterpret_cast<fr*>(b_gpu), b.size()/sizeof(uint64_t), stream);\
    return c;                                                                \
  }                                                                          \
  void name##_mod_cuda_(SyncedMemory self, SyncedMemory other, cudaStream_t stream) {             \
    void* self_gpu = self.mutable_gpu_data_async(stream);                                \
    void* other_gpu = other.mutable_gpu_data_async(stream);                              \
    if(!self.type_flag)                                                      \
      {                                                                      \
        name##_template_(reinterpret_cast<fq*>(self_gpu), reinterpret_cast<fq*>(other_gpu), other.size()/sizeof(uint64_t), stream);\
        return;                                                              \
      }                                                                      \
    name##_template_(reinterpret_cast<fr*>(self_gpu), reinterpret_cast<fr*>(other_gpu), other.size()/sizeof(uint64_t), stream);\
  }                                                                          \

#define SCALAR_OP_DEF(name)                                                  \
  SyncedMemory name##_mod_scalar_cuda(SyncedMemory a, SyncedMemory b, cudaStream_t stream) {      \
    SyncedMemory c(a.size());                                                \
    void* c_gpu = c.mutable_gpu_data_async(stream);                                      \
    void* a_gpu = a.mutable_gpu_data_async(stream);                                      \
    void* b_gpu = b.mutable_gpu_data_async(stream);                                      \
    name##_scalar_template(reinterpret_cast<fr*>(c_gpu), reinterpret_cast<fr*>(a_gpu), reinterpret_cast<fr*>(b_gpu), a.size()/sizeof(uint64_t), stream);\
    return c;                                                          \
  }                                                                    \
  void name##_mod_scalar_cuda_(SyncedMemory self, SyncedMemory other, cudaStream_t stream) { \
    void* self_gpu = self.mutable_gpu_data_async(stream);                          \
    void* other_gpu = other.mutable_gpu_data_async(stream);                        \
    name##_scalar_template_(reinterpret_cast<fr*>(self_gpu), reinterpret_cast<fr*>(other_gpu), self.size()/sizeof(uint64_t), stream);\
  }                                                                    \

  BIN_OP_DEF(add);
  BIN_OP_DEF(sub);
  BIN_OP_DEF(mul);
  BIN_OP_DEF(div);
  SCALAR_OP_DEF(add);
  SCALAR_OP_DEF(sub);
  SCALAR_OP_DEF(mul);
  SCALAR_OP_DEF(div); 

template <typename T>
__global__ void to_mont_kernel_(const int64_t N, T* data) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    data[i].to();
  }
}

template <typename T>
__global__ void to_base_kernel_(const int64_t N, T* data) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    data[i].from();
  }
}

template <typename T>
__launch_bounds__(1024) __global__ void inv_mod_kernel_(const int64_t N, T* data) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    data[i] = data[i].reciprocal();
  }
}

template <typename T>
__launch_bounds__(1024) __global__ void neg_mod_kernel_(const int64_t N, T* data) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    data[i].cneg(true);
  }
}

template <typename T>
__global__ void exp_mod_kernel_(const int64_t N, T* data, int exp) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    data[i] = data[i] ^ exp;
  }
}

template <typename T>
__global__ void one_kernel_(const int64_t N, T* data) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    data[i] = T::one();
  }
}

template <typename T>
__global__ void poly_eval_kernel(const int64_t N, const T* x, T* data) {
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) {
    data[tid] = T::one();
  } else if (tid == 1) {
    data[tid] = *x;
  } else if (tid < N) {
    data[tid] = (*x) ^ tid;
  }
}

template <typename T>
__global__ void poly_reduce_kernel_first(
    const int64_t N,
    const T* x,
    const T* coff,
    T* temp) {
  int64_t tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  T sum;
  sum.zero();
  for (size_t i = tid; i < N; i += BLOCK_SIZE * gridDim.x) {
    sum += x[i] * coff[i];
  }
  __shared__ T shared_sum[BLOCK_SIZE];
  shared_sum[threadIdx.x] = sum;
  __syncthreads();

  for (int i = BLOCK_SIZE / 2; i > 0; i >>= 1) {
    if (threadIdx.x < i) {
      shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    temp[blockIdx.x] = shared_sum[0];
  }
}

template <typename T>
__global__ void poly_reduce_kernel_second(
    const int64_t N,
    const T* temp,
    T* y) {
  int64_t tid = threadIdx.x;
  __shared__ T shared_sum[BLOCK_SIZE];
  if (tid < N) {
    shared_sum[threadIdx.x] = temp[tid];
  }
  __syncthreads();
  for (int i = blockDim.x / 2; i > 0; i >>= 1) {
    if (threadIdx.x < i) {
      shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    y[0] = shared_sum[0];
  }
}

template <typename T>
__global__ void exclusive_scan_add_kernel(const T* in, T* c, T* out, int64_t N, int64_t step){
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < step){
      out[N-1-tid] = in[N-1-tid];
    }
    if(tid < N-step){    
        if(step & 0xfffffffe){
          out[N-1-tid - step] = in[N-1-tid - step] + in[N-1-tid] * c[0];
        }
        else{
          out[N-1-tid - step] = in[N-1-tid - step] - in[N-1-tid] * c[0];
        }
    }
}

template <typename T>
__global__ void exclusive_scan_mul_kernel(const T* in, T* out, int64_t N, int64_t step){
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < step){
      out[tid] = in[tid];
    }
    if(tid < N-step){    
      out[tid + step] = in[tid + step] * in[tid];
    }
}

template <typename T>
__global__ void exclusive_scan_shift_one_kernel(const T* in, T* out, int64_t N){
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid == 0){
      out[0] = T::one();
    }
    else if(tid<N){
      out[tid] = in[tid-1];
    }
}

template <typename T>
__global__ void exclusive_scan_shift_zero_kernel(const T* in, T* out, int64_t N){
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N - 1){
      out[tid] = in[tid + 1];
    }
    else if(tid == N-1){
      out[tid].zero();
    }
}

template <typename T>
__global__ void repeat_kernel(const T* in, T* out, int64_t N){
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N){
      out[tid] = *in;
    }
}

template <typename T>
static void to_mont_cuda_template(T* self, int64_t numel) {
    int64_t N = numel/num_uint64(self);
    assert(N > 0 && N <= std::numeric_limits<int32_t>::max());
    int64_t grid = (N + block_work_size() - 1) / block_work_size();
    to_mont_kernel_<<<grid, block_work_size()>>>(N, self);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void to_base_cuda_template(T* self, int64_t numel, cudaStream_t stream) {
    int64_t N = numel/num_uint64(self);
    assert(N > 0 && N <= std::numeric_limits<int32_t>::max());
    int64_t grid = (N + block_work_size() - 1) / block_work_size();
    to_base_kernel_<<<grid, block_work_size(), 0, stream>>>(N, self);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
static void inv_mod_cuda_template(T* self, int64_t numel) {
    int64_t N = numel/num_uint64(self);
    assert(N > 0 && N <= std::numeric_limits<int32_t>::max());
    int64_t grid = (N + block_work_size() - 1) / block_work_size();
    inv_mod_kernel_<<<grid, block_work_size()>>>(N, self);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
static void neg_mod_cuda_template(T* self, int64_t numel) {
    int64_t N = numel/num_uint64(self);
    assert(N > 0 && N <= std::numeric_limits<int32_t>::max());
    int64_t grid = (N + block_work_size() - 1) / block_work_size();
    neg_mod_kernel_<<<grid, block_work_size()>>>(N, self);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
static void exp_mod_cuda_template(T* self, int exp, int64_t numel) {
  if (exp == 1) {
    return;
  }
  int64_t N = numel/num_uint64(self);
  assert(N > 0 && N <= std::numeric_limits<int32_t>::max());
  int64_t grid = (N + block_work_size() - 1) / block_work_size();
  if (exp == 0) {
    one_kernel_<<<grid, block_work_size()>>>(N, self);
    CUDA_CHECK(cudaGetLastError());
  } else {
    exp_mod_kernel_<<<grid, block_work_size()>>>(N, self, exp);
    CUDA_CHECK(cudaGetLastError());
  }
}

template <typename T>
static void poly_eval_cuda_template(T* x, T* poly, int64_t N, cudaStream_t stream = (cudaStream_t)0) {
    assert(N > 0 && N <= std::numeric_limits<int32_t>::max());
    int64_t grid = (N + block_work_size() - 1) / block_work_size();
    poly_eval_kernel<<<grid, block_work_size(), 0, stream>>>(N, x, poly);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
static void poly_reduce_cuda_template(
    T* x,
    T* coff,
    T* y,
    int64_t numel) {

    int64_t N = numel / num_uint64(x);
    assert(N > 0 && N <= std::numeric_limits<int32_t>::max());
    int64_t grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (N > (BLOCK_SIZE * MAX_NUM_BLOCKS)) {
      grid = MAX_NUM_BLOCKS;
    }
    SyncedMemory temp(grid * num_uint64(x) * sizeof(uint64_t));
    void* temp_gpu = temp.mutable_gpu_data();
    poly_reduce_kernel_first<<<grid, BLOCK_SIZE>>>(N, x, coff, reinterpret_cast<T*>(temp_gpu));
    CUDA_CHECK(cudaGetLastError());
    poly_reduce_kernel_second<<<1, grid>>>(grid,  reinterpret_cast<T*>(temp_gpu), y);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
static void poly_div_cuda_template(T* divid_poly, T* c, T* exclusive_sum, int64_t numel){
    int64_t N = numel/num_uint64(divid_poly);
    assert(N > 0 && N <= std::numeric_limits<int32_t>::max());
    int64_t grid = (N + block_work_size() - 1) / block_work_size();
    auto in_ptr = divid_poly;
    auto out_ptr = exclusive_sum;
    int step = 1;
    for(int step = 1; step<N; step*=2){
      exclusive_scan_add_kernel<<<grid, block_work_size()>>>(in_ptr, c, out_ptr, N, step);
      CUDA_CHECK(cudaGetLastError());
      mont_mul_mod_kernel_<<<1, 1>>>(1, c, c);
      CUDA_CHECK(cudaGetLastError());
      auto temp = out_ptr;
      out_ptr = in_ptr;
      in_ptr = temp;
    }
    if(in_ptr != exclusive_sum) {
      cudaMemcpy(
      exclusive_sum,
      in_ptr,
      numel * sizeof(uint64_t),
      cudaMemcpyDeviceToDevice);
      CUDA_CHECK(cudaGetLastError());
    }
    exclusive_scan_shift_zero_kernel<<<grid, block_work_size()>>>(exclusive_sum, divid_poly, N);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
static void accumulate_mul_poly_cuda_template(T* product_poly, T* accumulate_mul_poly, int64_t numel, cudaStream_t stream){

    int64_t N = numel/num_uint64(product_poly);
    assert(N > 0 && N <= std::numeric_limits<int32_t>::max());
    int64_t grid = (N + block_work_size() - 1) / block_work_size();

    auto in_ptr = product_poly;
    auto out_ptr = accumulate_mul_poly;
    int step = 1;
    for(int step = 1; step < N; step*=2){
      exclusive_scan_mul_kernel<<<grid, block_work_size(), 0, stream>>>(in_ptr, out_ptr, N, step);
      CUDA_CHECK(cudaGetLastError());
      auto temp = out_ptr;
      out_ptr = in_ptr;
      in_ptr = temp;
    }
    if(in_ptr != accumulate_mul_poly) {
      cudaMemcpyAsync(
      accumulate_mul_poly,
      in_ptr,
      numel * sizeof(uint64_t),
      cudaMemcpyDeviceToDevice,
      stream);
      CUDA_CHECK(cudaGetLastError());
    }
    exclusive_scan_shift_one_kernel<<<grid, block_work_size(), 0, stream>>>(accumulate_mul_poly, product_poly, N);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
static void repeat_to_poly_cuda_template(T* input, T* output, int64_t N, cudaStream_t stream) {
    assert(N > 0 && N <= std::numeric_limits<int32_t>::max());
    int64_t grid = (N + block_work_size() - 1) / block_work_size();
    repeat_kernel<<<grid, block_work_size(), 0, stream>>>(input, output, N);
    CUDA_CHECK(cudaGetLastError());
}


SyncedMemory to_mont_cuda(SyncedMemory input) {
  SyncedMemory output(input.size());
  void* out_gpu = output.mutable_gpu_data();
  void* in_gpu = input.mutable_gpu_data();
  int64_t numel = output.size()/sizeof(uint64_t);
  cudaMemcpy(
      out_gpu,
      in_gpu,
      input.size(),
      cudaMemcpyDeviceToDevice);
  CUDA_CHECK(cudaGetLastError());
  if(!input.type_flag){
    to_mont_cuda_template(reinterpret_cast<fq*>(out_gpu), numel);
    return output;
  }
  to_mont_cuda_template(reinterpret_cast<fr*>(out_gpu), numel);
  return output;
}

SyncedMemory to_base_cuda(SyncedMemory input, cudaStream_t stream) {
  SyncedMemory output(input.size());
  void* out_gpu = output.mutable_gpu_data_async(stream);
  void* in_gpu = input.mutable_gpu_data_async(stream);
  int64_t numel = output.size()/sizeof(uint64_t);
  cudaMemcpyAsync(
      out_gpu,
      in_gpu,
      input.size(),
      cudaMemcpyDeviceToDevice,
      stream);
  CUDA_CHECK(cudaGetLastError());
  if(!input.type_flag){
    auto out_ptr = reinterpret_cast<fq*>(out_gpu);
    to_base_cuda_template(out_ptr, numel, stream);
    return output;
  }
  auto out_ptr = reinterpret_cast<fr*>(out_gpu);
  to_base_cuda_template(out_ptr, numel, stream);
  return output;
}


SyncedMemory inv_mod_cuda(SyncedMemory input) {
  SyncedMemory output(input.size());
  void* out_gpu = output.mutable_gpu_data();
  void* in_gpu = input.mutable_gpu_data();
  int64_t numel = output.size()/sizeof(uint64_t);
  cudaMemcpy(
      out_gpu,
      in_gpu,
      input.size(),
      cudaMemcpyDeviceToDevice);
  CUDA_CHECK(cudaGetLastError());
  inv_mod_cuda_template(reinterpret_cast<fr*>(out_gpu), numel);
  return output;
}

SyncedMemory neg_mod_cuda(SyncedMemory input) {
  SyncedMemory output(input.size());
  void* out_gpu = output.mutable_gpu_data();
  void* in_gpu = input.mutable_gpu_data();
  int64_t numel = output.size()/sizeof(uint64_t);
  cudaMemcpy(
      out_gpu,
      in_gpu,
      input.size(),
      cudaMemcpyDeviceToDevice);
  CUDA_CHECK(cudaGetLastError());
  if(!input.type_flag)
      {
        output.type_flag = false;
        neg_mod_cuda_template(reinterpret_cast<fq*>(out_gpu), numel);
        return output;
      }
  neg_mod_cuda_template(reinterpret_cast<fr*>(out_gpu), numel);
  return output;
}

SyncedMemory exp_mod_cuda(SyncedMemory input, int64_t exp) {
  SyncedMemory output(input.size());
  void* out_gpu = output.mutable_gpu_data();
  void* in_gpu = input.mutable_gpu_data();
  int64_t numel = output.size()/sizeof(uint64_t);
  cudaMemcpy(
      out_gpu,
      in_gpu,
      input.size(),
      cudaMemcpyDeviceToDevice);
  CUDA_CHECK(cudaGetLastError());
  exp_mod_cuda_template(reinterpret_cast<fr*>(out_gpu), exp, numel);
  return output;
}

SyncedMemory pad_poly_cuda(SyncedMemory input, int64_t N, cudaStream_t stream) {
  SyncedMemory output(N * fr_LIMBS * sizeof(uint64_t)); 
  void* out_gpu = output.mutable_gpu_data_async(stream);
  cudaMemsetAsync(out_gpu, 0, N * fr_LIMBS * sizeof(uint64_t), stream);
  void* in_cpu = input.mutable_cpu_data_async(stream);
  cudaMemcpyAsync(
      out_gpu,
      in_cpu,
      input.size(),
      cudaMemcpyHostToDevice,
      stream);
  CUDA_CHECK(cudaGetLastError());
  return output;
}

void pad_poly_cuda_(SyncedMemory input, SyncedMemory output, int64_t N, cudaStream_t stream) {
  void* out_gpu = output.mutable_gpu_data_async(stream);
  cudaMemsetAsync(out_gpu, 0, N * fr_LIMBS * sizeof(uint64_t), stream);
  void* in_gpu = input.mutable_gpu_data_async(stream);
  cudaMemcpyAsync(
      out_gpu,
      in_gpu,
      input.size(),
      cudaMemcpyDeviceToDevice,
      stream);
  CUDA_CHECK(cudaGetLastError());
}

SyncedMemory repeat_to_poly_cuda(SyncedMemory input, int64_t N, cudaStream_t stream) {
  SyncedMemory output(N * fr_LIMBS * sizeof(uint64_t)); 
  void* out_gpu = output.mutable_gpu_data_async(stream);
  void* in_gpu = input.mutable_gpu_data_async(stream);
  repeat_to_poly_cuda_template(reinterpret_cast<fr*>(in_gpu), reinterpret_cast<fr*>(out_gpu), N, stream);
  return output;
}

void repeat_to_poly_cuda_(SyncedMemory input, SyncedMemory output, int64_t N, cudaStream_t stream){
  void* out_gpu = output.mutable_gpu_data_async(stream);
  void* in_gpu = input.mutable_gpu_data_async(stream);
  repeat_to_poly_cuda_template(reinterpret_cast<fr*>(in_gpu), reinterpret_cast<fr*>(out_gpu), N, stream);
}

SyncedMemory poly_eval_cuda(SyncedMemory x, int64_t N, cudaStream_t stream) {
  SyncedMemory poly(N * fr_LIMBS * sizeof(uint64_t)); 
  void* poly_gpu = poly.mutable_gpu_data();
  void* x_gpu = x.mutable_gpu_data();
  poly_eval_cuda_template(reinterpret_cast<fr*>(x_gpu), reinterpret_cast<fr*>(poly_gpu), N, stream);
  return poly;
}

void poly_eval_cuda_(SyncedMemory x, SyncedMemory y, int64_t N, cudaStream_t stream) {
  void* poly_gpu = y.mutable_gpu_data();
  void* x_gpu = x.mutable_gpu_data();
  poly_eval_cuda_template(reinterpret_cast<fr*>(x_gpu), reinterpret_cast<fr*>(poly_gpu), N, stream);
}

SyncedMemory poly_reduce_cuda(SyncedMemory x, SyncedMemory coeff) {
  SyncedMemory y(fr_LIMBS * sizeof(uint64_t)); 
  void* y_gpu = y.mutable_gpu_data();
  void* x_gpu = x.mutable_gpu_data();
  void* coeff_gpu = coeff.mutable_gpu_data();
  poly_reduce_cuda_template(reinterpret_cast<fr*>(x_gpu), reinterpret_cast<fr*>(coeff_gpu), reinterpret_cast<fr*>(y_gpu), coeff.size()/sizeof(uint64_t));
  return y;
}


SyncedMemory poly_div_cuda(SyncedMemory divid_poly, SyncedMemory c) {
  void* in_gpu = divid_poly.mutable_gpu_data();
  SyncedMemory tiring(c.size()); 
  void* tiring_gpu = tiring.mutable_gpu_data();
  void* c_gpu = c.mutable_gpu_data();
  cudaMemcpy(tiring_gpu, c_gpu, tiring.size(), cudaMemcpyDeviceToDevice);
  CUDA_CHECK(cudaGetLastError());
  SyncedMemory out(divid_poly.size()); 
  void* out_gpu = out.mutable_gpu_data();
  poly_div_cuda_template(reinterpret_cast<fr*>(in_gpu), reinterpret_cast<fr*>(tiring_gpu), reinterpret_cast<fr*>(out_gpu), divid_poly.size()/sizeof(uint64_t));
  return divid_poly;
}

SyncedMemory accumulate_mul_poly_cuda(SyncedMemory product_poly, cudaStream_t stream) {
  void* in_gpu = product_poly.mutable_gpu_data_async(stream);
  SyncedMemory out(product_poly.size()); 
  void* out_gpu = out.mutable_gpu_data_async(stream);
  accumulate_mul_poly_cuda_template(reinterpret_cast<fr*>(in_gpu), reinterpret_cast<fr*>(out_gpu), product_poly.size()/sizeof(uint64_t), stream);
  return product_poly;
}

void accumulate_mul_poly_cuda_(SyncedMemory product_poly, SyncedMemory output, cudaStream_t stream) {
  void* in_gpu = product_poly.mutable_gpu_data_async(stream);
  void* out_gpu = output.mutable_gpu_data_async(stream);
  accumulate_mul_poly_cuda_template(reinterpret_cast<fr*>(in_gpu), reinterpret_cast<fr*>(out_gpu), product_poly.size()/sizeof(uint64_t), stream);
}

}//namespace::cuda