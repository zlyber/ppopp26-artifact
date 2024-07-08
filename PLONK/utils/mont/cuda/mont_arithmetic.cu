#include "PLONK/utils/mont/cuda/mont_arithmetic.cuh"
#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace cuda{

#define BIN_OP(name)                                                         \
  SyncedMemory& name##_mod_cuda(SyncedMemory& a, SyncedMemory& b) {          \
    SyncedMemory c(a.size());                                                \
    void* c_gpu = c.mutable_gpu_data();                                      \
    void* a_gpu = a.mutable_gpu_data();                                      \
    void* b_gpu = b.mutable_gpu_data();                                      \
    name##_template(static_cast<fr*>(c_gpu), static_cast<fr*>(a_gpu), static_cast<fr*>(b_gpu), b.size()/sizeof(uint64_t));\
    return c;                                                                \
  }                                                                          \
  void name##_mod_cuda_(SyncedMemory& self, SyncedMemory& other) {           \
    void* self_gpu = self.mutable_gpu_data();                                \
    void* other_gpu = other.mutable_gpu_data();                              \
    name##_template_(static_cast<fr*>(self_gpu), static_cast<fr*>(other_gpu), other.size()/sizeof(uint64_t));\
  }                                                                          \

#define SCALAR_OP(name)                                                      \
  SyncedMemory& name##_mod_scalar_cuda(SyncedMemory& a, SyncedMemory& b) {   \
    SyncedMemory c(a.size());                                                \
    void* c_gpu = c.mutable_gpu_data();                                      \
    void* a_gpu = a.mutable_gpu_data();                                      \
    void* b_gpu = b.mutable_gpu_data();                                      \
    name##_scalar_template(static_cast<fr*>(c_gpu), static_cast<fr*>(a_gpu), static_cast<fr*>(b_gpu), a.size()/sizeof(uint64_t));\
    return c;                                                          \
  }                                                                    \
  void name##_mod_scalar_cuda_(SyncedMemory& self, SyncedMemory& other) { \
    void* self_gpu = self.mutable_gpu_data();                          \
    void* other_gpu = other.mutable_gpu_data();                        \
    name##_scalar_template_(static_cast<fr*>(self_gpu), static_cast<fr*>(other_gpu), self.size()/sizeof(uint64_t));\
  }                                                                    \

  BIN_OP(add);
  BIN_OP(sub);
  BIN_OP(mul);
  BIN_OP(div);
  SCALAR_OP(add);
  SCALAR_OP(sub);
  SCALAR_OP(mul);
  SCALAR_OP(div); 

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
__global__ void inv_mod_kernel_(const int64_t N, T* data) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    data[i] = data[i].reciprocal();
  }
}

template <typename T>
__global__ void neg_mod_kernel_(const int64_t N, T* data) {
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
    int64_t tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
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
    int64_t tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if(tid < step){
      out[tid] = in[tid];
    }
    if(tid < N-step){    
      out[tid + step] = in[tid + step] * in[tid];
    }
}

template <typename T>
__global__ void exclusive_scan_shift_one_kernel(const T* in, T* out, int64_t N){
    int64_t tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if(tid == 0){
      out[0] = T::one();
    }
    else if(tid<N){
      out[tid] = in[tid-1];
    }
}

template <typename T>
__global__ void exclusive_scan_shift_zero_kernel(const T* in, T* out, int64_t N){
    int64_t tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if(tid < N - 1){
      out[tid] = in[tid + 1];
    }
    else if(tid == N-1){
      out[tid].zero();
    }
}

template <typename T>
__global__ void repeat_kernel(const T* in, T* out, int64_t N){
    int64_t tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if(tid < N){
      out[tid] = *in;
    }
}

template <typename T>
static void to_mont_cuda_template(T* self, int64_t numel) {
    int64_t N = numel/num_uint64(self[0]);
    assert(N > 0 && N <= std::numeric_limits<int32_t>::max());
    int64_t grid = (N + block_work_size() - 1) / block_work_size();
    to_mont_kernel_<<<grid, block_work_size(), 0>>>(N, self);
}

template <typename T>
static void to_base_cuda_template(T* self, int64_t numel) {
    int64_t N = numel/num_uint64(self[0]);
    assert(N > 0 && N <= std::numeric_limits<int32_t>::max());
    int64_t grid = (N + block_work_size() - 1) / block_work_size();
    to_base_kernel_<<<grid, block_work_size(), 0>>>(N, self);
}

template <typename T>
static void inv_mod_cuda_template(T* self, int64_t numel) {
    int64_t N = numel/num_uint64(self[0]);
    assert(N > 0 && N <= std::numeric_limits<int32_t>::max());
    int64_t grid = (N + block_work_size() - 1) / block_work_size();
    inv_mod_kernel_<<<grid, block_work_size(), 0>>>(N, self);
}

template <typename T>
static void neg_mod_cuda_template(T* self, int64_t numel) {
    int64_t N = numel/num_uint64(self[0]);
    assert(N > 0 && N <= std::numeric_limits<int32_t>::max());
    int64_t grid = (N + block_work_size() - 1) / block_work_size();
    neg_mod_kernel_<<<grid, block_work_size(), 0>>>(N, self);
}

template <typename T>
static void exp_mod_cuda_template(T* self, int exp, int64_t numel) {
  if (exp == 1) {
    return;
  }
  int64_t N = numel/num_uint64(self[0]);
  assert(N > 0 && N <= std::numeric_limits<int32_t>::max());
  int64_t grid = (N + block_work_size() - 1) / block_work_size();
  if (exp == 0) {
    one_kernel_<<<grid, block_work_size(), 0>>>(N, self);
  } else {
    exp_mod_kernel_<<<grid, block_work_size(), 0>>>(N, self, exp);
  }
}

template <typename T>
static void poly_eval_cuda_template(T* x, T* poly, int64_t N) {
    assert(N > 0 && N <= std::numeric_limits<int32_t>::max());
    int64_t grid = (N + block_work_size() - 1) / block_work_size();
    poly_eval_kernel<<<grid, block_work_size(), 0>>>(N, x, poly);
}

template <typename T>
static void poly_reduce_cuda_template(
    T* x,
    T* coff,
    T* y,
    int64_t numel) {

    int64_t N = numel / num_uint64(x[0]);
    assert(N > 0 && N <= std::numeric_limits<int32_t>::max());
    int64_t grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (N > (BLOCK_SIZE * MAX_NUM_BLOCKS)) {
      grid = MAX_NUM_BLOCKS;
    }
    SyncedMemory temp(grid * num_uint64(x[0]) * sizeof(uint64_t));
    void* temp_gpu = temp.mutable_gpu_data();
    poly_reduce_kernel_first<<<grid, BLOCK_SIZE, 0>>>(N, x, coff, static_cast<T*>(temp_gpu));
    poly_reduce_kernel_second<<<1, grid, 0>>>(grid,  static_cast<T*>(temp_gpu), y);
}

template <typename T>
static void poly_div_cuda_template(T* divid_poly, T* c, T* exclusive_sum, int64_t numel){
    int64_t N = numel/num_uint64(divid_poly[0]);
    assert(N > 0 && N <= std::numeric_limits<int32_t>::max());
    int64_t grid = (N + block_work_size() - 1) / block_work_size();
    auto in_ptr = divid_poly;
    auto out_ptr = exclusive_sum;
    int step = 1;
    for(int step = 1; step<N; step*=2){
      exclusive_scan_add_kernel<<<grid, block_work_size(), 0>>>(in_ptr, c, out_ptr, N, step);
      mont_mul_mod_kernel_<<<1, 1, 0>>>(1, c, c);
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
    }
    exclusive_scan_shift_zero_kernel<<<grid, block_work_size(), 0>>>(exclusive_sum, divid_poly, N);
}

template <typename T>
static void accumulate_mul_poly_cuda_template(T* product_poly, T* accumulate_mul_poly, int64_t numel){

    int64_t N = numel/num_uint64(product_poly[0]);
    assert(N > 0 && N <= std::numeric_limits<int32_t>::max());
    int64_t grid = (N + block_work_size() - 1) / block_work_size();

    auto in_ptr = product_poly;
    auto out_ptr = accumulate_mul_poly;
    int step = 1;
    for(int step = 1; step < N; step*=2){
      exclusive_scan_mul_kernel<<<grid, block_work_size(), 0>>>(in_ptr, out_ptr, N, step);
      auto temp = out_ptr;
      out_ptr = in_ptr;
      in_ptr = temp;
    }
    if(in_ptr != accumulate_mul_poly) {
      cudaMemcpy(
      accumulate_mul_poly,
      in_ptr,
      numel * sizeof(uint64_t),
      cudaMemcpyDeviceToDevice);
    }
    exclusive_scan_shift_one_kernel<<<grid, block_work_size(), 0>>>(accumulate_mul_poly, product_poly, N);
}

template <typename T>
static void repeat_to_poly_cuda_template(T* input, T* output, int64_t N) {
    assert(N > 0 && N <= std::numeric_limits<int32_t>::max());
    int64_t grid = (N + block_work_size() - 1) / block_work_size();
    repeat_kernel<<<grid, block_work_size(), 0>>>(input, output, N);
}


SyncedMemory& to_mont_cuda(SyncedMemory& input) {
  SyncedMemory output(input.size());
  void* out_gpu = output.mutable_gpu_data();
  void* in_gpu = input.mutable_gpu_data();
  int64_t numel = output.size()/sizeof(uint64_t);
  cudaMemcpy(
      out_gpu,
      in_gpu,
      input.size(),
      cudaMemcpyDeviceToDevice);
  to_mont_cuda_template(static_cast<fr*>(out_gpu), numel);
  return output;
}

SyncedMemory& to_base_cuda(SyncedMemory& input) {
  SyncedMemory output(input.size());
  void* out_gpu = output.mutable_gpu_data();
  void* in_gpu = input.mutable_gpu_data();
  int64_t numel = output.size()/sizeof(uint64_t);
  cudaMemcpy(
      out_gpu,
      in_gpu,
      input.size(),
      cudaMemcpyDeviceToDevice);
  to_base_cuda_template(static_cast<fr*>(out_gpu), numel);
  return output;
}


SyncedMemory& inv_mod_cuda(SyncedMemory& input) {
  SyncedMemory output(input.size());
  void* out_gpu = output.mutable_gpu_data();
  void* in_gpu = input.mutable_gpu_data();
  int64_t numel = output.size()/sizeof(uint64_t);
  cudaMemcpy(
      out_gpu,
      in_gpu,
      input.size(),
      cudaMemcpyDeviceToDevice);
  inv_mod_cuda_template(static_cast<fr*>(out_gpu), numel);
  return output;
}

SyncedMemory& neg_mod_cuda(SyncedMemory& input) {
  SyncedMemory output(input.size());
  void* out_gpu = output.mutable_gpu_data();
  void* in_gpu = input.mutable_gpu_data();
  int64_t numel = output.size()/sizeof(uint64_t);
  cudaMemcpy(
      out_gpu,
      in_gpu,
      input.size(),
      cudaMemcpyDeviceToDevice);
  neg_mod_cuda_template(static_cast<fr*>(out_gpu), numel);
  return output;
}

SyncedMemory& exp_mod_cuda(SyncedMemory& input, int64_t exp) {
  SyncedMemory output(input.size());
  void* out_gpu = output.mutable_gpu_data();
  void* in_gpu = input.mutable_gpu_data();
  int64_t numel = output.size()/sizeof(uint64_t);
  cudaMemcpy(
      out_gpu,
      in_gpu,
      input.size(),
      cudaMemcpyDeviceToDevice);
  exp_mod_cuda_template(static_cast<fr*>(out_gpu), exp, numel);
  return output;
}

SyncedMemory& pad_poly_cuda(SyncedMemory& input, int64_t N) {
  SyncedMemory output(N * fr_LIMBS * sizeof(uint64_t)); 
  void* out_gpu = output.mutable_gpu_data();
  cudaMemset(out_gpu, 0, N * fr_LIMBS * sizeof(uint64_t));
  void* in_gpu = input.mutable_gpu_data();
  cudaMemcpy(
      out_gpu,
      in_gpu,
      input.size(),
      cudaMemcpyDeviceToDevice);
  return output;
}

SyncedMemory& repeat_to_poly_cuda(SyncedMemory& input, int64_t N) {
  SyncedMemory output(N * fr_LIMBS * sizeof(uint64_t)); 
  void* out_gpu = output.mutable_gpu_data();
  void* in_gpu = input.mutable_gpu_data();
  repeat_to_poly_cuda_template(static_cast<fr*>(out_gpu), static_cast<fr*>(in_gpu), N);
  return output;
}

SyncedMemory& poly_eval_cuda(SyncedMemory& x, int64_t N) {
  SyncedMemory poly(N * fr_LIMBS * sizeof(uint64_t)); 
  void* poly_gpu = poly.mutable_gpu_data();
  void* x_gpu = x.mutable_gpu_data();
  poly_eval_cuda_template(static_cast<fr*>(x_gpu), static_cast<fr*>(poly_gpu), N);
  return poly;
}

SyncedMemory& poly_reduce_cuda(SyncedMemory& x, SyncedMemory& coeff) {
  SyncedMemory y(fr_LIMBS * sizeof(uint64_t)); 
  void* y_gpu = y.mutable_gpu_data();
  void* x_gpu = x.mutable_gpu_data();
  void* coeff_gpu = coeff.mutable_gpu_data();
  poly_reduce_cuda_template(static_cast<fr*>(x_gpu), static_cast<fr*>(coeff_gpu), static_cast<fr*>(y_gpu), coeff.size()/sizeof(uint64_t));
  return y;
}


SyncedMemory& poly_div_cuda(SyncedMemory& divid_poly, SyncedMemory& c) {
  void* in_gpu = divid_poly.mutable_gpu_data();
  SyncedMemory tiring(c.size()); 
  void* tiring_gpu = tiring.mutable_gpu_data();
  void* c_gpu = c.mutable_gpu_data();
  cudaMemcpy(tiring_gpu, c_gpu, tiring.size(), cudaMemcpyDeviceToDevice);
  SyncedMemory out(divid_poly.size()); 
  void* out_gpu = out.mutable_gpu_data();
  poly_div_cuda_template(static_cast<fr*>(in_gpu), static_cast<fr*>(tiring_gpu), static_cast<fr*>(out_gpu), divid_poly.size()/sizeof(uint64_t));
  return divid_poly;
}

SyncedMemory& accumulate_mul_poly_cuda(SyncedMemory& product_poly) {
  void* in_gpu = product_poly.mutable_gpu_data();
  SyncedMemory out(product_poly.size()); 
  void* out_gpu = out.mutable_gpu_data();
  accumulate_mul_poly_cuda_template(static_cast<fr*>(in_gpu), static_cast<fr*>(out_gpu), product_poly.size()/sizeof(uint64_t));
  return product_poly;
}
}//namespace::cuda