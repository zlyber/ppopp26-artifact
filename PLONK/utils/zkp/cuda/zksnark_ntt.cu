#include "zksnark.cuh"
#include "zksnark_ntt/ntt_kernel/ntt.cuh"
#include "zksnark_ntt/parameters/parameters.cuh"
namespace cuda{
  // temporarily set device_id to 0, set InputOutputOrder to NN
  // TODO: optimize memory copy for inout data
  template<typename T>
  static void params_zkp_template(
    T* self,
    T* local_params,
    bool is_intt,
    cudaStream_t stream) {
    NTTParameters(is_intt, self, local_params, stream);
    CUDA_CHECK(cudaGetLastError());
  }

  SyncedMemory params_zkp_cuda(
    int64_t domain_size,
    bool is_intt,
    cudaStream_t stream) {
  auto partial_sz = WINDOW_NUM * WINDOW_SIZE;
  auto S1 = 2 * partial_sz;
  auto S2 = 32 + 64 + 128 + 256 + 512;
  auto S3 = 64 * 64 + 4096 * 64 + 128 * 128 + 256 * 256 + 512 * 512;
  auto S4 = domain_size + 1;

  auto double_roots = 2 * S4; // forward and inverse
  auto double_group_gen = 2; // group generator and inverse group generator

  SyncedMemory local_params((double_group_gen + double_roots) * fr_LIMBS * sizeof(uint64_t));
  SyncedMemory params((S1 + S2 + S3 + S4) * fr_LIMBS * sizeof(uint64_t));
  void* local_gpu = local_params.mutable_gpu_data_async(stream);
  CUDA_CHECK(cudaGetLastError());
  void* params_gpu = params.mutable_gpu_data_async(stream);
  CUDA_CHECK(cudaGetLastError());
  params_zkp_template(reinterpret_cast<fr*>(params_gpu), reinterpret_cast<fr*>(local_gpu), is_intt, stream);
  CUDA_CHECK(cudaGetLastError());
  return params;
  }

  template<typename T>
  static void ntt_zkp(
      T* self,
      T* params,
      bool is_intt,
      bool is_coset, int64_t numel,
      cudaStream_t stream = (cudaStream_t)0) {
    auto len = numel / fr_LIMBS;
    uint32_t lg_domain_size = log2(len);
    assert(len == 1 << lg_domain_size);
    auto L1 = WINDOW_NUM * WINDOW_SIZE;
    auto L2 = 2 * L1;
    auto L3 = L2 + (32 + 64 + 128 + 256 + 512);
    auto L4 = L3 + (64 * 64 + 4096 * 64 + 128 * 128 + 256 * 256 + 512 * 512);

    auto pt_ptr = params;
    auto pggp_ptr = params + L1;
    auto rp_ptr = params + L2;
    auto rpm_ptr = params + L3;
    auto size_inverse_ptr = params + L4;

    compute_ntt(
        0,
        self,
        pt_ptr,
        rp_ptr,
        rpm_ptr,
        pggp_ptr,
        size_inverse_ptr,
        lg_domain_size,
        InputOutputOrder::NN,
        is_intt ? Direction::inverse : Direction::forward,
        stream);
      CUDA_CHECK(cudaGetLastError());
  }

  template<typename T>
  static void ntt_zkp_step1(
      T* self,
      T* params,
      int lg_domain_size,
      bool is_intt,
      bool is_coset, int64_t numel, int stage, int chunk_id,
      cudaStream_t stream = (cudaStream_t)0) {
    auto len = numel / fr_LIMBS;
    uint32_t lg_chunk_size = log2(len);

    auto L1 = WINDOW_NUM * WINDOW_SIZE;
    auto L2 = 2 * L1;
    auto L3 = L2 + (32 + 64 + 128 + 256 + 512);
    auto L4 = L3 + (64 * 64 + 4096 * 64 + 128 * 128 + 256 * 256 + 512 * 512);

    auto pt_ptr = params;
    auto pggp_ptr = params + L1;
    auto rp_ptr = params + L2;
    auto rpm_ptr = params + L3;
    auto size_inverse_ptr = params + L4;

    ntt_step1(
        self,
        pt_ptr,
        rp_ptr,
        rpm_ptr,
        pggp_ptr,
        size_inverse_ptr,
        lg_domain_size,
        lg_chunk_size,
        stage,
        chunk_id,
        InputOutputOrder::NN,
        is_intt ? Direction::inverse : Direction::forward,
        stream);
      CUDA_CHECK(cudaGetLastError());
  }

  template<typename T>
  static void ntt_zkp_step2(
      T* self,
      T* params,
      int lg_domain_size,
      bool is_intt,
      bool is_coset, int64_t numel, int stage, int chunk_id,
      cudaStream_t stream = (cudaStream_t)0) {
    auto len = numel / fr_LIMBS;
    uint32_t lg_chunk_size = log2(len);

    auto L1 = WINDOW_NUM * WINDOW_SIZE;
    auto L2 = 2 * L1;
    auto L3 = L2 + (32 + 64 + 128 + 256 + 512);
    auto L4 = L3 + (64 * 64 + 4096 * 64 + 128 * 128 + 256 * 256 + 512 * 512);

    auto pt_ptr = params;
    auto pggp_ptr = params + L1;
    auto rp_ptr = params + L2;
    auto rpm_ptr = params + L3;
    auto size_inverse_ptr = params + L4;

    ntt_step2(
        self,
        pt_ptr,
        rp_ptr,
        rpm_ptr,
        pggp_ptr,
        size_inverse_ptr,
        lg_domain_size,
        lg_chunk_size,
        stage,
        chunk_id,
        InputOutputOrder::NN,
        is_intt ? Direction::inverse : Direction::forward,
        stream);
      CUDA_CHECK(cudaGetLastError());
  }

  template<typename T>
  static void ntt_zkp_step3(
      T* self,
      T* params,
      int lg_domain_size,
      bool is_intt,
      bool is_coset, int64_t numel, int stage, int chunk_id,
      cudaStream_t stream = (cudaStream_t)0) {
    auto len = numel / fr_LIMBS;
    uint32_t lg_chunk_size = log2(len);

    auto L1 = WINDOW_NUM * WINDOW_SIZE;
    auto L2 = 2 * L1;
    auto L3 = L2 + (32 + 64 + 128 + 256 + 512);
    auto L4 = L3 + (64 * 64 + 4096 * 64 + 128 * 128 + 256 * 256 + 512 * 512);

    auto pt_ptr = params;
    auto pggp_ptr = params + L1;
    auto rp_ptr = params + L2;
    auto rpm_ptr = params + L3;
    auto size_inverse_ptr = params + L4;

    ntt_step3(
        self,
        pt_ptr,
        rp_ptr,
        rpm_ptr,
        pggp_ptr,
        size_inverse_ptr,
        lg_domain_size,
        lg_chunk_size,
        stage,
        chunk_id,
        InputOutputOrder::NN,
        is_intt ? Direction::inverse : Direction::forward,
        stream);
      CUDA_CHECK(cudaGetLastError());
  }

  template<typename T>
  static void ntt_zkp_step1_internal(
      T* self,
      T* params,
      bool is_intt,
      bool is_coset, int64_t numel, int* stage,
      cudaStream_t stream = (cudaStream_t)0) {
    auto len = numel / fr_LIMBS;
    uint32_t lg_domain_size = log2(len);
    assert(len == 1 << lg_domain_size);
    auto L1 = WINDOW_NUM * WINDOW_SIZE;
    auto L2 = 2 * L1;
    auto L3 = L2 + (32 + 64 + 128 + 256 + 512);
    auto L4 = L3 + (64 * 64 + 4096 * 64 + 128 * 128 + 256 * 256 + 512 * 512);

    auto pt_ptr = params;
    auto pggp_ptr = params + L1;
    auto rp_ptr = params + L2;
    auto rpm_ptr = params + L3;
    auto size_inverse_ptr = params + L4;

    ntt_internal_step1(
        self,
        pt_ptr,
        rp_ptr,
        rpm_ptr,
        pggp_ptr,
        size_inverse_ptr,
        lg_domain_size,
        stage,
        InputOutputOrder::NN,
        is_intt ? Direction::inverse : Direction::forward,
        stream);
      CUDA_CHECK(cudaGetLastError());
  }

  template<typename T>
  static void ntt_zkp_step2_internal(
      T* self,
      T* params,
      bool is_intt,
      bool is_coset, int64_t numel, int* stage,
      cudaStream_t stream = (cudaStream_t)0) {
    auto len = numel / fr_LIMBS;
    uint32_t lg_domain_size = log2(len);
    assert(len == 1 << lg_domain_size);
    auto L1 = WINDOW_NUM * WINDOW_SIZE;
    auto L2 = 2 * L1;
    auto L3 = L2 + (32 + 64 + 128 + 256 + 512);
    auto L4 = L3 + (64 * 64 + 4096 * 64 + 128 * 128 + 256 * 256 + 512 * 512);

    auto pt_ptr = params;
    auto pggp_ptr = params + L1;
    auto rp_ptr = params + L2;
    auto rpm_ptr = params + L3;
    auto size_inverse_ptr = params + L4;

    ntt_internal_step2(
        self,
        pt_ptr,
        rp_ptr,
        rpm_ptr,
        pggp_ptr,
        size_inverse_ptr,
        lg_domain_size,
        stage,
        InputOutputOrder::NN,
        is_intt ? Direction::inverse : Direction::forward,
        stream);
      CUDA_CHECK(cudaGetLastError());
  }

  template<typename T>
  static void ntt_zkp_step3_internal(
      T* self,
      T* params,
      bool is_intt,
      bool is_coset, int64_t numel, int* stage,
      cudaStream_t stream = (cudaStream_t)0) {
    auto len = numel / fr_LIMBS;
    uint32_t lg_domain_size = log2(len);
    assert(len == 1 << lg_domain_size);
    auto L1 = WINDOW_NUM * WINDOW_SIZE;
    auto L2 = 2 * L1;
    auto L3 = L2 + (32 + 64 + 128 + 256 + 512);
    auto L4 = L3 + (64 * 64 + 4096 * 64 + 128 * 128 + 256 * 256 + 512 * 512);

    auto pt_ptr = params;
    auto pggp_ptr = params + L1;
    auto rp_ptr = params + L2;
    auto rpm_ptr = params + L3;
    auto size_inverse_ptr = params + L4;

    ntt_internal_step3(
        self,
        pt_ptr,
        rp_ptr,
        rpm_ptr,
        pggp_ptr,
        size_inverse_ptr,
        lg_domain_size,
        stage,
        InputOutputOrder::NN,
        is_intt ? Direction::inverse : Direction::forward,
        stream);
      CUDA_CHECK(cudaGetLastError());
  }

  SyncedMemory ntt_zkp_cuda(
    SyncedMemory input,
    SyncedMemory params,
    bool is_intt,
    cudaStream_t stream) {
  SyncedMemory output(input.size());
  void* out_gpu = output.mutable_gpu_data();
  void* in_gpu = input.mutable_gpu_data();
  cudaMemcpyAsync(
      out_gpu,
      in_gpu,
      input.size(),
      cudaMemcpyDeviceToDevice,
      stream);
  CUDA_CHECK(cudaGetLastError());
  void* params_gpu = params.mutable_gpu_data();
  int64_t numel = input.size()/sizeof(uint64_t);
  ntt_zkp(reinterpret_cast<fr*>(out_gpu), reinterpret_cast<fr*>(params_gpu), is_intt, false, numel, stream);
  return output;
  }

  void ntt_zkp_cuda(
    SyncedMemory input,
    SyncedMemory output,
    SyncedMemory params,
    bool is_intt,
    cudaStream_t stream) {
  void* out_gpu = output.mutable_gpu_data();
  void* in_gpu = input.mutable_gpu_data();
  cudaMemcpyAsync(
      out_gpu,
      in_gpu,
      input.size(),
      cudaMemcpyDeviceToDevice,
      stream);
  CUDA_CHECK(cudaGetLastError());
  void* params_gpu = params.mutable_gpu_data();
  int64_t numel = input.size()/sizeof(uint64_t);
  ntt_zkp(reinterpret_cast<fr*>(out_gpu), reinterpret_cast<fr*>(params_gpu), is_intt, false, numel, stream);
  }

  void ntt_zkp_cuda_(
    SyncedMemory inout,
    SyncedMemory params,
    bool is_intt,
    cudaStream_t stream) {
  void* in_gpu = inout.mutable_gpu_data();
  CUDA_CHECK(cudaGetLastError());
  void* params_gpu = params.mutable_gpu_data();
  int64_t numel = inout.size()/sizeof(uint64_t);
  ntt_zkp(reinterpret_cast<fr*>(in_gpu), reinterpret_cast<fr*>(params_gpu), is_intt, false, numel, stream);
  }

  void ntt_zkp_step1_cuda(
    SyncedMemory inout,
    SyncedMemory params,
    int lg_domain_size,
    bool is_intt,
    int stage,
    int chunk_id,
    cudaStream_t stream) {
  void* inout_ = inout.mutable_gpu_data();
  CUDA_CHECK(cudaGetLastError());
  void* params_gpu = params.mutable_gpu_data();
  int64_t numel = inout.size()/sizeof(uint64_t);
  ntt_zkp_step1(reinterpret_cast<fr*>(inout_), reinterpret_cast<fr*>(params_gpu), lg_domain_size, is_intt, false, numel, stage, chunk_id, stream);
  }

  void ntt_zkp_step2_cuda(
    SyncedMemory inout,
    SyncedMemory params,
    int lg_domain_size,
    bool is_intt,
    int stage,
    int chunk_id,
    cudaStream_t stream) {
  void* inout_ = inout.mutable_gpu_data();
  CUDA_CHECK(cudaGetLastError());
  void* params_gpu = params.mutable_gpu_data();
  int64_t numel = inout.size()/sizeof(uint64_t);
  ntt_zkp_step2(reinterpret_cast<fr*>(inout_), reinterpret_cast<fr*>(params_gpu), lg_domain_size, is_intt, false, numel, stage, chunk_id, stream);
  }

  void ntt_zkp_step3_cuda(
    SyncedMemory inout,
    SyncedMemory params,
    int lg_domain_size,
    bool is_intt,
    int stage,
    int chunk_id,
    cudaStream_t stream) {
  void* inout_ = inout.mutable_gpu_data();
  CUDA_CHECK(cudaGetLastError());
  void* params_gpu = params.mutable_gpu_data();
  int64_t numel = inout.size()/sizeof(uint64_t);
  ntt_zkp_step3(reinterpret_cast<fr*>(inout_), reinterpret_cast<fr*>(params_gpu), lg_domain_size, is_intt, false, numel, stage, chunk_id, stream);
  }

  void ntt_zkp_step1_internal_cuda(
    SyncedMemory input,
    SyncedMemory output,
    SyncedMemory params,
    bool is_intt,
    int* stage,
    cudaStream_t stream) {
  void* input_ = input.mutable_gpu_data();
  void* output_ = output.mutable_gpu_data();
  cudaMemcpyAsync(
    output_,
    input_,
    input.size(),
    cudaMemcpyDeviceToDevice,
    stream);
  CUDA_CHECK(cudaGetLastError());
  void* params_gpu = params.mutable_gpu_data();
  int64_t numel = input.size()/sizeof(uint64_t);
  ntt_zkp_step1_internal(reinterpret_cast<fr*>(output_), reinterpret_cast<fr*>(params_gpu), is_intt, false, numel, stage, stream);
  }

  void ntt_zkp_step2_internal_cuda(
    SyncedMemory inout,
    SyncedMemory params,
    bool is_intt,
    int* stage,
    cudaStream_t stream) {
  void* inout_ = inout.mutable_gpu_data();
  void* params_gpu = params.mutable_gpu_data();
  int64_t numel = inout.size()/sizeof(uint64_t);
  ntt_zkp_step2_internal(reinterpret_cast<fr*>(inout_), reinterpret_cast<fr*>(params_gpu), is_intt, false, numel, stage, stream);
  }

  void ntt_zkp_step3_internal_cuda(
    SyncedMemory inout,
    SyncedMemory params,
    bool is_intt,
    int* stage,
    cudaStream_t stream) {
  void* inout_ = inout.mutable_gpu_data();
  void* params_gpu = params.mutable_gpu_data();
  int64_t numel = inout.size()/sizeof(uint64_t);
  ntt_zkp_step3_internal(reinterpret_cast<fr*>(inout_), reinterpret_cast<fr*>(params_gpu), is_intt, false, numel, stage, stream);
  }

  template<typename T>
  static void ntt_zkp_bitrev(
      T* self,
      int lg_domain_size,
      bool is_intt,
      int64_t numel,
      cudaStream_t stream = (cudaStream_t)0) {
    auto len = numel / fr_LIMBS;
    uint32_t lg_chunk_size = log2(len);
    
    NTT_bitrev(
        self,
        lg_domain_size,
        lg_chunk_size,
        InputOutputOrder::NN,
        is_intt ? Direction::inverse : Direction::forward,
        stream);
      CUDA_CHECK(cudaGetLastError());
  }

  template<typename T>
  static void ntt_zkp_coset_init(
      T* self,
      T* params,
      int lg_domain_size,
      bool is_intt,
      int64_t numel,
      int chunk_id,
      cudaStream_t stream = (cudaStream_t)0) {
    auto len = numel / fr_LIMBS;
    uint32_t lg_chunk_size = log2(len);

    auto L1 = WINDOW_NUM * WINDOW_SIZE;
    auto L2 = 2 * L1;
    auto L3 = L2 + (32 + 64 + 128 + 256 + 512);
    auto L4 = L3 + (64 * 64 + 4096 * 64 + 128 * 128 + 256 * 256 + 512 * 512);

    auto pt_ptr = params;
    auto pggp_ptr = params + L1;
    auto rp_ptr = params + L2;
    auto rpm_ptr = params + L3;
    auto size_inverse_ptr = params + L4;
    
    NTT_LDE_init(
        self,
        pt_ptr,
        rp_ptr,
        rpm_ptr,
        pggp_ptr,
        size_inverse_ptr,
        lg_domain_size,
        lg_chunk_size,
        chunk_id,
        InputOutputOrder::NN,
        is_intt ? Direction::inverse : Direction::forward,
        stream);
      CUDA_CHECK(cudaGetLastError());
  }


  template<typename T>
  static void ntt_zkp_coset_step1(
      T* self,
      T* params,
      int lg_domain_size,
      bool is_intt,
      bool is_coset, int64_t numel,
      int stage,
      int chunk_id,
      cudaStream_t stream = (cudaStream_t)0) {
    auto len = numel / fr_LIMBS;
    uint32_t lg_chunk_size = log2(len);

    auto L1 = WINDOW_NUM * WINDOW_SIZE;
    auto L2 = 2 * L1;
    auto L3 = L2 + (32 + 64 + 128 + 256 + 512);
    auto L4 = L3 + (64 * 64 + 4096 * 64 + 128 * 128 + 256 * 256 + 512 * 512);

    auto pt_ptr = params;
    auto pggp_ptr = params + L1;
    auto rp_ptr = params + L2;
    auto rpm_ptr = params + L3;
    auto size_inverse_ptr = params + L4;
    
    NTT_LDE_step1(
        self,
        pt_ptr,
        rp_ptr,
        rpm_ptr,
        pggp_ptr,
        size_inverse_ptr,
        lg_domain_size,
        lg_chunk_size,
        stage,
        chunk_id,
        InputOutputOrder::NN,
        is_intt ? Direction::inverse : Direction::forward,
        stream);
      CUDA_CHECK(cudaGetLastError());
  }

  template<typename T>
  static void ntt_zkp_coset_init_with_bitrev(
    T* self,
    T* params,
    int lg_domain_size,
    bool is_intt,
    int64_t numel,
    cudaStream_t stream = (cudaStream_t)0) {
    auto len = numel / fr_LIMBS;
    uint32_t lg_chunk_size = log2(len);

    auto L1 = WINDOW_NUM * WINDOW_SIZE;
    auto L2 = 2 * L1;
    auto L3 = L2 + (32 + 64 + 128 + 256 + 512);
    auto L4 = L3 + (64 * 64 + 4096 * 64 + 128 * 128 + 256 * 256 + 512 * 512);

    auto pt_ptr = params;
    auto pggp_ptr = params + L1;
    auto rp_ptr = params + L2;
    auto rpm_ptr = params + L3;
    auto size_inverse_ptr = params + L4;
    
    NTT_LDE_init_with_bitrev(
        self,
        pt_ptr,
        rp_ptr,
        rpm_ptr,
        pggp_ptr,
        size_inverse_ptr,
        lg_domain_size,
        InputOutputOrder::NN,
        is_intt ? Direction::inverse : Direction::forward,
        stream);
      CUDA_CHECK(cudaGetLastError());
  }


  template<typename T>
  static void ntt_zkp_coset_step2(
      T* self,
      T* params,
      int lg_domain_size,
      bool is_intt,
      bool is_coset, int64_t numel,
      int stage,
      int chunk_id,
      cudaStream_t stream = (cudaStream_t)0) {
    auto len = numel / fr_LIMBS;
    uint32_t lg_chunk_size = log2(len);
    // assert(len == 1 << lg_domain_size);
    auto L1 = WINDOW_NUM * WINDOW_SIZE;
    auto L2 = 2 * L1;
    auto L3 = L2 + (32 + 64 + 128 + 256 + 512);
    auto L4 = L3 + (64 * 64 + 4096 * 64 + 128 * 128 + 256 * 256 + 512 * 512);

    auto pt_ptr = params;
    auto pggp_ptr = params + L1;
    auto rp_ptr = params + L2;
    auto rpm_ptr = params + L3;
    auto size_inverse_ptr = params + L4;
    
    NTT_LDE_step2(
        self,
        pt_ptr,
        rp_ptr,
        rpm_ptr,
        pggp_ptr,
        size_inverse_ptr,
        lg_domain_size,
        lg_chunk_size,
        stage,
        chunk_id,
        InputOutputOrder::NN,
        is_intt ? Direction::inverse : Direction::forward,
        stream);
      CUDA_CHECK(cudaGetLastError());
  }

  template<typename T>
  static void ntt_zkp_coset_step3(
      T* self,
      T* params,
      int lg_domain_size,
      bool is_intt,
      bool is_coset, int64_t numel,
      int stage,
      int chunk_id,
      cudaStream_t stream = (cudaStream_t)0) {
    auto len = numel / fr_LIMBS;
    uint32_t lg_chunk_size = log2(len);
    // assert(len == 1 << lg_domain_size);
    auto L1 = WINDOW_NUM * WINDOW_SIZE;
    auto L2 = 2 * L1;
    auto L3 = L2 + (32 + 64 + 128 + 256 + 512);
    auto L4 = L3 + (64 * 64 + 4096 * 64 + 128 * 128 + 256 * 256 + 512 * 512);

    auto pt_ptr = params;
    auto pggp_ptr = params + L1;
    auto rp_ptr = params + L2;
    auto rpm_ptr = params + L3;
    auto size_inverse_ptr = params + L4;
    
    NTT_LDE_step3(
        self,
        pt_ptr,
        rp_ptr,
        rpm_ptr,
        pggp_ptr,
        size_inverse_ptr,
        lg_domain_size,
        lg_chunk_size,
        stage,
        chunk_id,
        InputOutputOrder::NN,
        is_intt ? Direction::inverse : Direction::forward,
        stream);
      CUDA_CHECK(cudaGetLastError());
  }

  template<typename T>
  static void ntt_kcolumn_step1(
      T* self,
      T* params,
      int lg_domain_size,
      int i_size,
      int k,
      int* stage,
      bool is_intt,
      bool is_coset, int64_t numel,
      cudaStream_t stream = (cudaStream_t)0) {
    auto len = numel / fr_LIMBS;
    uint32_t lg_chunk_size = log2(len);
    // assert(len == 1 << lg_domain_size);
    auto L1 = WINDOW_NUM * WINDOW_SIZE;
    auto L2 = 2 * L1;
    auto L3 = L2 + (32 + 64 + 128 + 256 + 512);
    auto L4 = L3 + (64 * 64 + 4096 * 64 + 128 * 128 + 256 * 256 + 512 * 512);

    auto pt_ptr = params;
    auto pggp_ptr = params + L1;
    auto rp_ptr = params + L2;
    auto rpm_ptr = params + L3;
    auto size_inverse_ptr = params + L4;
    
    ntt_kcolumn_step1(
      self,
      pt_ptr,
      rp_ptr,
      rpm_ptr,
      pggp_ptr,
      size_inverse_ptr,
      lg_domain_size,
      i_size,
      k,
      stage,
      is_intt ? Direction::inverse : Direction::forward,
      stream);
      CUDA_CHECK(cudaGetLastError());
  }

  template<typename T>
  static void ntt_kcolumn_step2(
      T* self,
      T* params,
      int lg_domain_size,
      int j_size,
      int k,
      int* stage,
      bool is_intt,
      bool is_coset, int64_t numel,
      cudaStream_t stream = (cudaStream_t)0) {
    auto len = numel / fr_LIMBS;
    uint32_t lg_chunk_size = log2(len);
    // assert(len == 1 << lg_domain_size);
    auto L1 = WINDOW_NUM * WINDOW_SIZE;
    auto L2 = 2 * L1;
    auto L3 = L2 + (32 + 64 + 128 + 256 + 512);
    auto L4 = L3 + (64 * 64 + 4096 * 64 + 128 * 128 + 256 * 256 + 512 * 512);

    auto pt_ptr = params;
    auto pggp_ptr = params + L1;
    auto rp_ptr = params + L2;
    auto rpm_ptr = params + L3;
    auto size_inverse_ptr = params + L4;
    
    ntt_kcolumn_step2(
        self,
        pt_ptr,
        rp_ptr,
        rpm_ptr,
        pggp_ptr,
        size_inverse_ptr,
        lg_domain_size,
        j_size,
        k,
        stage,
        is_intt ? Direction::inverse : Direction::forward,
        stream);
      CUDA_CHECK(cudaGetLastError());
  }

  void ntt_zkp_bitrev_cuda_(
    SyncedMemory inout,
    int lg_domain_size,
    bool is_intt,
    cudaStream_t stream) {

  void* inout_ = inout.mutable_gpu_data();
  int64_t numel = inout.size()/sizeof(uint64_t);

  ntt_zkp_bitrev(reinterpret_cast<fr*>(inout_), lg_domain_size, is_intt, numel, stream);
  }

  void ntt_zkp_coset_init_cuda(
    SyncedMemory inout,
    SyncedMemory params,
    int lg_domain_size,
    bool is_intt,
    int chunk_id,
    cudaStream_t stream) {

  void* inout_ = inout.mutable_gpu_data();
  void* params_gpu = params.mutable_gpu_data();
  int64_t numel = inout.size()/sizeof(uint64_t);

  ntt_zkp_coset_init(reinterpret_cast<fr*>(inout_), reinterpret_cast<fr*>(params_gpu), lg_domain_size, is_intt, numel, chunk_id, stream);
  }

  void ntt_zkp_coset_init_with_bitrev_cuda(
    SyncedMemory inout,
    SyncedMemory params,
    int lg_domain_size,
    bool is_intt,
    cudaStream_t stream) {

  void* inout_ = inout.mutable_gpu_data();
  void* params_gpu = params.mutable_gpu_data();
  int64_t numel = inout.size()/sizeof(uint64_t);

  ntt_zkp_coset_init_with_bitrev(reinterpret_cast<fr*>(inout_), reinterpret_cast<fr*>(params_gpu), lg_domain_size, is_intt, numel, stream);
  }

  void ntt_zkp_coset_step1_cuda(
    SyncedMemory inout,
    SyncedMemory params,
    int lg_domain_size,
    bool is_intt,
    int stage,
    int chunk_id,
    cudaStream_t stream) {

  void* inout_ = inout.mutable_gpu_data();
  void* params_gpu = params.mutable_gpu_data();
  int64_t numel = inout.size()/sizeof(uint64_t);

  ntt_zkp_coset_step1(reinterpret_cast<fr*>(inout_), reinterpret_cast<fr*>(params_gpu), lg_domain_size, is_intt, true, numel, stage, chunk_id, stream);
  }

  void ntt_zkp_coset_step2_cuda(
    SyncedMemory input,
    SyncedMemory params,
    int lg_domain_size,
    bool is_intt,
    int stage,
    int chunk_id,
    cudaStream_t stream) {
  void* in_gpu = input.mutable_gpu_data();
  void* params_gpu = params.mutable_gpu_data();
  int64_t numel = input.size()/sizeof(uint64_t);

  ntt_zkp_coset_step2(reinterpret_cast<fr*>(in_gpu), reinterpret_cast<fr*>(params_gpu), lg_domain_size, is_intt, true, numel, stage, chunk_id, stream);
  }

  void ntt_zkp_coset_step3_cuda(
    SyncedMemory input,
    SyncedMemory params,
    int lg_domain_size,
    bool is_intt,
    int stage,
    int chunk_id,
    cudaStream_t stream) {
  void* in_gpu = input.mutable_gpu_data();
  void* params_gpu = params.mutable_gpu_data();
  int64_t numel = input.size()/sizeof(uint64_t);
      
  ntt_zkp_coset_step3(reinterpret_cast<fr*>(in_gpu), reinterpret_cast<fr*>(params_gpu), lg_domain_size, is_intt, true, numel, stage, chunk_id, stream);
  }

  void ntt_kcolumn_step1_cuda(
    SyncedMemory input,
    SyncedMemory params,
    int lg_domain_size,
    int i_size,
    int k,
    bool is_intt,
    int* stage,
    cudaStream_t stream) {
  void* in_gpu = input.mutable_gpu_data();
  void* params_gpu = params.mutable_gpu_data();
  int64_t numel = input.size()/sizeof(uint64_t);
      
  ntt_kcolumn_step1(reinterpret_cast<fr*>(in_gpu), reinterpret_cast<fr*>(params_gpu), lg_domain_size, i_size, k, stage, is_intt, false, numel, stream);
  }

  void ntt_kcolumn_step1_cuda_raw(
    uint64_t* input,
    SyncedMemory params,
    int64_t numel,
    int lg_domain_size,
    int i_size,
    int k,
    bool is_intt,
    int* stage,
    cudaStream_t stream) {

  void* params_gpu = params.mutable_gpu_data();
      
  ntt_kcolumn_step1(reinterpret_cast<fr*>(input), reinterpret_cast<fr*>(params_gpu), lg_domain_size, i_size, k, stage, is_intt, false, numel, stream);
  }

  void ntt_kcolumn_step2_cuda(
    SyncedMemory input,
    SyncedMemory params,
    int lg_domain_size,
    int j_size,
    int k,
    bool is_intt,
    int* stage,
    cudaStream_t stream) {
  void* in_gpu = input.mutable_gpu_data();
  void* params_gpu = params.mutable_gpu_data();
  int64_t numel = input.size()/sizeof(uint64_t);
      
  ntt_kcolumn_step2(reinterpret_cast<fr*>(in_gpu), reinterpret_cast<fr*>(params_gpu), lg_domain_size, j_size, k, stage, is_intt, false, numel, stream);
  }

}