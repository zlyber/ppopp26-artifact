#include "PLONK/utils/zkp/cuda/zksnark.cuh"
#include "PLONK/utils/zkp/cuda/zksnark_ntt/ntt_kernel/ntt.cuh"
#include "PLONK/utils/zkp/cuda/zksnark_ntt/parameters/parameters.cuh"
namespace cuda{
  // temporarily set device_id to 0, set InputOutputOrder to NN
  // TODO: optimize memory copy for inout data
  template<typename T>
  static void params_zkp_template(
    T* self,
    T* local_params,
    bool is_intt) {
    NTTParameters(is_intt, self, local_params);
  }

  SyncedMemory& params_zkp_cuda(
    int64_t domain_size,
    bool is_intt) {
  auto partial_sz = WINDOW_NUM * WINDOW_SIZE;
  auto S1 = 2 * partial_sz;
  auto S2 = 32 + 64 + 128 + 256 + 512;
  auto S3 = 64 * 64 + 4096 * 64 + 128 * 128 + 256 * 256 + 512 * 512;
  auto S4 = domain_size + 1;

  auto double_roots = 2 * domain_size; // forward and inverse
  auto double_group_gen = 2; // group generator and inverse group generator

  SyncedMemory local_params((double_group_gen + double_roots) * fr_LIMBS * sizeof(uint64_t));
  SyncedMemory params((S1 + S2 + S3 + S4) * fr_LIMBS * sizeof(uint64_t));
  void* local_gpu = local_params.mutable_gpu_data();
  void* params_gpu = params.mutable_gpu_data();
  params_zkp_template(static_cast<fr*>(params_gpu), static_cast<fr*>(params_gpu), is_intt);
  return params;
  }

  template<typename T>
  static void ntt_zkp(
      T* self,
      T* params,
      bool is_intt,
      bool is_coset, int64_t numel) {
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
        is_coset ? Type::coset : Type::standard);
  }

  SyncedMemory& ntt_zkp_cuda(
    SyncedMemory& input,
    SyncedMemory& params,
    bool is_intt,
    bool is_coset) {
  SyncedMemory output(input.size());
  void* out_gpu = output.mutable_gpu_data();
  void* in_gpu = input.mutable_gpu_data();
  cudaMemcpy(
      out_gpu,
      in_gpu,
      input.size(),
      cudaMemcpyDeviceToDevice);
  void* params_gpu = params.mutable_gpu_data();
  int64_t numel = input.size()/sizeof(uint64_t);
  ntt_zkp(static_cast<fr*>(out_gpu), static_cast<fr*>(params_gpu), is_intt, is_coset, numel);
  return output;
  }
}