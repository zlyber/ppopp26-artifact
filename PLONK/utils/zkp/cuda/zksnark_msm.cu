#include "PLONK/utils/zkp/cuda/zksnark.cuh"
#include "PLONK/utils/zkp/cuda/ec/jacobian_t.hpp"
#include "PLONK/utils/zkp/cuda/ec/xyzz_t.hpp"
#include "PLONK/utils/zkp/cuda/sppark_msm/pippenger.cuh"
namespace cuda{
  int lg2(size_t n) {
    int ret = 0;
    while (n >>= 1)
      ret++;
    return ret;
  }

  template<typename T>
  static void mult_pippenger_inf(
      T* self,
      T* points,
      T* scalars,
      uint64_t* workspace,
      int64_t smcount,
      int64_t blob_u64,
      int64_t numel) {
      using point_t = jacobian_t<T>;
      using bucket_t = xyzz_t<T>;
      using bucket_h = typename bucket_t::mem_t;
      using affine_t = typename bucket_t::affine_t;
      auto npoints = numel / (fq_LIMBS * 2);
      auto ffi_affine_sz = sizeof(affine_t); // affine mode (X,Y)
      auto bucket_ptr = reinterpret_cast<bucket_h*>(workspace);
      auto temp_ptr = reinterpret_cast<uint8_t*>(workspace + blob_u64);
      auto self_ptr = reinterpret_cast<bucket_t*>(self);
      auto point_ptr = reinterpret_cast<affine_t*>(points);
      auto scalar_ptr = reinterpret_cast<typename T::coeff_t*>(scalars);
      mult_pippenger<point_t>(
          self_ptr,
          point_ptr,
          npoints,
          scalar_ptr,
          bucket_ptr,
          temp_ptr,
          smcount,
          false,
          ffi_affine_sz);
  }

  SyncedMemory& msm_zkp_cuda(
      SyncedMemory& points,
      SyncedMemory& scalars,
      int64_t smcount) {
    auto wbits = 17;
    auto npoints = points.size() / (fq_LIMBS * 2 * sizeof(uint64_t));
    if (npoints > 192) {
      wbits = ::std::min(lg2(npoints + npoints / 2) - 8, 18);
      if (wbits < 10)
        wbits = 10;
    } else if (npoints > 0) {
      wbits = 10;
    }
    auto nbits = fr_BITS;
    auto nwins = (nbits - 1) / wbits + 1;
    uint32_t row_sz = 1U << (wbits - 1);
    size_t d_buckets_sz =
        (nwins * row_sz) + (smcount * BATCH_ADD_BLOCK_SIZE / WARP_SZ);
    size_t d_blob_sz =
        d_buckets_sz * sizeof(uint64_t) * fq_LIMBS * 4 +
        (nwins * row_sz * sizeof(uint32_t));
    uint32_t blob_u64 = d_blob_sz / sizeof(uint64_t);
    
    size_t digits_sz = nwins * npoints * sizeof(uint32_t);
    uint32_t temp_sz = npoints * sizeof(uint64_t) * fr_LIMBS + digits_sz;
    uint32_t temp_u64 = temp_sz / sizeof(uint64_t);
    SyncedMemory workspace((blob_u64 + temp_u64) * sizeof(uint64_t));
    SyncedMemory out((nwins * MSM_NTHREADS / 1 * 2 + smcount * BATCH_ADD_BLOCK_SIZE / WARP_SZ) * 4 * fq_LIMBS * sizeof(uint64_t));

    void* workspace_gpu = workspace.mutable_gpu_data();
    void* out_gpu = out.mutable_gpu_data();
    void* points_gpu = points.mutable_gpu_data();
    void* scalars_gpu = scalars.mutable_gpu_data();
    int64_t numel = points.size()/sizeof(uint64_t);

    mult_pippenger_inf(static_cast<fq*>(out_gpu), static_cast<fq*>(points_gpu), static_cast<fq*>(scalars_gpu),
                      static_cast<uint64_t*>(workspace_gpu), smcount, blob_u64, numel);
    return out;
  }
}//namespace::cuda