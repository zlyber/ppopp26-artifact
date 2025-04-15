#include "zksnark.cuh"
#include "ec/jacobian_t.hpp"
#include "ec/xyzz_t.hpp"
#include "sppark_msm/pippenger.cuh"

namespace cuda{
  int lg2(size_t n) {
    int ret = 0;
    while (n >>= 1)
      ret++;
    return ret;
  }

static void mult_pippenger_inf(
      fq* self,
      fq* points,
      fr* scalars,
      uint64_t* workspace,
      int64_t smcount,
      int64_t blob_u64,
      int64_t numel,
      cudaStream_t stream = (cudaStream_t)0) {
      using point_t = jacobian_t<fq>;
      using bucket_t = xyzz_t<fq>;
      using bucket_h = bucket_t::mem_t;
      using affine_t = bucket_t::affine_t;
      auto npoints = numel / (fq_LIMBS * 2);
      auto ffi_affine_sz = sizeof(affine_t); // affine mode (X,Y)
      auto bucket_ptr = reinterpret_cast<bucket_h*>(workspace);
      auto temp_ptr = reinterpret_cast<uint8_t*>(workspace + blob_u64);
      auto self_ptr = reinterpret_cast<bucket_t*>(self);
      auto point_ptr = reinterpret_cast<affine_t*>(points);
      auto scalar_ptr = scalars;
      mult_pippenger<point_t>(
          self_ptr,
          point_ptr,
          npoints,
          scalar_ptr,
          bucket_ptr,
          temp_ptr,
          smcount,
          stream,
          false,
          ffi_affine_sz);
  }

  SyncedMemory msm_zkp_cuda(
      SyncedMemory points,
      SyncedMemory scalars,
      int64_t smcount,
      cudaStream_t stream) {
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

    void* workspace_gpu = workspace.mutable_gpu_data_async(stream);
    void* out_gpu = out.mutable_gpu_data_async(stream);
    void* points_gpu = points.mutable_gpu_data_async(stream);
    void* scalars_gpu = scalars.mutable_gpu_data_async(stream);

    int64_t numel = points.size()/sizeof(uint64_t);

    mult_pippenger_inf(reinterpret_cast<fq*>(out_gpu), reinterpret_cast<fq*>(points_gpu), reinterpret_cast<fr*>(scalars_gpu),
                      reinterpret_cast<uint64_t*>(workspace_gpu), smcount, blob_u64, numel, stream);

    return out;
  }
  
  void msm_zkp_cuda_(
      SyncedMemory points,
      SyncedMemory scalars,
      SyncedMemory workspace,
      SyncedMemory out,
      int64_t smcount,
      cudaStream_t stream) {
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

    void* workspace_gpu = workspace.mutable_gpu_data_async(stream);
    void* out_gpu = out.mutable_gpu_data_async(stream);
    void* points_gpu = points.mutable_gpu_data_async(stream);
    void* scalars_gpu = scalars.mutable_gpu_data_async(stream);

    int64_t numel = points.size()/sizeof(uint64_t);

    mult_pippenger_inf(reinterpret_cast<fq*>(out_gpu), reinterpret_cast<fq*>(points_gpu), reinterpret_cast<fr*>(scalars_gpu),
                      reinterpret_cast<uint64_t*>(workspace_gpu), smcount, blob_u64, numel, stream);

  }

  void msm_zkp_chunk_cuda_(
    SyncedMemory points,
    SyncedMemory scalars,
    SyncedMemory workspace,
    SyncedMemory out,
    int64_t smcount,
    int chunk_id,
    uint64_t npoints,
    cudaStream_t stream) {
  auto wbits = 17;
  // auto npoints = points.size() / (fq_LIMBS * 2 * sizeof(uint64_t));
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

  void* workspace_gpu = workspace.mutable_gpu_data_async(stream);
  void* out_gpu = out.mutable_gpu_data_async(stream);
  void* points_gpu = points.mutable_gpu_data_async(stream);
  void* scalars_gpu = scalars.mutable_gpu_data_async(stream);

  int64_t numel = npoints * 12;

  mult_pippenger_inf(reinterpret_cast<fq*>(out_gpu), reinterpret_cast<fq*>(points_gpu) + chunk_id*npoints, reinterpret_cast<fr*>(scalars_gpu) + chunk_id*npoints,
                    reinterpret_cast<uint64_t*>(workspace_gpu), smcount, blob_u64, numel, stream);

}
}//namespace::cuda