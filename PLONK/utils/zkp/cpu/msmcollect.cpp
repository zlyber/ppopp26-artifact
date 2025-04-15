#include "msmcollect.hpp"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
namespace cpu{
  
  template<typename T>
  static void pippenger_collect(T* self, T* step1res, size_t npoints, int64_t numel) {
      using point_t = jacobian_t<T>;
      using bucket_t = xyzz_t<T>;
      using affine_t = typename bucket_t::affine_t;
      auto wbits = 17;
      if (npoints > 192) {
        wbits = std::min(lg2(npoints + npoints / 2) - 8, 18);
        if (wbits < 10)
          wbits = 10;
      } else if (npoints > 0) {
        wbits = 10;
      }
      uint32_t nbits = fr_BITS;
      uint32_t nwins = (nbits - 1) / wbits + 1;
      uint32_t lenofres = nwins * MSM_NTHREADS / 1 * 2;
      uint32_t lenofone = numel - lenofres;

      auto self_ptr =
          reinterpret_cast<point_t*>(self);
      auto res_ptr =
          reinterpret_cast<bucket_t*>(step1res);
      auto ones_ptr =
          reinterpret_cast<bucket_t*>(step1res) + lenofres;
      collect_t<bucket_t, point_t, affine_t, fr>
          msm_collect{npoints};
      msm_collect.collect(self_ptr, res_ptr, ones_ptr, lenofone);
  }

SyncedMemory msm_collect_cpu(SyncedMemory step1res, int64_t npoints) {
    SyncedMemory out(3 * fq_LIMBS, false);
    void* out_cpu = out.mutable_cpu_data();
    void* step1res_cpu = step1res.mutable_cpu_data();
    int64_t numel = step1res.size() / (fq_LIMBS * 4 * sizeof(uint64_t));
    pippenger_collect(static_cast<cpu::fq*>(out_cpu), static_cast<cpu::fq*>(step1res_cpu), npoints, numel);
    return out;
  }
}//namespace::cpu
