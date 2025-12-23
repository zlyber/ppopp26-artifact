#pragma once

#include <cuda.h>
#include <assert.h>
#include <iostream>
#include "../../../../mont/cuda/curve_def.cuh"
#include "algorithm.cuh"
#include "kernels.cuh"
namespace cuda{
enum class InputOutputOrder { NN, NR, RN, RR };
enum class Direction { forward, inverse };
enum class Type { standard, coset };
enum class Algorithm { GS, CT };

template <typename fr_t>
void bit_rev(fr_t* d_out, const fr_t* d_inp, uint32_t lg_domain_size, cudaStream_t stream = (cudaStream_t)0) {
  assert(lg_domain_size <= MAX_LG_DOMAIN_SIZE);
  size_t domain_size = (size_t)1 << lg_domain_size;

  if (domain_size <= WARP_SZ)
    bit_rev_permutation<<<1, domain_size, 0, stream>>>(d_out, d_inp, lg_domain_size);
  else if (d_out == d_inp)
    bit_rev_permutation<<<domain_size / WARP_SZ, WARP_SZ, 0, stream>>>(
        d_out, d_inp, lg_domain_size);
  else if (domain_size < 1024)
    bit_rev_permutation_aux<<<1, domain_size / 8, domain_size * sizeof(fr_t), stream>>>(
        d_out, d_inp, lg_domain_size);
  else
    bit_rev_permutation_aux<<<
        domain_size / 1024,
        1024 / 8,
        1024 * sizeof(fr_t),
        stream>>>(d_out, d_inp, lg_domain_size);
}


template <typename fr_t>
void LDE_powers(
    fr_t* inout,
    fr_t* pggp,
    bool bitrev,
    uint32_t lg_domain_size,
    uint32_t lg_chunk_size,
    uint32_t lg_blowup,
    bool ext_pow = false) {
  size_t chunk_size = (size_t)1 << lg_chunk_size;
  size_t sharedMemorySize = chunk_size * sizeof(fr_t);
  if (chunk_size < WARP_SZ)
    LDE_distribute_powers<<<1, chunk_size, sharedMemorySize>>>(
      inout, lg_blowup, lg_domain_size, 0, bitrev, pggp, ext_pow);
  else if (chunk_size < 512)
    LDE_distribute_powers<<<chunk_size / WARP_SZ, WARP_SZ, sharedMemorySize>>>(
        inout, lg_blowup, lg_domain_size, 0, bitrev, pggp, ext_pow);
  else
    LDE_distribute_powers<<<chunk_size / 512, 512, sharedMemorySize>>>(
        inout, lg_blowup, lg_domain_size, 0, bitrev, pggp, ext_pow);
}

template <typename fr_t>
void LDE_powers_chunk(
    fr_t* in,
    fr_t* pggp,
    bool bitrev,
    uint32_t lg_domain_size,
    uint32_t lg_chunk_size,
    uint32_t lg_blowup,
    int chunk_id,
    cudaStream_t stream,
    bool ext_pow = false) {
  size_t chunk_size = (size_t)1 << lg_chunk_size;
  size_t offset = chunk_id * chunk_size;
  if (chunk_size < WARP_SZ)
    LDE_distribute_powers<<<1, chunk_size, 0, stream>>>(
      in, lg_blowup, lg_domain_size, offset, bitrev, pggp, ext_pow);
  else if (chunk_size < 512)
    LDE_distribute_powers<<<chunk_size / WARP_SZ, WARP_SZ, 0, stream>>>(
        in, lg_blowup, lg_domain_size, offset, bitrev, pggp, ext_pow);
  else {
    LDE_distribute_powers<<<chunk_size / 512, 512, 0, stream>>>(
        in, lg_blowup, lg_domain_size, offset, bitrev, pggp, ext_pow);
  }
}


template <typename fr_t>
void LDE_powers_and_pad(
    fr_t* in,
    fr_t* out,
    fr_t* pggp,
    bool bitrev,
    uint32_t lg_domain_size,
    uint32_t lg_chunk_size,
    uint32_t lg_blowup,
    int chunk_id,
    int lambda,
    cudaStream_t stream,
    bool ext_pow = false) {
  size_t chunk_size = (size_t)1 << lg_chunk_size;
  size_t offset = chunk_id * chunk_size;
  if (chunk_size < WARP_SZ)
    LDE_distribute_powers<<<1, chunk_size, 0, stream>>>(
      in, lg_blowup, lg_domain_size, offset, bitrev, pggp, ext_pow);
  else if (chunk_size < 512)
    LDE_distribute_powers<<<chunk_size / WARP_SZ, WARP_SZ, 0, stream>>>(
        in, lg_blowup, lg_domain_size, offset, bitrev, pggp, ext_pow);
  else {
    LDE_distribute_powers<<<chunk_size / 512, 512, 0, stream>>>(
        in, lg_blowup, lg_domain_size, offset, bitrev, pggp, ext_pow);
    pad_and_transpose<<<chunk_size / 512, 512, 0, stream>>>(in, out, lambda);
  }
}

template <typename fr_t>
void NTT_internal(
    fr_t* d_inout,
    fr_t* partial_twiddles,
    fr_t* radix_twiddles,
    fr_t* radix_middles,
    fr_t* partial_group_gen_powers,
    fr_t* Domain_size_inverse,
    uint32_t lg_domain_size,
    InputOutputOrder order,
    Direction direction,
    cudaStream_t stream = (cudaStream_t)0,
    bool coset_ext_pow = false) {
  // Pick an NTT algorithm based on the input order and the desired output
  // order of the data. In certain cases, bit reversal can be avoided which
  // results in a considerable performance gain.

  const bool intt = direction == Direction::inverse;
  bool bitrev;
  Algorithm algorithm;

  switch (order) {
    case InputOutputOrder::NN:
      bit_rev(d_inout, d_inout, lg_domain_size, stream);
      bitrev = true;
      algorithm = Algorithm::CT;
      break;
    case InputOutputOrder::NR:
      bitrev = false;
      algorithm = Algorithm::GS;
      break;
    case InputOutputOrder::RN:
      bitrev = true;
      algorithm = Algorithm::CT;
      break;
    case InputOutputOrder::RR:
      bitrev = true;
      algorithm = Algorithm::GS;
      break;
    default:
      assert(false);
  }

  CT_NTT(
      d_inout,
      lg_domain_size,
      intt,
      partial_twiddles,
      radix_twiddles,
      radix_middles,
      partial_group_gen_powers,
      Domain_size_inverse,
      stream);


  if (order == InputOutputOrder::RR)
    bit_rev(d_inout, d_inout, lg_domain_size, stream);
}

template <typename fr_t>
void NTT_bitrev(
    fr_t* d_inout,
    uint32_t lg_domain_size,
    uint32_t lg_chunk_size,
    InputOutputOrder order,
    Direction direction,
    cudaStream_t stream,
    bool coset_ext_pow = false) {
  // Pick an NTT algorithm based on the input order and the desired output
  // order of the data. In certain cases, bit reversal can be avoided which
  // results in a considerable performance gain.

  const bool intt = direction == Direction::inverse;
  bool bitrev;
  Algorithm algorithm;

  switch (order) {
    case InputOutputOrder::NN:
      bit_rev(d_inout, d_inout, lg_domain_size, stream);
      bitrev = true;
      algorithm = Algorithm::CT;
      break;
    case InputOutputOrder::NR:
      bitrev = false;
      algorithm = Algorithm::GS;
      break;
    case InputOutputOrder::RN:
      bitrev = true;
      algorithm = Algorithm::CT;
      break;
    case InputOutputOrder::RR:
      bitrev = true;
      algorithm = Algorithm::GS;
      break;
    default:
      assert(false);
  }

}

template <typename fr_t>
void NTT_LDE_init(
    fr_t* d_inout,
    fr_t* partial_twiddles,
    fr_t* radix_twiddles,
    fr_t* radix_middles,
    fr_t* partial_group_gen_powers,
    fr_t* Domain_size_inverse,
    uint32_t lg_domain_size,
    uint32_t lg_chunk_size,
    int chunk_id,
    InputOutputOrder order,
    Direction direction,
    cudaStream_t stream,
    bool coset_ext_pow = false) {
  // Pick an NTT algorithm based on the input order and the desired output
  // order of the data. In certain cases, bit reversal can be avoided which
  // results in a considerable performance gain.

  const bool intt = direction == Direction::inverse;
  bool bitrev;
  Algorithm algorithm;

  switch (order) {
    case InputOutputOrder::NN:
      bit_rev(d_inout, d_inout, lg_chunk_size);
      bitrev = true;
      algorithm = Algorithm::CT;
      break;
    case InputOutputOrder::NR:
      bitrev = false;
      algorithm = Algorithm::GS;
      break;
    case InputOutputOrder::RN:
      bitrev = true;
      algorithm = Algorithm::CT;
      break;
    case InputOutputOrder::RR:
      bitrev = true;
      algorithm = Algorithm::GS;
      break;
    default:
      assert(false);
  }

  if (!intt)
    LDE_powers_chunk(
      d_inout,
      partial_group_gen_powers,
      bitrev,
      lg_domain_size,
      lg_chunk_size,
      0,
      chunk_id,
      stream,
      coset_ext_pow);
  else
    LDE_powers_chunk(
      d_inout,
      partial_group_gen_powers,
      !bitrev,
      lg_domain_size,
      lg_chunk_size,
      0,
      chunk_id,
      stream,
      coset_ext_pow);
}


template <typename fr_t>
void NTT_LDE_step1(
    fr_t* d_inout,
    fr_t* partial_twiddles,
    fr_t* radix_twiddles,
    fr_t* radix_middles,
    fr_t* partial_group_gen_powers,
    fr_t* Domain_size_inverse,
    uint32_t lg_domain_size,
    uint32_t lg_chunk_size,
    int stage,
    int chunk_id,
    InputOutputOrder order,
    Direction direction,
    cudaStream_t stream = (cudaStream_t)0,
    bool coset_ext_pow = false) {
  // Pick an NTT algorithm based on the input order and the desired output
  // order of the data. In certain cases, bit reversal can be avoided which
  // results in a considerable performance gain.

  const bool intt = direction == Direction::inverse;
  bit_rev(d_inout, d_inout, lg_chunk_size, stream);
  CUDA_CHECK(cudaGetLastError());
  // int step = lg_chunk_size / 2;
  // int rem = lg_chunk_size % 2;
  int step = lg_domain_size / 3;
  int rem = lg_domain_size % 3;
  LDE_powers_chunk(
    d_inout,
    partial_group_gen_powers,
    true,
    lg_domain_size,
    lg_chunk_size,
    0,
    chunk_id,
    stream,
    coset_ext_pow);
  
  CTkernel_(
      step,
      d_inout,
      partial_twiddles,
      radix_twiddles,
      radix_middles,
      partial_group_gen_powers,
      Domain_size_inverse,
      lg_domain_size,
      lg_chunk_size,
      intt,
      stage,
      chunk_id,
      stream);
  CUDA_CHECK(cudaGetLastError());
  CTkernel_(
    step,
    d_inout,
    partial_twiddles,
    radix_twiddles,
    radix_middles,
    partial_group_gen_powers,
    Domain_size_inverse,
    lg_domain_size,
    lg_chunk_size,
    intt,
    stage + step,
    chunk_id,
    stream);
  CUDA_CHECK(cudaGetLastError());
}

template <typename fr_t>
void NTT_LDE_init_and_step1(
    fr_t* d_in,
    fr_t* d_out,
    fr_t* partial_twiddles,
    fr_t* radix_twiddles,
    fr_t* radix_middles,
    fr_t* partial_group_gen_powers,
    fr_t* Domain_size_inverse,
    uint32_t lg_domain_size,
    uint32_t lg_chunk_size,
    int stage,
    int chunk_id,
    int lambda,
    InputOutputOrder order,
    Direction direction,
    cudaStream_t stream,
    bool coset_ext_pow = false) {
  // Pick an NTT algorithm based on the input order and the desired output
  // order of the data. In certain cases, bit reversal can be avoided which
  // results in a considerable performance gain.

  const bool intt = direction == Direction::inverse;
  bool bitrev;
  Algorithm algorithm;

  switch (order) {
    case InputOutputOrder::NN:
      bit_rev(d_in, d_in, lg_chunk_size, stream);
      bitrev = true;
      algorithm = Algorithm::CT;
      break;
    case InputOutputOrder::NR:
      bitrev = false;
      algorithm = Algorithm::GS;
      break;
    case InputOutputOrder::RN:
      bitrev = true;
      algorithm = Algorithm::CT;
      break;
    case InputOutputOrder::RR:
      bitrev = true;
      algorithm = Algorithm::GS;
      break;
    default:
      assert(false);
  }
  
  if (!intt)
    LDE_powers_and_pad(
      d_in,
      d_out,
      partial_group_gen_powers,
      bitrev,
      lg_domain_size,
      lg_chunk_size,
      0,
      chunk_id,
      lambda,
      stream,
      coset_ext_pow);

  int step = lg_domain_size / 3;
  int rem = lg_domain_size % 3;

  // int step = 8;
  // int rem = 8;
  CTkernel_(
      step,
      d_out,
      partial_twiddles,
      radix_twiddles,
      radix_middles,
      partial_group_gen_powers,
      Domain_size_inverse,
      lg_domain_size,
      lg_chunk_size + lambda,
      intt,
      stage,
      chunk_id,
      stream);

  CTkernel_(
      step,
      d_out,
      partial_twiddles,
      radix_twiddles,
      radix_middles,
      partial_group_gen_powers,
      Domain_size_inverse,
      lg_domain_size,
      lg_chunk_size + lambda,
      intt,
      stage+step,
      chunk_id,
      stream);

}

template <typename fr_t>
void NTT_LDE_step2(
    fr_t* d_inout,
    fr_t* partial_twiddles,
    fr_t* radix_twiddles,
    fr_t* radix_middles,
    fr_t* partial_group_gen_powers,
    fr_t* Domain_size_inverse,
    uint32_t lg_domain_size,
    uint32_t lg_chunk_size,
    int stage,
    int chunk_id,
    InputOutputOrder order,
    Direction direction,
    cudaStream_t stream = (cudaStream_t)0,
    bool coset_ext_pow = false) {
  // Pick an NTT algorithm based on the input order and the desired output
  // order of the data. In certain cases, bit reversal can be avoided which
  // results in a considerable performance gain.

  const bool intt = direction == Direction::inverse;
  bool bitrev = true;

  bit_rev(d_inout, d_inout, lg_chunk_size, stream);
  // 20 < lg_domain_size < 30
  int step = lg_domain_size / 3;
  int rem = lg_domain_size % 3;
  
  CTkernel_(
    step + (lg_domain_size == 29 ? 1 : 0),
    d_inout,
    partial_twiddles,
    radix_twiddles,
    radix_middles,
    partial_group_gen_powers,
    Domain_size_inverse,
    lg_domain_size,
    lg_chunk_size,
    intt,
    stage,
    chunk_id,
    stream);
  }


template <typename fr_t>
void NTT_LDE_step3(
    fr_t* d_in,
    fr_t* d_out,
    fr_t* partial_twiddles,
    fr_t* radix_twiddles,
    fr_t* radix_middles,
    fr_t* partial_group_gen_powers,
    fr_t* Domain_size_inverse,
    uint32_t lg_domain_size,
    uint32_t lg_chunk_size,
    int stage,
    int chunk_id,
    int lambda,
    InputOutputOrder order,
    Direction direction,
    cudaStream_t stream = (cudaStream_t)0,
    bool coset_ext_pow = false) {
  // Pick an NTT algorithm based on the input order and the desired output
  // order of the data. In certain cases, bit reversal can be avoided which
  // results in a considerable performance gain.

  const bool intt = direction == Direction::inverse;
  bool bitrev = true;

  // 20 < lg_domain_size < 30
  int step = lg_domain_size / 3;
  int rem = lg_domain_size % 3;
      
  CTkernel_lambda(
      step,
      d_in,
      d_out,
      partial_twiddles,
      radix_twiddles,
      radix_middles,
      partial_group_gen_powers,
      Domain_size_inverse,
      lg_domain_size,
      lg_chunk_size,
      intt,
      stage,
      chunk_id,
      lambda,
      stream);

}

template <typename fr_t>
void ntt_step1(fr_t* inout, fr_t* partial_twiddles, fr_t* radix_twiddles, fr_t* radix_middles,
              fr_t* partial_group_gen_powers, fr_t* Domain_size_inverse, int lg_domain_size, int lg_chunk_size, int stage, int chunk_id,
              InputOutputOrder ntt_order, Direction ntt_direction, cudaStream_t stream = (cudaStream_t)0) 
{
  const bool intt = ntt_direction == Direction::inverse;
  bool bitrev;

  switch (ntt_order) {
    case InputOutputOrder::NN:
      bitrev = true;
      break;
    case InputOutputOrder::NR:
      bitrev = false;
      break;
    case InputOutputOrder::RN:
      bitrev = true;
      break;
    case InputOutputOrder::RR:
      bitrev = true;
      break;
    default:
      assert(false);
  }
  int step = lg_domain_size / 3;
  int rem = lg_domain_size % 3;

  CTkernel_(
    step,
    inout,
    partial_twiddles,
    radix_twiddles,
    radix_middles,
    partial_group_gen_powers,
    Domain_size_inverse,
    lg_domain_size,
    lg_chunk_size,
    intt,
    stage,
    chunk_id,
    stream);
    CUDA_CHECK(cudaGetLastError());
}

template <typename fr_t>
void ntt_step2(fr_t* inout, fr_t* partial_twiddles, fr_t* radix_twiddles, fr_t* radix_middles,
              fr_t* partial_group_gen_powers, fr_t* Domain_size_inverse, int lg_domain_size, int lg_chunk_size, int stage, int chunk_id,
              InputOutputOrder ntt_order, Direction ntt_direction, cudaStream_t stream = (cudaStream_t)0) 
{
  const bool intt = ntt_direction == Direction::inverse;

  int step = lg_domain_size / 3;
  int rem = lg_domain_size % 3;

  CTkernel_(
    step + (lg_domain_size == 29 ? 1 : 0),
    inout,
    partial_twiddles,
    radix_twiddles,
    radix_middles,
    partial_group_gen_powers,
    Domain_size_inverse,
    lg_domain_size,
    lg_chunk_size,
    intt,
    stage,
    chunk_id,
    stream);
  CUDA_CHECK(cudaGetLastError());
}

template <typename fr_t>
void ntt_step3(fr_t* inout, fr_t* partial_twiddles, fr_t* radix_twiddles, fr_t* radix_middles,
              fr_t* partial_group_gen_powers, fr_t* Domain_size_inverse, int lg_domain_size, int lg_chunk_size, int stage, int chunk_id,
              InputOutputOrder ntt_order, Direction ntt_direction, cudaStream_t stream = (cudaStream_t)0) 
{
  const bool intt = ntt_direction == Direction::inverse;

  int step = lg_domain_size / 3;
  int rem = lg_domain_size % 3;

  CTkernel_(
    step + (lg_domain_size == 29 ? 1 : rem),
    inout,
    partial_twiddles,
    radix_twiddles,
    radix_middles,
    partial_group_gen_powers,
    Domain_size_inverse,
    lg_domain_size,
    lg_domain_size,
    intt,
    stage,
    chunk_id,
    stream);
}

template <typename fr_t>
void compute_ntt(
    size_t device_id,
    fr_t* inout,
    fr_t* partial_twiddles,
    fr_t* radix_twiddles,
    fr_t* radix_middles,
    fr_t* partial_group_gen_powers,
    fr_t* Domain_size_inverse,
    uint32_t lg_domain_size,
    InputOutputOrder ntt_order,
    Direction ntt_direction,
    cudaStream_t stream = (cudaStream_t)0) {
  assert(lg_domain_size != 0);

  NTT_internal(
      inout,
      partial_twiddles,
      radix_twiddles,
      radix_middles,
      partial_group_gen_powers,
      Domain_size_inverse,
      lg_domain_size,
      ntt_order,
      ntt_direction,
      stream);
}

template <typename fr_t>
void ntt_internal_step1(
    fr_t* inout,
    fr_t* partial_twiddles,
    fr_t* radix_twiddles,
    fr_t* radix_middles,
    fr_t* partial_group_gen_powers,
    fr_t* Domain_size_inverse,
    uint32_t lg_domain_size,
    int* stage,
    InputOutputOrder ntt_order,
    Direction ntt_direction,
    cudaStream_t stream = (cudaStream_t)0) {
  assert(lg_domain_size != 0);

  const bool intt = ntt_direction == Direction::inverse;
  bool bitrev;
  Algorithm algorithm;

  switch (ntt_order) {
    case InputOutputOrder::NN:
      bit_rev(inout, inout, lg_domain_size, stream);
      bitrev = true;
      algorithm = Algorithm::CT;
      break;
    case InputOutputOrder::NR:
      bitrev = false;
      algorithm = Algorithm::GS;
      break;
    case InputOutputOrder::RN:
      bitrev = true;
      algorithm = Algorithm::CT;
      break;
    case InputOutputOrder::RR:
      bitrev = true;
      algorithm = Algorithm::GS;
      break;
    default:
      assert(false);
  }
  uint32_t* d_map = nullptr;
  int step = lg_domain_size / 3;
  int rem = lg_domain_size % 3;
  CTkernel1(
      step,
      inout,
      d_map,
      partial_twiddles,
      radix_twiddles,
      radix_middles,
      partial_group_gen_powers,
      Domain_size_inverse,
      lg_domain_size,
      intt,
      stage,
      stream);
}

template <typename fr_t>
void ntt_internal_step2(
    fr_t* inout,
    fr_t* partial_twiddles,
    fr_t* radix_twiddles,
    fr_t* radix_middles,
    fr_t* partial_group_gen_powers,
    fr_t* Domain_size_inverse,
    uint32_t lg_domain_size,
    int* stage,
    InputOutputOrder ntt_order,
    Direction ntt_direction,
    cudaStream_t stream = (cudaStream_t)0) {

  const bool intt = ntt_direction == Direction::inverse;
  bool bitrev = true;
  uint32_t* map_ = nullptr;
  int step = lg_domain_size / 3;
  int rem = lg_domain_size % 3;
  CTkernel1(
      step + (lg_domain_size == 29 ? 1 : 0),
      inout,
      map_,
      partial_twiddles,
      radix_twiddles,
      radix_middles,
      partial_group_gen_powers,
      Domain_size_inverse,
      lg_domain_size,
      intt,
      stage,
      stream);
}

template <typename fr_t>
void ntt_internal_step3(
    fr_t* inout,
    fr_t* partial_twiddles,
    fr_t* radix_twiddles,
    fr_t* radix_middles,
    fr_t* partial_group_gen_powers,
    fr_t* Domain_size_inverse,
    uint32_t lg_domain_size,
    int* stage,
    InputOutputOrder ntt_order,
    Direction ntt_direction,
    cudaStream_t stream = (cudaStream_t)0) {

  const bool intt = ntt_direction == Direction::inverse;
  bool bitrev = true;
  uint32_t* map_ = nullptr;
  int step = lg_domain_size / 3;
  int rem = lg_domain_size % 3;
  CTkernel1(
    step + (lg_domain_size == 29 ? 1 : rem),
    inout,
    map_,
    partial_twiddles,
    radix_twiddles,
    radix_middles,
    partial_group_gen_powers,
    Domain_size_inverse,
    lg_domain_size,
    intt,
    stage,
    stream);
}

template <typename fr_t>
void ntt_kcolumn_step1(
    fr_t* inout,
    fr_t* partial_twiddles,
    fr_t* radix_twiddles,
    fr_t* radix_middles,
    fr_t* partial_group_gen_powers,
    fr_t* Domain_size_inverse,
    uint32_t lg_domain_size,
    uint32_t i_size,
    uint32_t k,
    int* stage,
    Direction ntt_direction,
    cudaStream_t stream = (cudaStream_t)0) {
  assert(lg_domain_size != 0);

  const bool intt = ntt_direction == Direction::inverse;
  bool bitrev;
  Algorithm algorithm;
  bitrev = true;
  algorithm = Algorithm::CT;
    
  uint32_t* d_map = nullptr;
  int step = lg_domain_size / 3;
  int rem = lg_domain_size % 3;
  // int step = (i_size +k) / 2;
  // int rem = (i_size +k) % 2;
  bit_rev(inout, inout, i_size + k, stream);

  CTkernel(
      step,
      inout,
      partial_twiddles,
      radix_twiddles,
      radix_middles,
      partial_group_gen_powers,
      Domain_size_inverse,
      i_size + k,
      intt,
      stage,
      stream);

  CTkernel(
      step + (lg_domain_size == 29 ? 1 : 0),
      inout,
      partial_twiddles,
      radix_twiddles,
      radix_middles,
      partial_group_gen_powers,
      Domain_size_inverse,
      i_size + k,
      intt,
      stage,
      stream);

}

template <typename fr_t>
void ntt_kcolumn_step2(
  fr_t* inout,
  fr_t* partial_twiddles,
  fr_t* radix_twiddles,
  fr_t* radix_middles,
  fr_t* partial_group_gen_powers,
  fr_t* Domain_size_inverse,
  uint32_t lg_domain_size,
  uint32_t j_size,
  int k,
  int* stage,
  Direction ntt_direction,
  cudaStream_t stream = (cudaStream_t)0) {
  assert(lg_domain_size != 0);

  const bool intt = ntt_direction == Direction::inverse;
  bool bitrev;
  Algorithm algorithm;

  bitrev = true;
  algorithm = Algorithm::CT;
      
  int step = lg_domain_size / 3;
  int rem = lg_domain_size % 3;
  bit_rev(inout, inout, j_size + k, stream);
  CUDA_CHECK(cudaGetLastError());
  CTkernel(
      step + (lg_domain_size == 29 ? 1 : rem),
      inout,
      partial_twiddles,
      radix_twiddles,
      radix_middles,
      partial_group_gen_powers,
      Domain_size_inverse,
      j_size + k,
      intt,
      stage,
      stream);
  CUDA_CHECK(cudaGetLastError());
}

}