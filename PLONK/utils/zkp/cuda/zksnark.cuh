#pragma once
#include <math.h>
#include <stdint.h>
#include <assert.h>
#include <stdexcept>
#include <limits>
#include "caffe/syncedmem.hpp"
#include "PLONK/utils/mont/cuda/curve_def.cuh"
#include <iostream>

using namespace caffe;

namespace cuda
{   
    SyncedMemory& compress_cuda(SyncedMemory& f_0, SyncedMemory& f_1, 
                                SyncedMemory& f_2, SyncedMemory& f_3, SyncedMemory& challenge);

    SyncedMemory& compress_cuda_2(SyncedMemory& concatenated_f, SyncedMemory& challenge, int64_t numel);

    SyncedMemory& compute_query_table_cuda(SyncedMemory& padded_q_lookup, 
                                    SyncedMemory& w_l_scalar, SyncedMemory& w_r_scalar, SyncedMemory& w_o_scalar, SyncedMemory& w_4_scalar,
                                    SyncedMemory& t_poly);
    
    SyncedMemory& msm_zkp_cuda(SyncedMemory& points, SyncedMemory& scalars, int64_t smcount);

    SyncedMemory& params_zkp_cuda(int64_t domain_size, bool is_intt);

    SyncedMemory& ntt_zkp_cuda(SyncedMemory& input, SyncedMemory& params, bool is_intt, bool is_coset);
} // namespace cuda
