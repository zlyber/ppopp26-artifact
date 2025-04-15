#pragma once
#include <math.h>
#include <stdint.h>
#include <assert.h>
#include <stdexcept>
#include <limits>
#include "../../../../caffe/interface.hpp"
#include "../../mont/cuda/curve_def.cuh"
#include <iostream>

using namespace caffe;

namespace cuda
{   
    void compress_cuda(SyncedMemory output, SyncedMemory f_0, SyncedMemory f_1, 
                                SyncedMemory f_2, SyncedMemory f_3, SyncedMemory challenge, cudaStream_t stream = (cudaStream_t)0);

    void compress_cuda_2(SyncedMemory concatenated_f, SyncedMemory output, SyncedMemory challenge, int64_t numel, cudaStream_t stream = (cudaStream_t)0);

    void compute_query_table_cuda(SyncedMemory output, SyncedMemory padded_q_lookup, 
                                    SyncedMemory w_l_scalar, SyncedMemory w_r_scalar, SyncedMemory w_o_scalar, SyncedMemory w_4_scalar,
                                    SyncedMemory t_poly, cudaStream_t stream = (cudaStream_t)0);
    SyncedMemory msm_zkp_cuda(
      SyncedMemory points,
      SyncedMemory scalars,
      int64_t smcount,
      cudaStream_t stream = (cudaStream_t)0);

    void msm_zkp_cuda_(
      SyncedMemory points,
      SyncedMemory scalars,
      SyncedMemory workspace,
      SyncedMemory out,
      int64_t smcount, cudaStream_t stream = (cudaStream_t)0);

    void msm_zkp_chunk_cuda_(
      SyncedMemory points,
      SyncedMemory scalars,
      SyncedMemory workspace,
      SyncedMemory out,
      int64_t smcount,
      int chunk_id,
      uint64_t npoints,
      cudaStream_t stream = (cudaStream_t)0);

    SyncedMemory params_zkp_cuda(int64_t domain_size, bool is_intt, cudaStream_t stream = (cudaStream_t)0);

    SyncedMemory ntt_zkp_cuda(SyncedMemory input, SyncedMemory params, bool is_intt, cudaStream_t stream = (cudaStream_t)0);

    void ntt_zkp_bitrev_cuda_(SyncedMemory inout, int lg_domain_size, bool is_intt, cudaStream_t stream = (cudaStream_t)0);

    void ntt_zkp_cuda(SyncedMemory input, SyncedMemory output, SyncedMemory params, bool is_intt, cudaStream_t stream = (cudaStream_t)0);

    void ntt_zkp_cuda_(SyncedMemory inout, SyncedMemory params, bool is_intt, cudaStream_t stream = (cudaStream_t)0);
    
    void ntt_zkp_step1_cuda(SyncedMemory inout, SyncedMemory params, int lg_domain_size, bool is_intt, int stage, int chunk_id, cudaStream_t stream = (cudaStream_t)0); 

    void ntt_kcolumn_step1_cuda(SyncedMemory input, SyncedMemory params, int lg_domain_size, int i_size, int k, bool is_intt, int* stage, cudaStream_t stream = (cudaStream_t)0);
    
    void ntt_kcolumn_step1_cuda_raw(
      uint64_t* input,
      SyncedMemory params,
      int64_t numel,
      int lg_domain_size,
      int i_size,
      int k,
      bool is_intt,
      int* stage,
      cudaStream_t stream = (cudaStream_t)0);

    void ntt_zkp_step2_cuda(SyncedMemory inout, SyncedMemory params, int lg_domain_size, bool is_intt, int stage, int chunk_id, cudaStream_t stream = (cudaStream_t)0); 

    void ntt_kcolumn_step2_cuda(SyncedMemory input, SyncedMemory params, int lg_domain_size, int j_size, int k, bool is_intt, int* stage, cudaStream_t stream = (cudaStream_t)0);

    void ntt_zkp_step3_cuda(SyncedMemory inout, SyncedMemory params, int lg_domain_size, bool is_intt, int stage, int chunk_id, cudaStream_t stream = (cudaStream_t)0); 

    void ntt_zkp_step1_internal_cuda(SyncedMemory input, SyncedMemory output, SyncedMemory params, bool is_intt, int* stage, cudaStream_t stream);

    void ntt_zkp_step2_internal_cuda(SyncedMemory inout, SyncedMemory params, bool is_intt, int* stage, cudaStream_t stream);

    void ntt_zkp_step3_internal_cuda(SyncedMemory inout, SyncedMemory params, bool is_intt, int* stage, cudaStream_t stream);

    void ntt_zkp_coset_init_cuda(SyncedMemory inout, SyncedMemory params, int lg_domain_size, bool is_intt, int stage, cudaStream_t stream = (cudaStream_t)0);

    void ntt_zkp_coset_init_with_bitrev_cuda(SyncedMemory inout, SyncedMemory params, int lg_domain_size, bool is_intt, cudaStream_t stream = (cudaStream_t)0);
    
    void ntt_zkp_coset_step1_cuda(SyncedMemory input, SyncedMemory params, int lg_domain_size, bool is_intt, int stage, int chunk_id, cudaStream_t stream = (cudaStream_t)0);
    
    void ntt_zkp_coset_step2_cuda(SyncedMemory input, SyncedMemory params, int lg_domain_size, bool is_intt, int stage, int chunk_id, cudaStream_t stream = (cudaStream_t)0);

    void ntt_zkp_coset_step3_cuda(SyncedMemory input, SyncedMemory params, int lg_domain_size, bool is_intt, int stage, int chunk_id, cudaStream_t stream = (cudaStream_t)0);

    SyncedMemory make_tensor(SyncedMemory input, uint64_t pad_len);

    void lookup_ratio_step1_cuda_(SyncedMemory h_1, SyncedMemory h_2, SyncedMemory h_1_next, 
      SyncedMemory one, SyncedMemory delta, SyncedMemory epsilon, SyncedMemory result, cudaStream_t stream = (cudaStream_t)0);

    void lookup_ratio_step2_cuda_(SyncedMemory f, SyncedMemory t, SyncedMemory t_next, 
      SyncedMemory one, SyncedMemory delta, SyncedMemory epsilon, SyncedMemory result, cudaStream_t stream = (cudaStream_t)0);
    
    SyncedMemory compute_gate_constraint_allmerge_cuda(SyncedMemory a_val, SyncedMemory b_val, SyncedMemory c_val, SyncedMemory d_val, 
      SyncedMemory a_next_eval, SyncedMemory b_next_eval, SyncedMemory d_next_eval, 
      SyncedMemory q_m, SyncedMemory q_l, SyncedMemory q_r, SyncedMemory q_o, 
      SyncedMemory q_4, SyncedMemory q_c, SyncedMemory q_hl, SyncedMemory q_hr, SyncedMemory q_h4, SyncedMemory q_arith, 
      SyncedMemory range, SyncedMemory logic, SyncedMemory fixed_group_add, SyncedMemory variable_group_add, SyncedMemory pi_eval_8n, 
      SyncedMemory four, SyncedMemory one, SyncedMemory two, SyncedMemory three, 
      SyncedMemory kappa_range, SyncedMemory kappa_sq_range, SyncedMemory kappa_cu_range, SyncedMemory range_challenge, 
      SyncedMemory P_D, SyncedMemory nine, SyncedMemory kappa_cu_logic, 
      SyncedMemory eighteen, SyncedMemory eightyone, SyncedMemory eightythree, SyncedMemory kappa_qu_logic, 
      SyncedMemory logic_challenge, SyncedMemory P_A, SyncedMemory kappa_fb, SyncedMemory kappa_sq_fb, 
      SyncedMemory kappa_cu_fb, SyncedMemory fb_challenge, SyncedMemory kappa_vb, SyncedMemory kappa_sq_vb, 
      SyncedMemory vb_challenge, int64_t SBOX_ALPHA, cudaStream_t stream = (cudaStream_t)0);

    void compute_quotient_identity_range_check_i(SyncedMemory x,
      SyncedMemory w_l,  SyncedMemory w_r, SyncedMemory w_o, SyncedMemory w_4, 
      SyncedMemory z, SyncedMemory alpha, SyncedMemory beta, SyncedMemory gamma, 
      SyncedMemory k1, SyncedMemory k2, SyncedMemory k3, SyncedMemory result, cudaStream_t stream = (cudaStream_t)0);

    void compute_quotient_copy_range_check_i(SyncedMemory left_sigma, SyncedMemory right_sigma,
      SyncedMemory out_sigma, SyncedMemory fourth_sigma, SyncedMemory mod,
      SyncedMemory w_l,  SyncedMemory w_r, SyncedMemory w_o, SyncedMemory w_4, 
      SyncedMemory z_next, SyncedMemory alpha, SyncedMemory beta, SyncedMemory gamma, 
      SyncedMemory result, cudaStream_t stream = (cudaStream_t)0);

    void compute_quotient_term_check_one_i(SyncedMemory z, SyncedMemory l1, SyncedMemory one, SyncedMemory result, cudaStream_t stream = (cudaStream_t)0);

    void compute_quotient_i(SyncedMemory compress_tuple, 
      SyncedMemory f, SyncedMemory table, SyncedMemory table_next, SyncedMemory h1, SyncedMemory h1_next, SyncedMemory h2, 
      SyncedMemory z2, SyncedMemory z2_next,
      SyncedMemory l1,
      SyncedMemory q_lookup, SyncedMemory mod,
      SyncedMemory delta, SyncedMemory epsilon, SyncedMemory zeta, SyncedMemory one,
      SyncedMemory lookup_seq, SyncedMemory lookup_seq_sq, SyncedMemory lookup_seq_cu,  
      SyncedMemory result,  cudaStream_t stream = (cudaStream_t)0);
} // namespace cuda
