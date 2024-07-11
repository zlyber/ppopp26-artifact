#include <vector>
#include <tuple>
#include "caffe/syncedmem.hpp"
#include "PLONK/src/bls12_381/fr.hpp"
#include "PLONK/plonk_core/src/proof_system/widget/mod.cu"
#include "PLONK/utils/function.cuh"
#include "PLONK/src/arithmetic.cu"
class Lookup {
public:
    
        std::tuple<SyncedMemory&, SyncedMemory&> q_lookup;

        SyncedMemory& table_1;

        SyncedMemory&  table_2;

        SyncedMemory& table_3;

        SyncedMemory&  table_4;
};

SyncedMemory&  _compute_quotient_i(
        SyncedMemory&  w_l_i,
        SyncedMemory&  w_r_i,
        SyncedMemory&  w_o_i,
        SyncedMemory&  w_4_i,
        SyncedMemory&  f_poly,
        SyncedMemory&  table_poly,
        SyncedMemory&  h1_poly,
        SyncedMemory&  h2_poly,
        SyncedMemory&  z2_poly,
        SyncedMemory&  l1_poly,
        SyncedMemory&  delta,
        SyncedMemory&  epsilon,
        SyncedMemory&  zeta,
        SyncedMemory&  lookup_sep,
        SyncedMemory&  proverkey_q_lookup,
        int size
    ) {  
    Ntt_coset coset_ntt(fr::TWO_ADICITY, size);

    // q_lookup(X) * (a(X) + zeta * b(X) + (zeta^2 * c(X)) + (zeta^3 * d(X) - f(X))) * α_1
    SyncedMemory&  one = fr::one();
    
   
    SyncedMemory&  lookup_sep_sq = mul_mod(lookup_sep, lookup_sep);
    SyncedMemory&  lookup_sep_cu = mul_mod(lookup_sep_sq, lookup_sep);
    SyncedMemory&  one_plus_delta = add_mod(delta, one); 
    SyncedMemory&  epsilon_one_plus_delta = mul_mod(epsilon, one_plus_delta); 
    
    void* epsilon_gpu_data=epsilon.mutable_gpu_data();
    void* delta_gpu_data=delta.mutable_gpu_data();
    void* zeta_gpu_data=zeta.mutable_gpu_data();
    void* lookup_sep_gpu_data=lookup_sep.mutable_gpu_data();
    void* lookup_sep_sq_gpu_data=lookup_sep_sq.mutable_gpu_data();
    void* lookup_sep_cu_gpu_data=lookup_sep_cu.mutable_gpu_data();
    void* one_plus_delta_gpu_data=one_plus_delta.mutable_gpu_data();
    void* epsilon_one_plus_delta_gpu_data=epsilon_one_plus_delta.mutable_gpu_data();
    void* one_gpu_data=one.mutable_gpu_data();
    
    
    SyncedMemory&  compressed_tuple = torch::compress(w_l_i, w_r_i, w_o_i, w_4_i, zeta); 
    SyncedMemory&  f_i = coset_ntt.forward(f_poly);
    SyncedMemory&  mid = sub_mod(compressed_tuple, f_i);
    mid = mul_mod(proverkey_q_lookup, mid);
    SyncedMemory&  a = mul_mod_scalar(mid, lookup_sep);
    // mid = nullptr; 

   
    SyncedMemory&  b_0 = add_mod_scalar(f_i, epsilon);
    // f_i = nullptr; 

    SyncedMemory&  table_i = coset_ntt.forward(table_poly);
    SyncedMemory&  b_1_1 = add_mod_scalar(table_i, epsilon_one_plus_delta);
    SyncedMemory&  table_i_next = torch::cat({table_i, table_i.slice(0, 8)}, 0).slice(8);
    SyncedMemory&  b_1_2 = mul_mod_scalar(table_i_next, delta);
    // table_i = nullptr; 
    // table_i_next = nullptr; 

    SyncedMemory&  b_1 = add_mod(b_1_1, b_1_2);
    // b_1_1 = nullptr; 
    // b_1_2 = nullptr; 

    SyncedMemory&  z2_i = coset_ntt.forward(z2_poly);
    mid = mul_mod_scalar(z2_i, one_plus_delta);
    mid = mul_mod(mid, b_0);
    mid = mul_mod(mid, b_1);
    SyncedMemory&  b = mul_mod_scalar(mid, lookup_sep_sq);

    // mid = nullptr; 
    // b_0 = nullptr; 
    // b_1 = nullptr; 
    SyncedMemory&  res = add_mod(a, b);
    // a = nullptr; 
    // b = nullptr; 


   SyncedMemory& h1_i = coset_ntt.forward(h1_poly);
   SyncedMemory& c_0_1 = add_mod_scalar(h1_i, epsilon_one_plus_delta);
   SyncedMemory& h2_i = coset_ntt.forward(h2_poly);
   SyncedMemory& c_0_2 = mul_mod_scalar(h2_i, delta);
   SyncedMemory& c_0 = add_mod(c_0_1, c_0_2);
    // c_0_1 = nullptr; 
    // c_0_2 = nullptr; 

   SyncedMemory& extend_mod = repeat_to_poly(fr::MODULUS().to("cuda"), size);
   SyncedMemory& z2_i_next = torch::cat({z2_i, z2_i.slice(0, 8)}, 0).slice(8);
   SyncedMemory& neg_z2_next = sub_mod(extend_mod, z2_i_next);
    // z2_i_next = nullptr; 

    mid = mul_mod(neg_z2_next, c_0);
    c_0 = nullptr; 
    // extend_mod = nullptr; 

   SyncedMemory& c_1_1 = add_mod_scalar(h2_i, epsilon_one_plus_delta);
    // h2_i = nullptr; 

   SyncedMemory& h1_i_next = torch::cat({h1_i, h1_i.slice(0, 8)}, 0).slice(8);
   SyncedMemory&c_1_2 = mul_mod_scalar(h1_i_next, delta);
    // h1_i = nullptr; 
    // h1_i_next = nullptr; 

   SyncedMemory& c_1 = add_mod(c_1_1, c_1_2);
    // c_1_1 = nullptr; 
    // c_1_2 = nullptr; 

    mid = mul_mod(mid, c_1);
    // c_1 = nullptr; 
   SyncedMemory& c = mul_mod_scalar(mid, lookup_sep_sq);
    res = add_mod(res, c);
    // mid = nullptr; 
    // c = nullptr; 

   
   SyncedMemory& d_1 = sub_mod_scalar(z2_i, one);
   SyncedMemory& l1_i = coset_ntt.forward(l1_poly);
   SyncedMemory& d_2 = mul_mod_scalar(l1_i, lookup_sep_cu);
    // z2_i = nullptr; 
    // l1_i = nullptr; 

   SyncedMemory& d = mul_mod(d_1, d_2);
    // d_1 = nullptr; 
    // d_2 = nullptr; 

   
    res = add_mod(res, d);
    return res;
}

SyncedMemory& compute_linearisation_lookup(
    SyncedMemory&  l1_eval,
    SyncedMemory&  a_eval,
    SyncedMemory&  b_eval,
    SyncedMemory&  c_eval,
    SyncedMemory&  d_eval,
    SyncedMemory&  f_eval,
    SyncedMemory&  table_eval,
    SyncedMemory&  table_next_eval,
    SyncedMemory&  h1_next_eval,
    SyncedMemory&  h2_eval,
    SyncedMemory&  z2_next_eval,
    SyncedMemory&  delta,
    SyncedMemory&  epsilon,
    SyncedMemory&  zeta,
    SyncedMemory&  z2_poly,
    SyncedMemory&  h1_poly,
    SyncedMemory&  lookup_sep,
    SyncedMemory&  pk_q_lookup
) {
    SyncedMemory&  lookup_sep_sq = mul_mod(lookup_sep, lookup_sep);
    SyncedMemory&  lookup_sep_cu = mul_mod(lookup_sep_sq, lookup_sep);
    SyncedMemory& one= fr::one();
    void* one_gpu_data= one.mutable_gpu_data();
    SyncedMemory&  one_plus_delta = add_mod(delta,one);
    SyncedMemory&  epsilon_one_plus_delta = mul_mod(epsilon, one_plus_delta);

    SyncedMemory&  compressed_tuple = lc({a_eval, b_eval, c_eval, d_eval}, zeta);
    SyncedMemory&  compressed_tuple_sub_f_eval = sub_mod(compressed_tuple, f_eval);
    SyncedMemory&  const1 = mul_mod(compressed_tuple_sub_f_eval, lookup_sep);
    SyncedMemory& a = poly_mul_const(pk_q_lookup, const1);

    // z2(X) * (1 + δ) * (ε + f_bar) * (ε(1+δ) + t_bar + δ*tω_bar) * lookup_sep^2
    SyncedMemory&  b_0 = add_mod(epsilon, f_eval);
    SyncedMemory&  epsilon_one_plus_delta_plus_table_eval = add_mod(epsilon_one_plus_delta, table_eval);
    SyncedMemory&  delta_times_table_next_eval = mul_mod(delta, table_next_eval);
    SyncedMemory&  b_1 = add_mod(epsilon_one_plus_delta_plus_table_eval, delta_times_table_next_eval);
    SyncedMemory&  b_2 = mul_mod(l1_eval, lookup_sep_cu);
    SyncedMemory&  one_plus_delta_b_0 = mul_mod(one_plus_delta, b_0);
    SyncedMemory&  one_plus_delta_b_0_b_1 = mul_mod(one_plus_delta_b_0, b_1);
    SyncedMemory&  one_plus_delta_b_0_b_1_lookup = mul_mod(one_plus_delta_b_0_b_1, lookup_sep_sq);
    SyncedMemory&  const2 = add_mod(one_plus_delta_b_0_b_1_lookup, b_2);
    SyncedMemory& b = poly_mul_const(z2_poly, const2);

    // h1(X) * (−z2ω_bar) * (ε(1+δ) + h2_bar  + δh1ω_bar) * lookup_sep^2
    SyncedMemory&  modular=fr::MODULUS();
    void* modular_gpu_data= modular.mutable_gpu_data();
    SyncedMemory&  neg_z2_next_eval = sub_mod(modular, z2_next_eval);
    SyncedMemory&  c_0 = mul_mod(neg_z2_next_eval, lookup_sep_sq);
    SyncedMemory&  epsilon_one_plus_delta_h2_eval = add_mod(epsilon_one_plus_delta, h2_eval);
    SyncedMemory&  delta_h1_next_eval = add_mod(delta, h1_next_eval);
    SyncedMemory&  c_1 = add_mod(epsilon_one_plus_delta_h2_eval, delta_h1_next_eval);
    SyncedMemory&  c0_c1 = mul_mod(c_0, c_1);
    SyncedMemory& c = poly_mul_const(h1_poly, c0_c1);
    SyncedMemory& ab = poly_add_poly(a, b);
    SyncedMemory& abc = poly_add_poly(ab, c);

    return abc;
}

SyncedMemory& compute_lookup_quotient_term(
    int&  n,
    SyncedMemory&  wl_eval_8n,
    SyncedMemory&  wr_eval_8n,
    SyncedMemory&  wo_eval_8n,
    SyncedMemory&  w4_eval_8n,
    SyncedMemory&  f_poly,
    SyncedMemory&  table_poly,
    SyncedMemory&  h1_poly,
    SyncedMemory&  h2_poly,
    SyncedMemory&  z2_poly,
    SyncedMemory&  l1_poly,
    SyncedMemory&  delta,
    SyncedMemory&  epsilon,
    SyncedMemory&  zeta,
    SyncedMemory&  lookup_sep,
    SyncedMemory&  pk_lookup_qlookup_evals
) {
    int   size = 8 * n;

    SyncedMemory wl_eval_8n_size(size*sizeof(uint64_t));
    SyncedMemory wr_eval_8n_size(size*sizeof(uint64_t));
    SyncedMemory wo_eval_8n_size(size*sizeof(uint64_t));
    SyncedMemory w4_eval_8n_size(size*sizeof(uint64_t));
    void* wl_eval_8n_size_gpu_data = wl_eval_8n_size.mutable_gpu_data();
    void* wr_eval_8n_size_gpu_data = wr_eval_8n_size.mutable_gpu_data();
    void* wo_eval_8n_size_gpu_data = wo_eval_8n_size.mutable_gpu_data();
    void* w4_eval_8n_size_gpu_data = w4_eval_8n_size.mutable_gpu_data();
    void* wl_eval_8n_gpu_data = wl_eval_8n.mutable_gpu_data();
    void* wr_eval_8n_gpu_data = wr_eval_8n.mutable_gpu_data();
    void* wo_eval_8n_gpu_data = wo_eval_8n.mutable_gpu_data();
    void* w4_eval_8n_gpu_data = w4_eval_8n.mutable_gpu_data();
    caffe_gpu_memcpy(8*n*sizeof(uint64_t), wl_eval_8n_gpu_data, wl_eval_8n_size_gpu_data);
    caffe_gpu_memcpy(8*n*sizeof(uint64_t), wr_eval_8n_gpu_data, wr_eval_8n_size_gpu_data);
    caffe_gpu_memcpy(8*n*sizeof(uint64_t), wo_eval_8n_gpu_data, wo_eval_8n_size_gpu_data);
    caffe_gpu_memcpy(8*n*sizeof(uint64_t), w4_eval_8n_gpu_data, w4_eval_8n_size_gpu_data);
    SyncedMemory& quotient = _compute_quotient_i(
        wl_eval_8n_size,
        wr_eval_8n_size,
        wo_eval_8n_size,
        w4_eval_8n_size,
        f_poly,
        table_poly,
        h1_poly,
        h2_poly,
        z2_poly,
        l1_poly,
        delta,
        epsilon,
        zeta,
        lookup_sep,
        pk_lookup_qlookup_evals,
        size
    );

    return quotient;
}