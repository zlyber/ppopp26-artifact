#include <vector>
#include <tuple>
#include <iostream>
#include <cuda_runtime.h>
#include "PLONK/src/plonk_core/src/permutation/constants.cu"
#include "PLONK/utils/function.cuh"
#include "caffe/syncedmem.hpp"
#include "PLONK/src/structure.cuh"
#include "PLONK/src/bls12_381/fr.hpp"
#include "PLONK/src/bls12_381/fq.hpp"
#include "PLONK/src/domain.cuh"
#include "PLONK/src/domain.cu"
#include "PLONK/src/plonk_core/src/constaraint_system/hash.cu"
#include "PLONK/src/arithmetic.cu"
#include "PLONK/src/plonk_core/src/proof_system/widget/mod.cu"
#define Byte_LEN 8
struct Arith {
    SyncedMemory& q_m;
    SyncedMemory& q_l;
    SyncedMemory& q_r;
    SyncedMemory& q_o;
    SyncedMemory& q_4;
    SyncedMemory& q_hl;
    SyncedMemory& q_hr;
    SyncedMemory& q_h4;
    SyncedMemory& q_c;
    SyncedMemory& q_arith;
};

 SyncedMemory& compute_quotient_i( Arith& arithmetics_evals,  WitnessValues& wit_vals) {
    SyncedMemory& mult = mul_mod(wit_vals.a_val, wit_vals.b_val);
    SyncedMemory& mult = mul_mod(mult, arithmetics_evals.q_m);
    SyncedMemory& left = mul_mod(wit_vals.a_val, arithmetics_evals.q_l);
    SyncedMemory& right = mul_mod(wit_vals.b_val, arithmetics_evals.q_r);
    SyncedMemory& out = mul_mod(wit_vals.c_val, arithmetics_evals.q_o);
    SyncedMemory& fourth = mul_mod(wit_vals.d_val, arithmetics_evals.q_4);

    SyncedMemory& mid_temp_1 = add_mod(mult, left);
    left.~SyncedMemory();
    SyncedMemory& mid_temp_2 = add_mod(mid_temp_1, right);
    right.~SyncedMemory();
    SyncedMemory& mid_temp_3 = add_mod(mid_temp_2, out);
    out.~SyncedMemory();
    SyncedMemory& mid_temp_4 = add_mod(mid_temp_3, fourth);
    fourth.~SyncedMemory();

    SyncedMemory& a_high_temp = exp_mod(wit_vals.a_val, SBOX_ALPHA);
    SyncedMemory& a_high = mul_mod(a_high_temp, arithmetics_evals.q_hl);
    SyncedMemory& mid_temp_5 = add_mod(mid_temp_4, a_high);
    a_high.~SyncedMemory();

    SyncedMemory& b_high_temp = exp_mod(wit_vals.b_val, SBOX_ALPHA);
    SyncedMemory& b_high = mul_mod(b_high_temp, arithmetics_evals.q_hr);
    SyncedMemory& mid_temp_6 = add_mod(mid_temp_5, b_high);
    b_high.~SyncedMemory();

    SyncedMemory& f_high_temp = exp_mod(wit_vals.d_val, SBOX_ALPHA);
    SyncedMemory& f_high = mul_mod(f_high_temp, arithmetics_evals.q_h4);
    SyncedMemory& mid_temp_7 = add_mod(mid_temp_6, f_high);
    f_high.~SyncedMemory();

    SyncedMemory& mid = add_mod(mid_temp_7, arithmetics_evals.q_c);
    SyncedMemory& arith_val = mul_mod(mid, arithmetics_evals.q_arith);
    return arith_val;
}


 SyncedMemory& compute_linearisation_arithmetic(
     SyncedMemory& a_eval,
     SyncedMemory& b_eval,
     SyncedMemory& c_eval,
     SyncedMemory& d_eval,
     SyncedMemory& q_arith_eval,
     Arith& prover_key_arithmetic) {

     SyncedMemory& mid1_1 =mul_mod(a_eval, b_eval);

     SyncedMemory& mid1 = poly_mul_const(prover_key_arithmetic.q_m, mid1_1);
     SyncedMemory& mid2 = poly_mul_const(prover_key_arithmetic.q_l, a_eval);
     SyncedMemory& mid3 = poly_mul_const(prover_key_arithmetic.q_r, b_eval);
     SyncedMemory& mid4 = poly_mul_const(prover_key_arithmetic.q_o, c_eval);
     SyncedMemory& mid5 = poly_mul_const(prover_key_arithmetic.q_4, d_eval);
     SyncedMemory& mid6_1 = exp_mod(a_eval, SBOX_ALPHA);
     SyncedMemory& mid6 = poly_mul_const(prover_key_arithmetic.q_hl, mid6_1);
     SyncedMemory& mid7_1 = exp_mod(b_eval, SBOX_ALPHA);
     SyncedMemory& mid7 = poly_mul_const(prover_key_arithmetic.q_hr, mid7_1);
     SyncedMemory& mid8_1 = exp_mod(d_eval, SBOX_ALPHA);
     SyncedMemory& mid8 = poly_mul_const(prover_key_arithmetic.q_h4, mid8_1);

     SyncedMemory& add1 = poly_add_poly(mid1, mid2);
     SyncedMemory& add2 = poly_add_poly(add1, mid3);
     SyncedMemory& add3 = poly_add_poly(add2, mid4);
     SyncedMemory& add4 = poly_add_poly(add3, mid5);
     SyncedMemory& add5 = poly_add_poly(add4, mid6);
     SyncedMemory& add6 = poly_add_poly(add5, mid7);
     SyncedMemory& add7 = poly_add_poly(add6, mid8);
     SyncedMemory& add8 = poly_add_poly(add7, prover_key_arithmetic.q_c);

     SyncedMemory& result = poly_mul_const(add8, q_arith_eval);
    return result;
}
SyncedMemory& move(SyncedMemory& a_eval){
    SyncedMemory res(a_eval.size());
    void* a_eval_gpu_data = a_eval.mutable_gpu_data();
    void* res_gpu_data = res.mutable_gpu_data();
    caffe_gpu_memcpy(a_eval.size()-2*Byte_LEN,a_eval_gpu_data+2*Byte_LEN,res_gpu_data);
    caffe_gpu_memcpy(2*Byte_LEN,a_eval_gpu_data,res_gpu_data);
    return res;
}
SyncedMemory& cat(SyncedMemory& a_eval,SyncedMemory& b_eval){
    SyncedMemory res(a_eval.size());
    void* a_eval_gpu_data = a_eval.mutable_gpu_data();
    void* b_eval_gpu_data = b_eval.mutable_gpu_data();
    void* res_gpu_data = res.mutable_gpu_data();
    caffe_gpu_memcpy(a_eval.size(),a_eval_gpu_data,res_gpu_data);
    caffe_gpu_memcpy(b_eval.size(),b_eval_gpu_data,res_gpu_data);
    return res;
}
SyncedMemory& get_tail(SyncedMemory& a_eval,int size){
    SyncedMemory a_eval_tail(a_eval.size()-(size/4*sizeof(uint64_t)));
    void* a_eval_tail_gpu_data=a_eval_tail.mutable_gpu_data();
    void* a_eval_gpu_data=a_eval.mutable_gpu_data();
    caffe_gpu_memcpy(a_eval.size()-(size/4*sizeof(uint64_t)),a_eval_gpu_data,a_eval_tail_gpu_data);
    return a_eval_tail;
}
SyncedMemory& get_head(SyncedMemory& a_eval,int size){
    SyncedMemory a_eval_head(a_eval.size()-(size/4*sizeof(uint64_t)));
    void* a_eval_head_gpu_data=a_eval_head.mutable_gpu_data();
    void* a_eval_gpu_data=a_eval.mutable_gpu_data();
    caffe_gpu_memcpy(size/4*sizeof(uint64_t),a_eval_gpu_data,a_eval_head_gpu_data);
    return a_eval_head;
}