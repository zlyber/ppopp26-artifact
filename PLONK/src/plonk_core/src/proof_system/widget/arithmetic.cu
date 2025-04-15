#include "mod.cuh"
#include "../../../../structure.cuh"
#include "../../constaraint_system/hash.h"

#define Byte_LEN 8

// SyncedMemory compute_quotient_i(Arithmetic arithmetics_evals,  WitnessValues wit_vals) {
//     SyncedMemory mult = mul_mod(wit_vals.a_val, wit_vals.b_val);
//     mul_mod_(mult, arithmetics_evals.q_m);
//     SyncedMemory mid;
//     {
//         SyncedMemory left = mul_mod(wit_vals.a_val, arithmetics_evals.q_l);
//         SyncedMemory right = mul_mod(wit_vals.b_val, arithmetics_evals.q_r);
//         SyncedMemory out = mul_mod(wit_vals.c_val, arithmetics_evals.q_o);
//         SyncedMemory fourth = mul_mod(wit_vals.d_val, arithmetics_evals.q_4);

//         mid = add_mod(mult, left);
//         add_mod_(mid, right);
//         add_mod_(mid, out);
//         add_mod_(mid, fourth);
//     }

//     {
//         SyncedMemory a_high = exp_mod(wit_vals.a_val, SBOX_ALPHA);
//         mul_mod_(a_high, arithmetics_evals.q_hl);
//         add_mod_(mid, a_high);
//     }

//     {
//         SyncedMemory b_high = exp_mod(wit_vals.b_val, SBOX_ALPHA);
//         mul_mod_(b_high, arithmetics_evals.q_hr);
//         add_mod_(mid, b_high);
//     }

//     {
//         SyncedMemory f_high = exp_mod(wit_vals.d_val, SBOX_ALPHA);
//         mul_mod_(f_high, arithmetics_evals.q_h4);
//         add_mod_(mid, f_high);
//     }

//     add_mod_(mid, arithmetics_evals.q_c);
//     SyncedMemory arith_val = mul_mod(mid, arithmetics_evals.q_arith);
    
//     return arith_val;
// }

// SyncedMemory compute_linearisation_arithmetic(
//      SyncedMemory a_eval,
//      SyncedMemory b_eval,
//      SyncedMemory c_eval,
//      SyncedMemory d_eval,
//      SyncedMemory q_arith_eval,
//      Arithmetic prover_key_arithmetic) {

//      SyncedMemory mid1_1 = mul_mod(a_eval, b_eval);

//      SyncedMemory mid1 = poly_mul_const(prover_key_arithmetic.q_m, mid1_1);
//      SyncedMemory mid2 = poly_mul_const(prover_key_arithmetic.q_l, a_eval);
//      SyncedMemory mid3 = poly_mul_const(prover_key_arithmetic.q_r, b_eval);
//      SyncedMemory mid4 = poly_mul_const(prover_key_arithmetic.q_o, c_eval);
//      SyncedMemory mid5 = poly_mul_const(prover_key_arithmetic.q_4, d_eval);
//      SyncedMemory mid6_1 = exp_mod(a_eval, SBOX_ALPHA);
//      SyncedMemory mid6 = poly_mul_const(prover_key_arithmetic.q_hl, mid6_1);
//      SyncedMemory mid7_1 = exp_mod(b_eval, SBOX_ALPHA);
//      SyncedMemory mid7 = poly_mul_const(prover_key_arithmetic.q_hr, mid7_1);
//      SyncedMemory mid8_1 = exp_mod(d_eval, SBOX_ALPHA);
//      SyncedMemory mid8 = poly_mul_const(prover_key_arithmetic.q_h4, mid8_1);

//      SyncedMemory add1 = poly_add_poly(mid1, mid2);
//      SyncedMemory add2 = poly_add_poly(add1, mid3);
//      SyncedMemory add3 = poly_add_poly(add2, mid4);
//      SyncedMemory add4 = poly_add_poly(add3, mid5);
//      SyncedMemory add5 = poly_add_poly(add4, mid6);
//      SyncedMemory add6 = poly_add_poly(add5, mid7);
//      SyncedMemory add7 = poly_add_poly(add6, mid8);
//      SyncedMemory add8 = poly_add_poly(add7, prover_key_arithmetic.q_c);

//      SyncedMemory result = poly_mul_const(add8, q_arith_eval);
//     return result;
// }

SyncedMemory move(SyncedMemory a_eval){
    SyncedMemory res(a_eval.size());
    void* a_eval_gpu_data = a_eval.mutable_gpu_data();
    void* res_gpu_data = res.mutable_gpu_data();
    caffe_gpu_memcpy(a_eval.size()-2*Byte_LEN,a_eval_gpu_data+2*Byte_LEN,res_gpu_data);
    caffe_gpu_memcpy(2*Byte_LEN,a_eval_gpu_data,res_gpu_data);
    return res;
}

SyncedMemory get_tail(SyncedMemory a_eval,int size){
    SyncedMemory a_eval_tail(a_eval.size()-(size/4*sizeof(uint64_t)));
    void* a_eval_tail_gpu_data=a_eval_tail.mutable_gpu_data();
    void* a_eval_gpu_data=a_eval.mutable_gpu_data();
    caffe_gpu_memcpy(a_eval.size()-(size/4*sizeof(uint64_t)),a_eval_gpu_data,a_eval_tail_gpu_data);
    return a_eval_tail;
}