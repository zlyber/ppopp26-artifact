#include <vector>
#include <tuple>
#include <iostream>
#include <cuda_runtime.h>
#include "PLONK/plonk_core/src/permutation/constants.cu"
#include "PLONK/utils/function.cuh"
#include "caffe/syncedmem.hpp"
#include "PLONK/src/structure.cuh"
#include "PLONK/src/bls12_381/fr.hpp"
#include "PLONK/src/bls12_381/fq.hpp"
#include "PLONK/src/domain.cuh"
#include "PLONK/src/domain.cu"
#include "PLONK/plonk_core/src/constaraint_system/hash.cu"
#include "/PLONK/src/arithmetic.cu"
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

 SyncedMemory& compute_quotient_i(const Arith& arithmetics_evals, const WitnessValues& wit_vals) {
    SyncedMemory& mult = mul_mod(wit_vals.a_val, wit_vals.b_val);
    SyncedMemory& mult = mul_mod(mult, arithmetics_evals.q_m);
    SyncedMemory& left = mul_mod(wit_vals.a_val, arithmetics_evals.q_l);
    SyncedMemory& right = mul_mod(wit_vals.b_val, arithmetics_evals.q_r);
    SyncedMemory& out = mul_mod(wit_vals.c_val, arithmetics_evals.q_o);
    SyncedMemory& fourth = mul_mod(wit_vals.d_val, arithmetics_evals.q_4);

    SyncedMemory& mid = add_mod(mult, left);
    // delete left;
    mid = add_mod(mid, right);
    // delete right;
    mid = add_mod(mid, out);
    // delete out;
    mid = add_mod(mid, fourth);
    // delete fourth;

    SyncedMemory& a_high = exp_mod(wit_vals.a_val, SBOX_ALPHA);
    a_high = mul_mod(a_high, arithmetics_evals.q_hl);
    mid = add_mod(mid, a_high);
    // delete a_high;

    SyncedMemory& b_high = exp_mod(wit_vals.b_val, SBOX_ALPHA);
    b_high = mul_mod(b_high, arithmetics_evals.q_hr);
    mid = add_mod(mid, b_high);
    // delete b_high;

    SyncedMemory& f_high = exp_mod(wit_vals.d_val, SBOX_ALPHA);
    f_high = mul_mod(f_high, arithmetics_evals.q_h4);
    mid = add_mod(mid, f_high);
    // delete f_high;

    mid = add_mod(mid, arithmetics_evals.q_c);
    SyncedMemory& arith_val = mul_mod(mid, arithmetics_evals.q_arith);
    return arith_val;
}

// 计算给定评估点处线性化多项式的算术门贡献
 SyncedMemory& compute_linearisation_arithmetic(
     SyncedMemory& a_eval,
     SyncedMemory& b_eval,
     SyncedMemory& c_eval,
     SyncedMemory& d_eval,
     SyncedMemory& q_arith_eval,
    const Arith& prover_key_arithmetic) {

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