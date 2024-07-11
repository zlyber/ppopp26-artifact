#include <stdint.h>
#include <vector>
#include <iostream>
#include "caffe/syncedmem.hpp"
#include "PLONK/src/structure.cuh"
#include "PLONK/plonk_core/src/proof_system/widget/mod.cu"
#include "PLONK/src/arithmetic.cu"
#include "PLONK/src/bls12_381/edwards.h"
class CAValues {
public:
    SyncedMemory& a_next_val;
    SyncedMemory& b_next_val;
    SyncedMemory& d_next_val;

public:
    CAValues(SyncedMemory& a, SyncedMemory& b, SyncedMemory& d)
        : a_next_val(a), b_next_val(b), d_next_val(d) {}

    static CAValues from_evaluations(std::map<std::string, SyncedMemory&> custom_evals) {
        SyncedMemory& a_next_val = custom_evals["a_next_eval"];
        SyncedMemory& b_next_val = custom_evals["b_next_eval"];
        SyncedMemory& d_next_val = custom_evals["d_next_eval"];
        
        return CAValues{a_next_val, b_next_val, d_next_val};
    }
};

class CAGate {
public:
    static SyncedMemory& constraints(SyncedMemory& separation_challenge, const WitnessValues& wit_vals, const CAValues& custom_vals) {
        SyncedMemory& x_1 = wit_vals.a_val;
        SyncedMemory& x_3 = custom_vals.a_next_val;
        SyncedMemory& y_1 = wit_vals.b_val;
        SyncedMemory& y_3 = custom_vals.b_next_val;
        SyncedMemory& x_2 = wit_vals.c_val;
        SyncedMemory& y_2 = wit_vals.d_val;
        SyncedMemory& x1_y2 = custom_vals.d_next_val;
        SyncedMemory& kappa = mul_mod(separation_challenge, separation_challenge);
        SyncedMemory& x1y2 = mul_mod(x_1, y_2);
        SyncedMemory& xy_consistency =sub_mod(x1y2, x1_y2);
        SyncedMemory& y1_x2 = mul_mod(y_1, x_2);
        SyncedMemory& y1_y2 = mul_mod(y_1, y_2);
        SyncedMemory& x1_x2 = mul_mod(x_1, x_2);
        SyncedMemory& x3_lhs = add_mod(x1_y2, y1_x2);
        SyncedMemory& x_3_D = mul_mod(x_3, COEFF_D());
        SyncedMemory& x_3_D_x1_y2 = mul_mod(x_3_D, x1_y2);
        SyncedMemory& x_3_D_x1_y2_y1_x2 = mul_mod(x_3_D_x1_y2, y1_x2);
        SyncedMemory& x3_rhs = add_mod(x_3, x_3_D_x1_y2_y1_x2);
        SyncedMemory& x3_l_sub_r = sub_mod(x3_lhs, x3_rhs);
        SyncedMemory& x3_consistency = mul_mod(x3_l_sub_r, kappa);
        SyncedMemory& x1_x2_A = mul_mod(COEFF_A(), x1_x2);
        SyncedMemory& y3_lhs = sub_mod(y1_y2, x1_x2_A);
        SyncedMemory& y_3_D = mul_mod(y_3, COEFF_D());
        SyncedMemory& y_3_D_x1_y2 = mul_mod(y_3_D, x1_y2);
        SyncedMemory& y_3_D_x1_y2_y1_x2 = mul_mod(y_3_D_x1_y2, y1_x2);
        SyncedMemory& y3_rhs = sub_mod(y_3, y_3_D_x1_y2_y1_x2);
        SyncedMemory& y3_l_sub_r = sub_mod(y3_lhs, y3_rhs);
        SyncedMemory& kappa2 = mul_mod(kappa, kappa);
        SyncedMemory& y3_consistency = mul_mod(y3_l_sub_r, kappa2);

        SyncedMemory& mid1 = add_mod(xy_consistency, x3_consistency);
        SyncedMemory& mid2 = add_mod(mid1, y3_consistency);
        SyncedMemory& result = mul_mod(mid2, separation_challenge);
        return result;
    }

    static SyncedMemory& quotient_term(SyncedMemory& selector, SyncedMemory& separation_challenge, const WitnessValues& wit_vals, const CAValues& custom_vals) {
        SyncedMemory& kappa = mul_mod(separation_challenge, separation_challenge);
        SyncedMemory& kappasq = mul_mod(kappa, kappa);

        
        SyncedMemory& y1_x2 = mul_mod(custom_vals.a_next_val, wit_vals.c_val);
        SyncedMemory& coeff_d=COEFF_D();
        void* coeff_d_gpu_data = coeff_d.mutable_gpu_data();
        SyncedMemory& mid = mul_mod_scalar(custom_vals.a_next_val, coeff_d);
        mid = mul_mod(mid, custom_vals.d_next_val);
        mid = mul_mod(mid, y1_x2);
        SyncedMemory& x3_rhs = add_mod(custom_vals.a_next_val, mid);
        SyncedMemory& x3_lhs = add_mod(custom_vals.d_next_val, y1_x2);
        SyncedMemory& x3_l_sub_r = sub_mod(x3_lhs, x3_rhs);
        // delete x3_lhs; delete x3_rhs;
        mid = mul_mod_scalar(custom_vals.b_next_val, coeff_d);
        mid = mul_mod(mid, custom_vals.d_next_val);
        mid = mul_mod(mid, y1_x2);
        // delete y1_x2;
        void* kappa_gpu_data = kappa.mutable_gpu_data();
        SyncedMemory& x3_consistency = mul_mod_scalar(x3_l_sub_r, kappa);
        // delete x3_l_sub_r;

        SyncedMemory& coeff_a=COEFF_A();
        void* coeff_a_gpu_data = coeff_a.mutable_gpu_data();
        SyncedMemory& x1_x2 = mul_mod(wit_vals.a_val, wit_vals.c_val);
        x1_x2 = mul_mod_scalar(x1_x2, coeff_a);
        SyncedMemory& y1_y2 = mul_mod(custom_vals.a_next_val, wit_vals.d_val);
        SyncedMemory& y3_lhs = sub_mod(y1_y2, x1_x2);
        // delete y1_y2; delete x1_x2;

        SyncedMemory& y3_rhs = sub_mod(custom_vals.b_next_val, mid);
        SyncedMemory& y3_l_sub_r = sub_mod(y3_lhs, y3_rhs);
        void* kappasq_gpu_data=kappasq.mutable_gpu_data();
        SyncedMemory& y3_consistency = mul_mod_scalar(y3_l_sub_r, kappasq);
        // delete mid; delete y3_lhs; delete y3_rhs; delete y3_l_sub_r;

    
        SyncedMemory& x1y2 = mul_mod(wit_vals.a_val, wit_vals.d_val);
        SyncedMemory& xy_consistency = sub_mod(x1y2, custom_vals.d_next_val);

        SyncedMemory& res = add_mod(xy_consistency, x3_consistency);
        res = add_mod(res, y3_consistency);
        void* separation_challenge_gpu_data=separation_challenge.mutable_gpu_data();
        res = mul_mod_scalar(res, separation_challenge);
        res = mul_mod(selector, res);
        return res;
    }

    static SyncedMemory& linearisation_term(SyncedMemory& selector_poly,  SyncedMemory& separation_challenge,  WitnessValues& wit_vals,  CAValues& custom_vals) {
        SyncedMemory& temp = CAGate::constraints(separation_challenge, wit_vals, custom_vals);
        SyncedMemory& res = poly_mul_const(selector_poly, temp);
        return res;
    }
};