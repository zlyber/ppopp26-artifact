#include "caffe/syncedmem.hpp"
#include "PLONK/src/bls12_381/fr.hpp"
#include "PLONK/plonk_core/src/proof_system/widget/mod.cu"
#include "PLONK/src/bls12_381/edwards.h"
#include "PLONK/src/arithmetic.cu"
#include "PLONK/plonk_core/src/proof_system/widget/custom_class.cu"
class FBSMValues {
public:
    
    SyncedMemory& a_next_val;
    SyncedMemory& b_next_val;
    SyncedMemory& d_next_val;
    SyncedMemory& q_l_val;
    SyncedMemory& q_r_val;
    SyncedMemory& q_c_val;

    FBSMValues(SyncedMemory& a_next_val, SyncedMemory& b_next_val, SyncedMemory& d_next_val, SyncedMemory& q_l_val, SyncedMemory& q_r_val, SyncedMemory& q_c_val) :
        a_next_val(a_next_val), b_next_val(b_next_val), d_next_val(d_next_val), q_l_val(q_l_val), q_r_val(q_r_val), q_c_val(q_c_val) {}

    static FBSMValues from_evaluations(const Custom_class& custom_evals) {
        SyncedMemory& a_next_val = custom_evals.a_next_eval;
        SyncedMemory& b_next_val = custom_evals.b_next_eval;
        SyncedMemory& d_next_val = custom_evals.d_next_eval;
        SyncedMemory& q_l_val = custom_evals.q_l_eval;
        SyncedMemory& q_r_val = custom_evals.q_r_eval;
        SyncedMemory& q_c_val = custom_evals.q_c_eval;

        return FBSMValues(a_next_val, b_next_val, d_next_val, q_l_val, q_r_val, q_c_val);
    }
};

class FBSMGate {
public:
    SyncedMemory& constraints(
        SyncedMemory& separation_challenge, const WitnessValues& wit_vals, const FBSMValues& custom_vals
    ) {

        SyncedMemory& kappa = mul_mod(separation_challenge, separation_challenge);
        SyncedMemory& kappa_sq = mul_mod(kappa, kappa);
        SyncedMemory& kappa_cu = mul_mod(kappa_sq, kappa);
        SyncedMemory& one = fr::one();
        void* one_gpu_data =one.mutable_gpu_data();
    

        SyncedMemory& acc_x = wit_vals.a_val;
        SyncedMemory& acc_x_next = custom_vals.a_next_val;
        SyncedMemory& acc_y = wit_vals.b_val;
        SyncedMemory& acc_y_next = custom_vals.b_next_val;

        SyncedMemory& xy_alpha = wit_vals.c_val;

        SyncedMemory& accumulated_bit = wit_vals.d_val;
        SyncedMemory& accumulated_bit_next = custom_vals.d_next_val;
        SyncedMemory& bit = extract_bit(accumulated_bit, accumulated_bit_next);

        
        SyncedMemory& bit_consistency = check_bit_consistency(bit, one);

        SyncedMemory& y_beta_sub_one =sub_mod(custom_vals.q_r_val, one);
        SyncedMemory& bit2 = mul_mod(bit, bit);
        SyncedMemory& y_alpha_1 = mul_mod(bit2, y_beta_sub_one);
        SyncedMemory& y_alpha = add_mod(y_alpha_1, one);
        SyncedMemory& x_alpha = mul_mod(custom_vals.q_l_val, bit);

        
        // custom_vals.q_c_val=torch::tensor(from_gmpy_list_1(custom_vals.q_c_val),dtype=torch::BLS12_381_Fr_G1_Mont)
        SyncedMemory& bit_times_q_c_val = mul_mod(bit, custom_vals.q_c_val);
        SyncedMemory& xy_consistency_temp = sub_mod(bit_times_q_c_val, xy_alpha);
        SyncedMemory& xy_consistency = mul_mod(xy_consistency_temp, kappa);

        
        SyncedMemory& x_3 = acc_x_next;
        SyncedMemory& x_3_times_xy_alpha = mul_mod(x_3, xy_alpha);
        SyncedMemory& x_3_times_xy_alpha_times_acc_x = mul_mod(x_3_times_xy_alpha, acc_x);
        SyncedMemory& x_3_times_xy_alpha_times_acc_x_times_acc_y = mul_mod(
            x_3_times_xy_alpha_times_acc_x, acc_y
        );
        SyncedMemory& x_3_times_xy_alpha_times_acc_x_times_acc_y_times_coeff_d = mul_mod(
            x_3_times_xy_alpha_times_acc_x_times_acc_y, COEFF_D()
        );
        SyncedMemory& lhs_x = add_mod(x_3, x_3_times_xy_alpha_times_acc_x_times_acc_y_times_coeff_d);
        SyncedMemory& x_3_times_acc_y = mul_mod(x_alpha, acc_y);
        SyncedMemory& y_alpha_times_acc_x = mul_mod(y_alpha, acc_x);
        SyncedMemory& rhs_x = add_mod(x_3_times_acc_y, y_alpha_times_acc_x);
        SyncedMemory& x_acc_consistency_temp = sub_mod(lhs_x, rhs_x);
        SyncedMemory& x_acc_consistency = mul_mod(x_acc_consistency_temp, kappa_sq);

        
        SyncedMemory& y_3 = acc_y_next;
        SyncedMemory& y_3_times_xy_alpha = mul_mod(y_3, xy_alpha);
        SyncedMemory& y_3_times_xy_alpha_times_acc_x = mul_mod(y_3_times_xy_alpha, acc_x);
        SyncedMemory& y_3_times_xy_alpha_times_acc_x_times_acc_y = mul_mod(
            y_3_times_xy_alpha_times_acc_x, acc_y
        );
        SyncedMemory& y_3_times_xy_alpha_times_acc_x_times_acc_y_times_coeff_d = mul_mod(
            y_3_times_xy_alpha_times_acc_x_times_acc_y, COEFF_D()
        );
        SyncedMemory& lhs_y =sub_mod(y_3, y_3_times_xy_alpha_times_acc_x_times_acc_y_times_coeff_d);
        SyncedMemory& y_alpha_times_acc_y = mul_mod(y_alpha, acc_y);
        SyncedMemory& coeff_A_times_x_alpha = mul_mod(COEFF_A(), x_alpha);
        SyncedMemory& coeff_A_times_x_alpha_times_acc_x =mul_mod(coeff_A_times_x_alpha, acc_x);
        SyncedMemory& rhs_y = sub_mod(y_alpha_times_acc_y, coeff_A_times_x_alpha_times_acc_x);
        SyncedMemory& y_acc_consistency_temp = sub_mod(lhs_y, rhs_y);
        SyncedMemory& y_acc_consistency = mul_mod(y_acc_consistency_temp, kappa_cu);

        SyncedMemory& mid1 = add_mod(bit_consistency, x_acc_consistency);
        SyncedMemory& mid2 = add_mod(mid1, y_acc_consistency);
        SyncedMemory& checks = add_mod(mid2, xy_consistency);
        SyncedMemory& res = mul_mod(checks, separation_challenge);
        return res;
    }
    
    SyncedMemory& quotient_term( SyncedMemory& selector,SyncedMemory& separation_challenge,WitnessValues wit_vals,FBSMValues custom_vals) 
    {
        SyncedMemory& bit_consistency = mul_mod_scalar(selector, custom_vals.q_c_val);

    SyncedMemory& kappa = mul_mod(separation_challenge, separation_challenge);
    SyncedMemory& kappa_sq = mul_mod(kappa, kappa);
    SyncedMemory& kappa_cu = mul_mod(kappa_sq, kappa);
    SyncedMemory& one = fr::one();
    void* one_gpu_data =one.mutable_gpu_data();

    SyncedMemory& bit = extract_bit(wit_vals.d_val, custom_vals.d_next_val);
    SyncedMemory& y_alpha_temp_1 = sub_mod_scalar(custom_vals.q_r_val, one);
    SyncedMemory& bit2 = mul_mod(bit, bit);
    SyncedMemory& y_alpha_temp_2 = mul_mod(bit2, y_alpha_temp_1);
    // bit2.reset();
    SyncedMemory& y_alpha = add_mod_scalar(y_alpha_temp_2, one);
    SyncedMemory& x_alpha = mul_mod(custom_vals.q_l_val, bit);

    SyncedMemory& mid_temp_1 = mul_mod(custom_vals.a_next_val, wit_vals.c_val);
    SyncedMemory& mid_temp_2 = mul_mod(mid_temp_1, wit_vals.a_val);
    SyncedMemory& mid_temp_3 = mul_mod(mid_temp_2, wit_vals.b_val);
    SyncedMemory& coeff_d=COEFF_D();
    void* coeff_d_gpu_data =coeff_d.mutable_gpu_data();
    SyncedMemory& mid_temp_4 = mul_mod_scalar(mid_temp_3, coeff_d);
    SyncedMemory& lhs_x = add_mod(custom_vals.a_next_val, mid_temp_4);
    // mid.reset();

    SyncedMemory& rhs_x_temp = mul_mod(x_alpha, wit_vals.b_val);
    SyncedMemory& y_alpha_times_acc_x = mul_mod(y_alpha, wit_vals.a_val);
    SyncedMemory& rhs_x = add_mod(rhs_x_temp, y_alpha_times_acc_x);
    SyncedMemory& rhs_y = mul_mod(y_alpha, wit_vals.b_val);
    // y_alpha.reset();
    // y_alpha_times_acc_x.reset();

    SyncedMemory& x_acc_consistency_temp = sub_mod(lhs_x, rhs_x);
    // lhs_x.reset();
    // rhs_x.reset();
    void* kappa_sq_gpu_data=kappa_sq.mutable_gpu_data();
    SyncedMemory& x_acc_consistency = mul_mod_scalar(x_acc_consistency_temp, kappa_sq);
    SyncedMemory& bit_consistency = check_bit_consistency(bit, one);
    SyncedMemory& mid1_temp_1 = add_mod(bit_consistency, x_acc_consistency);
    SyncedMemory& xy_consistency_temp_1 = mul_mod(bit, custom_vals.q_c_val);
    // x_acc_consistency.reset();
    // bit.reset();
    // bit_consistency.reset();

    SyncedMemory& mid2_temp_1 = mul_mod(custom_vals.b_next_val, wit_vals.c_val);
    SyncedMemory& mid2_temp_2 = mul_mod(mid2_temp_1, wit_vals.a_val);
    SyncedMemory& mid2_temp_3 = mul_mod(mid2_temp_2, wit_vals.b_val);
    SyncedMemory& mid = mul_mod_scalar(mid2_temp_3, coeff_d);
    SyncedMemory& lhs_y = sub_mod(custom_vals.b_next_val, mid);
    SyncedMemory& coeff_a = COEFF_A();
    void* coeff_a_gpu_data = coeff_a.mutable_gpu_data();
    SyncedMemory& mid3_temp_1 = mul_mod_scalar(x_alpha, coeff_a);
    // x_alpha.reset();

    SyncedMemory& mid3 = mul_mod(mid3_temp_1, wit_vals.a_val);
    SyncedMemory& rhs_y = sub_mod(rhs_y, mid3);
    // mid.reset();

    SyncedMemory& y_acc_consistency_temp = sub_mod(lhs_y, rhs_y);
    // lhs_y.reset();
    // rhs_y.reset();
    void* kappa_cu_gpu_data=kappa_cu.mutable_gpu_data();
    void* kappa_gpu_data=kappa.mutable_gpu_data();
    SyncedMemory& y_acc_consistency = mul_mod_scalar(y_acc_consistency_temp, kappa_cu);
    SyncedMemory& xy_consistency_temp_2 = sub_mod(xy_consistency_temp_1, wit_vals.c_val);
    SyncedMemory& xy_consistency = mul_mod_scalar(xy_consistency_temp_2, kappa);
    SyncedMemory& mid1_temp_2 = add_mod(mid1_temp_1, y_acc_consistency);
    SyncedMemory& res_temp_1 = add_mod(mid1_temp_2, xy_consistency);
    // mid1.reset();
    // y_acc_consistency.reset();
    // xy_consistency.reset();
    void* separation_challenge_gpu_data=separation_challenge.mutable_gpu_data();
    SyncedMemory& res_temp_2 = mul_mod_scalar(res_temp_1, separation_challenge);
    SyncedMemory& res = mul_mod(selector, res_temp_2);
    return res;
}

    SyncedMemory& linearisation_term(SyncedMemory&selector_poly, SyncedMemory& separation_challenge, WitnessValues&wit_vals, FBSMValues&custom_vals) {
        
        SyncedMemory&temp = FBSMGate::constraints(separation_challenge, wit_vals, custom_vals);
        SyncedMemory&res = poly_mul_const(selector_poly, temp);
        return res;
    }
};
SyncedMemory& extract_bit(SyncedMemory&curr_acc, SyncedMemory& next_acc) {
    
    SyncedMemory& res_temp = sub_mod(next_acc, curr_acc);
    SyncedMemory& res = sub_mod(res_temp, curr_acc);
    return res;
}


SyncedMemory& check_bit_consistency(SyncedMemory& bit, SyncedMemory& one) {

    SyncedMemory& mid_temp = sub_mod_scalar(bit, one);
    SyncedMemory& res_temp = mul_mod(mid_temp, bit);
    SyncedMemory& mid = add_mod_scalar(bit, one);
    SyncedMemory& res = mul_mod(res_temp, mid);
    return res;
}