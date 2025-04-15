#include "mod.cuh"
#include "../../../../bls12_381/edwards.cuh"

FBSMValues::FBSMValues(SyncedMemory a_next_val, SyncedMemory b_next_val, SyncedMemory d_next_val, SyncedMemory q_l_val, SyncedMemory q_r_val, SyncedMemory q_c_val) :
    a_next_val(a_next_val), b_next_val(b_next_val), d_next_val(d_next_val), q_l_val(q_l_val), q_r_val(q_r_val), q_c_val(q_c_val) {}

FBSMValues FBSMValues::from_evaluations(CustomGate custom_evals) {
    SyncedMemory a_next_val = custom_evals.a_next;
    SyncedMemory b_next_val = custom_evals.b_next;
    SyncedMemory d_next_val = custom_evals.d_next;
    SyncedMemory q_l_val = custom_evals.q_l;
    SyncedMemory q_r_val = custom_evals.q_r;
    SyncedMemory q_c_val = custom_evals.q_c;
    FBSMValues res = FBSMValues(a_next_val, b_next_val, d_next_val, q_l_val, q_r_val, q_c_val);
    return res;
}


SyncedMemory extract_bit(SyncedMemory curr_acc, SyncedMemory next_acc) {
    SyncedMemory res = sub_mod(next_acc, curr_acc);
    sub_mod_(res, curr_acc);
    return res;
}


SyncedMemory check_bit_consistency(SyncedMemory bit, SyncedMemory one) {
    SyncedMemory mid1 = sub_mod_scalar(bit, one);
    SyncedMemory res = mul_mod(mid1, bit);
    SyncedMemory mid2 = add_mod_scalar(bit, one);
    mul_mod_(res, mid2);
    return res;
}

SyncedMemory FBSMGate_constraints(
    SyncedMemory separation_challenge,  
    WitnessValues wit_vals, 
    const FBSMValues custom_vals) 
{

    SyncedMemory kappa = mul_mod(separation_challenge, separation_challenge);
    SyncedMemory kappa_sq = mul_mod(kappa, kappa);
    SyncedMemory kappa_cu = mul_mod(kappa_sq, kappa);
    SyncedMemory one = fr::one();
    void* one_gpu_data =one.mutable_gpu_data();

    SyncedMemory acc_x = wit_vals.a_val;
    SyncedMemory acc_x_next = custom_vals.a_next_val;
    SyncedMemory acc_y = wit_vals.b_val;
    SyncedMemory acc_y_next = custom_vals.b_next_val;

    SyncedMemory xy_alpha = wit_vals.c_val;

    SyncedMemory accumulated_bit = wit_vals.d_val;
    SyncedMemory accumulated_bit_next = custom_vals.d_next_val;
    SyncedMemory bit = extract_bit(accumulated_bit, accumulated_bit_next);

    SyncedMemory bit_consistency = check_bit_consistency(bit, one);

    SyncedMemory y_beta_sub_one =sub_mod(custom_vals.q_r_val, one);
    SyncedMemory bit2 = mul_mod(bit, bit);
    SyncedMemory y_alpha_1 = mul_mod(bit2, y_beta_sub_one);
    SyncedMemory y_alpha = add_mod(y_alpha_1, one);
    SyncedMemory x_alpha = mul_mod(custom_vals.q_l_val, bit);

    SyncedMemory bit_times_q_c_val = mul_mod(bit, custom_vals.q_c_val);
    SyncedMemory xy_consistency_temp = sub_mod(bit_times_q_c_val, xy_alpha);
    SyncedMemory xy_consistency = mul_mod(xy_consistency_temp, kappa);
    
    SyncedMemory x_3 = acc_x_next;
    SyncedMemory x_3_times_xy_alpha = mul_mod(x_3, xy_alpha);
    SyncedMemory x_3_times_xy_alpha_times_acc_x = mul_mod(x_3_times_xy_alpha, acc_x);
    SyncedMemory x_3_times_xy_alpha_times_acc_x_times_acc_y = mul_mod(
        x_3_times_xy_alpha_times_acc_x, acc_y
    );
    SyncedMemory x_3_times_xy_alpha_times_acc_x_times_acc_y_times_coeff_d = mul_mod(
        x_3_times_xy_alpha_times_acc_x_times_acc_y, COEFF_D()
    );
    SyncedMemory lhs_x = add_mod(x_3, x_3_times_xy_alpha_times_acc_x_times_acc_y_times_coeff_d);
    SyncedMemory x_3_times_acc_y = mul_mod(x_alpha, acc_y);
    SyncedMemory y_alpha_times_acc_x = mul_mod(y_alpha, acc_x);
    SyncedMemory rhs_x = add_mod(x_3_times_acc_y, y_alpha_times_acc_x);
    SyncedMemory x_acc_consistency = sub_mod(lhs_x, rhs_x);
    mul_mod_(x_acc_consistency, kappa_sq);

    SyncedMemory y_3 = acc_y_next;
    SyncedMemory y_3_times_xy_alpha = mul_mod(y_3, xy_alpha);
    SyncedMemory y_3_times_xy_alpha_times_acc_x = mul_mod(y_3_times_xy_alpha, acc_x);
    SyncedMemory y_3_times_xy_alpha_times_acc_x_times_acc_y = mul_mod(
        y_3_times_xy_alpha_times_acc_x, acc_y
    );
    SyncedMemory y_3_times_xy_alpha_times_acc_x_times_acc_y_times_coeff_d = mul_mod(
        y_3_times_xy_alpha_times_acc_x_times_acc_y, COEFF_D()
    );
    SyncedMemory lhs_y =sub_mod(y_3, y_3_times_xy_alpha_times_acc_x_times_acc_y_times_coeff_d);
    SyncedMemory y_alpha_times_acc_y = mul_mod(y_alpha, acc_y);
    SyncedMemory coeff_A_times_x_alpha = mul_mod(COEFF_A(), x_alpha);
    SyncedMemory coeff_A_times_x_alpha_times_acc_x =mul_mod(coeff_A_times_x_alpha, acc_x);
    SyncedMemory rhs_y = sub_mod(y_alpha_times_acc_y, coeff_A_times_x_alpha_times_acc_x);
    SyncedMemory y_acc_consistency = sub_mod(lhs_y, rhs_y);
    mul_mod_(y_acc_consistency, kappa_cu);

    SyncedMemory mid1 = add_mod(bit_consistency, x_acc_consistency);
    SyncedMemory mid2 = add_mod(mid1, y_acc_consistency);
    SyncedMemory checks = add_mod(mid2, xy_consistency);
    SyncedMemory res = mul_mod(checks, separation_challenge);
    return res;
}

SyncedMemory FBSMGate_quotient_term( SyncedMemory selector,SyncedMemory separation_challenge,WitnessValues wit_vals, FBSMValues custom_vals) {

    SyncedMemory kappa = mul_mod(separation_challenge, separation_challenge);
    void* kappa_gpu = kappa.mutable_gpu_data();
    SyncedMemory kappa_sq = mul_mod(kappa, kappa);
    SyncedMemory kappa_cu = mul_mod(kappa_sq, kappa);
    SyncedMemory one = fr::one();
    void* one_gpu_data =one.mutable_gpu_data();

    SyncedMemory bit = extract_bit(wit_vals.d_val, custom_vals.d_next_val);
    SyncedMemory y_alpha = sub_mod_scalar(custom_vals.q_r_val, one);
    SyncedMemory bit2 = mul_mod(bit, bit);
    mul_mod_(y_alpha, bit2);
    bit2 = SyncedMemory();
    add_mod_scalar_(y_alpha, one);
    SyncedMemory x_alpha = mul_mod(custom_vals.q_l_val, bit);

    SyncedMemory mid_1 = mul_mod(custom_vals.a_next_val, wit_vals.c_val);
    mul_mod_(mid_1, wit_vals.a_val);
    mul_mod_(mid_1, wit_vals.b_val);
    SyncedMemory coeff_d = COEFF_D();
    mul_mod_scalar_(mid_1, coeff_d);
    SyncedMemory lhs_x = add_mod(custom_vals.a_next_val, mid_1);
    mid_1 = SyncedMemory();

    SyncedMemory rhs_x = mul_mod(x_alpha, wit_vals.b_val);
    SyncedMemory y_alpha_times_acc_x = mul_mod(y_alpha, wit_vals.a_val);
    add_mod_(rhs_x, y_alpha_times_acc_x);
    SyncedMemory rhs_y = mul_mod(y_alpha, wit_vals.b_val);
    y_alpha = SyncedMemory();
    y_alpha_times_acc_x = SyncedMemory();

    SyncedMemory x_acc_consistency = sub_mod(lhs_x, rhs_x);
    lhs_x = SyncedMemory();
    rhs_x = SyncedMemory();

    mul_mod_scalar_(x_acc_consistency, kappa_sq);
    SyncedMemory bit_consistency = check_bit_consistency(bit, one);
    SyncedMemory mid1 = add_mod(bit_consistency, x_acc_consistency);
    SyncedMemory xy_consistency = mul_mod(bit, custom_vals.q_c_val);
    x_acc_consistency = SyncedMemory();
    bit = SyncedMemory();
    bit_consistency = SyncedMemory();

    SyncedMemory mid_2 = mul_mod(custom_vals.b_next_val, wit_vals.c_val);
    mul_mod_(mid_2, wit_vals.a_val);
    mul_mod_(mid_2, wit_vals.b_val);
    mul_mod_scalar_(mid_2, coeff_d);
    SyncedMemory lhs_y = sub_mod(custom_vals.b_next_val, mid_2);
    SyncedMemory coeff_a = COEFF_A();
    SyncedMemory mid_3 = mul_mod_scalar(x_alpha, coeff_a);
    x_alpha = SyncedMemory();
    mid_2 = SyncedMemory();

    mul_mod_(mid_3, wit_vals.a_val);
    sub_mod_(rhs_y, mid_3);
    mid_3 = SyncedMemory();

    SyncedMemory y_acc_consistency = sub_mod(lhs_y, rhs_y);
    lhs_y = SyncedMemory();
    rhs_y = SyncedMemory();

    mul_mod_scalar_(y_acc_consistency, kappa_cu);
    sub_mod_(xy_consistency, wit_vals.c_val);
    mul_mod_scalar_(xy_consistency, kappa);
    add_mod_(mid1, y_acc_consistency);
    SyncedMemory res = add_mod(mid1, xy_consistency);
    y_acc_consistency = SyncedMemory();
    xy_consistency = SyncedMemory();
    void* separation_challenge_gpu_data = separation_challenge.mutable_gpu_data();
    mul_mod_scalar_(res, separation_challenge);
    mul_mod_(res, selector);
    return res;
}

SyncedMemory FBSMGate_linearisation_term(
    SyncedMemory selector_poly, 
    SyncedMemory separation_challenge, 
    WitnessValues wit_vals, 
    FBSMValues custom_vals) 
{    
    SyncedMemory temp = FBSMGate_constraints(separation_challenge, wit_vals, custom_vals);
    SyncedMemory res = poly_mul_const(selector_poly, temp);
    return res;
}
