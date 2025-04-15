#include "mod.cuh"
#include "../../../../bls12_381/edwards.cuh"

CAValues::CAValues(SyncedMemory a, SyncedMemory b, SyncedMemory d)
    : a_next_val(a), b_next_val(b), d_next_val(d) {}

CAValues CAValues::from_evaluations(CustomGate custom_evals) {
    SyncedMemory a_next_val = custom_evals.a_next;
    SyncedMemory b_next_val = custom_evals.b_next;
    SyncedMemory d_next_val = custom_evals.d_next;
    CAValues res = CAValues{a_next_val, b_next_val, d_next_val};
    return res;
}


static SyncedMemory CAGate_constraints(SyncedMemory separation_challenge, const WitnessValues wit_vals, CAValues custom_vals) {
    SyncedMemory x_1 = wit_vals.a_val;
    SyncedMemory x_3 = custom_vals.a_next_val;
    SyncedMemory y_1 = wit_vals.b_val;
    SyncedMemory y_3 = custom_vals.b_next_val;
    SyncedMemory x_2 = wit_vals.c_val;
    SyncedMemory y_2 = wit_vals.d_val;
    SyncedMemory x1_y2 = custom_vals.d_next_val;
    SyncedMemory kappa = mul_mod(separation_challenge, separation_challenge);
    SyncedMemory x1y2 = mul_mod(x_1, y_2);
    SyncedMemory xy_consistency =sub_mod(x1y2, x1_y2);
    SyncedMemory y1_x2 = mul_mod(y_1, x_2);
    SyncedMemory y1_y2 = mul_mod(y_1, y_2);
    SyncedMemory x1_x2 = mul_mod(x_1, x_2);
    SyncedMemory x3_lhs = add_mod(x1_y2, y1_x2);
    SyncedMemory x_3_D = mul_mod(x_3, COEFF_D());
    SyncedMemory x_3_D_x1_y2 = mul_mod(x_3_D, x1_y2);
    SyncedMemory x_3_D_x1_y2_y1_x2 = mul_mod(x_3_D_x1_y2, y1_x2);
    SyncedMemory x3_rhs = add_mod(x_3, x_3_D_x1_y2_y1_x2);
    SyncedMemory x3_l_sub_r = sub_mod(x3_lhs, x3_rhs);
    SyncedMemory x3_consistency = mul_mod(x3_l_sub_r, kappa);
    SyncedMemory x1_x2_A = mul_mod(COEFF_A(), x1_x2);
    SyncedMemory y3_lhs = sub_mod(y1_y2, x1_x2_A);
    SyncedMemory y_3_D = mul_mod(y_3, COEFF_D());
    SyncedMemory y_3_D_x1_y2 = mul_mod(y_3_D, x1_y2);
    SyncedMemory y_3_D_x1_y2_y1_x2 = mul_mod(y_3_D_x1_y2, y1_x2);
    SyncedMemory y3_rhs = sub_mod(y_3, y_3_D_x1_y2_y1_x2);
    SyncedMemory y3_l_sub_r = sub_mod(y3_lhs, y3_rhs);
    SyncedMemory kappa2 = mul_mod(kappa, kappa);
    SyncedMemory y3_consistency = mul_mod(y3_l_sub_r, kappa2);

    SyncedMemory mid1 = add_mod(xy_consistency, x3_consistency);
    SyncedMemory mid2 = add_mod(mid1, y3_consistency);
    SyncedMemory result = mul_mod(mid2, separation_challenge);
    return result;
}

SyncedMemory CAGate_quotient_term(SyncedMemory selector, SyncedMemory separation_challenge, WitnessValues wit_vals, CAValues custom_vals) {

    void* separation_challenge_gpu_data=separation_challenge.mutable_gpu_data();
    SyncedMemory kappa = mul_mod(separation_challenge, separation_challenge);
    SyncedMemory kappasq = mul_mod(kappa, kappa);

    SyncedMemory y1_x2 = mul_mod(custom_vals.a_next_val, wit_vals.c_val);
    SyncedMemory coeff_d = COEFF_D();
    SyncedMemory mid = mul_mod_scalar(custom_vals.a_next_val, coeff_d);
    mul_mod_(mid, custom_vals.d_next_val);
    mul_mod_(mid, y1_x2);
    SyncedMemory x3_rhs = add_mod(custom_vals.a_next_val, mid);
    SyncedMemory x3_lhs = add_mod(custom_vals.d_next_val, y1_x2);
    SyncedMemory x3_l_sub_r = sub_mod(x3_lhs, x3_rhs);
    x3_lhs = SyncedMemory();
    x3_rhs = SyncedMemory();

    SyncedMemory temp = mul_mod_scalar(custom_vals.b_next_val, coeff_d);
    mul_mod_(temp, custom_vals.d_next_val);
    mul_mod_(temp, y1_x2);
    y1_x2 = SyncedMemory();
    temp = SyncedMemory();
    
    SyncedMemory x3_consistency = mul_mod_scalar(x3_l_sub_r, kappa);
    x3_l_sub_r = SyncedMemory();

    SyncedMemory coeff_a = COEFF_A();
    SyncedMemory x1_x2 = mul_mod(wit_vals.a_val, wit_vals.c_val);
    mul_mod_scalar_(x1_x2, coeff_a);
    SyncedMemory y1_y2 = mul_mod(custom_vals.a_next_val, wit_vals.d_val);
    SyncedMemory y3_lhs = sub_mod(y1_y2, x1_x2);
    y1_y2 = SyncedMemory();
    x1_x2 = SyncedMemory();

    SyncedMemory y3_rhs = sub_mod(custom_vals.b_next_val, mid);
    SyncedMemory y3_l_sub_r = sub_mod(y3_lhs, y3_rhs);
    SyncedMemory y3_consistency = mul_mod_scalar(y3_l_sub_r, kappasq);
    mid = SyncedMemory();
    y3_lhs = SyncedMemory();
    y3_rhs = SyncedMemory();
    y3_l_sub_r = SyncedMemory();

    SyncedMemory x1y2 = mul_mod(wit_vals.a_val, wit_vals.d_val);
    SyncedMemory xy_consistency = sub_mod(x1y2, custom_vals.d_next_val);

    SyncedMemory res = add_mod(xy_consistency, x3_consistency);
    add_mod_(res, y3_consistency);
    
    mul_mod_scalar_(res, separation_challenge);
    mul_mod_(res, selector);
    return res;
}

SyncedMemory CAGate_linearisation_term(
    SyncedMemory selector_poly, 
    SyncedMemory separation_challenge, 
    WitnessValues wit_vals, 
    CAValues custom_vals) 
{
    SyncedMemory temp = CAGate_constraints(separation_challenge, wit_vals, custom_vals);
    SyncedMemory res = poly_mul_const(selector_poly, temp);
    return res;
}
