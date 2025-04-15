#include "mod.cuh"

SyncedMemory range_constraints(SyncedMemory separation_challenge,  WitnessValues wit_vals, CustomGate custom_vals) {

    SyncedMemory four = fr::make_tensor(4);
    void* four_gpu_data=four.mutable_gpu_data();
   
    SyncedMemory kappa = mul_mod(separation_challenge, separation_challenge);
    SyncedMemory kappa_sq = mul_mod(kappa, kappa);
    SyncedMemory kappa_cu = mul_mod(kappa_sq, kappa);

    SyncedMemory b_1_1 = mul_mod_scalar(wit_vals.d_val, four);
    SyncedMemory f_b1 = sub_mod(wit_vals.c_val, b_1_1);
    SyncedMemory b_1 = delta(f_b1);

    SyncedMemory b_2_1 = mul_mod(four, wit_vals.c_val);
    SyncedMemory b_2_2 = sub_mod(wit_vals.b_val, b_2_1);
    SyncedMemory f_b2 = delta(b_2_2);
    SyncedMemory b_2 = mul_mod(f_b2, kappa);
    
    SyncedMemory b_3_1 = mul_mod(four, wit_vals.b_val);
    SyncedMemory b_3_2 = sub_mod(wit_vals.a_val, b_3_1);
    SyncedMemory f_b3 = delta(b_3_2);
    SyncedMemory b_3 = mul_mod(f_b3, kappa_sq);
  
    SyncedMemory b_4_1 = mul_mod(four, wit_vals.a_val);
    SyncedMemory b_4_2 = sub_mod(custom_vals.d_next, b_4_1);
    SyncedMemory f_b4 = delta(b_4_2);
    SyncedMemory b_4 = mul_mod(f_b4, kappa_cu);
   
    SyncedMemory mid1 = add_mod(b_1, b_2);
    SyncedMemory mid2 = add_mod(mid1, b_3);
    SyncedMemory mid3 = add_mod(mid2, b_4);
    SyncedMemory res = mul_mod(mid3, separation_challenge);

    return res;
}

SyncedMemory range_quotient_term(SyncedMemory selector, SyncedMemory separation_challenge,  WitnessValues wit_vals,  CustomGate custom_vals) {
 
    SyncedMemory four = fr::make_tensor(4);
    void* four_gpu_data = four.mutable_gpu_data();

    SyncedMemory kappa = mul_mod(separation_challenge, separation_challenge); 
    void* kappa_gpu = kappa.mutable_gpu_data();
    SyncedMemory kappa_sq = mul_mod(kappa, kappa);
    SyncedMemory kappa_cu = mul_mod(kappa_sq, kappa);

    SyncedMemory mid_temp_1 = mul_mod_scalar(wit_vals.d_val, four);
    SyncedMemory mid_temp_2 = sub_mod(wit_vals.c_val, mid_temp_1);
    SyncedMemory b_1 = delta(mid_temp_2);

    mid_temp_1 = SyncedMemory();
    mid_temp_2 = SyncedMemory();
    
    SyncedMemory mid_temp_3 = mul_mod_scalar(wit_vals.c_val, four);
    SyncedMemory mid_temp_4 = sub_mod(wit_vals.b_val, mid_temp_3);
    SyncedMemory mid_temp_5 = delta(mid_temp_4);
    SyncedMemory b_2 = mul_mod_scalar(mid_temp_5, kappa);

    mid_temp_3 = SyncedMemory();
    mid_temp_4 = SyncedMemory();
    mid_temp_5 = SyncedMemory();

    SyncedMemory mid = add_mod(b_1, b_2);
    b_1 = SyncedMemory();
    b_2 = SyncedMemory();

    SyncedMemory mid_temp_6 = mul_mod_scalar(wit_vals.b_val, four);
    SyncedMemory mid_temp_7 = sub_mod(wit_vals.a_val, mid_temp_6);
    SyncedMemory mid_temp_8 = delta(mid_temp_7);
    SyncedMemory b_3 = mul_mod_scalar(mid_temp_8, kappa_sq);

    mid_temp_6 = SyncedMemory();
    mid_temp_7 = SyncedMemory();
    mid_temp_8 = SyncedMemory();

    add_mod_(mid, b_3);
    b_3 = SyncedMemory();

    SyncedMemory mid_temp_9 = mul_mod_scalar(wit_vals.a_val, four);
    SyncedMemory mid_temp_10 = sub_mod(custom_vals.d_next, mid_temp_9);
    SyncedMemory mid_temp_11 = delta(mid_temp_10);
    SyncedMemory b_4 = mul_mod_scalar(mid_temp_11, kappa_cu);

    mid_temp_9 = SyncedMemory();
    mid_temp_10 = SyncedMemory();
    mid_temp_11 = SyncedMemory();

    add_mod_(mid, b_4);
    b_4 = SyncedMemory();

    SyncedMemory temp = mul_mod_scalar(mid, separation_challenge);

    SyncedMemory res = mul_mod(selector, temp);

    return res;
}

SyncedMemory range_linearisation_term(SyncedMemory selector_poly, SyncedMemory separation_challenge, WitnessValues wit_vals, CustomGate custom_vals) {
    SyncedMemory temp = range_constraints(separation_challenge, wit_vals, custom_vals);
    if (selector_poly.size() == 0) {
        SyncedMemory res((size_t)0);
        return res;
    } else {
       SyncedMemory res = mul_mod_scalar(selector_poly, temp);
       return res;
    }
}