#include "mod.cuh"
SyncedMemory _delta_xor_and(
    SyncedMemory a,
    SyncedMemory b,
    SyncedMemory w,
    SyncedMemory c,
    SyncedMemory q_c
) {
    
    SyncedMemory two = fr::make_tensor(2);
    SyncedMemory three = fr::make_tensor(3);
    SyncedMemory four = fr::make_tensor(4);
    SyncedMemory nine = fr::make_tensor(9);
    SyncedMemory eighteen = fr::make_tensor(18);
    SyncedMemory eighty_one = fr::make_tensor(81);
    SyncedMemory eighty_three = fr::make_tensor(83);
    void* two_gpu_data = two.mutable_gpu_data();
    void* three_gpu_data = three.mutable_gpu_data();
    void* four_gpu_data = four.mutable_gpu_data();
    void* nine_gpu_data = nine.mutable_gpu_data();
    void* eighteen_gpu_data = eighteen.mutable_gpu_data();
    void* eighty_one_gpu_data = eighty_one.mutable_gpu_data();
    void* eighty_three_gpu_data = eighty_three.mutable_gpu_data();

    
    SyncedMemory a_plus_b = add_mod(a, b);
    SyncedMemory f_1_1 = mul_mod_scalar(w, four);
    SyncedMemory f_1_2 = mul_mod_scalar(a_plus_b, eighteen);
    SyncedMemory f_1 = sub_mod(f_1_1, f_1_2);
    add_mod_scalar_(f_1, eighty_one);
    mul_mod_(f_1, w);
    
    SyncedMemory f_2_1 = mul_mod(a, a);
    SyncedMemory f_2_2 = mul_mod(b, b);
    SyncedMemory f_2 = add_mod(f_2_1, f_2_2);
    mul_mod_scalar_(f_2, eighteen);
    f_2_1 = SyncedMemory();
    f_2_2 = SyncedMemory();
    
    SyncedMemory f = add_mod(f_1, f_2);
    SyncedMemory f_3 = mul_mod_scalar(a_plus_b, eighty_one);
    sub_mod_(f, f_3);
    f_1 = SyncedMemory();
    f_2 = SyncedMemory();
    f_3 = SyncedMemory();

    SyncedMemory e_1= add_mod(a_plus_b, c);
    SyncedMemory b_2 = mul_mod_scalar(a_plus_b, three);
    SyncedMemory b_1 = mul_mod_scalar(c, nine);
    SyncedMemory b_ = sub_mod(b_1, b_2);
    mul_mod_(b_, q_c);
    a_plus_b = SyncedMemory();
    b_1 = SyncedMemory();
    b_2 = SyncedMemory();
    
    add_mod_scalar_(f, eighty_three);
    mul_mod_(f, w);
    mul_mod_scalar_(e_1, three);
    SyncedMemory e_2 = mul_mod_scalar(f, two);
    SyncedMemory e = sub_mod(e_1, e_2);
    e_1 = SyncedMemory();
    e_2 = SyncedMemory();
    
    SyncedMemory res = add_mod(b_, e);
    return res;
}
SyncedMemory logic_constraints(SyncedMemory separation_challenge, WitnessValues wit_vals, CustomGate custom_vals) {

    
    SyncedMemory four = fr::make_tensor(4);
    void* four_gpu_data=four.mutable_gpu_data();
    

    SyncedMemory kappa = mul_mod(separation_challenge, separation_challenge);
    SyncedMemory kappa_sq = mul_mod(kappa, kappa);
    SyncedMemory kappa_cu = mul_mod(kappa_sq, kappa);
    SyncedMemory kappa_qu = mul_mod(kappa_cu, kappa);

    
    SyncedMemory a_1 = mul_mod_scalar(wit_vals.a_val, four);
    SyncedMemory a = sub_mod(custom_vals.a_next, a_1);
    SyncedMemory c_0 = delta(a);

  
    SyncedMemory b_1 = mul_mod_scalar(wit_vals.b_val, four);
    SyncedMemory b = sub_mod(custom_vals.b_next, b_1);
    SyncedMemory c_1 = delta(b);

    
    SyncedMemory d_1 = mul_mod_scalar(wit_vals.d_val, four);
    SyncedMemory d = sub_mod(custom_vals.d_next, d_1);
    SyncedMemory c_2 = delta(d);

   
    SyncedMemory w = wit_vals.c_val;
    SyncedMemory w_1 = mul_mod(a, b);
    SyncedMemory w_2 = sub_mod(w, w_1);
    SyncedMemory c_3 = mul_mod(w_2, kappa_cu);

    
    SyncedMemory c_4_1 = _delta_xor_and(a, b, w, d, custom_vals.q_c);
    SyncedMemory c_4 = mul_mod(c_4_1, kappa_qu);

   
    SyncedMemory mid1 = add_mod(c_0, c_1);
    SyncedMemory mid2 = add_mod(mid1, c_2);
    SyncedMemory mid3 = add_mod(mid2, c_3);
    SyncedMemory mid4 = add_mod(mid3, c_4);
    SyncedMemory res = mul_mod(mid4, separation_challenge);

    return res;
}

SyncedMemory logic_quotient_term(
    SyncedMemory selector,
    SyncedMemory separation_challenge,
    WitnessValues wit_vals,
    CustomGate custom_vals
) {
    SyncedMemory four = fr::make_tensor(4);
    void* four_gpu_data=four.mutable_gpu_data();
    
    SyncedMemory kappa = mul_mod(separation_challenge, separation_challenge);
    void* kappa_gpu = kappa.mutable_gpu_data();
    SyncedMemory kappa_sq = mul_mod(kappa, kappa);
    SyncedMemory kappa_cu = mul_mod(kappa_sq, kappa);
    SyncedMemory kappa_qu = mul_mod(kappa_cu, kappa);

    SyncedMemory a_temp = mul_mod_scalar(wit_vals.a_val, four);
    SyncedMemory a = sub_mod(custom_vals.a_next, a_temp);
    a_temp = SyncedMemory();

    SyncedMemory b_temp = mul_mod_scalar(wit_vals.b_val, four);
    SyncedMemory b = sub_mod(custom_vals.b_next, b_temp);
    b_temp = SyncedMemory();

    SyncedMemory d_temp = mul_mod_scalar(wit_vals.d_val, four);
    SyncedMemory d = sub_mod(custom_vals.d_next, d_temp);
    d_temp = SyncedMemory();
    
    SyncedMemory c_4 = _delta_xor_and(a, b, wit_vals.c_val, d, custom_vals.q_c);
    SyncedMemory c_2 = delta(d);
    d = SyncedMemory();

    SyncedMemory c_0 = delta(a);
    SyncedMemory c_1 = delta(b);
    SyncedMemory c_3_temp = mul_mod(a, b);
    a = SyncedMemory();
    b = SyncedMemory();

    SyncedMemory res = add_mod(c_0, c_1);
    c_0 = SyncedMemory();
    c_1 = SyncedMemory();
    add_mod_(res, c_2);
    c_2 = SyncedMemory();

    SyncedMemory c_3 = sub_mod(wit_vals.c_val, c_3_temp);
    c_3_temp = SyncedMemory();
    mul_mod_scalar_(c_3, kappa_cu);
    add_mod_(res, c_3);
    c_3 = SyncedMemory();

    void* kappa_qu_data=kappa_qu.mutable_gpu_data();
    mul_mod_scalar_(c_4, kappa_qu);
    add_mod_(res, c_4);
    c_4 = SyncedMemory();

    mul_mod_scalar_(res, separation_challenge);
    mul_mod_(res, selector);
    return res;
}

SyncedMemory logic_linearisation_term(
    SyncedMemory selector_poly, 
    SyncedMemory separation_challenge, 
    WitnessValues wit_vals, 
    CustomGate custom_vals) 
{
    SyncedMemory temp = logic_constraints(separation_challenge, wit_vals, custom_vals);
    SyncedMemory res = poly_mul_const(selector_poly, temp);
    return res;
}