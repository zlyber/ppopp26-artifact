#include "caffe/syncedmem.hpp"
#include "PLONK/src/bls12_381/fr.hpp"
#include "PLONK/src/plonk_core/src/proof_system/widget/mod.cu"
#include "PLONK/src/plonk_core/src/proof_system/widget/fixed_base_scalar_mul.cu"
#include "PLONK/src/plonk_core/src/proof_system/widget/curve_addition.cu"
#include "PLONK/src/plonk_core/src/proof_system/widget/custom_class.cu"
SyncedMemory& _delta_xor_and(
    SyncedMemory& a,
    SyncedMemory& b,
    SyncedMemory& w,
    SyncedMemory& c,
    SyncedMemory& q_c
) {
    
    SyncedMemory& two = fr::make_tensor(2);
    SyncedMemory& three = fr::make_tensor(3);
    SyncedMemory& four = fr::make_tensor(4);
    SyncedMemory& nine = fr::make_tensor(9);
    SyncedMemory& eighteen = fr::make_tensor(18);
    SyncedMemory& eighty_one = fr::make_tensor(81);
    SyncedMemory& eighty_three = fr::make_tensor(83);
    void* two_gpu_data= two.mutable_gpu_data();
    void* three_gpu_data= three.mutable_gpu_data();
    void* four_gpu_data= four.mutable_gpu_data();
    void* nine_gpu_data= nine.mutable_gpu_data();
    void* eighteen_gpu_data= eighteen.mutable_gpu_data();
    void* eighty_one_gpu_data= eighty_one.mutable_gpu_data();
    void* eighty_three_gpu_data= eighty_three.mutable_gpu_data();

    
    SyncedMemory& a_plus_b = add_mod(a, b);
    SyncedMemory& f_1_1 = mul_mod_scalar(w, four);
    SyncedMemory& f_1_2 = mul_mod_scalar(a_plus_b, eighteen);
    SyncedMemory& f_1_temp_1 = sub_mod(f_1_1, f_1_2);
    SyncedMemory& f_1_temp_2 = add_mod_scalar(f_1_temp_1, eighty_one);
    SyncedMemory& f_1 = mul_mod(f_1_temp_2, w);
    f_1_1.~SyncedMemory();
    f_1_2.~SyncedMemory();

    
    SyncedMemory& f_2_1_1 = mul_mod(a, a);
    SyncedMemory& f_2_1_2 = mul_mod(b, b);
    SyncedMemory& f_2_temp = add_mod(f_2_1_1, f_2_1_2);
    SyncedMemory& f_2 = mul_mod_scalar(f_2_temp, eighteen);
    f_2_1_1.~SyncedMemory();
    f_2_1_2.~SyncedMemory();
    
    SyncedMemory& f_temp_1 = add_mod(f_1, f_2);
    SyncedMemory& f_3 = mul_mod_scalar(a_plus_b, eighty_one);
    SyncedMemory& f_temp_2 = sub_mod(f_temp_1, f_3);
    f_temp_1.~SyncedMemory();
    f_1.~SyncedMemory();
    f_2.~SyncedMemory();
    f_3.~SyncedMemory();

    SyncedMemory& e_1_temp_1 = add_mod(a_plus_b, c);
    SyncedMemory& b_2 = mul_mod_scalar(a_plus_b, three);
    SyncedMemory& b_1 = mul_mod_scalar(c, nine);
    SyncedMemory& b_res_temp = sub_mod(b_1, b_2);
    SyncedMemory&  b_res = mul_mod(q_c, b_res_temp);
    b_1.~SyncedMemory();
    b_2.~SyncedMemory();
    
    SyncedMemory& f_temp_3 = add_mod_scalar(f_temp_2, eighty_three);
    SyncedMemory& f = mul_mod(w, f_temp_3);
    SyncedMemory& e_1 = mul_mod_scalar(e_1_temp_1, three);
    SyncedMemory& e_2 = mul_mod_scalar(f, two);
    SyncedMemory& e = sub_mod(e_1, e_2);
    e_1_temp_1.~SyncedMemory();
    e_1.~SyncedMemory();
    e_2.~SyncedMemory();
    
    SyncedMemory& res = add_mod(b_res, e);
    return res;
}
SyncedMemory& logic_constraints(SyncedMemory& separation_challenge, WitnessValues& wit_vals, Custom_class& custom_vals) {

    
    SyncedMemory& four = fr::make_tensor(4);
    void* four_gpu_data=four.mutable_gpu_data();
    

    SyncedMemory& kappa = mul_mod(separation_challenge, separation_challenge);
    SyncedMemory& kappa_sq = mul_mod(kappa, kappa);
    SyncedMemory& kappa_cu = mul_mod(kappa_sq, kappa);
    SyncedMemory& kappa_qu = mul_mod(kappa_cu, kappa);

    
    SyncedMemory& a_1 = mul_mod_scalar(wit_vals.a_val, four);
    SyncedMemory& a = sub_mod(custom_vals.a_next_eval, a_1);
    SyncedMemory& c_0 = delta(a);

  
    SyncedMemory& b_1 = mul_mod_scalar(wit_vals.b_val, four);
    SyncedMemory& b = sub_mod(custom_vals.b_next_eval, b_1);
    SyncedMemory& c_1 = delta(b);

    
    SyncedMemory& d_1 = mul_mod_scalar(wit_vals.d_val, four);
    SyncedMemory& d = sub_mod(custom_vals.d_next_eval, d_1);
    SyncedMemory& c_2 = delta(d);

   
    SyncedMemory& w = wit_vals.c_val;
    SyncedMemory& w_1 = mul_mod(a, b);
    SyncedMemory& w_2 = sub_mod(w, w_1);
    SyncedMemory& c_3 = mul_mod(w_2, kappa_cu);

    
    SyncedMemory& c_4_1 = _delta_xor_and(a, b, w, d, custom_vals.q_c_eval);
    SyncedMemory& c_4 = mul_mod(c_4_1, kappa_qu);

   
    SyncedMemory& mid1 = add_mod(c_0, c_1);
    SyncedMemory& mid2 = add_mod(mid1, c_2);
    SyncedMemory& mid3 = add_mod(mid2, c_3);
    SyncedMemory& mid4 = add_mod(mid3, c_4);
    SyncedMemory& res = mul_mod(mid4, separation_challenge);

    return res;
}

SyncedMemory& logic_quotient_term(
    SyncedMemory& selector,
    SyncedMemory& separation_challenge,
    WitnessValues& wit_vals,
    Custom_class& custom_vals
) {
    SyncedMemory& four = fr::make_tensor(4);
    void* four_gpu_data=four.mutable_gpu_data();
    
    SyncedMemory& kappa = mul_mod(separation_challenge, separation_challenge);
    SyncedMemory& kappa_sq = mul_mod(kappa, kappa);
    SyncedMemory& kappa_cu = mul_mod(kappa_sq, kappa);
    SyncedMemory& kappa_qu = mul_mod(kappa_cu, kappa);

    SyncedMemory& a_temp = mul_mod_scalar(wit_vals.a_val, four);
    SyncedMemory& a = sub_mod(custom_vals.a_next_eval, a_temp);
    SyncedMemory& c_0 = delta(a);

    SyncedMemory& b_temp = mul_mod_scalar(wit_vals.b_val, four);
    SyncedMemory& b = sub_mod(custom_vals.b_next_eval, b_temp);
    SyncedMemory& c_1 = delta(b);
    SyncedMemory& res_temp_1 = add_mod(c_0, c_1);
    c_0.~SyncedMemory();
    c_1.~SyncedMemory();

    SyncedMemory& d_temp = mul_mod_scalar(wit_vals.d_val, four);
    SyncedMemory& d = sub_mod(custom_vals.d_next_eval, d_temp);
    SyncedMemory& c_2 = delta(d);
    SyncedMemory& res_temp_2 = add_mod(res_temp_1, c_2);
    c_2.~SyncedMemory();

    SyncedMemory& c_3_temp_1 = mul_mod(a, b);
    SyncedMemory& c_4_temp_1 = _delta_xor_and(a, b, wit_vals.c_val, d, custom_vals.q_c_eval);
    a.~SyncedMemory();
    b.~SyncedMemory();
    d.~SyncedMemory();

    SyncedMemory& c_3_temp_2 = sub_mod(wit_vals.c_val, c_3_temp_1);
    void* kappa_cu_data=kappa_cu.mutable_gpu_data();
    SyncedMemory& c_3 = mul_mod_scalar(c_3_temp_2, kappa_cu);
    SyncedMemory& res_temp_3 = add_mod(res_temp_2, c_3);
    c_3.~SyncedMemory();
    c_3_temp_1.~SyncedMemory();
    c_3_temp_2.~SyncedMemory();
    void* kappa_qu_data=kappa_qu.mutable_gpu_data();
    SyncedMemory& c_4 = mul_mod_scalar(c_4_temp_1, kappa_qu);
    SyncedMemory& res_temp_4 = add_mod(res_temp_3, c_4);
    c_4.~SyncedMemory();
    c_4_temp_1.~SyncedMemory();
    void* separation_challenge_gpu_data=separation_challenge.mutable_gpu_data();
    SyncedMemory& res_temp_5 = mul_mod_scalar(res_temp_4, separation_challenge);
    SyncedMemory& res = mul_mod(selector, res_temp_5);
    return res;
}

 SyncedMemory& logic_linearisation_term(SyncedMemory& selector_poly,SyncedMemory& separation_challenge,  WitnessValues& wit_vals, Custom_class& custom_vals) {
    
     SyncedMemory& temp = logic_constraints(separation_challenge, wit_vals, custom_vals);
     SyncedMemory& res = poly_mul_const(selector_poly, temp);
    return res;
}