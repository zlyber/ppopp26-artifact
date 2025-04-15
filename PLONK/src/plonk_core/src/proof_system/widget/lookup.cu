#include "mod.cuh"

// SyncedMemory _compute_quotient_i(
//     SyncedMemory w_l_i,
//     SyncedMemory w_r_i,
//     SyncedMemory w_o_i,
//     SyncedMemory w_4_i,
//     SyncedMemory f_poly,
//     SyncedMemory table_poly,
//     SyncedMemory h1_poly,
//     SyncedMemory h2_poly,
//     SyncedMemory z2_poly,
//     SyncedMemory l1_poly,
//     SyncedMemory delta,
//     SyncedMemory epsilon,
//     SyncedMemory zeta,
//     SyncedMemory lookup_sep,
//     SyncedMemory proverkey_q_lookup,
//     uint64_t size)
// {
//     Ntt_coset NTT_coset{fr::TWO_ADICITY, size};

//     // q_lookup(X) * (a(X) + zeta * b(X) + (zeta^2 * c(X)) + (zeta^3 * d(X) - f(X))) * α_1
//     SyncedMemory one = fr::one();

//     SyncedMemory lookup_sep_sq = mul_mod(lookup_sep, lookup_sep);
//     SyncedMemory lookup_sep_cu = mul_mod(lookup_sep_sq, lookup_sep);
//     SyncedMemory one_plus_delta = add_mod(delta, one);
//     SyncedMemory epsilon_one_plus_delta = mul_mod(epsilon, one_plus_delta);

//     void *epsilon_gpu_data = epsilon.mutable_gpu_data();
//     void *delta_gpu_data = delta.mutable_gpu_data();
//     void *zeta_gpu_data = zeta.mutable_gpu_data();
//     void *lookup_sep_gpu_data = lookup_sep.mutable_gpu_data();
//     void *lookup_sep_sq_gpu_data = lookup_sep_sq.mutable_gpu_data();
//     void *lookup_sep_cu_gpu_data = lookup_sep_cu.mutable_gpu_data();
//     void *one_plus_delta_gpu_data = one_plus_delta.mutable_gpu_data();
//     void *epsilon_one_plus_delta_gpu_data = epsilon_one_plus_delta.mutable_gpu_data();
//     void *one_gpu_data = one.mutable_gpu_data();

//     SyncedMemory compressed_tuple(w_l_i.size());
//     compress(compressed_tuple, w_l_i, w_r_i, w_o_i, w_4_i, zeta);
//     SyncedMemory f_i = NTT_coset.forward(f_poly);
//     SyncedMemory mid_1 = sub_mod(compressed_tuple, f_i);
//     compressed_tuple = SyncedMemory();
//     mul_mod_(mid_1, proverkey_q_lookup);   
//     SyncedMemory a = mul_mod_scalar(mid_1, lookup_sep);
//     mid_1 = SyncedMemory();


//     SyncedMemory b_0 = add_mod_scalar(f_i, epsilon);
//     f_i = SyncedMemory();

//     SyncedMemory table_i = NTT_coset.forward(table_poly);
//     SyncedMemory b_1_1 = add_mod_scalar(table_i, epsilon_one_plus_delta);
//     SyncedMemory table_i_slice = slice(table_i, 8, true);
//     SyncedMemory table_i_next = slice(cat(table_i, table_i_slice), size, false);
//     SyncedMemory b_1_2 = mul_mod_scalar(table_i_next, delta);
//     table_i_next = SyncedMemory();
//     table_i = SyncedMemory();
//     table_i_slice = SyncedMemory();

//     SyncedMemory b_1 = add_mod(b_1_1, b_1_2);
//     b_1_1 = SyncedMemory();
//     b_1_2 = SyncedMemory();

//     SyncedMemory z2_i = NTT_coset.forward(z2_poly);
//     SyncedMemory mid_2 = mul_mod_scalar(z2_i, one_plus_delta);
//     mul_mod_(mid_2, b_0);
//     mul_mod_(mid_2, b_1);
//     SyncedMemory b = mul_mod_scalar(mid_2, lookup_sep_sq);
//     mid_2 = SyncedMemory();
//     b_0 = SyncedMemory();
//     b_1 = SyncedMemory();
//     SyncedMemory res = add_mod(a, b);
//     a = SyncedMemory();
//     b = SyncedMemory();

//     SyncedMemory h1_i = NTT_coset.forward(h1_poly);
//     SyncedMemory c_0_1 = add_mod_scalar(h1_i, epsilon_one_plus_delta);
//     SyncedMemory h2_i = NTT_coset.forward(h2_poly);
//     SyncedMemory c_0_2 = mul_mod_scalar(h2_i, delta);
//     SyncedMemory c_0 = add_mod(c_0_1, c_0_2);
//     c_0_1 = SyncedMemory();
//     c_0_2 = SyncedMemory();

//     SyncedMemory fr_mod = fr::MODULUS();
//     void *fr_mod_gpu = fr_mod.mutable_gpu_data();
//     SyncedMemory extend_mod = repeat_to_poly(fr_mod, size);
//     SyncedMemory z2_i_slice = slice(z2_i, 8, true);
//     SyncedMemory z2_i_next = slice(cat(z2_i, z2_i_slice), size, false);
//     SyncedMemory neg_z2_next = sub_mod(extend_mod, z2_i_next);
//     z2_i_next = SyncedMemory();
//     z2_i_slice = SyncedMemory();
//     extend_mod = SyncedMemory();

//     SyncedMemory mid_3 = mul_mod(neg_z2_next, c_0);
//     c_0 = SyncedMemory();
//     neg_z2_next = SyncedMemory();

//     SyncedMemory c_1_1 = add_mod_scalar(h2_i, epsilon_one_plus_delta);
//     h2_i = SyncedMemory();

//     SyncedMemory h1_i_slice = slice(h1_i, 8, true);
//     SyncedMemory h1_i_next = slice(cat(h1_i, h1_i_slice), size, false);
//     SyncedMemory c_1_2 = mul_mod_scalar(h1_i_next, delta);

//     h1_i = SyncedMemory();
//     h1_i_slice = SyncedMemory();
//     h1_i_next = SyncedMemory();

//     SyncedMemory c_1 = add_mod(c_1_1, c_1_2);
//     c_1_1 = SyncedMemory();
//     c_1_2 = SyncedMemory();

//     mul_mod_(mid_3, c_1);
//     c_1 = SyncedMemory();
//     SyncedMemory c = mul_mod_scalar(mid_3, lookup_sep_sq);
//     add_mod_(res, c);
//     mid_3 = SyncedMemory();
//     c = SyncedMemory();

//     SyncedMemory d_1 = sub_mod_scalar(z2_i, one);
//     SyncedMemory l1_i = NTT_coset.forward(l1_poly);
//     SyncedMemory d_2 = mul_mod_scalar(l1_i, lookup_sep_cu);
//     z2_i = SyncedMemory();
//     l1_i = SyncedMemory();

//     SyncedMemory d = mul_mod(d_1, d_2);
//     d_1 = SyncedMemory();
//     d_2 = SyncedMemory();
//     add_mod_(res, d);
    
//     return res;
// }

SyncedMemory compute_linearisation_lookup(
    SyncedMemory l1_eval,
    SyncedMemory a_eval,
    SyncedMemory b_eval,
    SyncedMemory c_eval,
    SyncedMemory d_eval,
    SyncedMemory f_eval,
    SyncedMemory table_eval,
    SyncedMemory table_next_eval,
    SyncedMemory h1_next_eval,
    SyncedMemory h2_eval,
    SyncedMemory z2_next_eval,
    SyncedMemory delta,
    SyncedMemory epsilon,
    SyncedMemory zeta,
    SyncedMemory z2_poly,
    SyncedMemory h1_poly,
    SyncedMemory lookup_sep,
    SyncedMemory pk_q_lookup)
{
    SyncedMemory lookup_sep_sq = mul_mod(lookup_sep, lookup_sep);
    SyncedMemory lookup_sep_cu = mul_mod(lookup_sep_sq, lookup_sep);
    SyncedMemory one = fr::one();
    void *one_gpu_data = one.mutable_gpu_data();
    SyncedMemory one_plus_delta = add_mod(delta, one);
    SyncedMemory epsilon_one_plus_delta = mul_mod(epsilon, one_plus_delta);
    std::vector<SyncedMemory> vec;
    vec.push_back(a_eval);
    vec.push_back(b_eval);
    vec.push_back(c_eval);
    vec.push_back(d_eval);
    SyncedMemory compressed_tuple = lc(vec, zeta);
    SyncedMemory compressed_tuple_sub_f_eval = sub_mod(compressed_tuple, f_eval);
    SyncedMemory const1 = mul_mod(compressed_tuple_sub_f_eval, lookup_sep);
    SyncedMemory a = poly_mul_const(pk_q_lookup, const1);

    // z2(X) * (1 + δ) * (ε + f_bar) * (ε(1+δ) + t_bar + δ*tω_bar) * lookup_sep^2
    SyncedMemory b_0 = add_mod(epsilon, f_eval);
    SyncedMemory epsilon_one_plus_delta_plus_table_eval = add_mod(epsilon_one_plus_delta, table_eval);
    SyncedMemory delta_times_table_next_eval = mul_mod(delta, table_next_eval);
    SyncedMemory b_1 = add_mod(epsilon_one_plus_delta_plus_table_eval, delta_times_table_next_eval);
    SyncedMemory b_2 = mul_mod(l1_eval, lookup_sep_cu);
    SyncedMemory one_plus_delta_b_0 = mul_mod(one_plus_delta, b_0);
    SyncedMemory one_plus_delta_b_0_b_1 = mul_mod(one_plus_delta_b_0, b_1);
    SyncedMemory one_plus_delta_b_0_b_1_lookup = mul_mod(one_plus_delta_b_0_b_1, lookup_sep_sq);
    SyncedMemory const2 = add_mod(one_plus_delta_b_0_b_1_lookup, b_2);
    SyncedMemory b = poly_mul_const(z2_poly, const2);

    // h1(X) * (−z2ω_bar) * (ε(1+δ) + h2_bar  + δh1ω_bar) * lookup_sep^2
    SyncedMemory modular = fr::MODULUS();
    void *modular_gpu_data = modular.mutable_gpu_data();
    SyncedMemory neg_z2_next_eval = sub_mod(modular, z2_next_eval);
    SyncedMemory c_0 = mul_mod(neg_z2_next_eval, lookup_sep_sq);
    SyncedMemory epsilon_one_plus_delta_h2_eval = add_mod(epsilon_one_plus_delta, h2_eval);
    SyncedMemory delta_h1_next_eval = add_mod(delta, h1_next_eval);
    SyncedMemory c_1 = add_mod(epsilon_one_plus_delta_h2_eval, delta_h1_next_eval);
    SyncedMemory c0_c1 = mul_mod(c_0, c_1);
    SyncedMemory c = poly_mul_const(h1_poly, c0_c1);
    
    SyncedMemory ab = poly_add_poly(a, b);
    SyncedMemory abc = poly_add_poly(ab, c);

    return abc;
}

// SyncedMemory compute_lookup_quotient_term(
//     uint64_t n,
//     SyncedMemory wl_eval_8n,
//     SyncedMemory wr_eval_8n,
//     SyncedMemory wo_eval_8n,
//     SyncedMemory w4_eval_8n,
//     SyncedMemory f_poly,
//     SyncedMemory table_poly,
//     SyncedMemory h1_poly,
//     SyncedMemory h2_poly,
//     SyncedMemory z2_poly,
//     SyncedMemory l1_poly,
//     SyncedMemory delta,
//     SyncedMemory epsilon,
//     SyncedMemory zeta,
//     SyncedMemory lookup_sep,
//     SyncedMemory pk_lookup_qlookup_evals){

//     uint64_t size = 8 * n;

//     SyncedMemory wl_eval_8n_ = slice(wl_eval_8n, size, true);
//     SyncedMemory wr_eval_8n_ = slice(wr_eval_8n, size, true);
//     SyncedMemory wo_eval_8n_ = slice(wo_eval_8n, size, true);
//     SyncedMemory w4_eval_8n_ = slice(w4_eval_8n, size, true);

//     SyncedMemory quotient = _compute_quotient_i(
//         wl_eval_8n_,
//         wr_eval_8n_,
//         wo_eval_8n_,
//         w4_eval_8n_,
//         f_poly,
//         table_poly,
//         h1_poly,
//         h2_poly,
//         z2_poly,
//         l1_poly,
//         delta,
//         epsilon,
//         zeta,
//         lookup_sep,
//         pk_lookup_qlookup_evals,
//         size);

//     return quotient;
// }