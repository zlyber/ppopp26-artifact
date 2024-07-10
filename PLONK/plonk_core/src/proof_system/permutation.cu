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
// (a(x) + beta * X + gamma) (b(X) + beta * k1 * X + gamma) (c(X) + beta *
// k2 * X + gamma)(d(X) + beta * k3 * X + gamma)z(X) * alpha
SyncedMemory& compute_quotient_identity_range_check_i(
    SyncedMemory& x,
    SyncedMemory& w_l_i, SyncedMemory& w_r_i, SyncedMemory& w_o_i, SyncedMemory& w_4_i,
    SyncedMemory& z_i, SyncedMemory& alpha, SyncedMemory& beta, SyncedMemory& gamma) {

    SyncedMemory& k1 = K1();
    SyncedMemory& k2 = K2();
    SyncedMemory& k3 = K3();

    // 单个标量操作在 CPU 上
    SyncedMemory& mid2 = mul_mod(beta, k1);
    SyncedMemory& mid3 = mul_mod(beta, k2);
    SyncedMemory& mid4 = mul_mod(beta, k3);

    void* alpha_gpu_data=alpha.mutable_gpu_data();
    void* beta_gpu_data=beta.mutable_gpu_data();
    void* gamma_gpu_data=gamma.mutable_gpu_data();

    SyncedMemory& mid1 =mul_mod_scalar(x, beta);
    SyncedMemory& mid1 = add_mod(w_l_i, mid1);
    SyncedMemory& mid1 = add_mod_scalar(mid1, gamma);

    void* mid2_gpu_data= mid2.mutable_gpu_data();
    SyncedMemory& mid2 = mul_mod_scalar(x, mid2);
    SyncedMemory& mid2 = add_mod(w_r_i, mid2);
    SyncedMemory& mid2 = add_mod_scalar(mid2, gamma);

    SyncedMemory& mid = mul_mod(mid1, mid2);
    // delete mid1, mid2;

    void* mid3_gpu_data= mid3.mutable_gpu_data();
    SyncedMemory& mid3 = mul_mod_scalar(x, mid3);
    SyncedMemory& mid3 = add_mod(w_o_i, mid3);
    SyncedMemory& mid3 = add_mod_scalar(mid3, gamma);
    SyncedMemory& mid = mul_mod(mid, mid3);
    // delete mid3;

    void* mid4_gpu_data= mid4.mutable_gpu_data();
    SyncedMemory& mid4 = mul_mod_scalar(x, mid4);
    SyncedMemory& mid4 = add_mod(w_4_i, mid4);
    SyncedMemory& mid4 = add_mod_scalar(mid4, gamma);
    SyncedMemory& mid = mul_mod(mid, mid4);
    // delete mid4;

    SyncedMemory& res = mul_mod(mid, z_i);
    SyncedMemory& res = mul_mod_scalar(res, alpha);
    return res;
}

// # Computes the following:
// # (a(x) + beta* Sigma1(X) + gamma) (b(X) + beta * Sigma2(X) + gamma) (c(X)
// # + beta * Sigma3(X) + gamma)(d(X) + beta * Sigma4(X) + gamma) Z(X.omega) *
// # alpha
SyncedMemory& compute_quotient_copy_range_check_i(
    int size,
    SyncedMemory& pk_left_sigma_evals,
    SyncedMemory& pk_right_sigma_evals,
    SyncedMemory& pk_out_sigma_evals,
    SyncedMemory& pk_fourth_sigma_evals,
    SyncedMemory& w_l_i,
    SyncedMemory& w_r_i,
    SyncedMemory& w_o_i,
    SyncedMemory& w_4_i,
    SyncedMemory& z_i_next,
    SyncedMemory& alpha,
    SyncedMemory& beta,
    SyncedMemory& gamma
) {

    void* alpha_gpu_data=alpha.mutable_gpu_data();
    void* beta_gpu_data=beta.mutable_gpu_data();
    void* gamma_gpu_data=gamma.mutable_gpu_data();

    SyncedMemory& mid1 = mul_mod_scalar(pk_left_sigma_evals, beta);
    mid1 =  add_mod(w_l_i, mid1);
    mid1 =  add_mod_scalar(mid1, gamma);

    SyncedMemory& mid2 =  mul_mod_scalar(pk_right_sigma_evals, beta);
    mid2 =  add_mod(w_r_i, mid2);
    mid2 =  add_mod_scalar(mid2, gamma);

    SyncedMemory& res =  mul_mod(mid1, mid2);
    // delete mid1;
    // delete mid2;

    SyncedMemory& mid3 =  mul_mod_scalar(pk_out_sigma_evals, beta);
    mid3 =  add_mod(w_o_i, mid3);
    mid3 =  add_mod_scalar(mid3, gamma);
    res =  mul_mod(res, mid3);
    // delete mid3;

    SyncedMemory& mid4 =  mul_mod_scalar(pk_fourth_sigma_evals, beta);
    mid4 =  add_mod(w_4_i, mid4);
    mid4 =  add_mod_scalar(mid4, gamma);

    res =  mul_mod(res, mid4);
    delete mid4;
    res =  mul_mod(res, z_i_next);
    res =  mul_mod_scalar(res, alpha);

    // 获取 modulus 并扩展到多项式形式
    void* fr_MODULUS_gpu_data=fr::MODULUS().mutable_gpu_data();
    SyncedMemory& extend_mod = repeat_to_poly(fr::MODULUS(), size);
    res =  sub_mod(extend_mod, res);

    return res;
}

//  Computes the following:
//  L_1(X)[Z(X) - 1]
SyncedMemory& compute_quotient_term_check_one_i(SyncedMemory& z_i, SyncedMemory& l1_alpha_sq) {
    SyncedMemory& one = fr::one();  
    SyncedMemory& res = sub_mod_scalar(z_i, one);  
    res = mul_mod(res, l1_alpha_sq);  
    return res;  
}

SyncedMemory& compute_linearisation_permutation(
     SyncedMemory& z_challenge, 
     vector<SyncedMemory&>& challengTuple, 
     vector<SyncedMemory&>& wireTuple, 
     vector<SyncedMemory&>& sigmaTuple, 
     SyncedMemory& z_eval, 
     SyncedMemory& z_poly, 
     Radix2EvaluationDomain&  domain,
     SyncedMemory& pk_fourth_sigma_coeff) {

    SyncedMemory& a = compute_lineariser_identity_range_check(
        wireTuple[0], wireTuple[1], wireTuple[2], wireTuple[3],
        z_challenge,
        challengTuple[0], challengTuple[1], challengTuple[2],
        z_poly
    );

    SyncedMemory& mod = fr::MODULUS();
    void* mod_gpu_data=mod.mutable_gpu_data();
    SyncedMemory& b = compute_lineariser_copy_range_check(
        mod,
        wireTuple[0], wireTuple[1], wireTuple[2],
        z_eval,
        sigmaTuple[0], sigmaTuple[1], sigmaTuple[2],
        challengTuple[0], challengTuple[1], challengTuple[2],
        pk_fourth_sigma_coeff
    );

 
    SyncedMemory& alpha2 = mul_mod(challengTuple[0], challengTuple[0]);
    void* alpha2_gpu_data=alpha2.mutable_gpu_data();

    SyncedMemory& c = compute_lineariser_check_is_one(
        domain,
        z_challenge,
        alpha2,
        z_poly
    );

     SyncedMemory& ab =  poly_add_poly(a, b);
     SyncedMemory& abc = poly_add_poly(ab, c);


    return abc;
}

SyncedMemory& compute_lineariser_identity_range_check(
     SyncedMemory& a_eval,  SyncedMemory& b_eval,  SyncedMemory& c_eval,  SyncedMemory& d_eval,
     SyncedMemory& z_challenge,
     SyncedMemory& alpha,  SyncedMemory& beta,  SyncedMemory& gamma,
    SyncedMemory& z_poly
) {
    SyncedMemory& beta_z = mul_mod(beta, z_challenge);
    
    // a_eval + beta * z_challenge + gamma
    SyncedMemory& a_0 = add_mod(a_eval, beta_z);
    a_0 = add_mod(a_0, gamma);

    // b_eval + beta * K1 * z_challenge + gamma
    SyncedMemory&  k1 = K1();
    void* k1_data=k1.mutable_gpu_data();

    SyncedMemory& beta_z_k1 = mul_mod(k1, beta_z);
    SyncedMemory& a_1 = add_mod(b_eval, beta_z_k1);
    a_1 = add_mod(a_1, gamma);

    // c_eval + beta * K2 * z_challenge + gamma
    SyncedMemory&  k2 = K2();
    void* k2_data=k2.mutable_gpu_data();
    SyncedMemory& beta_z_k2 = mul_mod(k2, beta_z);
    SyncedMemory& a_2 = add_mod(c_eval, beta_z_k2);
    a_2 = add_mod(a_2, gamma);

    // d_eval + beta * K3 * z_challenge + gamma
    SyncedMemory&  k3 = K3();
    void* k3_data=k3.mutable_gpu_data();
    SyncedMemory& beta_z_k3 = mul_mod(k3, beta_z);
    SyncedMemory& a_3 = add_mod(d_eval, beta_z_k3);
    a_3 = add_mod(a_3, gamma);

    SyncedMemory& a = mul_mod(a_0, a_1);
    a = mul_mod(a, a_2);
    a = mul_mod(a, a_3);
    a = mul_mod(a, alpha);

    SyncedMemory& res = poly_mul_const(z_poly, a);
    return res;
}

SyncedMemory& compute_lineariser_copy_range_check(
     SyncedMemory& mod,
     SyncedMemory& a_eval,  SyncedMemory& b_eval,  SyncedMemory& c_eval,
     SyncedMemory& z_eval,
     SyncedMemory& sigma_1_eval,
     SyncedMemory& sigma_2_eval,
     SyncedMemory& sigma_3_eval,
     SyncedMemory& alpha,  SyncedMemory& beta,  SyncedMemory& gamma,
     SyncedMemory& fourth_sigma_poly
) {
    // a_eval + beta * sigma_1 + gamma
    SyncedMemory&beta_sigma_1 = mul_mod(beta, sigma_1_eval);
    SyncedMemory&a_0 = add_mod(a_eval, beta_sigma_1);
    a_0 = add_mod(a_0, gamma);

    // b_eval + beta * sigma_2 + gamma
    SyncedMemory&beta_sigma_2 = mul_mod(beta, sigma_2_eval);
    SyncedMemory&a_1 = add_mod(b_eval, beta_sigma_2);
    a_1 = add_mod(a_1, gamma);

    // c_eval + beta * sigma_3 + gamma
    SyncedMemory&beta_sigma_3 = mul_mod(beta, sigma_3_eval);
    SyncedMemory&a_2 =add_mod(c_eval, beta_sigma_3);
    a_2 = add_mod(a_2, gamma);

    SyncedMemory&beta_z_eval = mul_mod(beta, z_eval);
    SyncedMemory&a = mul_mod(a_0, a_1);
    a = mul_mod(a, a_2);
    a = mul_mod(a, beta_z_eval);
    a = mul_mod(a, alpha);
    SyncedMemory&neg_a = sub_mod(mod, a);

    SyncedMemory&res = poly_mul_const(fourth_sigma_poly, neg_a);
    return res;
}

SyncedMemory& compute_lineariser_check_is_one(
    Radix2EvaluationDomain& domain, 
    SyncedMemory& z_challenge, 
    SyncedMemory& alpha_sq, 
    SyncedMemory& z_coeffs) {

    SyncedMemory& lagrange_coefficients = domain.evaluate_all_lagrange_coefficients(z_challenge);
    void* lagrange_coefficients_gpu_data = lagrange_coefficients.mutable_gpu_data();
    SyncedMemory l_1_z(sizeof(uint64_t)) ;
    void* l_1_z_gpu_data= l_1_z.mutable_gpu_data();
    caffe_gpu_memcpy(5 * sizeof(uint64_t), lagrange_coefficients_gpu_data, l_1_z_gpu_data);
    SyncedMemory& const_num =mul_mod(l_1_z, alpha_sq);
    SyncedMemory& res = poly_mul_const(z_coeffs, const_num);
    

    return res;
}

 SyncedMemory& permutation_compute_quotient(
         int size,
         SyncedMemory& pk_linear_evaluations_evals,
         SyncedMemory& pk_left_sigma_evals,
         SyncedMemory& pk_right_sigma_evals,
         SyncedMemory& pk_out_sigma_evals,
         SyncedMemory& pk_fourth_sigma_evals,
         SyncedMemory& w_l_i,  SyncedMemory& w_r_i,  SyncedMemory& w_o_i,  SyncedMemory& w_4_i,
         SyncedMemory& z_i,  SyncedMemory& z_i_next,
         SyncedMemory& alpha,  SyncedMemory& l1_alpha_sq,
         SyncedMemory& beta,  SyncedMemory& gamma) 
{

        SyncedMemory& a = compute_quotient_identity_range_check_i(
          pk_linear_evaluations_evals, w_l_i, w_r_i, w_o_i, w_4_i, z_i, alpha, beta, gamma
        );

        SyncedMemory& b = compute_quotient_copy_range_check_i(
            size,
            pk_left_sigma_evals,
            pk_right_sigma_evals,
            pk_out_sigma_evals,
            pk_fourth_sigma_evals,  w_l_i, w_r_i, w_o_i, w_4_i, z_i_next, alpha, beta, gamma
        );

        SyncedMemory& c = compute_quotient_term_check_one_i(z_i, l1_alpha_sq);

        SyncedMemory& res = add_mod(a, b);
        res = add_mod(res, c);
        return res;
}