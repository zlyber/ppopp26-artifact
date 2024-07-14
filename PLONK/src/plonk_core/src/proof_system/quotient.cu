#include <vector>
#include <tuple>
#include "caffe/syncedmem.hpp"
#include "PLONK/src/bls12_381/fr.hpp"
#include "PLONK/src/plonk_core/src/proof_system/widget/mod.cu"
#include "PLONK/utils/function.cuh"
#include "PLONK/src/arithmetic.cu"
#include "PLONK/src/plonk_core/src/proof_system/widget/custom_class.cu"
#include "PLONK/src/plonk_core/utils.cu"
#include "PLONK/src/load.cu"
#include "PLONK/src/plonk_core/src/proof_system/widget/lookup.cu"
#include "PLONK/src/plonk_core/src/proof_system/widget/range.cu"
#include "PLONK/src/plonk_core/src/proof_system/widget/logic.cu"
#include "PLONK/src/plonk_core/src/proof_system/widget/fixed_base_scalar_mul.cu"
SyncedMemory& compute_first_lagrange_poly_scaled(int n, SyncedMemory& scale) {
    Intt intt(n);
    SyncedMemory& x_evals = pad_poly(scale, n);
    void* x_evals_gpu_data=x_evals.mutable_gpu_data();
    SyncedMemory& x_coeffs = intt.forward(x_evals);
    return x_coeffs;
}

SyncedMemory& compute_gate_constraint_satisfiability(
    Ntt_coset& coset_NTT,
    SyncedMemory& range_challenge,
    SyncedMemory& logic_challenge,
    SyncedMemory& fixed_base_challenge,
    SyncedMemory& var_base_challenge,
    Arith& arithmetics_evals,
    ProverKey& pk,
    SyncedMemory& wl_eval_8n,
    SyncedMemory& wr_eval_8n,
    SyncedMemory& wo_eval_8n,
    SyncedMemory& w4_eval_8n,
    SyncedMemory& pi_poly) {

    void* pi_poly_gpu_data=pi_poly.mutable_gpu_data();
    SyncedMemory& pi_eval_8n = coset_NTT.forward(pi_poly);
    int size=coset_NTT.Size;
    SyncedMemory& wl_eval_8n_head=get_head(wl_eval_8n,size);
    SyncedMemory& wr_eval_8n_head=get_head(wr_eval_8n,size);
    SyncedMemory& wo_eval_8n_head=get_head(wo_eval_8n,size);
    SyncedMemory& w4_eval_8n_head=get_head(w4_eval_8n,size);

    WitnessValues wit_vals = {
        wl_eval_8n_head,
        wr_eval_8n_head,
        wo_eval_8n_head,
        w4_eval_8n_head,
    };

    SyncedMemory& wl_eval_8n_slice_tail=get_tail(wl_eval_8n,size);
    SyncedMemory& wr_eval_8n_slice_tail=get_tail(wr_eval_8n,size);
    SyncedMemory& wo_eval_8n_slice_tail=get_tail(wo_eval_8n,size);
    SyncedMemory& w4_eval_8n_slice_tail=get_tail(w4_eval_8n,size);
    Custom_class custom_vals(
        wl_eval_8n_slice_tail,
        wr_eval_8n_slice_tail,
        wo_eval_8n_slice_tail,
        w4_eval_8n_slice_tail,
        arithmetics_evals.q_l,
        arithmetics_evals.q_r,
        arithmetics_evals.q_c,
        arithmetics_evals.q_hl,
        arithmetics_evals.q_hr,
        arithmetics_evals.q_h4
    );

    SyncedMemory& arithmetic = compute_quotient_i(arithmetics_evals, wit_vals);

    SyncedMemory& range_term = range_constraint_quotient_term(
        pk.range_selector_coeffs,
        range_challenge,
        wit_vals,
        custom_vals
    );

    SyncedMemory& logic_term = logic_constraint_quotient_term(
        pk.logic_selector_coeffs,
        logic_challenge,
        wit_vals,
        custom_vals
    );

    SyncedMemory& fixed_base_scalar_mul_term = FBSMGate_quotient_term(
        pk.fixed_group_add_selector_coeffs,
        fixed_base_challenge,
        wit_vals,
        FBSMValues::from_evaluations(custom_vals)
    );

    SyncedMemory& curve_addition_term = CAGate_quotient_term(
        pk.variable_group_add_selector_coeffs,
        var_base_challenge,
        wit_vals,
        CAValues::from_evaluations(custom_vals)
    );

    SyncedMemory& gate_contributions_temp_1=add_mod(arithmetic, pi_eval_8n);
    SyncedMemory& gate_contributions_temp_2=add_mod(gate_contributions_temp_1, range_term);
    SyncedMemory& gate_contributions_temp_3=add_mod(gate_contributions_temp_2, logic_term);
    SyncedMemory& gate_contributions_temp_4=add_mod(gate_contributions_temp_3, fixed_base_scalar_mul_term);
    SyncedMemory& gate_contributions=add_mod(gate_contributions_temp_4, curve_addition_term);

    return gate_contributions;
}

SyncedMemory& compute_permutation_checks(
    int n,
    Ntt_coset& coset_ntt,
    SyncedMemory& linear_evaluations_evals,
    SyncedMemory& permutations_evals,
    SyncedMemory& wl_eval_8n,
    SyncedMemory& wr_eval_8n,
    SyncedMemory& wo_eval_8n,
    SyncedMemory& w4_eval_8n,
    SyncedMemory& z_eval_8n,
    SyncedMemory& alpha,
    SyncedMemory& beta,
    SyncedMemory& gamma,
    ProverKey& pk) {

    int size = 8 * n;

    SyncedMemory& alpha2 = mul_mod(alpha, alpha);
    void* alpha2_gpu_data=alpha2.mutable_gpu_data();
    SyncedMemory& l1_poly_alpha = compute_first_lagrange_poly_scaled(n, alpha2);
    SyncedMemory& l1_alpha_sq_evals = coset_ntt.forward(l1_poly_alpha);
    l1_poly_alpha.~SyncedMemory();


    SyncedMemory& wl_eval_8n_slice_head=get_head(wl_eval_8n,size);
    SyncedMemory& wr_eval_8n_slice_head=get_head(wr_eval_8n,size);
    SyncedMemory& wo_eval_8n_slice_head=get_head(wo_eval_8n,size);
    SyncedMemory& w4_eval_8n_slice_head=get_head(w4_eval_8n,size);
    SyncedMemory& z_eval_8n_slice_head=get_head(z_eval_8n,size);
    SyncedMemory& l1_alpha_sq_evals_slice_head=get_head(l1_alpha_sq_evals,size);
    SyncedMemory& l1_alpha_sq_evals_slice_tail=get_tail(l1_alpha_sq_evals,size);
    SyncedMemory& quotient = permutation_compute_quotient(
        size,
        linear_evaluations_evals,
        pk,
        wl_eval_8n_slice_head,
        wr_eval_8n_slice_head,
        wo_eval_8n_slice_head,
        w4_eval_8n_slice_head,
        z_eval_8n_slice_head,
        l1_alpha_sq_evals_slice_tail,
        alpha,
        l1_alpha_sq_evals_slice_head,
        beta,
        gamma
    );

    return quotient;
}

SyncedMemory& compute_quotient_poly(
    int n,
    ProverKey& pk_new,
    SyncedMemory& z_poly,
    SyncedMemory& z2_poly,
    SyncedMemory& w_l_poly,
    SyncedMemory& w_r_poly,
    SyncedMemory& w_o_poly,
    SyncedMemory& w_4_poly,
    SyncedMemory& public_inputs_poly,
    SyncedMemory& f_poly,
    SyncedMemory& table_poly,
    SyncedMemory& h1_poly,
    SyncedMemory& h2_poly,
    SyncedMemory& alpha,
    SyncedMemory& beta,
    SyncedMemory& gamma,
    SyncedMemory& delta,
    SyncedMemory& epsilon,
    SyncedMemory& zeta,
    SyncedMemory& range_challenge,
    SyncedMemory& logic_challenge,
    SyncedMemory& fixed_base_challenge,
    SyncedMemory& var_base_challenge,
    SyncedMemory& lookup_challenge) {

    int coset_size = 8 * n;
    SyncedMemory& one = fr::one();
    void* one_gpu_data= one.mutable_gpu_data();
    SyncedMemory& l1_poly = compute_first_lagrange_poly_scaled(n, one);
    Ntt_coset ntt_coset(fr::TWO_ADICITY(),coset_size);
    void* w_l_poly_gpu_data=w_l_poly.mutable_gpu_data();
    void* w_r_poly_gpu_data=w_r_poly.mutable_gpu_data();
    void* w_o_poly_gpu_data=w_o_poly.mutable_gpu_data();
    void* w_4_poly_gpu_data=w_4_poly.mutable_gpu_data();

    SyncedMemory& wl_eval_8n_temp = ntt_coset.forward(w_l_poly);
    SyncedMemory wl_eval_8n_head(2*sizeof(uint64_t));
    void* wl_eval_8n_head_gpu_data=wl_eval_8n_head.mutable_gpu_data();
    void* wl_eval_8n_gpu_data= wl_eval_8n_temp.mutable_gpu_data();
    caffe_gpu_memcpy(wl_eval_8n_head.size(),wl_eval_8n_gpu_data,wl_eval_8n_head_gpu_data);
    SyncedMemory& wl_eval_8n = cat(wl_eval_8n_temp,wl_eval_8n_head);

    SyncedMemory wr_eval_8n_temp = ntt_coset.forward(w_r_poly);
    SyncedMemory wr_eval_8n_head(2*sizeof(uint64_t));
    void* wr_eval_8n_head_gpu_data=wr_eval_8n_head.mutable_gpu_data();
    void* wr_eval_8n_temp_gpu_data=wr_eval_8n_temp.mutable_gpu_data();
    caffe_gpu_memcpy(wr_eval_8n_head.size(),wr_eval_8n_temp_gpu_data,wr_eval_8n_head_gpu_data);
    SyncedMemory& wr_eval_8n = cat(wr_eval_8n_temp,wr_eval_8n_head);

    
    SyncedMemory& wo_eval_8n = ntt_coset.forward(w_o_poly);
    SyncedMemory& w4_eval_8n_temp = ntt_coset.forward(w_4_poly);
    SyncedMemory w4_eval_8n_head(2*sizeof(uint64_t));
    void* w4_eval_8n_head_gpu_data=w4_eval_8n_head.mutable_gpu_data();
    caffe_gpu_memcpy(w4_eval_8n_head.size(),w4_eval_8n_temp,wr_eval_8n_head_gpu_data);
    SyncedMemory& w4_eval_8n = cat(w4_eval_8n,w4_eval_8n_head);

    SyncedMemory& gate_constraints = compute_gate_constraint_satisfiability(
        ntt_coset,
        range_challenge,
        logic_challenge,
        fixed_base_challenge,
        var_base_challenge,
        pk_new.arithmetics_evals,
        pk_new.selectors_evals,
        wl_eval_8n,
        wr_eval_8n,
        wo_eval_8n,
        w4_eval_8n,
        public_inputs_poly
    );
    void* z_poly_gpu_data= z_poly.mutable_gpu_data();
    SyncedMemory& z_eval_8n_temp = ntt_coset,forward(z_poly);
    SyncedMemory& z_eval_8n_head = get_head(z_eval_8n_temp,8);
    SyncedMemory& z_eval_8n=cat(z_eval_8n_temp,z_eval_8n_head);

    SyncedMemory& permutation = compute_permutation_checks(
        n,
        coset_NTT,
        pk_new.linear_evaluations,
        pk_new.permutations_evals,
        wl_eval_8n,
        wr_eval_8n,
        wo_eval_8n,
        w4_eval_8n,
        z_eval_8n,
        alpha,
        beta,
        gamma
    );

    z_eval_8n.~SyncedMemory();

    void* f_poly_gpu_data= f_poly.mutable_gpu_data();
    void* table_poly_gpu_data= table_poly.mutable_gpu_data();
    void* h1_poly_gpu_data= h1_poly.mutable_gpu_data();
    void* h2_poly_gpu_data= h2_poly.mutable_gpu_data();
    void* z2_poly_gpu_data= z2_poly.mutable_gpu_data();
    void* l1_poly_gpu_data= l1_poly.mutable_gpu_data();
    SyncedMemory& lookup = compute_lookup_quotient_term(
        n,
        wl_eval_8n,
        wr_eval_8n,
        wo_eval_8n,
        w4_eval_8n,
        f_poly,
        table_poly,
        h1_poly,
        h2_poly,
        z2_poly,
        l1_poly,
        delta,
        epsilon,
        zeta,
        lookup_challenge,
        pk_new.q_lookup_evals
    );

    wl_eval_8n.~SyncedMemory();
    wr_eval_8n.~SyncedMemory();
    wo_eval_8n.~SyncedMemory();
    w4_eval_8n.~SyncedMemory();
    SyncedMemory&  numerator_temp_1 = add_mod(gate_constraints, permutation);
    SyncedMemory&  gate_constraints.~SyncedMemory();
    SyncedMemory&  permutation.~SyncedMemory();
    SyncedMemory& numerator_temp_2 = add_mod(numerator_temp_1, lookup);
    lookup.~SyncedMemory();
    SyncedMemory& denominator =inv_mod(pk_new.v_h_coset_8n_evals);
    SyncedMemory& res_temp = mul_mod(numerator_temp_2, denominator);
    numerator.~SyncedMemory();
    denominator.~SyncedMemory();
    Intt_coset intt_coset(fr::TWO_ADICITY());
    SyncedMemory& res = intt_coset.forward(res_temp);
    return res;
    }