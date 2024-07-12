#include <vector>
#include <tuple>
#include "caffe/syncedmem.hpp"
#include "PLONK/src/bls12_381/fr.hpp"
#include "PLONK/plonk_core/src/proof_system/widget/mod.cu"
#include "PLONK/utils/function.cuh"
#include "PLONK/src/arithmetic.cu"
#include "PLONK/plonk_core/src/proof_system/widget/custom_class.cu"
#include "PLONK/plonk_core/utils.cu"
#include "PLONK/src/load.cu"
#include "PLONK/plonk_core/src/proof_system/widget/lookup.cu"
#include "PLONK/plonk_core/src/proof_system/widget/range.cu"
#include "PLONK/plonk_core/src/proof_system/widget/logic.cu"
#include "PLONK/plonk_core/src/proof_system/widget/fixed_base_scalar_mul.cu"
SyncedMemory& compute_first_lagrange_poly_scaled(int n, SyncedMemory& scale) {
    Intt intt(n);
    SyncedMemory& x_evals = pad_poly(scale, n);
    void* x_evals_gpu_data=x_evals.mutable_gpu_data();
    SyncedMemory& x_coeffs = intt.forward(x_evals);
    return x_coeffs;
}

std::vector<SyncedMemory&> compute_gate_constraint_satisfiability(
    Ntt_coset& coset_NTT,
    SyncedMemory& range_challenge,
    SyncedMemory& logic_challenge,
    SyncedMemory& fixed_base_challenge,
    SyncedMemory& var_base_challenge,
    SyncedMemory& arithmetics_evals,
    SyncedMemory& selectors_evals,
    SyncedMemory& wl_eval_8n,
    SyncedMemory& wr_eval_8n,
    SyncedMemory& wo_eval_8n,
    SyncedMemory& w4_eval_8n,
    SyncedMemory& pi_poly) {

    void* pi_poly_gpu_data=pi_poly.mutable_gpu_data();
    SyncedMemory& pi_eval_8n = coset_NTT::forward(pi_poly);
    WitnessValues wit_vals = {
        wl_eval_8n.slice(0, 0, coset_NTT->Size),
        wr_eval_8n.slice(0, 0, coset_NTT->Size),
        wo_eval_8n.slice(0, 0, coset_NTT->Size),
        w4_eval_8n.slice(0, 0, coset_NTT->Size),
    };


    std::unordered_map<std::string, SyncedMemory&> custom_vals = {
        {"a_next_eval", wl_eval_8n.slice(0, 8)},
        {"b_next_eval", wr_eval_8n.slice(0, 8)},
        {"d_next_eval", w4_eval_8n.slice(0, 8)},
        {"q_l_eval", arithmetics_evals.q_l},
        {"q_r_eval", arithmetics_evals.q_r},
        {"q_c_eval", arithmetics_evals.q_c},
        {"q_hl_eval", arithmetics_evals.q_hl},
        {"q_hr_eval", arithmetics_evals.q_hr},
        {"q_h4_eval", arithmetics_evals.q_h4},
    };

    SyncedMemory& arithmetic = compute_quotient_i(arithmetics_evals, wit_vals);

    SyncedMemory& range_term = range_constraint.quotient_term(
        selectors_evals.range,
        range_challenge,
        wit_vals,
        custom_vals
    );

    SyncedMemory& logic_term = logic_constraint.quotient_term(
        selectors_evals.logic,
        logic_challenge,
        wit_vals,
        custom_vals
    );

    SyncedMemory& fixed_base_scalar_mul_term = FBSMGate_quotient_term(
        selectors_evals.fixed_group_add,
        fixed_base_challenge,
        wit_vals,
        FBSMValues::from_evaluations(custom_vals)
    );

    SyncedMemory& curve_addition_term = CAGate_quotient_term(
        selectors_evals.variable_group_add,
        var_base_challenge,
        wit_vals,
        CAValues::from_evaluations(custom_vals)
    );

    SyncedMemory& gate_contributions_temp_1=add_mod(arithmetic, pi_eval_8n));
    SyncedMemory& gate_contributions_temp_2=add_mod(gate_contributions_temp_1, range_term));
    SyncedMemory& gate_contributions_temp_3=add_mod(gate_contributions_temp_2, logic_term));
    SyncedMemory& gate_contributions_temp_4=add_mod(gate_contributions_temp_3, fixed_base_scalar_mul_term));
    SyncedMemory& gate_contributions=add_mod(gate_contributions_temp_4, curve_addition_term));

    return gate_contributions;
}

std::vector<SyncedMemory&> compute_permutation_checks(
    int n,
    SyncedMemory& coset_ntt,
    SyncedMemory& linear_evaluations_evals,
    SyncedMemory& permutations_evals,
    SyncedMemory& wl_eval_8n,
    SyncedMemory& wr_eval_8n,
    SyncedMemory& wo_eval_8n,
    SyncedMemory& w4_eval_8n,
    SyncedMemory& z_eval_8n,
    SyncedMemory& alpha,
    SyncedMemory& beta,
    SyncedMemory& gamma) {

    int size = 8 * n;

    SyncedMemory& alpha2 = mul_mod(alpha, alpha);
    void* alpha2_gpu_data=alpha2.mutable_gpu_data();
    SyncedMemory& l1_poly_alpha = compute_first_lagrange_poly_scaled(n, alpha2);
    SyncedMemory& l1_alpha_sq_evals = coset_ntt.forward(l1_poly_alpha);
    l1_poly_alpha.clear();

    SyncedMemory& quotient = permutation_compute_quotient(
        size,
        linear_evaluations_evals,
        permutations_evals.left_sigma,
        permutations_evals.right_sigma,
        permutations_evals.out_sigma,
        permutations_evals.fourth_sigma,
        wl_eval_8n.slice(0, 0, size),
        wr_eval_8n.slice(0, 0, size),
        wo_eval_8n.slice(0, 0, size),
        w4_eval_8n.slice(0, 0, size),
        z_eval_8n.slice(0, 0, size),
        z_eval_8n.slice(0, 8),
        alpha,
        l1_alpha_sq_evals.slice(0, 0, size),
        beta,
        gamma
    );

    return quotient;
}

std::vector<SyncedMemory&> compute_quotient_poly(
    int n,
    SyncedMemory& pk_new,
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
    SyncedMemory& wl_eval_8n = ntt_coset::forward(w_l_poly);
    wl_eval_8n = torch::cat({wl_eval_8n, wl_eval_8n.slice(0, 0, 8)}, 0);

    SyncedMemory& wr_eval_8n = ntt_coset::forward(w_r_poly.to(torch::kCUDA));
    wr_eval_8n = torch::cat({wr_eval_8n, wr_eval_8n.slice(0, 0, 8)}, 0);

    SyncedMemory& wo_eval_8n = coset_NTT->forward(w_o_poly.to(torch::kCUDA));

    SyncedMemory& w4_eval_8n = coset_NTT->forward(w_4_poly.to(torch::kCUDA));
    w4_eval_8n = torch::cat({w4_eval_8n, w4_eval_8n.slice(0, 0, 8)}, 0);

    SyncedMemory& gate_constraints = compute_gate_constraint_satisfiability(
        coset_NTT,
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
    SyncedMemory& z_eval_8n = coset_NTT->forward(z_poly);
    z_eval_8n = torch::cat({z_eval_8n, z_eval_8n.slice(0, 0, 8)}, 0);

    SyncedMemory& permutation = compute_permutation_checks(
        n,
        coset_NTT,
        pk_new.linear_evaluations_evals,
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

    z_eval_8n.reset();

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
        pk_new.lookups_evals.q_lookup
    );

    wl_eval_8n.reset();
    wr_eval_8n.reset();
    wo_eval_8n.reset();
    w4_eval_8n.reset