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
#include "PLONK/src/plonk_core/src/proof_system/widget/arithmetic.cu"
#include "PLONK/src/plonk_core/src/proof_system/permutation.cu"
struct Linearisation_poly {
    SyncedMemory& a;
    ProofEvaluations & proof_eval;
        // Constructor
    Linearisation_poly(SyncedMemory& a, ProofEvaluations& proof_eval)
        : a(a), proof_eval(proof_eval) {}

};
class WireEvaluations {
public:
    SyncedMemory& a_eval;
    SyncedMemory& b_eval;
    SyncedMemory& c_eval;
    SyncedMemory& d_eval;

    WireEvaluations(SyncedMemory& a, SyncedMemory& b, SyncedMemory& c, SyncedMemory& d)
        : a_eval(a), b_eval(b), c_eval(c), d_eval(d) {}
};

class PermutationEvaluations {
public:
    SyncedMemory& left_sigma_eval;
    SyncedMemory& right_sigma_eval;
    SyncedMemory& out_sigma_eval;
    SyncedMemory& permutation_eval;

    PermutationEvaluations(SyncedMemory& left_sigma, SyncedMemory& right_sigma, SyncedMemory& out_sigma, SyncedMemory& permutation)
        : left_sigma_eval(left_sigma), right_sigma_eval(right_sigma), out_sigma_eval(out_sigma), permutation_eval(permutation) {}
};

class LookupEvaluations {
public:
    SyncedMemory& q_lookup_eval;
    SyncedMemory& z2_next_eval;
    SyncedMemory& h1_eval;
    SyncedMemory& h1_next_eval;
    SyncedMemory& h2_eval;
    SyncedMemory& f_eval;
    SyncedMemory& table_eval;
    SyncedMemory& table_next_eval;

    LookupEvaluations(SyncedMemory& q_lookup, SyncedMemory& z2_next, SyncedMemory& h1, SyncedMemory& h1_next, SyncedMemory& h2, SyncedMemory& f, SyncedMemory& table, SyncedMemory& table_next)
        : q_lookup_eval(q_lookup), z2_next_eval(z2_next), h1_eval(h1), h1_next_eval(h1_next), h2_eval(h2), f_eval(f), table_eval(table), table_next_eval(table_next) {}
};

class ProofEvaluations {
public:
    WireEvaluations wire_evals;
    PermutationEvaluations perm_evals;
    LookupEvaluations lookup_evals;
    Custom_class custom_evals;

    ProofEvaluations(WireEvaluations wire, PermutationEvaluations perm, LookupEvaluations lookup, Custom_class custom)
        : wire_evals(wire), perm_evals(perm), lookup_evals(lookup), custom_evals(custom) {}
};


Linearisation_poly& compute_linearisation_poly(
    Radix2EvaluationDomain domain,
    ProverKey& pk,
    SyncedMemory& alpha,
    SyncedMemory& beta,
    SyncedMemory& gamma,
    SyncedMemory& delta,
    SyncedMemory& epsilon,
    SyncedMemory& zeta,
    SyncedMemory& range_separation_challenge,
    SyncedMemory& logic_separation_challenge,
    SyncedMemory& fixed_base_separation_challenge,
    SyncedMemory& var_base_separation_challenge,
    SyncedMemory& lookup_separation_challenge,
    SyncedMemory& z_challenge,
    SyncedMemory& w_l_poly,
    SyncedMemory& w_r_poly,
    SyncedMemory& w_o_poly,
    SyncedMemory& w_4_poly,
    SyncedMemory& t_1_poly,
    SyncedMemory& t_2_poly,
    SyncedMemory& t_3_poly,
    SyncedMemory& t_4_poly,
    SyncedMemory& t_5_poly,
    SyncedMemory& t_6_poly,
    SyncedMemory& t_7_poly,
    SyncedMemory& t_8_poly,
    SyncedMemory& z_poly,
    SyncedMemory& z2_poly,
    SyncedMemory& f_poly,
    SyncedMemory& h1_poly,
    SyncedMemory& h2_poly,
    SyncedMemory& table_poly
) {
    int n = domain.size;
    SyncedMemory& omega = domain.group_gen;
    Radix2EvaluationDomain domain_permutation(n);
    
    
    SyncedMemory& one = fr::one();
    SyncedMemory& mod = fr::MODULUS();
    SyncedMemory& neg_one = sub_mod(mod, one);
    SyncedMemory& shifted_z_challenge = mul_mod(z_challenge, omega);
    SyncedMemory& vanishing_poly_eval = domain.evaluate_vanishing_polynomial(z_challenge);
    SyncedMemory& z_challenge_to_n = add_mod(vanishing_poly_eval, one);
    
    
    SyncedMemory& l1_eval = compute_first_lagrange_evaluation(
        domain.size, vanishing_poly_eval, z_challenge
    );

    void* w_l_poly_gpu_data=w_l_poly.mutable_gpu_data();
    void* w_r_poly_gpu_data=w_r_poly.mutable_gpu_data();
    void* w_o_poly_gpu_data=w_o_poly.mutable_gpu_data();
    void* w_4_poly_gpu_data=w_4_poly.mutable_gpu_data();
    void* t_1_poly_gpu_data=t_1_poly.mutable_gpu_data();
    void* t_2_poly_gpu_data=t_2_poly.mutable_gpu_data();
    void* t_3_poly_gpu_data=t_3_poly.mutable_gpu_data();
    void* t_4_poly_gpu_data=t_4_poly.mutable_gpu_data();
    void* t_5_poly_gpu_data=t_5_poly.mutable_gpu_data();
    void* t_6_poly_gpu_data=t_6_poly.mutable_gpu_data();
    void* t_7_poly_gpu_data=t_7_poly.mutable_gpu_data();
    void* t_8_poly_gpu_data=t_8_poly.mutable_gpu_data();
    void* z_poly_gpu_data=z_poly.mutable_gpu_data();
    void* z2_poly_gpu_data=z2_poly.mutable_gpu_data();
    void* f_poly_gpu_data=f_poly.mutable_gpu_data();
    void* h1_poly_gpu_data=h1_poly.mutable_gpu_data();
    void* h2_poly_gpu_data=h2_poly.mutable_gpu_data();
    void* table_poly_gpu_data=table_poly.mutable_gpu_data();
    void* z_challenge_gpu_data=z_challenge.mutable_gpu_data();
    void* shifted_z_challenge_gpu_data=shifted_z_challenge.mutable_gpu_data();



    SyncedMemory& a_eval = evaluate(w_l_poly, z_challenge);
    SyncedMemory& b_eval = evaluate(w_r_poly, z_challenge);
    SyncedMemory& c_eval = evaluate(w_o_poly, z_challenge);
    SyncedMemory& d_eval = evaluate(w_4_poly, z_challenge);

    WireEvaluations wire_evals(a_eval, b_eval, c_eval, d_eval);

    SyncedMemory& left_sigma_eval = evaluate(pk.permutation_coeffs.left_sigma, z_challenge);
    SyncedMemory& right_sigma_eval = evaluate(pk.permutation_coeffs.right_sigma, z_challenge);
    SyncedMemory& out_sigma_eval = evaluate(pk.permutation_coeffs.out_sigma, z_challenge);
    SyncedMemory& permutation_eval = evaluate(z_poly, shifted_z_challenge);

    PermutationEvaluations perm_evals(
        left_sigma_eval, right_sigma_eval, out_sigma_eval, permutation_eval
    );

  
    SyncedMemory& q_arith_eval = evaluate(pk.arithmetic_coeffs.q_arith, z_challenge);
    SyncedMemory& q_lookup_eval = evaluate(pk.lookup_coeffs.q_lookup, z_challenge);
    SyncedMemory& q_c_eval = evaluate(pk.arithmetic_coeffs.q_c, z_challenge);
    SyncedMemory& q_l_eval = evaluate(pk.arithmetic_coeffs.q_l, z_challenge);
    SyncedMemory& q_r_eval = evaluate(pk.arithmetic_coeffs.q_r, z_challenge);
    SyncedMemory& a_next_eval = evaluate(w_l_poly, shifted_z_challenge);
    SyncedMemory& b_next_eval = evaluate(w_r_poly, shifted_z_challenge);
    SyncedMemory& d_next_eval = evaluate(w_4_poly, shifted_z_challenge);


    SyncedMemory& q_hl_eval = evaluate(pk.arithmetic_coeffs.q_hl, z_challenge);
    SyncedMemory& q_hr_eval = evaluate(pk.arithmetic_coeffs.q_hr, z_challenge);
    SyncedMemory& q_h4_eval = evaluate(pk.arithmetic_coeffs.q_4, z_challenge);

    Custom_class custom_evals(q_arith_eval, q_lookup_eval, q_c_eval, q_l_eval, q_r_eval, q_hl_eval, q_hr_eval, q_h4_eval, a_next_eval, b_next_eval, d_next_eval);


    SyncedMemory& z2_next_eval = evaluate(z2_poly, shifted_z_challenge);
    SyncedMemory& h1_eval = evaluate(h1_poly, z_challenge);
    SyncedMemory& h1_next_eval = evaluate(h1_poly, shifted_z_challenge);
    SyncedMemory& h2_eval = evaluate(h2_poly, z_challenge);
    SyncedMemory& f_eval = evaluate(f_poly, z_challenge);
    SyncedMemory& table_eval = evaluate(table_poly, z_challenge);
    SyncedMemory& table_next_eval = evaluate(table_poly, shifted_z_challenge);

    LookupEvaluations lookup_evals(
        q_lookup_eval,
        z2_next_eval,
        h1_eval,
        h1_next_eval,
        h2_eval,
        f_eval,
        table_eval,
        table_next_eval
    );

    SyncedMemory& gate_constraints = compute_gate_constraint_satisfiability(
        range_separation_challenge,
        logic_separation_challenge,
        fixed_base_separation_challenge,
        var_base_separation_challenge,
        wire_evals,
        q_arith_eval,
        custom_evals,
        pk
    );

    void* l1_eval_gpu_data =l1_eval.mutable_gpu_data();
    void* delta_gpu_data = delta.mutable_gpu_data();
    void* epsilon_gpu_data = epsilon.mutable_gpu_data();
    void* zeta_gpu_data = zeta.mutable_gpu_data();
    void* lookup_separation_challenge_gpu_data = lookup_separation_challenge.mutable_gpu_data();
    SyncedMemory& lookup = compute_linearisation_lookup(
        l1_eval,
        a_eval,
        b_eval,
        c_eval,
        d_eval,
        f_eval,
        table_eval,
        table_next_eval,
        h1_next_eval,
        h2_eval,
        z2_next_eval,
        delta,
        epsilon,
        zeta,
        z2_poly,
        h1_poly,
        lookup_separation_challenge,
        pk.lookup_coeffs.q_lookup
    );
    void* alpha_gpu_data = alpha.mutable_gpu_data();
    void* beta_gpu_data = beta.mutable_gpu_data();
    void* gamma_gpu_data = gamma.mutable_gpu_data();
    
    vector<SyncedMemory&> challengTuple={alpha, beta, gamma};
    vector<SyncedMemory&> wireTuple={a_eval, b_eval, c_eval, d_eval};
    vector<SyncedMemory&> sigmaTuple={left_sigma_eval, right_sigma_eval, out_sigma_eval};
    SyncedMemory& permutation = compute_linearisation_permutation(
        z_challenge,
        challengTuple,
        wireTuple,
        sigmaTuple,
        permutation_eval,
        z_poly,
        domain_permutation,
        pk.permutation_coeffs.fourth_sigma
    );

    void* z_challenge_to_n_gpu_data = z_challenge_to_n.mutable_gpu_data();
    
    //  t_8_poly * z_challenge_to_n
    SyncedMemory& term_1 = poly_mul_const(t_8_poly, z_challenge_to_n);

    //  (term_1 + t_7_poly) * z_challenge_to_n
    SyncedMemory& term_2_1 = poly_add_poly(term_1, t_7_poly);
    SyncedMemory& term_2 = poly_mul_const(term_2_1, z_challenge_to_n);

    //  (term_2 + t_6_poly) * z_challenge_to_n
    SyncedMemory& term_3_1 = poly_add_poly(term_2, t_6_poly);
    SyncedMemory& term_3 = poly_mul_const(term_3_1, z_challenge_to_n);

    //  (term_3 + t_5_poly) * z_challenge_to_n
    SyncedMemory& term_4_1 = poly_add_poly(term_3, t_5_poly);
    SyncedMemory& term_4 = poly_mul_const(term_4_1, z_challenge_to_n);

    //  (term_4 + t_4_poly) * z_challenge_to_n
    SyncedMemory& term_5_1 = poly_add_poly(term_4, t_4_poly);
    SyncedMemory& term_5 = poly_mul_const(term_5_1, z_challenge_to_n);

    //  (term_5 + t_3_poly) * z_challenge_to_n
    SyncedMemory& term_6_1 = poly_add_poly(term_5, t_3_poly);
    SyncedMemory& term_6 = poly_mul_const(term_6_1, z_challenge_to_n);

    //  (term_6 + t_2_poly) * z_challenge_to_n
    SyncedMemory& term_7_1 = poly_add_poly(term_6, t_2_poly);
    SyncedMemory& term_7 = poly_mul_const(term_7_1, z_challenge_to_n);

    //  (term_7 + t_1_poly) * vanishing_poly_eval
    SyncedMemory& term_8_1 = poly_add_poly(term_7, t_1_poly);

    void* vanishing_poly_eval_gpu_data=vanishing_poly_eval.mutable_gpu_data();
    SyncedMemory& quotient_term = poly_mul_const(term_8_1, vanishing_poly_eval);

    void* neg_one_gpu_data=neg_one.mutable_gpu_data();
    SyncedMemory& negative_quotient_term = poly_mul_const(quotient_term, neg_one);
    SyncedMemory& linearisation_polynomial_term_1 = poly_add_poly(gate_constraints, permutation);
    SyncedMemory& linearisation_polynomial_term_2 = poly_add_poly(lookup, negative_quotient_term);
    SyncedMemory& linearisation_polynomial = poly_add_poly(
        linearisation_polynomial_term_1, linearisation_polynomial_term_2
    );

    ProofEvaluations proof_evaluations = ProofEvaluations(
        wire_evals, perm_evals, lookup_evals, custom_evals
    );
    Linearisation_poly res(linearisation_polynomial, proof_evaluations);
    return res;
}

SyncedMemory& compute_gate_constraint_satisfiability(
    SyncedMemory& range_separation_challenge,
    SyncedMemory& logic_separation_challenge,
    SyncedMemory& fixed_base_separation_challenge,
    SyncedMemory& var_base_separation_challenge,
    WireEvaluations& wire_evals,
    SyncedMemory& q_arith_eval,
    Custom_class& custom_evals,
    ProverKey& pk
) {
    
    WitnessValues wit_vals (       
       wire_evals.a_eval,
       wire_evals.b_eval,
       wire_evals.c_eval,
       wire_evals.d_eval);


    
    SyncedMemory& arithmetic = compute_linearisation_arithmetic(
        wire_evals.a_eval,
        wire_evals.b_eval,
        wire_evals.c_eval,
        wire_evals.d_eval,
        q_arith_eval,
        pk.arithmetic_coeffs
    );

    
    void* range_separation_challenge_gpu_data=range_separation_challenge.mutable_gpu_data();
    SyncedMemory& range = range_linearisation_term(
        pk.range_selector_coeffs,
        range_separation_challenge,
        wit_vals,
        custom_evals
    );

    void* logic_separation_challenge_gpu_data=logic_separation_challenge.mutable_gpu_data();
    SyncedMemory& logic = logic_linearisation_term(
        pk.logic_selector_coeffs,
        logic_separation_challenge,
        wit_vals,
        custom_evals
    );


    void*  fixed_base_separation_challenge_gpu_data=fixed_base_separation_challenge.mutable_gpu_data();
    SyncedMemory& fixed_base_scalar_mul = FBSMGate_linearisation_term(
        pk.fixed_group_add_selector_coeffs,
        fixed_base_separation_challenge,
        wit_vals,
        FBSMValues::from_evaluations(custom_evals)
    );

    void* var_base_separation_challenge_gpu_data=var_base_separation_challenge.mutable_gpu_data();
   
    SyncedMemory& curve_addition = CAGate_linearisation_term(
        pk.variable_group_add_selector_coeffs,
        var_base_separation_challenge,
        wit_vals,
        CAValues::from_evaluations(custom_evals)
    );

    
    SyncedMemory& mid1 = poly_add_poly(arithmetic, range);
    SyncedMemory& mid2 = poly_add_poly(mid1, logic);
    SyncedMemory& mid3 = poly_add_poly(mid2, fixed_base_scalar_mul);
    SyncedMemory& res = poly_add_poly(mid3, curve_addition);
    
    return res;
}