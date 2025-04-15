#pragma once
#include "../../../arithmetic.cuh"
#include "../permutation/constants.cuh"
#include "../../../domain.cuh"
#include "widget/mod.cuh"

struct WireEvaluations
{
    SyncedMemory a_eval;
    SyncedMemory b_eval;
    SyncedMemory c_eval;
    SyncedMemory d_eval;
    WireEvaluations(SyncedMemory a, SyncedMemory b, SyncedMemory c, SyncedMemory d)
        : a_eval(a), b_eval(b), c_eval(c), d_eval(d) {}
};

struct PermutationEvaluations {
    SyncedMemory left_sigma_eval;
    SyncedMemory right_sigma_eval;
    SyncedMemory out_sigma_eval;
    SyncedMemory permutation_eval;
    PermutationEvaluations(SyncedMemory left_sigma, SyncedMemory right_sigma, SyncedMemory out_sigma, SyncedMemory permutation)
        : left_sigma_eval(left_sigma), right_sigma_eval(right_sigma), out_sigma_eval(out_sigma), permutation_eval(permutation) {}
};

class CustomEvaluations {
    public:
        SyncedMemory q_arith_eval;
        SyncedMemory q_c_eval;
        SyncedMemory q_l_eval;
        SyncedMemory q_r_eval;
        SyncedMemory q_hl_eval;
        SyncedMemory q_hr_eval;
        SyncedMemory q_h4_eval;
        SyncedMemory a_next_eval;
        SyncedMemory b_next_eval;
        SyncedMemory d_next_eval;

        CustomEvaluations(
            SyncedMemory qa_eval,
            SyncedMemory qc_eval,
            SyncedMemory ql_eval,
            SyncedMemory qr_eval,
            SyncedMemory qhl_eval,
            SyncedMemory qhr_eval,
            SyncedMemory qh4_eval,
            SyncedMemory anext_eval,
            SyncedMemory bnext_eval,
            SyncedMemory dnext_eval)
            :q_arith_eval(qa_eval), q_c_eval(qc_eval), q_l_eval(ql_eval), q_r_eval(qr_eval),
            q_hl_eval(qhl_eval), q_hr_eval(qhr_eval), q_h4_eval(qh4_eval),
            a_next_eval(anext_eval), b_next_eval(bnext_eval), d_next_eval(dnext_eval){}

        CustomGate from_eval() {
            CustomGate res = CustomGate(a_next_eval, b_next_eval, d_next_eval,
                              q_l_eval, q_r_eval, q_c_eval,
                              q_hl_eval, q_hr_eval, q_h4_eval);
            return res;
        }
};

struct LookupEvaluations {
    SyncedMemory q_lookup_eval;
    SyncedMemory z2_next_eval;
    SyncedMemory h1_eval;
    SyncedMemory h1_next_eval;
    SyncedMemory h2_eval;
    SyncedMemory f_eval;
    SyncedMemory table_eval;
    SyncedMemory table_next_eval;
    LookupEvaluations(
        SyncedMemory q_lookup, 
        SyncedMemory z2_next, 
        SyncedMemory h1, SyncedMemory h1_next, 
        SyncedMemory h2, 
        SyncedMemory f, 
        SyncedMemory table, SyncedMemory table_next)
        : q_lookup_eval(q_lookup),
         z2_next_eval(z2_next), 
         h1_eval(h1), h1_next_eval(h1_next), 
         h2_eval(h2), 
         f_eval(f), 
         table_eval(table), table_next_eval(table_next) {}
};

struct ProofEvaluations {
    WireEvaluations wire_evals;
    PermutationEvaluations perm_evals;
    LookupEvaluations lookup_evals;
    CustomEvaluations custom_evals;
    ProofEvaluations(WireEvaluations wire, PermutationEvaluations perm, LookupEvaluations lookup, CustomEvaluations custom)
        : wire_evals(wire), perm_evals(perm), lookup_evals(lookup), custom_evals(custom) {}
    ProofEvaluationsC CtoR() {
        void* wire_a = wire_evals.a_eval.mutable_cpu_data();
        void* wire_b = wire_evals.b_eval.mutable_cpu_data();
        void* wire_c = wire_evals.c_eval.mutable_cpu_data();
        void* wire_d = wire_evals.d_eval.mutable_cpu_data();

        void* perm_l = perm_evals.left_sigma_eval.mutable_cpu_data();
        void* perm_r = perm_evals.right_sigma_eval.mutable_cpu_data();
        void* perm_o = perm_evals.out_sigma_eval.mutable_cpu_data();
        void* perm_p = perm_evals.permutation_eval.mutable_cpu_data();

        void* custom_arith = custom_evals.q_arith_eval.mutable_cpu_data();
        void* custom_qc = custom_evals.q_c_eval.mutable_cpu_data();
        void* custom_ql = custom_evals.q_l_eval.mutable_cpu_data();
        void* custom_qr = custom_evals.q_r_eval.mutable_cpu_data();
        void* custom_hl = custom_evals.q_hl_eval.mutable_cpu_data();
        void* custom_hr = custom_evals.q_hr_eval.mutable_cpu_data();
        void* custom_h4 = custom_evals.q_h4_eval.mutable_cpu_data();
        void* custom_a = custom_evals.a_next_eval.mutable_cpu_data();
        void* custom_b = custom_evals.b_next_eval.mutable_cpu_data();
        void* custom_d = custom_evals.d_next_eval.mutable_cpu_data();

        void* lookup_lookup = lookup_evals.q_lookup_eval.mutable_cpu_data();
        void* lookup_z2 = lookup_evals.z2_next_eval.mutable_cpu_data();
        void* lookup_h1 = lookup_evals.h1_eval.mutable_cpu_data();
        void* lookup_h1n = lookup_evals.h1_next_eval.mutable_cpu_data();
        void* lookup_h2 = lookup_evals.h2_eval.mutable_cpu_data();
        void* lookup_f = lookup_evals.f_eval.mutable_cpu_data();
        void* lookup_table = lookup_evals.table_eval.mutable_cpu_data();
        void* lookup_tablen = lookup_evals.table_next_eval.mutable_cpu_data();

        ProofEvaluationsC res = ProofEvaluationsC(
            WireEvaluationsC(reinterpret_cast<uint64_t*>(wire_a),
                             reinterpret_cast<uint64_t*>(wire_b),
                             reinterpret_cast<uint64_t*>(wire_c),
                             reinterpret_cast<uint64_t*>(wire_d)),
            PermutationEvaluationsC(reinterpret_cast<uint64_t*>(perm_l),
                                    reinterpret_cast<uint64_t*>(perm_r),
                                    reinterpret_cast<uint64_t*>(perm_o),
                                    reinterpret_cast<uint64_t*>(perm_p)),
            LookupEvaluationsC(reinterpret_cast<uint64_t*>(lookup_lookup),
                               reinterpret_cast<uint64_t*>(lookup_z2),
                               reinterpret_cast<uint64_t*>(lookup_h1),
                               reinterpret_cast<uint64_t*>(lookup_h1n),
                               reinterpret_cast<uint64_t*>(lookup_h2),
                               reinterpret_cast<uint64_t*>(lookup_f),
                               reinterpret_cast<uint64_t*>(lookup_table),
                               reinterpret_cast<uint64_t*>(lookup_tablen)),
            CustomEvaluationsC(reinterpret_cast<uint64_t*>(custom_arith),
                               reinterpret_cast<uint64_t*>(custom_qc),
                               reinterpret_cast<uint64_t*>(custom_ql),
                               reinterpret_cast<uint64_t*>(custom_qr),
                               reinterpret_cast<uint64_t*>(custom_hl),
                               reinterpret_cast<uint64_t*>(custom_hr),
                               reinterpret_cast<uint64_t*>(custom_h4),
                               reinterpret_cast<uint64_t*>(custom_a),
                               reinterpret_cast<uint64_t*>(custom_b),
                               reinterpret_cast<uint64_t*>(custom_d)));
            return res;
    }
};

struct linear {
    public:
        SyncedMemory linear_poly;
        ProofEvaluations evaluations;
        linear(SyncedMemory lin_poly, 
               WireEvaluations& w_eval,
               PermutationEvaluations& p_evals,
               LookupEvaluations& l_evals,
               CustomEvaluations& c_evals)
               :linear_poly(lin_poly), evaluations(ProofEvaluations(w_eval, p_evals, l_evals, c_evals)){}
};

SyncedMemory permutation_compute_quotient(
        uint64_t size,
        SyncedMemory pk_linear_evaluations_evals,
        SyncedMemory pk_left_sigma_evals,
        SyncedMemory pk_right_sigma_evals,
        SyncedMemory pk_out_sigma_evals,
        SyncedMemory pk_fourth_sigma_evals,
        SyncedMemory w_l_i, SyncedMemory w_r_i, SyncedMemory w_o_i, SyncedMemory w_4_i,
        SyncedMemory z_i, SyncedMemory z_i_next,
        SyncedMemory alpha, SyncedMemory l1_alpha_sq,
        SyncedMemory beta, SyncedMemory gamma);

SyncedMemory compute_linearisation_permutation(
    SyncedMemory z_challenge, 
    std::vector<SyncedMemory> challengTuple, 
    std::vector<SyncedMemory> wireTuple, 
    std::vector<SyncedMemory> sigmaTuple, 
    SyncedMemory z_eval, 
    SyncedMemory z_poly, 
    Radix2EvaluationDomain& domain,
    SyncedMemory pk_fourth_sigma_coeff);