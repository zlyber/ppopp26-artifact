#pragma once
#include "../../../../bls12_381/fr.cuh"
#include "../../constaraint_system/hash.h"
#include "../../../utils.cuh"
#include "../../../../arithmetic.cuh"

struct WitnessValues {
    SyncedMemory a_val;
    SyncedMemory b_val;
    SyncedMemory c_val;
    SyncedMemory d_val;

    WitnessValues( SyncedMemory a_val,  SyncedMemory b_val,  SyncedMemory c_val,  SyncedMemory d_val) 
        : a_val(a_val), b_val(b_val), c_val(c_val), d_val(d_val) {}
};

struct CustomGate {
    public:
        SyncedMemory a_next;
        SyncedMemory b_next;
        SyncedMemory d_next;
        SyncedMemory q_l;
        SyncedMemory q_r;
        SyncedMemory q_c;
        SyncedMemory q_hl;
        SyncedMemory q_hr;
        SyncedMemory q_h4;
        CustomGate(SyncedMemory a, SyncedMemory b, SyncedMemory d, SyncedMemory ql, SyncedMemory qr, SyncedMemory qc,
                    SyncedMemory qhl, SyncedMemory qhr, SyncedMemory qh4)
            : a_next(a), b_next(b), d_next(d), q_l(ql), q_r(qr), q_c(qc),
            q_hl(qhl), q_hr(qhr), q_h4(qh4) {}
};

class FBSMValues {
public:
    SyncedMemory a_next_val;
    SyncedMemory b_next_val;
    SyncedMemory d_next_val;
    SyncedMemory q_l_val;
    SyncedMemory q_r_val;
    SyncedMemory q_c_val;

    FBSMValues(SyncedMemory a_next_val, SyncedMemory b_next_val, SyncedMemory d_next_val, SyncedMemory q_l_val, SyncedMemory q_r_val, SyncedMemory q_c_val);

    static FBSMValues from_evaluations(CustomGate custom_evals);
};

class CAValues {
public:
    SyncedMemory a_next_val;
    SyncedMemory b_next_val;
    SyncedMemory d_next_val;

    CAValues(SyncedMemory a, SyncedMemory b, SyncedMemory d);
    static CAValues from_evaluations(CustomGate custom_evals);
};

SyncedMemory delta(SyncedMemory f);

SyncedMemory compute_quotient_i(Arithmetic arithmetics_evals, WitnessValues wit_vals);

SyncedMemory range_quotient_term(
    SyncedMemory selector,
    SyncedMemory separation_challenge,
    WitnessValues wit_vals,  
    CustomGate custom_vals
);

SyncedMemory logic_quotient_term(
    SyncedMemory selector,
    SyncedMemory separation_challenge,
    WitnessValues wit_vals,
    CustomGate custom_vals
);

SyncedMemory FBSMGate_quotient_term(
    SyncedMemory selector,
    SyncedMemory separation_challenge,
    WitnessValues wit_vals,
    FBSMValues custom_vals
);

SyncedMemory CAGate_quotient_term(
    SyncedMemory selector,
    SyncedMemory separation_challenge, 
    WitnessValues wit_vals, 
    CAValues custom_vals
);

SyncedMemory compute_lookup_quotient_term(
    uint64_t n,
    SyncedMemory wl_eval_8n,
    SyncedMemory wr_eval_8n,
    SyncedMemory wo_eval_8n,
    SyncedMemory w4_eval_8n,
    SyncedMemory f_poly,
    SyncedMemory table_poly,
    SyncedMemory h1_poly,
    SyncedMemory h2_poly,
    SyncedMemory z2_poly,
    SyncedMemory l1_poly,
    SyncedMemory delta,
    SyncedMemory epsilon,
    SyncedMemory zeta,
    SyncedMemory lookup_sep,
    SyncedMemory pk_lookup_qlookup_evals);

SyncedMemory compute_linearisation_arithmetic(
     SyncedMemory a_eval,
     SyncedMemory b_eval,
     SyncedMemory c_eval,
     SyncedMemory d_eval,
     SyncedMemory q_arith_eval,
     Arithmetic prover_key_arithmetic);

SyncedMemory range_linearisation_term(
    SyncedMemory selector_poly, 
    SyncedMemory separation_challenge, 
    WitnessValues wit_vals, 
    CustomGate custom_vals);

SyncedMemory logic_linearisation_term(
    SyncedMemory selector_poly, 
    SyncedMemory separation_challenge, 
    WitnessValues wit_vals, 
    CustomGate custom_vals);

SyncedMemory FBSMGate_linearisation_term(
    SyncedMemory selector_poly, 
    SyncedMemory separation_challenge, 
    WitnessValues wit_vals, 
    FBSMValues custom_vals);

SyncedMemory CAGate_linearisation_term(
    SyncedMemory selector_poly, 
    SyncedMemory separation_challenge, 
    WitnessValues wit_vals, 
    CAValues custom_vals);

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
    SyncedMemory pk_q_lookup);