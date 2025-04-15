#include <stdint.h>
#include <vector>
#include <string>
#include <iostream>

typedef struct {
    uint64_t wire_evals[4][4];  
    uint64_t perm_evals[4][4];
    uint64_t lookup_evals[8][4];
    uint64_t custom_evals[2][4];
} ProofEvaluations;

typedef struct {
    uint64_t a_comm[2][6];
    uint64_t b_comm[2][6];
    uint64_t c_comm[2][6];
    uint64_t d_comm[2][6];
    uint64_t z_comm[2][6];
    uint64_t f_comm[2][6];
    uint64_t h_1_comm[2][6];
    uint64_t h_2_comm[2][6];
    uint64_t z_2_comm[2][6];
    uint64_t t_1_comm[2][6];
    uint64_t t_2_comm[2][6];
    uint64_t t_3_comm[2][6];
    uint64_t t_4_comm[2][6];
    uint64_t t_5_comm[2][6];
    uint64_t t_6_comm[2][6];
    uint64_t t_7_comm[2][6];
    uint64_t t_8_comm[2][6];
    uint64_t aw_opening[2][6];
    uint64_t saw_opening[2][6];
    ProofEvaluations evaluations;
} ProofC;

typedef struct {
    uint64_t n;
    uint64_t lookup_len;
    uint64_t intended_pi_pos;
    uint64_t* cs_q_lookup;
    uint64_t* w_l;
    uint64_t* w_r;
    uint64_t* w_o;
    uint64_t* w_4;
} CircuitC;


typedef struct {
    uint64_t* q_m_coeffs;
    uint64_t* q_m_evals;

    uint64_t* q_l_coeffs;
    uint64_t* q_l_evals;

    uint64_t* q_r_coeffs;
    uint64_t* q_r_evals;

    uint64_t* q_o_coeffs;
    uint64_t* q_o_evals;

    uint64_t* q_4_coeffs;
    uint64_t* q_4_evals;

    uint64_t* q_c_coeffs;
    uint64_t* q_c_evals;

    uint64_t* q_hl_coeffs;
    uint64_t* q_hl_evals;

    uint64_t* q_hr_coeffs;
    uint64_t* q_hr_evals;

    uint64_t* q_h4_coeffs;
    uint64_t* q_h4_evals;

    uint64_t* q_arith_coeffs;
    uint64_t* q_arith_evals;

    uint64_t* range_selector_coeffs;
    uint64_t* range_selector_evals;

    uint64_t* logic_selector_coeffs;
    uint64_t* logic_selector_evals;

    uint64_t* fixed_group_add_selector_coeffs;
    uint64_t* fixed_group_add_selector_evals;

    uint64_t* variable_group_add_selector_coeffs;
    uint64_t* variable_group_add_selector_evals;

    uint64_t* q_lookup_coeffs;
    uint64_t* q_lookup_evals;
    uint64_t* table1;
    uint64_t* table2;
    uint64_t* table3;
    uint64_t* table4;

    uint64_t* left_sigma_coeffs;
    uint64_t* left_sigma_evals;

    uint64_t* right_sigma_coeffs;
    uint64_t* right_sigma_evals;

    uint64_t* out_sigma_coeffs;
    uint64_t* out_sigma_evals;

    uint64_t* fourth_sigma_coeffs;
    uint64_t* fourth_sigma_evals;

    uint64_t* linear_evaluations;

    uint64_t* v_h_coset_8n;
} ProverKeyC;

