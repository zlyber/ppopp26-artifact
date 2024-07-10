#pragma once
#include <iostream>
#include <vector>
#include "caffe/syncedmem.hpp"
#include "PLONK/src/bls12_381/fq.hpp"
#include "PLONK/utils/function.cuh"

constexpr int COEFF_A = 0;

class AffinePointG1{
    public:
        SyncedMemory& x;
        SyncedMemory& y;
        AffinePointG1(SyncedMemory& a, SyncedMemory& b);
        static bool is_zero(AffinePointG1 self);
};

class ProjectivePointG1{
    public:
        SyncedMemory& x;
        SyncedMemory& y;
        SyncedMemory& z;
        ProjectivePointG1(SyncedMemory& a, SyncedMemory& b, SyncedMemory& c);
        static bool is_zero(ProjectivePointG1 self);
};

AffinePointG1 to_affine(ProjectivePointG1 G1);

ProjectivePointG1 add_assign(ProjectivePointG1 self, ProjectivePointG1 other);

ProjectivePointG1 double_ProjectivePointG1(ProjectivePointG1 self);

ProjectivePointG1 add_assign_mixed(ProjectivePointG1 self, AffinePointG1 other);

class BTreeMap{
    public:
    SyncedMemory& item;
    uint64_t pos;
    BTreeMap(SyncedMemory& item_, uint64_t pos_) : item(item_), pos(pos_) {}

    static BTreeMap& new_instance(SyncedMemory& item_, uint64_t pos_){
        BTreeMap newmap = BTreeMap(item_,pos_);
        return newmap;
    }
};


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
    uint64_t* powers_of_g;
    uint64_t* powers_of_gamma_g;
}CommitKeyC;

uint64_t next_power_of_2(uint64_t x);

uint64_t total_size(CircuitC circuit);

uint64_t circuit_bound(CircuitC circuit);


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


class CommitKey {
    public:
        SyncedMemory& powers_of_g;
        SyncedMemory& powers_of_gamma_g;
        CommitKey(
            SyncedMemory& powers_of_g,
            SyncedMemory& powers_of_gamma_g
        );
};

class Circuit {
    public:
        uint64_t n;
        uint64_t lookup_len;
        uint64_t intended_pi_pos;
        SyncedMemory& cs_q_lookup;
        SyncedMemory& w_l;
        SyncedMemory& w_r;
        SyncedMemory& w_o;
        SyncedMemory& w_4;
        Circuit(
        uint64_t n,
        uint64_t lookup_len,
        uint64_t intended_pi_pos,
        SyncedMemory& cs_q_lookup,
        SyncedMemory& w_l,
        SyncedMemory& w_r,
        SyncedMemory& w_o,
        SyncedMemory& w_4
    );
};


class ProverKey{
    public:
        SyncedMemory& q_m_coeffs;
        SyncedMemory& q_m_evals;

        SyncedMemory& q_l_coeffs;
        SyncedMemory& q_l_evals;

        SyncedMemory& q_r_coeffs;
        SyncedMemory& q_r_evals;

        SyncedMemory& q_o_coeffs;
        SyncedMemory& q_o_evals;

        SyncedMemory& q_4_coeffs;
        SyncedMemory& q_4_evals;

        SyncedMemory& q_c_coeffs;
        SyncedMemory& q_c_evals;

        SyncedMemory& q_hl_coeffs;
        SyncedMemory& q_hl_evals;

        SyncedMemory& q_hr_coeffs;
        SyncedMemory& q_hr_evals;

        SyncedMemory& q_h4_coeffs;
        SyncedMemory& q_h4_evals;

        SyncedMemory& q_arith_coeffs;
        SyncedMemory& q_arith_evals;

        SyncedMemory& range_selector_coeffs;
        SyncedMemory& range_selector_evals;

        SyncedMemory& logic_selector_coeffs;
        SyncedMemory& logic_selector_evals;

        SyncedMemory& fixed_group_add_selector_coeffs;
        SyncedMemory& fixed_group_add_selector_evals;

        SyncedMemory& variable_group_add_selector_coeffs;
        SyncedMemory& variable_group_add_selector_evals;

        SyncedMemory& q_lookup_coeffs;
        SyncedMemory& q_lookup_evals;
        SyncedMemory& table1;
        SyncedMemory& table2;
        SyncedMemory& table3;
        SyncedMemory& table4;

        SyncedMemory& left_sigma_coeffs;
        SyncedMemory& left_sigma_evals;

        SyncedMemory& right_sigma_coeffs;
        SyncedMemory& right_sigma_evals;

        SyncedMemory& out_sigma_coeffs;
        SyncedMemory& out_sigma_evals;

        SyncedMemory& fourth_sigma_coeffs;
        SyncedMemory& fourth_sigma_evals;

        SyncedMemory& linear_evaluations;

        SyncedMemory& v_h_coset_8n;
        ProverKey(
        SyncedMemory& q_m_coeffs, SyncedMemory& q_m_evals,
        SyncedMemory& q_l_coeffs, SyncedMemory& q_l_evals,
        SyncedMemory& q_r_coeffs, SyncedMemory& q_r_evals,
        SyncedMemory& q_o_coeffs, SyncedMemory& q_o_evals,
        SyncedMemory& q_4_coeffs, SyncedMemory& q_4_evals,
        SyncedMemory& q_c_coeffs, SyncedMemory& q_c_evals,
        SyncedMemory& q_hl_coeffs, SyncedMemory& q_hl_evals,
        SyncedMemory& q_hr_coeffs, SyncedMemory& q_hr_evals,
        SyncedMemory& q_h4_coeffs, SyncedMemory& q_h4_evals,
        SyncedMemory& q_arith_coeffs, SyncedMemory& q_arith_evals,
        SyncedMemory& range_selector_coeffs, SyncedMemory& range_selector_evals,
        SyncedMemory& logic_selector_coeffs, SyncedMemory& logic_selector_evals,
        SyncedMemory& fixed_group_add_selector_coeffs, SyncedMemory& fixed_group_add_selector_evals,
        SyncedMemory& variable_group_add_selector_coeffs, SyncedMemory& variable_group_add_selector_evals,
        SyncedMemory& q_lookup_coeffs, SyncedMemory& q_lookup_evals,
        SyncedMemory& table1, SyncedMemory& table2, SyncedMemory& table3, SyncedMemory& table4,
        SyncedMemory& left_sigma_coeffs, SyncedMemory& left_sigma_evals,
        SyncedMemory& right_sigma_coeffs, SyncedMemory& right_sigma_evals,
        SyncedMemory& out_sigma_coeffs, SyncedMemory& out_sigma_evals,
        SyncedMemory& fourth_sigma_coeffs, SyncedMemory& fourth_sigma_evals,
        SyncedMemory& linear_evaluations,
        SyncedMemory& v_h_coset_8n);
};

ProverKey load_pk(ProverKeyC pk, uint64_t n);

Circuit load_cs(CircuitC cs, uint64_t n);

CommitKey load_ck(CommitKeyC ck, uint64_t n);