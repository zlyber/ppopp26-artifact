#include "PLONK/src/structure.cuh"
#include "PLONK/src/bls12_381/fr.hpp"

ProverKey::ProverKey(
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
        SyncedMemory& v_h_coset_8n) : 
        q_m_coeffs(q_m_coeffs), q_m_evals(q_m_evals),
        q_l_coeffs(q_l_coeffs), q_l_evals(q_l_evals),
        q_r_coeffs(q_r_coeffs), q_r_evals(q_r_evals),
        q_o_coeffs(q_o_coeffs), q_o_evals(q_o_evals),
        q_4_coeffs(q_4_coeffs), q_4_evals(q_4_evals),
        q_c_coeffs(q_c_coeffs), q_c_evals(q_c_evals),
        q_hl_coeffs(q_hl_coeffs), q_hl_evals(q_hl_evals),
        q_hr_coeffs(q_hr_coeffs), q_hr_evals(q_hr_evals),
        q_h4_coeffs(q_h4_coeffs), q_h4_evals(q_h4_evals),
        q_arith_coeffs(q_arith_coeffs), q_arith_evals(q_arith_evals),
        range_selector_coeffs(range_selector_coeffs), range_selector_evals(range_selector_evals),
        logic_selector_coeffs(logic_selector_coeffs), logic_selector_evals(logic_selector_evals),
        fixed_group_add_selector_coeffs(fixed_group_add_selector_coeffs), fixed_group_add_selector_evals(fixed_group_add_selector_evals),
        variable_group_add_selector_coeffs(variable_group_add_selector_coeffs), variable_group_add_selector_evals(variable_group_add_selector_evals),
        q_lookup_coeffs(q_lookup_coeffs), q_lookup_evals(q_lookup_evals),
        table1(table1), table2(table2), table3(table3), table4(table4),
        left_sigma_coeffs(left_sigma_coeffs), left_sigma_evals(left_sigma_evals),
        right_sigma_coeffs(right_sigma_coeffs), right_sigma_evals(right_sigma_evals),
        out_sigma_coeffs(out_sigma_coeffs), out_sigma_evals(out_sigma_evals),
        fourth_sigma_coeffs(fourth_sigma_coeffs), fourth_sigma_evals(fourth_sigma_evals),
        linear_evaluations(linear_evaluations),
        v_h_coset_8n(v_h_coset_8n) {}

ProverKey load_pk(ProverKeyC pk, uint64_t n) {
    uint64_t coeff_size = n;
    uint64_t eval_size = 8 * n;

    SyncedMemory q_m_coeffs(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* q_m_coeffs_gpu = q_m_coeffs.mutable_gpu_data();
    caffe_gpu_memcpy(q_m_coeffs.size(), q_m_coeffs_gpu, pk.q_m_coeffs);

    SyncedMemory q_m_evals(eval_size*fr::Limbs*sizeof(uint64_t));
    void* q_m_evals_gpu = q_m_evals.mutable_gpu_data();
    caffe_gpu_memcpy(q_m_evals.size(), q_m_evals_gpu, pk.q_m_evals);

    SyncedMemory q_l_coeffs(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* q_l_coeffs_gpu = q_l_coeffs.mutable_gpu_data();
    caffe_gpu_memcpy(q_l_coeffs.size(), q_l_coeffs_gpu, pk.q_l_coeffs);

    SyncedMemory q_l_evals(eval_size*fr::Limbs*sizeof(uint64_t));
    void* q_l_evals_gpu = q_l_evals.mutable_gpu_data();
    caffe_gpu_memcpy(q_l_evals.size(), q_l_evals_gpu, pk.q_l_evals);

    SyncedMemory q_r_coeffs(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* q_r_coeffs_gpu = q_r_coeffs.mutable_gpu_data();
    caffe_gpu_memcpy(q_r_coeffs.size(), q_r_coeffs_gpu, pk.q_r_coeffs);

    SyncedMemory q_r_evals(eval_size*fr::Limbs*sizeof(uint64_t));
    void* q_r_evals_gpu = q_r_evals.mutable_gpu_data();
    caffe_gpu_memcpy(q_r_evals.size(), q_r_evals_gpu, pk.q_r_evals);

    SyncedMemory q_o_coeffs(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* q_o_coeffs_gpu = q_o_coeffs.mutable_gpu_data();
    caffe_gpu_memcpy(q_o_coeffs.size(), q_o_coeffs_gpu, pk.q_o_coeffs);

    SyncedMemory q_o_evals(eval_size*fr::Limbs*sizeof(uint64_t));
    void* q_o_evals_gpu = q_o_evals.mutable_gpu_data();
    caffe_gpu_memcpy(q_o_evals.size(), q_o_evals_gpu, pk.q_o_evals);

    SyncedMemory q_4_coeffs(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* q_4_coeffs_gpu = q_4_coeffs.mutable_gpu_data();
    caffe_gpu_memcpy(q_4_coeffs.size(), q_4_coeffs_gpu, pk.q_4_coeffs);

    SyncedMemory q_4_evals(eval_size*fr::Limbs*sizeof(uint64_t));
    void* q_4_evals_gpu = q_4_evals.mutable_gpu_data();
    caffe_gpu_memcpy(q_4_evals.size(), q_4_evals_gpu, pk.q_4_evals);

    SyncedMemory q_c_coeffs(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* q_c_coeffs_gpu = q_c_coeffs.mutable_gpu_data();
    caffe_gpu_memcpy(q_c_coeffs.size(), q_c_coeffs_gpu, pk.q_c_coeffs);

    SyncedMemory q_c_evals(eval_size*fr::Limbs*sizeof(uint64_t));
    void* q_c_evals_gpu = q_c_evals.mutable_gpu_data();
    caffe_gpu_memcpy(q_c_evals.size(), q_c_evals_gpu, pk.q_c_evals);

    SyncedMemory q_hl_coeffs(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* q_hl_coeffs_gpu = q_hl_coeffs.mutable_gpu_data();
    caffe_gpu_memcpy(q_hl_coeffs.size(), q_hl_coeffs_gpu, pk.q_hl_coeffs);

    SyncedMemory q_hl_evals(eval_size*fr::Limbs*sizeof(uint64_t));
    void* q_hl_evals_gpu = q_hl_evals.mutable_gpu_data();
    caffe_gpu_memcpy(q_hl_evals.size(), q_hl_evals_gpu, pk.q_hl_evals);

    SyncedMemory q_hr_coeffs(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* q_hr_coeffs_gpu = q_hr_coeffs.mutable_gpu_data();
    caffe_gpu_memcpy(q_hr_coeffs.size(), q_hr_coeffs_gpu, pk.q_hr_coeffs);

    SyncedMemory q_hr_evals(eval_size*fr::Limbs*sizeof(uint64_t));
    void* q_hr_evals_gpu = q_hr_evals.mutable_gpu_data();
    caffe_gpu_memcpy(q_hr_evals.size(), q_hr_evals_gpu, pk.q_hr_evals);


    SyncedMemory q_h4_coeffs(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* q_h4_coeffs_gpu = q_h4_coeffs.mutable_gpu_data();
    caffe_gpu_memcpy(q_h4_coeffs.size(), q_h4_coeffs_gpu, pk.q_h4_coeffs);

    SyncedMemory q_h4_evals(eval_size*fr::Limbs*sizeof(uint64_t));
    void* q_h4_evals_gpu = q_h4_evals.mutable_gpu_data();
    caffe_gpu_memcpy(q_h4_evals.size(), q_h4_evals_gpu, pk.q_h4_evals);

    SyncedMemory q_arith_coeffs(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* q_arith_coeffs_gpu = q_arith_coeffs.mutable_gpu_data();
    caffe_gpu_memcpy(q_arith_coeffs.size(), q_arith_coeffs_gpu, pk.q_arith_coeffs);

    SyncedMemory q_arith_evals(eval_size*fr::Limbs*sizeof(uint64_t));
    void* q_arith_evals_gpu = q_arith_evals.mutable_gpu_data();
    caffe_gpu_memcpy(q_arith_evals.size(), q_arith_evals_gpu, pk.q_arith_evals);

    SyncedMemory range_selector_coeffs(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* range_selector_coeffs_gpu = range_selector_coeffs.mutable_gpu_data();
    caffe_gpu_memcpy(range_selector_coeffs.size(), range_selector_coeffs_gpu, pk.range_selector_coeffs);

    SyncedMemory range_selector_evals(eval_size*fr::Limbs*sizeof(uint64_t));
    void* range_selector_evals_gpu = range_selector_evals.mutable_gpu_data();
    caffe_gpu_memcpy(range_selector_evals.size(), range_selector_evals_gpu, pk.range_selector_evals);

    SyncedMemory logic_selector_coeffs(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* logic_selector_coeffs_gpu = logic_selector_coeffs.mutable_gpu_data();
    caffe_gpu_memcpy(logic_selector_coeffs.size(), logic_selector_coeffs_gpu, pk.logic_selector_coeffs);

    SyncedMemory logic_selector_evals(eval_size*fr::Limbs*sizeof(uint64_t));
    void* logic_selector_evals_gpu = logic_selector_evals.mutable_gpu_data();
    caffe_gpu_memcpy(logic_selector_evals.size(), logic_selector_evals_gpu, pk.logic_selector_evals);

    SyncedMemory fixed_group_add_selector_coeffs(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* fixed_group_add_selector_coeffs_gpu = fixed_group_add_selector_coeffs.mutable_gpu_data();
    caffe_gpu_memcpy(fixed_group_add_selector_coeffs.size(), fixed_group_add_selector_coeffs_gpu, pk.fixed_group_add_selector_coeffs);

    SyncedMemory fixed_group_add_selector_evals(eval_size*fr::Limbs*sizeof(uint64_t));
    void* fixed_group_add_selector_evals_gpu = fixed_group_add_selector_evals.mutable_gpu_data();
    caffe_gpu_memcpy(fixed_group_add_selector_evals.size(), fixed_group_add_selector_evals_gpu, pk.fixed_group_add_selector_evals);

    SyncedMemory variable_group_add_selector_coeffs(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* variable_group_add_selector_coeffs_gpu = variable_group_add_selector_coeffs.mutable_gpu_data();
    caffe_gpu_memcpy(variable_group_add_selector_coeffs.size(), variable_group_add_selector_coeffs_gpu, pk.variable_group_add_selector_coeffs);

    SyncedMemory variable_group_add_selector_evals(eval_size*fr::Limbs*sizeof(uint64_t));
    void* variable_group_add_selector_evals_gpu = variable_group_add_selector_evals.mutable_gpu_data();
    caffe_gpu_memcpy(variable_group_add_selector_evals.size(), variable_group_add_selector_evals_gpu, pk.variable_group_add_selector_evals);

    SyncedMemory q_lookup_coeffs(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* q_lookup_coeffs_gpu = q_lookup_coeffs.mutable_gpu_data();
    caffe_gpu_memcpy(q_lookup_coeffs.size(), q_lookup_coeffs_gpu, pk.q_lookup_coeffs);

    SyncedMemory q_lookup_evals(eval_size*fr::Limbs*sizeof(uint64_t));
    void* q_lookup_evals_gpu = q_lookup_evals.mutable_gpu_data();
    caffe_gpu_memcpy(q_lookup_evals.size(), q_lookup_evals_gpu, pk.q_lookup_evals);

    SyncedMemory table1(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* table1_gpu = table1.mutable_gpu_data();
    caffe_gpu_memcpy(table1.size(), table1_gpu, pk.table1);

    SyncedMemory table2(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* table2_gpu = table2.mutable_gpu_data();
    caffe_gpu_memcpy(table2.size(), table2_gpu, pk.table2);

    SyncedMemory table3(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* table3_gpu = table3.mutable_gpu_data();
    caffe_gpu_memcpy(table3.size(), table3_gpu, pk.table3);

    SyncedMemory table4(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* table4_gpu = table4.mutable_gpu_data();
    caffe_gpu_memcpy(table4.size(), table4_gpu, pk.table4);

    SyncedMemory left_sigma_coeffs(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* left_sigma_coeffs_gpu = left_sigma_coeffs.mutable_gpu_data();
    caffe_gpu_memcpy(left_sigma_coeffs.size(), left_sigma_coeffs_gpu, pk.left_sigma_coeffs);

    SyncedMemory left_sigma_evals(eval_size*fr::Limbs*sizeof(uint64_t));
    void* left_sigma_evals_gpu = left_sigma_evals.mutable_gpu_data();
    caffe_gpu_memcpy(left_sigma_evals.size(), left_sigma_evals_gpu, pk.left_sigma_evals);

    SyncedMemory right_sigma_coeffs(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* right_sigma_coeffs_gpu = right_sigma_coeffs.mutable_gpu_data();
    caffe_gpu_memcpy(right_sigma_coeffs.size(), right_sigma_coeffs_gpu, pk.right_sigma_coeffs);

    SyncedMemory right_sigma_evals(eval_size*fr::Limbs*sizeof(uint64_t));
    void* right_sigma_evals_gpu = right_sigma_evals.mutable_gpu_data();
    caffe_gpu_memcpy(right_sigma_evals.size(), right_sigma_evals_gpu, pk.right_sigma_evals);

    SyncedMemory out_sigma_coeffs(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* out_sigma_coeffs_gpu = out_sigma_coeffs.mutable_gpu_data();
    caffe_gpu_memcpy(out_sigma_coeffs.size(), out_sigma_coeffs_gpu, pk.out_sigma_coeffs);

    SyncedMemory out_sigma_evals(eval_size*fr::Limbs*sizeof(uint64_t));
    void* out_sigma_evals_gpu = out_sigma_evals.mutable_gpu_data();
    caffe_gpu_memcpy(out_sigma_evals.size(), out_sigma_evals_gpu, pk.out_sigma_evals);

    SyncedMemory fourth_sigma_coeffs(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* fourth_sigma_coeffs_gpu = fourth_sigma_coeffs.mutable_gpu_data();
    caffe_gpu_memcpy(fourth_sigma_coeffs.size(), fourth_sigma_coeffs_gpu, pk.fourth_sigma_coeffs);

    SyncedMemory fourth_sigma_evals(eval_size*fr::Limbs*sizeof(uint64_t));
    void* fourth_sigma_evals_gpu = fourth_sigma_evals.mutable_gpu_data();
    caffe_gpu_memcpy(fourth_sigma_evals.size(), fourth_sigma_evals_gpu, pk.fourth_sigma_evals);

    SyncedMemory linear_evaluations(eval_size*fr::Limbs*sizeof(uint64_t));
    void* linear_evaluations_gpu = linear_evaluations.mutable_gpu_data();
    caffe_gpu_memcpy(linear_evaluations.size(), linear_evaluations_gpu, pk.linear_evaluations);


    SyncedMemory v_h_coset_8n(eval_size*fr::Limbs*sizeof(uint64_t));
    void* v_h_coset_8n_gpu = v_h_coset_8n.mutable_gpu_data();
    caffe_gpu_memcpy(v_h_coset_8n.size(), v_h_coset_8n_gpu, pk.v_h_coset_8n);

    ProverKey proverkey = ProverKey(q_m_coeffs,
         q_m_evals,
         q_l_coeffs,
         q_l_evals,
         q_r_coeffs,
         q_r_evals,
         q_o_coeffs,
         q_o_evals,
         q_4_coeffs,
         q_4_evals,
         q_c_coeffs,
         q_c_evals,
         q_hl_coeffs,
         q_hl_evals,
         q_hr_coeffs,
         q_hr_evals,
         q_h4_coeffs,
         q_h4_evals,
         q_arith_coeffs,
         q_arith_evals,
         range_selector_coeffs,
         range_selector_evals,
         logic_selector_coeffs,
         logic_selector_evals,
         fixed_group_add_selector_coeffs,
         fixed_group_add_selector_evals,
         variable_group_add_selector_coeffs,
         variable_group_add_selector_evals,
         q_lookup_coeffs,
         q_lookup_evals,
         table1,
         table2,
         table3,
         table4,
         left_sigma_coeffs,
         left_sigma_evals,
         right_sigma_coeffs,
         right_sigma_evals,
         out_sigma_coeffs,
         out_sigma_evals,
         fourth_sigma_coeffs,
         fourth_sigma_evals,
         linear_evaluations,
         v_h_coset_8n);
    return proverkey;
}

Circuit::Circuit(
        uint64_t n,
        uint64_t lookup_len,
        uint64_t intended_pi_pos,
        SyncedMemory& public_inputs,
        SyncedMemory& cs_q_lookup,
        SyncedMemory& w_l,
        SyncedMemory& w_r,
        SyncedMemory& w_o,
        SyncedMemory& w_4
    ) : n(n), lookup_len(lookup_len),
        intended_pi_pos(intended_pi_pos),
        public_inputs(public_inputs),
        cs_q_lookup(cs_q_lookup),
        w_l(w_l),
        w_r(w_r),
        w_o(w_o),
        w_4(w_4)
    {}

Circuit load_cs(CircuitC cs, uint64_t n){
    SyncedMemory q_lookup(n*fr::Limbs*sizeof(uint64_t));
    void* q_lookup_gpu = q_lookup.mutable_gpu_data();
    caffe_gpu_memcpy(q_lookup.size(), q_lookup_gpu, cs.cs_q_lookup);

    SyncedMemory pi(fr::Limbs*sizeof(uint64_t));
    void* pi_gpu = q_lookup.mutable_gpu_data();
    caffe_gpu_memcpy(pi.size(), pi_gpu, cs.public_inputs);

    SyncedMemory w_l(n*fr::Limbs*sizeof(uint64_t));
    void* w_l_gpu = w_l.mutable_gpu_data();
    caffe_gpu_memcpy(w_l.size(), w_l_gpu, cs.w_l);

    SyncedMemory w_r(n*fr::Limbs*sizeof(uint64_t));
    void* w_r_gpu = w_r.mutable_gpu_data();
    caffe_gpu_memcpy(w_r.size(), w_r_gpu, cs.w_r);

    SyncedMemory w_o(n*fr::Limbs*sizeof(uint64_t));
    void* w_o_gpu = w_o.mutable_gpu_data();
    caffe_gpu_memcpy(w_o.size(), w_o_gpu, cs.w_o);

    SyncedMemory w_4(n*fr::Limbs*sizeof(uint64_t));
    void* w_4_gpu = w_o.mutable_gpu_data();
    caffe_gpu_memcpy(w_4.size(), w_4_gpu, cs.w_4);

    return Circuit(cs.n,cs.lookup_len,cs.intended_pi_pos, pi,
                   q_lookup, w_l, w_r, w_o, w_4);
}

CommitKey::CommitKey(
        SyncedMemory& powers_of_g,
        SyncedMemory& powers_of_gamma_g
    ) : powers_of_g(powers_of_g),
        powers_of_gamma_g(powers_of_gamma_g)
    {}

CommitKey load_ck(CommitKeyC ck, uint64_t n){
    SyncedMemory powers_of_g(2 * n * fq::Limbs * sizeof(uint64_t));
    void* powers_of_g_gpu = powers_of_g.mutable_gpu_data();
    caffe_gpu_memcpy(powers_of_g.size(), powers_of_g_gpu, ck.powers_of_g);

    SyncedMemory powers_of_gamma_g(2 * fq::Limbs * sizeof(uint64_t));
    void* powers_of_gamma_g_gpu = powers_of_gamma_g.mutable_gpu_data();
    caffe_gpu_memcpy(powers_of_gamma_g.size(), powers_of_gamma_g_gpu, ck.powers_of_gamma_g);

    return CommitKey(powers_of_g, powers_of_gamma_g);
}
