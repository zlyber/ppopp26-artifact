#include "structure.cuh"
#include "bls12_381/fr.cuh"


std::vector<SyncedMemory> chunk(uint64_t* input, size_t size, int chunk_num)
{
   std::vector<SyncedMemory> output;
   output.reserve(chunk_num);
   for(int i = 0; i< chunk_num; i++){
      SyncedMemory out(size);
      void* out_ = out.mutable_cpu_data();
      memcpy(out_, reinterpret_cast<char*>(input) + i*out.size(), out.size());
      output.push_back(out);
   }
   return output;
}

std::vector<SyncedMemory> chunk_gpu(SyncedMemory input, int chunk_num, cudaStream_t stream)
{
   std::vector<SyncedMemory> output;
   void* input_ = input.mutable_gpu_data_async(stream);
   output.reserve(chunk_num);
   size_t size = input.size()/chunk_num;
   for(int i = 0; i< chunk_num; i++){
      SyncedMemory out(size);
      void* out_ = out.mutable_gpu_data_async(stream);
      caffe_gpu_memcpy_async(out.size(), input_ + i*out.size(), out_, stream);
      output.push_back(out);
   }
   return output;
}

std::vector<SyncedMemory> copy(std::vector<SyncedMemory> input, uint64_t chunk_size)
{
   int chunk_num = input.size();
   std::vector<SyncedMemory> output;
   output.reserve(chunk_num);
   int type = input[0].head();
   if(type == 1){
    for(int i =0; i<chunk_num;i++){
        SyncedMemory out(chunk_size);
        void* out_ = out.mutable_cpu_data();
        void* in_ = input[i].mutable_cpu_data();
        memcpy(out_, in_, chunk_size);
        output.push_back(out);
    }
   }
   else if(type == 2){
    for(int i =0; i<chunk_num;i++){
        SyncedMemory out(chunk_size);
        void* out_ = out.mutable_gpu_data();
        void* in_ = input[i].mutable_gpu_data();
        caffe_gpu_memcpy(chunk_size, in_, out_);
        output.push_back(out);
    }
   }
    return output;
}

LookupTable::LookupTable(SyncedMemory ql, std::vector<SyncedMemory> t1, 
                         std::vector<SyncedMemory> t2, std::vector<SyncedMemory> t3, 
                         std::vector<SyncedMemory> t4)
    : q_lookup(ql), table1(t1), table2(t2), table3(t3), table4(t4) {}


Arithmetic::Arithmetic(std::vector<SyncedMemory> qm, std::vector<SyncedMemory> ql, std::vector<SyncedMemory> qr,
               std::vector<SyncedMemory> qo, std::vector<SyncedMemory> q4, std::vector<SyncedMemory> qc,
               std::vector<SyncedMemory> qhl, std::vector<SyncedMemory> qhr, std::vector<SyncedMemory> qh4,
               std::vector<SyncedMemory> qarith)
        : q_m(qm), q_l(ql), q_r(qr), q_o(qo), q_4(q4),
          q_c(qc), q_hl(qhl), q_hr(qhr), q_h4(qh4), q_arith(qarith) {}


Selectors::Selectors(std::vector<SyncedMemory> rs, std::vector<SyncedMemory> ls, 
                     std::vector<SyncedMemory> fs, std::vector<SyncedMemory> vs)
        : range_selector(rs), logic_selector(ls),
          fixed_group_add_selector(fs), variable_group_add_selector(vs) {}


ProverKey::ProverKey(
        std::vector<SyncedMemory> q_m_coeffs, std::vector<SyncedMemory> q_m_evals,
        std::vector<SyncedMemory> q_l_coeffs, std::vector<SyncedMemory> q_l_evals,
        std::vector<SyncedMemory> q_r_coeffs, std::vector<SyncedMemory> q_r_evals,
        std::vector<SyncedMemory> q_o_coeffs, std::vector<SyncedMemory> q_o_evals,
        std::vector<SyncedMemory> q_4_coeffs, std::vector<SyncedMemory> q_4_evals,
        std::vector<SyncedMemory> q_c_coeffs, std::vector<SyncedMemory> q_c_evals,
        std::vector<SyncedMemory> q_hl_coeffs, std::vector<SyncedMemory> q_hl_evals,
        std::vector<SyncedMemory> q_hr_coeffs, std::vector<SyncedMemory> q_hr_evals,
        std::vector<SyncedMemory> q_h4_coeffs, std::vector<SyncedMemory> q_h4_evals,
        std::vector<SyncedMemory> q_arith_coeffs, std::vector<SyncedMemory> q_arith_evals,
        std::vector<SyncedMemory> range_selector_coeffs, std::vector<SyncedMemory> range_selector_evals,
        std::vector<SyncedMemory> logic_selector_coeffs, std::vector<SyncedMemory> logic_selector_evals,
        std::vector<SyncedMemory> fixed_group_add_selector_coeffs, std::vector<SyncedMemory> fixed_group_add_selector_evals,
        std::vector<SyncedMemory> variable_group_add_selector_coeffs, std::vector<SyncedMemory> variable_group_add_selector_evals,
        SyncedMemory q_lookup_coeffs, std::vector<SyncedMemory> q_lookup_evals,
        std::vector<SyncedMemory> table1, std::vector<SyncedMemory> table2, std::vector<SyncedMemory> table3, std::vector<SyncedMemory> table4,
        std::vector<SyncedMemory> left_sigma_coeffs, std::vector<SyncedMemory> left_sigma_evals,
        std::vector<SyncedMemory> right_sigma_coeffs, std::vector<SyncedMemory> right_sigma_evals,
        std::vector<SyncedMemory> out_sigma_coeffs, std::vector<SyncedMemory> out_sigma_evals,
        std::vector<SyncedMemory> fourth_sigma_coeffs, std::vector<SyncedMemory> fourth_sigma_evals,
        std::vector<SyncedMemory> linear_evaluations,
        std::vector<SyncedMemory> v_h_coset_8n) : 
        arithmetic_coeffs(Arithmetic(q_m_coeffs, q_l_coeffs, q_r_coeffs, q_o_coeffs, 
                                     q_4_coeffs, q_c_coeffs, q_hl_coeffs, q_hr_coeffs, q_h4_coeffs, q_arith_coeffs)),
        arithmetic_evals(Arithmetic(q_m_evals, q_l_evals, q_r_evals, q_o_evals, 
                                    q_4_evals, q_c_evals, q_hl_evals, q_hr_evals, q_h4_evals, q_arith_evals)),
        selectors_coeffs(Selectors(range_selector_coeffs, logic_selector_coeffs,
                                   fixed_group_add_selector_coeffs, variable_group_add_selector_coeffs)),
        selectors_evals(Selectors(range_selector_evals, logic_selector_evals,
                                   fixed_group_add_selector_evals, variable_group_add_selector_evals)),
        lookup_coeffs(LookupTable(q_lookup_coeffs, table1, table2, table3, table4)), lookup_evals(q_lookup_evals),
        permutation_coeffs(Permutation(left_sigma_coeffs, right_sigma_coeffs, out_sigma_coeffs, fourth_sigma_coeffs)),
        permutation_evals(Permutation(left_sigma_evals, right_sigma_evals, out_sigma_evals, fourth_sigma_evals)),
        linear_evaluations(linear_evaluations),
        v_h_coset_8n(v_h_coset_8n) {}


ProverKey load_pk(ProverKeyC pk, uint64_t n, int chunk_num) {

    size_t size = n*fr::Limbs*sizeof(uint64_t)/chunk_num;
    int coset_chunk_num = chunk_num*8;

    std::vector<SyncedMemory> q_m_coeffs;
    std::vector<SyncedMemory> q_m_evals = chunk(pk.q_m_evals, size, coset_chunk_num);

    std::vector<SyncedMemory> q_l_coeffs = chunk(pk.q_l_coeffs, size, chunk_num);
    std::vector<SyncedMemory> q_l_evals = chunk(pk.q_l_evals, size, coset_chunk_num);

    std::vector<SyncedMemory> q_r_coeffs = chunk(pk.q_r_coeffs, size, chunk_num);
    std::vector<SyncedMemory> q_r_evals = chunk(pk.q_r_evals, size, coset_chunk_num);

    std::vector<SyncedMemory> q_o_coeffs = chunk(pk.q_o_coeffs, size, chunk_num);
    std::vector<SyncedMemory> q_o_evals = chunk(pk.q_o_evals, size, coset_chunk_num);

    std::vector<SyncedMemory> q_4_coeffs = chunk(pk.q_4_coeffs, size, chunk_num);
    std::vector<SyncedMemory> q_4_evals = chunk(pk.q_4_evals, size, coset_chunk_num);

    std::vector<SyncedMemory> q_c_coeffs = chunk(pk.q_c_coeffs, size, chunk_num);
    std::vector<SyncedMemory> q_c_evals = chunk(pk.q_c_evals, size, coset_chunk_num);

    std::vector<SyncedMemory> q_hl_coeffs = chunk(pk.q_hl_coeffs, size, chunk_num);
    std::vector<SyncedMemory> q_hl_evals = chunk(pk.q_hl_evals, size, coset_chunk_num);

    std::vector<SyncedMemory> q_hr_coeffs = chunk(pk.q_hr_coeffs, size, chunk_num);
    std::vector<SyncedMemory> q_hr_evals = chunk(pk.q_hr_evals, size, coset_chunk_num);

    std::vector<SyncedMemory> q_h4_coeffs = chunk(pk.q_h4_coeffs, size, chunk_num);
    std::vector<SyncedMemory> q_h4_evals = chunk(pk.q_h4_evals, size, coset_chunk_num);

    std::vector<SyncedMemory> q_arith_coeffs = chunk(pk.q_arith_coeffs, size, chunk_num);
    std::vector<SyncedMemory> q_arith_evals = chunk(pk.q_arith_evals, size, coset_chunk_num);

    std::vector<SyncedMemory> range_selector_coeffs;
    std::vector<SyncedMemory> range_selector_evals = chunk(pk.range_selector_evals, size, coset_chunk_num);

    std::vector<SyncedMemory> logic_selector_coeffs;
    std::vector<SyncedMemory> logic_selector_evals = chunk(pk.logic_selector_evals, size, coset_chunk_num);

    std::vector<SyncedMemory> fixed_group_add_selector_coeffs;
    std::vector<SyncedMemory> fixed_group_add_selector_evals = chunk(pk.fixed_group_add_selector_evals, size, coset_chunk_num);

    std::vector<SyncedMemory> variable_group_add_selector_coeffs;
    std::vector<SyncedMemory> variable_group_add_selector_evals = chunk(pk.variable_group_add_selector_evals, size, coset_chunk_num);

    SyncedMemory q_lookup_coeffs(0);

    std::vector<SyncedMemory> table1 = chunk(pk.table1, size, chunk_num);

    std::vector<SyncedMemory> table2 = chunk(pk.table2, size, chunk_num);

    std::vector<SyncedMemory> table3 = chunk(pk.table3, size, chunk_num);

    std::vector<SyncedMemory> table4 = chunk(pk.table4, size, chunk_num);
    

    // std::vector<SyncedMemory> q_lookup_coeffs = chunk(pk.q_lookup_coeffs, size, chunk_num);
    std::vector<SyncedMemory> q_lookup_evals = chunk(pk.q_lookup_evals, size, coset_chunk_num);

    // std::vector<SyncedMemory> table1 = chunk(pk.table1, size, chunk_num);
    // std::vector<SyncedMemory> table2 = chunk(pk.table2, size, chunk_num);
    // std::vector<SyncedMemory> table3 = chunk(pk.table3, size, chunk_num);
    // std::vector<SyncedMemory> table4 = chunk(pk.table4, size, chunk_num);

    std::vector<SyncedMemory> left_sigma_coeffs = chunk(pk.left_sigma_coeffs, size, chunk_num);
    std::vector<SyncedMemory> left_sigma_evals = chunk(pk.left_sigma_evals, size, coset_chunk_num);

    std::vector<SyncedMemory> right_sigma_coeffs = chunk(pk.right_sigma_coeffs, size, chunk_num);
    std::vector<SyncedMemory> right_sigma_evals = chunk(pk.right_sigma_evals, size, coset_chunk_num);

    std::vector<SyncedMemory> out_sigma_coeffs = chunk(pk.out_sigma_coeffs, size, chunk_num);
    std::vector<SyncedMemory> out_sigma_evals = chunk(pk.out_sigma_evals, size, coset_chunk_num);

    std::vector<SyncedMemory> fourth_sigma_coeffs = chunk(pk.fourth_sigma_coeffs, size, chunk_num);
    std::vector<SyncedMemory> fourth_sigma_evals = chunk(pk.fourth_sigma_evals, size, coset_chunk_num);

    std::vector<SyncedMemory> linear_evaluations = chunk(pk.linear_evaluations, size, coset_chunk_num);
    std::vector<SyncedMemory> v_h_coset_8n = chunk(pk.v_h_coset_8n, size, coset_chunk_num);


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
        uint64_t intended_pi_pos,
        std::vector<SyncedMemory> cs_q_lookup,
        SyncedMemory public_inputs,
        std::vector<SyncedMemory> w_l,
        std::vector<SyncedMemory> w_r,
        std::vector<SyncedMemory> w_o,
        std::vector<SyncedMemory> w_4
    ) : n(n), 
        intended_pi_pos(intended_pi_pos),
        cs_q_lookup(cs_q_lookup),
        public_inputs(public_inputs),
        w_l(w_l),
        w_r(w_r),
        w_o(w_o),
        w_4(w_4)
    {}

Circuit load_cs(CircuitC cs, int chunk_num){
    uint64_t n = 1<<26;
    size_t size = n*fr::Limbs*sizeof(uint64_t)/chunk_num;
    std::vector<SyncedMemory> q_lookup = chunk(cs.cs_q_lookup, size, chunk_num);
    std::vector<SyncedMemory> w_l = chunk(cs.w_l, size, chunk_num);
    std::vector<SyncedMemory> w_r = chunk(cs.w_r, size, chunk_num);
    std::vector<SyncedMemory> w_o = chunk(cs.w_o, size, chunk_num);
    std::vector<SyncedMemory> w_4 = chunk(cs.w_4, size, chunk_num);
    SyncedMemory pi(fr::Limbs*sizeof(uint64_t));
    uint64_t pi_64[4] = {4096197448552212605, 13022290890408443230, 10052896554840191078, 333484707567578785};
    void* pi_cpu = pi.mutable_cpu_data();
    memcpy(pi_cpu, pi_64, pi.size());
    uint64_t intended_pi_pos = 50593603;
    return Circuit(n, intended_pi_pos, 
                   q_lookup, pi, w_l, w_r, w_o, w_4);
}

CommitKey::CommitKey(
        SyncedMemory powers_of_g,
        SyncedMemory powers_of_gamma_g
    ) : powers_of_g(powers_of_g),
        powers_of_gamma_g(powers_of_gamma_g)
    {}


