// #include "mod.cuh"
# include "quotient.cuh"

SyncedMemory compute_first_lagrange_poly_scaled(uint64_t n, SyncedMemory scale) {
    Intt INTT(fr::TWO_ADICITY);
    SyncedMemory x_evals = pad_poly(scale, n);
    SyncedMemory x_coeffs = INTT.forward(x_evals);
    return x_coeffs;
}

void read_file1(const char* filename, void* data){
   // 打开文件
   std::ifstream file(filename, std::ios::binary);
   if (!file) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
   }
   file.seekg(0, std::ios::end);
   size_t fileSize = file.tellg();
   file.seekg(0, std::ios::beg);

   file.read(reinterpret_cast<char*>(data), fileSize);
   if (!file) {
    std::cerr << "Error reading file: " << filename << std::endl;
   }
   file.close();
}  

std::vector<SyncedMemory> compute_gate_constraint_satisfiability(
    SyncedMemory range_challenge,
    SyncedMemory logic_challenge,
    SyncedMemory fixed_base_challenge,
    SyncedMemory var_base_challenge,
    ProverKey pk,
    std::vector<SyncedMemory> ck,
    std::vector<SyncedMemory> global_buffer,
    SyncedMemory w_l_8n, 
    SyncedMemory w_r_8n, 
    SyncedMemory w_o_8n, 
    SyncedMemory w_4_8n,
    SyncedMemory pi_8n,
    int chunk_num,
    cudaStream_t stream1,
    cudaStream_t stream2) {
    
    std::vector<SyncedMemory> gate_contribution;
    gate_contribution.reserve(chunk_num);

    int SBOX_ALPHA = 5;
    uint64_t chunk_size = global_buffer[0].size();
    SyncedMemory P_A = COEFF_A();
    SyncedMemory P_D = COEFF_D();
    SyncedMemory one = fr::one();
    SyncedMemory two = fr::make_tensor(2);
    SyncedMemory three = fr::make_tensor(3);
    SyncedMemory four = fr::make_tensor(4);
    SyncedMemory nine = fr::make_tensor(9);
    SyncedMemory eighteen = fr::make_tensor(18);
    SyncedMemory eighty_one = fr::make_tensor(81);
    SyncedMemory eighty_three = fr::make_tensor(83);

    SyncedMemory kappa_range = mul_mod(range_challenge, range_challenge);
    SyncedMemory kappa_sq_range = mul_mod(kappa_range, kappa_range);
    SyncedMemory kappa_cu_range = mul_mod(kappa_sq_range, kappa_range);
    SyncedMemory kappa_fb = mul_mod(fixed_base_challenge, fixed_base_challenge);
    SyncedMemory kappa_sq_fb = mul_mod(kappa_fb, kappa_fb);
    SyncedMemory kappa_cu_fb = mul_mod(kappa_sq_fb, kappa_fb);
    SyncedMemory kappa_logic = mul_mod(logic_challenge, logic_challenge);
    SyncedMemory kappa_sq_logic = mul_mod(kappa_logic, kappa_logic);
    SyncedMemory kappa_cu_logic = mul_mod(kappa_sq_logic, kappa_logic);
    SyncedMemory kappa_qu_logic = mul_mod(kappa_cu_logic, kappa_logic);
    SyncedMemory kappa_vb = mul_mod(var_base_challenge, var_base_challenge);
    SyncedMemory kappa_sq_vb = mul_mod(kappa_vb, kappa_vb);
    
    cudaEvent_t event;
    cudaEventCreate(&event);

    for(int i=0; i<chunk_num;i++){
        void* buffer0_ = global_buffer[0].mutable_gpu_data();
        void* buffer1_ = global_buffer[1].mutable_gpu_data();
        void* buffer2_ = global_buffer[2].mutable_gpu_data();
        void* buffer3_ = global_buffer[3].mutable_gpu_data();
        void* buffer4_ = global_buffer[4].mutable_gpu_data();
        void* buffer5_ = global_buffer[5].mutable_gpu_data();
        void* buffer6_ = global_buffer[6].mutable_gpu_data();
        void* buffer7_ = global_buffer[7].mutable_gpu_data();
        void* buffer8_ = global_buffer[8].mutable_gpu_data();
        void* buffer9_ = global_buffer[9].mutable_gpu_data();
        void* buffer10_ = global_buffer[10].mutable_gpu_data();
        void* buffer11_ = global_buffer[11].mutable_gpu_data();
        void* buffer12_ = global_buffer[12].mutable_gpu_data();
        void* buffer13_ = global_buffer[13].mutable_gpu_data();

        void* q_m_ = pk.arithmetic_evals.q_m[i].mutable_cpu_data();
        void* q_l_ = pk.arithmetic_evals.q_l[i].mutable_cpu_data();
        void* q_r_ = pk.arithmetic_evals.q_r[i].mutable_cpu_data();
        void* q_o_ = pk.arithmetic_evals.q_o[i].mutable_cpu_data();
        void* q_4_ = pk.arithmetic_evals.q_4[i].mutable_cpu_data();
        void* q_c_ = pk.arithmetic_evals.q_c[i].mutable_cpu_data();
        void* q_hl_ = pk.arithmetic_evals.q_hl[i].mutable_cpu_data();
        void* q_hr_ = pk.arithmetic_evals.q_hr[i].mutable_cpu_data();
        void* q_h4_ = pk.arithmetic_evals.q_h4[i].mutable_cpu_data();
        void* q_arith_ = pk.arithmetic_evals.q_arith[i].mutable_cpu_data();
        void* range_ = pk.selectors_evals.range_selector[i].mutable_cpu_data();
        void* logic_ = pk.selectors_evals.logic_selector[i].mutable_cpu_data();
        void* fb_ = pk.selectors_evals.fixed_group_add_selector[i].mutable_cpu_data();
        void* vb_ = pk.selectors_evals.variable_group_add_selector[i].mutable_cpu_data();
        
        caffe_gpu_memcpy_async(chunk_size, q_m_, buffer0_, stream1);
        caffe_gpu_memcpy_async(chunk_size, q_l_, buffer1_, stream1);
        caffe_gpu_memcpy_async(chunk_size, q_r_, buffer2_, stream1);
        caffe_gpu_memcpy_async(chunk_size, q_o_, buffer3_, stream1);
        caffe_gpu_memcpy_async(chunk_size, q_4_, buffer4_, stream1);
        caffe_gpu_memcpy_async(chunk_size, q_c_, buffer5_, stream1);
        caffe_gpu_memcpy_async(chunk_size, q_hl_, buffer6_, stream1);
        caffe_gpu_memcpy_async(chunk_size, q_hr_, buffer7_, stream1);
        caffe_gpu_memcpy_async(chunk_size, q_h4_, buffer8_, stream1);
        caffe_gpu_memcpy_async(chunk_size, q_arith_, buffer9_, stream1);
        caffe_gpu_memcpy_async(chunk_size, range_, buffer10_, stream1);
        caffe_gpu_memcpy_async(chunk_size, logic_, buffer11_, stream1);
        caffe_gpu_memcpy_async(chunk_size, fb_, buffer12_, stream1);
        caffe_gpu_memcpy_async(chunk_size, vb_, buffer13_, stream1);

        void* buffer14_ = global_buffer[14].mutable_gpu_data();
        void* buffer15_ = global_buffer[15].mutable_gpu_data();
        void* buffer16_ = global_buffer[16].mutable_gpu_data();
        void* buffer17_ = global_buffer[17].mutable_gpu_data();
        void* buffer18_ = global_buffer[18].mutable_gpu_data();
        void* buffer19_ = global_buffer[19].mutable_gpu_data();
        void* buffer20_ = global_buffer[20].mutable_gpu_data();
        void* buffer21_ = global_buffer[21].mutable_gpu_data();

        void* w_l_ = w_l_8n.mutable_cpu_data();
        void* w_r_ = w_r_8n.mutable_cpu_data();
        void* w_o_ = w_o_8n.mutable_cpu_data();
        void* w_4_ = w_4_8n.mutable_cpu_data();
        void* pi_ = pi_8n.mutable_cpu_data();

        caffe_gpu_memcpy_async(chunk_size, w_l_ + i*chunk_size, buffer14_, stream1);
        caffe_gpu_memcpy_async(chunk_size, w_r_ + i*chunk_size, buffer15_, stream1);
        caffe_gpu_memcpy_async(chunk_size, w_o_ + i*chunk_size, buffer16_, stream1);
        caffe_gpu_memcpy_async(chunk_size, w_4_ + i*chunk_size, buffer17_, stream1);
        caffe_gpu_memcpy_async(chunk_size, pi_ + i*chunk_size, buffer18_, stream1);

        if(i != chunk_num -1){
            caffe_gpu_memcpy_async(chunk_size, w_l_ + 8*32 + i*chunk_size, buffer19_, stream1);
            caffe_gpu_memcpy_async(chunk_size, w_r_ + 8*32 + i*chunk_size, buffer20_, stream1);
            caffe_gpu_memcpy_async(chunk_size, w_4_ + 8*32 + i*chunk_size, buffer21_, stream1);
        }
        else{
            caffe_gpu_memcpy_async(chunk_size - 8*32, w_l_ + i*chunk_size, buffer19_, stream1);
            caffe_gpu_memcpy_async(chunk_size - 8*32, w_r_ + i*chunk_size, buffer20_, stream1);
            caffe_gpu_memcpy_async(chunk_size - 8*32, w_4_ + i*chunk_size, buffer21_, stream1);

            caffe_gpu_memcpy_async((size_t)8*32, w_l_, buffer19_ + chunk_size - 8*32, stream1);
            caffe_gpu_memcpy_async((size_t)8*32, w_r_, buffer20_ + chunk_size - 8*32, stream1);
            caffe_gpu_memcpy_async((size_t)8*32, w_4_, buffer21_ + chunk_size - 8*32, stream1);
        }
        cudaEventRecord(event, stream1);
        cudaStreamWaitEvent(stream2, event);
        gate_contribution.push_back(cuda::compute_gate_constraint_allmerge_cuda(
                    global_buffer[14], global_buffer[15], global_buffer[16], global_buffer[17], 
                    global_buffer[19], global_buffer[20], global_buffer[21], global_buffer[0], 
                    global_buffer[1], global_buffer[2], global_buffer[3], global_buffer[4], 
                    global_buffer[5], global_buffer[6], global_buffer[7], global_buffer[8], 
                    global_buffer[9], global_buffer[10], global_buffer[11], global_buffer[12], 
                    global_buffer[13], global_buffer[18], four, one, two, three, 
                    kappa_range, kappa_sq_range, kappa_cu_range, range_challenge, 
                    P_D, nine, kappa_cu_logic, 
                    eighteen, eighty_one, eighty_three, kappa_qu_logic, 
                    logic_challenge, P_A, kappa_fb, kappa_sq_fb, 
                    kappa_cu_fb, fixed_base_challenge, kappa_vb, kappa_sq_vb, 
                    var_base_challenge, SBOX_ALPHA, stream2));
    }
    return gate_contribution;
}

void compute_permutation_checks(
    std::vector<SyncedMemory> gate_constraints,
    SyncedMemory alpha,
    SyncedMemory beta,
    SyncedMemory gamma,
    ProverKey pk,
    std::vector<SyncedMemory> ck,
    std::vector<SyncedMemory> global_buffer,
    SyncedMemory w_l_8n, 
    SyncedMemory w_r_8n, 
    SyncedMemory w_o_8n, 
    SyncedMemory w_4_8n,
    SyncedMemory z_8n,
    SyncedMemory l1_8n,
    int chunk_num,
    cudaStream_t stream1,
    cudaStream_t stream2){
    
    uint64_t chunk_size = global_buffer[0].size();
    
    SyncedMemory k1 = K1();
    SyncedMemory k2 = K2();
    SyncedMemory k3 = K3();
    SyncedMemory mod = fr::MODULUS();
    SyncedMemory one = fr::one();

    void* k1_ = k1.mutable_gpu_data();
    void* k2_ = k2.mutable_gpu_data();
    void* k3_ = k3.mutable_gpu_data();
    void* mod_ = mod.mutable_gpu_data();
    void* one_ = one.mutable_gpu_data();
    
    void* buffer0_ = global_buffer[0].mutable_gpu_data();
    void* buffer1_ = global_buffer[1].mutable_gpu_data();
    void* buffer2_ = global_buffer[2].mutable_gpu_data();
    void* buffer3_ = global_buffer[3].mutable_gpu_data();
    void* buffer4_ = global_buffer[4].mutable_gpu_data();
    void* buffer5_ = global_buffer[5].mutable_gpu_data();
    void* buffer6_ = global_buffer[6].mutable_gpu_data();
    void* buffer7_ = global_buffer[7].mutable_gpu_data();
    void* buffer8_ = global_buffer[8].mutable_gpu_data();
    void* buffer9_ = global_buffer[9].mutable_gpu_data();
    void* buffer10_ = global_buffer[10].mutable_gpu_data();
    void* buffer11_ = global_buffer[11].mutable_gpu_data();
    void* buffer14_ = global_buffer[14].mutable_gpu_data();

    void* w_l_ = w_l_8n.mutable_cpu_data();
    void* w_r_ = w_r_8n.mutable_cpu_data();
    void* w_o_ = w_o_8n.mutable_cpu_data();
    void* w_4_ = w_4_8n.mutable_cpu_data();
    void* z_ = z_8n.mutable_cpu_data();
    void* l1_ = l1_8n.mutable_cpu_data();

    // extend_mod
    repeat_to_poly_(mod, global_buffer[12], chunk_size/(sizeof(uint64_t)*fr::Limbs), stream2);
    // extend_one
    repeat_to_poly_(one, global_buffer[13], chunk_size/(sizeof(uint64_t)*fr::Limbs), stream2);
    
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);

    for(int i=0; i<chunk_num;i++){

        void* left_ = pk.permutation_evals[0][i].mutable_cpu_data();
        void* right_ = pk.permutation_evals[1][i].mutable_cpu_data();
        void* out_ = pk.permutation_evals[2][i].mutable_cpu_data();
        void* fourth_ = pk.permutation_evals[3][i].mutable_cpu_data();
        void* linear_ = pk.linear_evaluations[i].mutable_cpu_data();

        caffe_gpu_memcpy_async(chunk_size, linear_, buffer10_, stream1);

        caffe_gpu_memcpy_async(chunk_size, w_l_ + i*chunk_size, buffer0_, stream1);
        caffe_gpu_memcpy_async(chunk_size, w_r_ + i*chunk_size, buffer1_, stream1);
        caffe_gpu_memcpy_async(chunk_size, w_o_ + i*chunk_size, buffer2_, stream1);
        caffe_gpu_memcpy_async(chunk_size, w_4_ + i*chunk_size, buffer3_, stream1);
        caffe_gpu_memcpy_async(chunk_size, z_ + i*chunk_size, buffer4_, stream1);

        cudaEventRecord(event1, stream1);
        

        caffe_gpu_memcpy_async(chunk_size, l1_ + i*chunk_size, buffer5_, stream1);
        caffe_gpu_memcpy_async(chunk_size, left_, buffer6_, stream1);
        caffe_gpu_memcpy_async(chunk_size, right_, buffer7_, stream1);
        caffe_gpu_memcpy_async(chunk_size, out_, buffer8_, stream1);
        caffe_gpu_memcpy_async(chunk_size, fourth_, buffer9_, stream1);
        cudaEventRecord(event2, stream1);

        if(i != chunk_num -1){
            caffe_gpu_memcpy_async(chunk_size, z_ + 8*32 + i*chunk_size, buffer11_, stream1);
        }
        else{
            caffe_gpu_memcpy_async(chunk_size - 8*32, z_ + i*chunk_size, buffer11_, stream1);
            caffe_gpu_memcpy_async((size_t)8*32, z_, buffer11_ + chunk_size - 8*32, stream1);
        }

        cudaStreamWaitEvent(stream2, event1);

        cuda::compute_quotient_identity_range_check_i(global_buffer[10],
            global_buffer[0],  global_buffer[1], global_buffer[2], global_buffer[3], 
            global_buffer[4], alpha, beta, gamma, 
            k1, k2, k3, gate_constraints[i], stream2);

        cudaStreamWaitEvent(stream2, event2);
        cuda::compute_quotient_copy_range_check_i(global_buffer[6], global_buffer[7], global_buffer[8], global_buffer[9], global_buffer[12],
                                                  global_buffer[0], global_buffer[1], global_buffer[2], global_buffer[3], global_buffer[11],
                                                  alpha, beta, gamma, gate_constraints[i], stream2);
        
        cuda::compute_quotient_term_check_one_i(global_buffer[4], global_buffer[5], global_buffer[13], gate_constraints[i], stream2);
 
    }
    cudaEventDestroy(event1);
    cudaEventDestroy(event2);
}

void compute_quotient_i(
    std::vector<SyncedMemory> gate_constraints,
    SyncedMemory delta, SyncedMemory epsilon, SyncedMemory zeta, SyncedMemory lookup_seq,
    ProverKey pk,
    std::vector<SyncedMemory> global_buffer, 
    SyncedMemory w_l_8n, 
    SyncedMemory w_r_8n, 
    SyncedMemory w_o_8n, 
    SyncedMemory w_4_8n, 
    SyncedMemory f_8n, 
    SyncedMemory table_8n, 
    SyncedMemory h1_8n, 
    SyncedMemory h2_8n, 
    SyncedMemory z2_8n,
    SyncedMemory l1_8n,
    int chunk_num,
    cudaStream_t stream1 = (cudaStream_t)0,
    cudaStream_t stream2 = (cudaStream_t)0)
{
    uint64_t chunk_size = global_buffer[0].size();

    SyncedMemory mod = fr::MODULUS();
    SyncedMemory one = fr::one();

    SyncedMemory lookup_seq_sq = mul_mod(lookup_seq, lookup_seq);
    SyncedMemory lookup_seq_cu = mul_mod(lookup_seq_sq, lookup_seq);

    void* mod_ = mod.mutable_gpu_data();
    void* one_ = one.mutable_gpu_data();

    void* buffer0_ = global_buffer[0].mutable_gpu_data();
    void* buffer1_ = global_buffer[1].mutable_gpu_data();
    void* buffer2_ = global_buffer[2].mutable_gpu_data();
    void* buffer3_ = global_buffer[3].mutable_gpu_data();
    void* buffer4_ = global_buffer[4].mutable_gpu_data();
    void* buffer5_ = global_buffer[5].mutable_gpu_data();
    void* buffer6_ = global_buffer[6].mutable_gpu_data();
    void* buffer7_ = global_buffer[7].mutable_gpu_data();
    void* buffer8_ = global_buffer[8].mutable_gpu_data();
    void* buffer9_ = global_buffer[9].mutable_gpu_data();
    void* buffer10_ = global_buffer[10].mutable_gpu_data();
    void* buffer11_ = global_buffer[11].mutable_gpu_data();
    void* buffer12_ = global_buffer[12].mutable_gpu_data();
    void* buffer13_ = global_buffer[13].mutable_gpu_data();
    void* buffer14_ = global_buffer[14].mutable_gpu_data();
    
    void* w_l_ = w_l_8n.mutable_cpu_data();
    void* w_r_ = w_r_8n.mutable_cpu_data();
    void* w_o_ = w_o_8n.mutable_cpu_data();
    void* w_4_ = w_4_8n.mutable_cpu_data();
    void* f_ = f_8n.mutable_cpu_data();
    void* table_ = table_8n.mutable_cpu_data();
    void* z2_ = z2_8n.mutable_cpu_data();
    void* l1_ = l1_8n.mutable_cpu_data();
    void* h1_ = h1_8n.mutable_cpu_data();
    void* h2_ = h2_8n.mutable_cpu_data();

    // extend_mod
    repeat_to_poly_(mod, global_buffer[15], chunk_size/(sizeof(uint64_t)*fr::Limbs), stream2);

    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);

    for(int i =0; i<chunk_num;i++){

        void* q_lookup_ = pk.lookup_evals[i].mutable_cpu_data();

        caffe_gpu_memcpy_async(chunk_size, w_l_ + i*chunk_size, buffer0_, stream1);
        caffe_gpu_memcpy_async(chunk_size, w_r_ + i*chunk_size, buffer1_, stream1);
        caffe_gpu_memcpy_async(chunk_size, w_o_ + i*chunk_size, buffer2_, stream1);
        caffe_gpu_memcpy_async(chunk_size, w_4_ + i*chunk_size, buffer3_, stream1);

        cudaEventRecord(event1, stream1);
        cudaStreamWaitEvent(stream2, event1);
        compress(global_buffer[16], global_buffer[0], global_buffer[1], global_buffer[2], global_buffer[3], zeta, stream2);

        caffe_gpu_memcpy_async(chunk_size, f_ + i*chunk_size, buffer5_, stream1);
        caffe_gpu_memcpy_async(chunk_size, table_ + i*chunk_size, buffer6_, stream1);
        caffe_gpu_memcpy_async(chunk_size, z2_ + i*chunk_size, buffer7_, stream1);
        caffe_gpu_memcpy_async(chunk_size, h1_ + i*chunk_size, buffer8_, stream1);
        caffe_gpu_memcpy_async(chunk_size, h2_ + i*chunk_size, buffer9_, stream1);
        caffe_gpu_memcpy_async(chunk_size, l1_ + i*chunk_size, buffer10_, stream1);
        caffe_gpu_memcpy_async(chunk_size, q_lookup_ + i*chunk_size, buffer11_, stream1);

        if(i != chunk_num -1){
            caffe_gpu_memcpy_async(chunk_size, h1_ + 8*32 + i*chunk_size, buffer12_, stream1);

            caffe_gpu_memcpy_async(chunk_size, table_ + 8*32 + i*chunk_size, buffer13_, stream1);

            caffe_gpu_memcpy_async(chunk_size, z2_ + 8*32 + i*chunk_size, buffer14_, stream1);
        }
        else{
            caffe_gpu_memcpy_async(chunk_size - 8*32, h1_ + i*chunk_size, buffer12_, stream1);
            caffe_gpu_memcpy_async((size_t)8*32, h1_, buffer12_ + chunk_size - 8*32, stream1);

            caffe_gpu_memcpy_async(chunk_size - 8*32, table_ + i*chunk_size, buffer13_, stream1);
            caffe_gpu_memcpy_async((size_t)8*32, table_, buffer13_ + chunk_size - 8*32, stream1);

            caffe_gpu_memcpy_async(chunk_size - 8*32, z2_ + i*chunk_size, buffer14_, stream1);
            caffe_gpu_memcpy_async((size_t)8*32, z2_, buffer14_ + chunk_size - 8*32, stream1);
        }

        cudaEventRecord(event2, stream1);
        cudaStreamWaitEvent(stream2, event2);

        cuda::compute_quotient_i(global_buffer[16], global_buffer[5], global_buffer[6], global_buffer[13], global_buffer[8], global_buffer[12],
                                 global_buffer[9], global_buffer[7], global_buffer[14], global_buffer[10], global_buffer[11], global_buffer[15],
                                delta, epsilon, zeta, one, lookup_seq, lookup_seq_sq, lookup_seq_cu, gate_constraints[i], stream2);
    }
    
}
std::vector<SyncedMemory> compute_quotient_poly(
    uint64_t n,
    ProverKey pk,
    std::vector<SyncedMemory> ck,
    SyncedMemory chunk_msm_workspace, SyncedMemory chunk_msm_out,
    std::vector<SyncedMemory> wl_commits_step1,
    std::vector<SyncedMemory> wr_commits_step1,
    std::vector<SyncedMemory> wo_commits_step1,
    std::vector<SyncedMemory> w4_commits_step1,
    std::vector<SyncedMemory> global_buffer,
    SyncedMemory w_l, 
    SyncedMemory w_r, 
    SyncedMemory w_o, 
    SyncedMemory w_4,
    SyncedMemory pi,
    SyncedMemory w_l_8n,
    SyncedMemory w_r_8n,
    SyncedMemory w_o_8n,
    SyncedMemory w_4_8n,
    SyncedMemory pi_8n,
    SyncedMemory z_eval_8n,
    SyncedMemory z2_eval_8n,
    SyncedMemory table_eval_8n,
    SyncedMemory h1_eval_8n,
    SyncedMemory h2_eval_8n,
    SyncedMemory l1_eval_8n,
    SyncedMemory f_eval_8n,
    std::vector<SyncedMemory> z_chunks,
    std::vector<SyncedMemory> z2_chunks,
    std::vector<SyncedMemory> f_chunks,
    std::vector<SyncedMemory> table_chunks,
    std::vector<SyncedMemory> h1_chunks,
    std::vector<SyncedMemory> h2_chunks,
    SyncedMemory alpha,
    SyncedMemory beta,
    SyncedMemory gamma,
    SyncedMemory delta,
    SyncedMemory epsilon,
    SyncedMemory zeta,
    SyncedMemory range_challenge,
    SyncedMemory logic_challenge,
    SyncedMemory fixed_base_challenge,
    SyncedMemory var_base_challenge,
    SyncedMemory lookup_challenge,
    uint32_t* ntt_step1_map,
    uint32_t* ntt_step2_map,
    uint32_t* ntt_step3_map,
    SyncedMemory lde_cpu_map,
    SyncedMemory lde_input,
    std::vector<SyncedMemory> lde_cpu_buffer,
    int lg_LDE,
    int chunk_num,
    Transcript transcript,
    cudaStream_t stream1,
    cudaStream_t stream2) {

    int coset_chunk_num = 8 * chunk_num;
    void* map_ = lde_cpu_map.mutable_cpu_data();
    Ntt_coset NTT(fr::TWO_ADICITY, lg_LDE);
    uint64_t chunk_size = global_buffer[0].size();
    
    // need to refresh map to all-zero before each LDE, this can be replaced with more cpu memory...
    LDE_forward_with_commit(NTT, lg_LDE, w_l, ck, chunk_msm_workspace, chunk_msm_out, wl_commits_step1, global_buffer, lde_cpu_buffer, ntt_step1_map, ntt_step2_map, ntt_step3_map, lde_cpu_map, lde_input, true, stream1, stream2);
    void* w_l_8n_ = w_l_8n.mutable_cpu_data();
    for(int i = 0;i<8;i++){
        void* d_ = global_buffer[i].mutable_gpu_data();
        caffe_gpu_memcpy_async(chunk_size, d_, w_l_8n_ + i * chunk_size, stream2);
    }
    SyncedMemory step2_res = cpu::msm_collect_cpu(wl_commits_step1[0], chunk_size/(fr::Limbs*sizeof(uint64_t)));
    ProjectivePointG1 commit_pro = to_point(step2_res);
    AffinePointG1 commit_affine = to_affine(commit_pro);
    transcript.append_chunk("w_l", commit_affine, 0);

    for(int i = 8;i<16;i++){
        void* d_ = global_buffer[i].mutable_gpu_data();
        caffe_gpu_memcpy_async(chunk_size, d_, w_l_8n_ + i * chunk_size, stream2);
    }
    step2_res = cpu::msm_collect_cpu(wl_commits_step1[1], chunk_size/(fr::Limbs*sizeof(uint64_t)));
    commit_pro = to_point(step2_res);
    commit_affine = to_affine(commit_pro);
    transcript.append_chunk("w_l", commit_affine, 1);

    LDE_forward_with_commit(NTT, lg_LDE, w_r, ck, chunk_msm_workspace, chunk_msm_out, wr_commits_step1, global_buffer, lde_cpu_buffer, ntt_step1_map, ntt_step2_map, ntt_step3_map, lde_cpu_map, lde_input, true, stream1, stream2);
    void* w_r_8n_ = w_r_8n.mutable_cpu_data();
    for(int i = 0;i<8;i++){
        void* d_ = global_buffer[i].mutable_gpu_data();
        caffe_gpu_memcpy_async(chunk_size, d_, w_r_8n_ + i * chunk_size, stream2);
    }
    step2_res = cpu::msm_collect_cpu(wr_commits_step1[0], chunk_size/(fr::Limbs*sizeof(uint64_t)));
    commit_pro = to_point(step2_res);
    commit_affine = to_affine(commit_pro);
    transcript.append_chunk("w_r", commit_affine, 0);

    for(int i = 8;i<16;i++){
        void* d_ = global_buffer[i].mutable_gpu_data();
        caffe_gpu_memcpy_async(chunk_size, d_, w_r_8n_ + i * chunk_size, stream2);
    }
    step2_res = cpu::msm_collect_cpu(wr_commits_step1[1], chunk_size/(fr::Limbs*sizeof(uint64_t)));
    commit_pro = to_point(step2_res);
    commit_affine = to_affine(commit_pro);
    transcript.append_chunk("w_r", commit_affine, 1);

    LDE_forward_with_commit(NTT, lg_LDE, w_o, ck, chunk_msm_workspace, chunk_msm_out, wo_commits_step1, global_buffer, lde_cpu_buffer, ntt_step1_map, ntt_step2_map, ntt_step3_map, lde_cpu_map, lde_input, false, stream1, stream2);
    void* w_o_8n_ = w_o_8n.mutable_cpu_data();
    for(int i = 0;i<8;i++){
        void* d_ = global_buffer[i].mutable_gpu_data();
        caffe_gpu_memcpy_async(chunk_size, d_, w_o_8n_ + i * chunk_size, stream2);
    }
    step2_res = cpu::msm_collect_cpu(wr_commits_step1[0], chunk_size/(fr::Limbs*sizeof(uint64_t)));
    commit_pro = to_point(step2_res);
    commit_affine = to_affine(commit_pro);
    transcript.append_chunk("w_o", commit_affine, 0);

    for(int i = 8;i<16;i++){
        void* d_ = global_buffer[i].mutable_gpu_data();
        caffe_gpu_memcpy_async(chunk_size, d_, w_o_8n_ + i * chunk_size, stream2);
    }
    step2_res = cpu::msm_collect_cpu(wo_commits_step1[1], chunk_size/(fr::Limbs*sizeof(uint64_t)));
    commit_pro = to_point(step2_res);
    commit_affine = to_affine(commit_pro);
    transcript.append_chunk("w_o", commit_affine, 1);

    LDE_forward_with_commit(NTT, lg_LDE, w_4, ck, chunk_msm_workspace, chunk_msm_out, w4_commits_step1, global_buffer, lde_cpu_buffer, ntt_step1_map, ntt_step2_map, ntt_step3_map, lde_cpu_map, lde_input, true, stream1, stream2);
    void* w_4_8n_ = w_4_8n.mutable_cpu_data();
    for(int i = 0;i<8;i++){
        void* d_ = global_buffer[i].mutable_gpu_data();
        caffe_gpu_memcpy_async(chunk_size, d_, w_4_8n_ + i * chunk_size, stream2);
    }
    step2_res = cpu::msm_collect_cpu(w4_commits_step1[0], chunk_size/(fr::Limbs*sizeof(uint64_t)));
    commit_pro = to_point(step2_res);
    commit_affine = to_affine(commit_pro);
    transcript.append_chunk("w_4", commit_affine, 0);

    for(int i = 8;i<16;i++){
        void* d_ = global_buffer[i].mutable_gpu_data();
        caffe_gpu_memcpy_async(chunk_size, d_, w_4_8n_ + i * chunk_size, stream2);
    }
    step2_res = cpu::msm_collect_cpu(w4_commits_step1[1], chunk_size/(fr::Limbs*sizeof(uint64_t)));
    commit_pro = to_point(step2_res);
    commit_affine = to_affine(commit_pro);
    transcript.append_chunk("w_4", commit_affine, 1);

    LDE_forward(NTT, lg_LDE, pi, pi_8n, global_buffer, lde_cpu_buffer, ntt_step1_map, ntt_step2_map, ntt_step3_map, lde_cpu_map, lde_input, false, stream1, stream2);
    
    cudaDeviceSynchronize();

    std::vector<SyncedMemory> gate_constraints = compute_gate_constraint_satisfiability(
        range_challenge,
        logic_challenge,
        fixed_base_challenge,
        var_base_challenge,
        pk,
        ck,
        global_buffer,
        w_l_8n, 
        w_r_8n, 
        w_o_8n, 
        w_4_8n,
        pi_8n,
        coset_chunk_num,
        stream1,
        stream2
    );
    cudaDeviceSynchronize();

    cudaEvent_t event;
    cudaEventCreate(&event);
    Intt INTT(fr::TWO_ADICITY, stream2);
    SyncedMemory alpha2 = mul_mod(alpha, alpha, stream2);
    pad_poly_(alpha2, global_buffer[14], n/2, stream2);
    caffe_gpu_memset_async(chunk_size, 0, global_buffer[15].mutable_gpu_data(), stream1);
    cudaEventRecord(event, stream1);
    INTT._forward_(global_buffer[14], stream2);
    cudaStreamWaitEvent(stream2, event);
    INTT._forward_(global_buffer[15], stream2);
    std::vector<SyncedMemory> l1;
    l1.push_back(global_buffer[14]);
    l1.push_back(global_buffer[15]);
    INTT.Params = SyncedMemory();

    LDE_forward_chunk(NTT, lg_LDE, chunk_num, l1, l1_eval_8n, global_buffer, lde_cpu_buffer, ntt_step1_map, ntt_step2_map, ntt_step3_map, lde_cpu_map, lde_input, false, stream1, stream2);
    
    LDE_forward_chunk(NTT, lg_LDE, chunk_num, z_chunks, z_eval_8n, global_buffer, lde_cpu_buffer, ntt_step1_map, ntt_step2_map, ntt_step3_map, lde_cpu_map, lde_input, true, stream1, stream2);

    cudaDeviceSynchronize();

    compute_permutation_checks(gate_constraints, alpha, beta, gamma, pk, ck, global_buffer, w_l_8n, w_r_8n, w_o_8n, w_4_8n, z_eval_8n, l1_eval_8n, coset_chunk_num, stream1, stream2);
    
    LDE_forward_chunk(NTT, lg_LDE, chunk_num, z2_chunks, z2_eval_8n, global_buffer, lde_cpu_buffer, ntt_step1_map, ntt_step2_map, ntt_step3_map, lde_cpu_map, lde_input, true, stream1, stream2);

    LDE_forward_chunk(NTT, lg_LDE, chunk_num, table_chunks, table_eval_8n, global_buffer, lde_cpu_buffer, ntt_step1_map, ntt_step2_map, ntt_step3_map, lde_cpu_map, lde_input, true, stream1, stream2);

    LDE_forward_chunk(NTT, lg_LDE, chunk_num, h1_chunks, h1_eval_8n, global_buffer, lde_cpu_buffer, ntt_step1_map, ntt_step2_map, ntt_step3_map, lde_cpu_map, lde_input, true, stream1, stream2);

    LDE_forward_chunk(NTT, lg_LDE, chunk_num, h2_chunks, h2_eval_8n, global_buffer, lde_cpu_buffer, ntt_step1_map, ntt_step2_map, ntt_step3_map, lde_cpu_map, lde_input, false, stream1, stream2);

    LDE_forward_chunk(NTT, lg_LDE, chunk_num, f_chunks, f_eval_8n, global_buffer, lde_cpu_buffer, ntt_step1_map, ntt_step2_map, ntt_step3_map, lde_cpu_map, lde_input, false, stream1, stream2);

    NTT.Params = SyncedMemory();
    compute_quotient_i(
    gate_constraints, 
    delta, epsilon, zeta, lookup_challenge, 
    pk, 
    global_buffer, 
    w_l_8n, w_r_8n, w_o_8n, w_4_8n, 
    f_eval_8n, table_eval_8n, h1_eval_8n, h2_eval_8n, z2_eval_8n, l1_eval_8n, 
    coset_chunk_num, 
    stream1, stream2);
    
    return gate_constraints;
}