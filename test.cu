#pragma once
#include <functional>
#include <fstream>
#include <iostream>
// #include "PLONK/src/transcript/transcript.cuh"
#include "PLONK/src/KZG/kzg10.cuh"
#include "PLONK/src/plonk_core/src/permutation/mod.cuh"
#include "PLONK/src/plonk_core/src/proof_system/pi.cuh"
#include "PLONK/src/plonk_core/src/proof_system/quotient.cuh"
#include "PLONK/src/plonk_core/src/proof_system/linearisation.cuh"
#include <bitset>

typedef struct msmstruct
{
    uint64_t msm_worksize;
    uint64_t msm_outsize;
};

void read_file(const char* filename, void* data){
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

msmstruct msm_init(uint64_t npoints){
    auto wbits = 17;
    auto smcount = 114;
    auto  BATCH_ADD_BLOCK_SIZE = 256;

    if (npoints > 192) {
      wbits = ::std::min(lg2(npoints + npoints / 2) - 8, 18);
      if (wbits < 10)
        wbits = 10;
    } else if (npoints > 0) {
      wbits = 10;
    }
    auto nbits = fr::MODULUS_BITS;
    auto nwins = (nbits - 1) / wbits + 1;
    uint32_t row_sz = 1U << (wbits - 1);
    size_t d_buckets_sz =
        (nwins * row_sz) + (smcount * BATCH_ADD_BLOCK_SIZE / WARP_SZ);
    size_t d_blob_sz =
        d_buckets_sz * sizeof(uint64_t) * fq::Limbs * 4 +
        (nwins * row_sz * sizeof(uint32_t));
    uint32_t blob_u64 = d_blob_sz / sizeof(uint64_t);
    
    size_t digits_sz = nwins * npoints * sizeof(uint32_t);
    uint32_t temp_sz = npoints * sizeof(uint64_t) * fr::Limbs + digits_sz;
    uint32_t temp_u64 = temp_sz / sizeof(uint64_t);

    msmstruct result;
    result.msm_worksize = (blob_u64 + temp_u64) * sizeof(uint64_t);
    result.msm_outsize = (nwins * MSM_NTHREADS / 1 * 2 + smcount * BATCH_ADD_BLOCK_SIZE / WARP_SZ) * 4 * fq::Limbs * sizeof(uint64_t);
    return result;
    // SyncedMemory msm_workspace((blob_u64 + temp_u64) * sizeof(uint64_t));
    // SyncedMemory msm_out((nwins * MSM_NTHREADS / 1 * 2 + smcount * BATCH_ADD_BLOCK_SIZE / WARP_SZ) * 4 * fq::Limbs * sizeof(uint64_t));
}
void prove(ProverKeyC pkc, Circuit cs, std::vector<SyncedMemory> ck, SyncedMemory pp){
    uint64_t size = cs.n;
    int chunk_num = 2;
    int coset_chunk_num = chunk_num*8;
    int lg_chunk = 21;
    int lg_N = 22;
    int lg_LDE = 25;
    ProverKey pk = load_pk(pkc, size, chunk_num);
    Radix2EvaluationDomain domain = newdomain(size);
    uint64_t n = domain.size;
    uint64_t lde_size = 1<<lg_LDE;
    uint64_t chunk_size = n/chunk_num*sizeof(uint64_t)*fr::Limbs;
    uint64_t full_size = n*sizeof(uint64_t)*fr::Limbs;
    int smcount = 114;
    char transcript_init[] = "Merkle tree";
    Transcript transcript = Transcript(transcript_init);
    char pi[] = "pi";
    transcript.append_pi(pi, cs.public_inputs, cs.intended_pi_pos);

    uint32_t* ntt_step1_map = new uint32_t[n];
    uint32_t* ntt_step2_map = new uint32_t[n];
    uint32_t* ntt_step3_map = new uint32_t[n];

    uint32_t* lde_step1_map = new uint32_t[lde_size];
    uint32_t* lde_step2_map = new uint32_t[lde_size];
    uint32_t* lde_step3_map = new uint32_t[lde_size];

    int stage1 = 0;
    int iterations1 = 7;
    int stage2 = 7;
    int iterations2 = 7;
    int stage3 = 14;
    int iterations3 = 8;

    uint32_t block_size2 = 64;
    uint32_t num_threads2 = 1 << (lg_N - 1);
    uint32_t num_block2 = (num_threads2 + block_size2 - 1) / block_size2;

    uint32_t block_size3 = 128;
    uint32_t num_threads3 = 1 << (lg_N - 1);
    uint32_t num_block3 = (num_threads3 + block_size3 - 1) / block_size3;

    calculate_map1(ntt_step1_map, lg_N);
    calculate_map1(lde_step1_map, lg_LDE);

    for(int i=0;i<num_block2;i++){
        for(int j=0;j<block_size2;j++){
            uint32_t tid = j + i*block_size2;
            calculate_indices(tid, stage1, iterations1, stage2, iterations2, ntt_step2_map);
        }
    }

    for(int i=0;i<num_block3;i++){
        for(int j=0;j<block_size3;j++){
            uint32_t tid = j + i*block_size3;
            calculate_indices(tid, stage1, iterations3, stage3, iterations3, ntt_step3_map);
        }
    }

    stage1 = 0;
    iterations1 = 8;
    stage2 = 8;
    iterations2 = 8;
    stage3 = 16;
    iterations3 = 9;

    block_size2 = 128;
    num_threads2 = 1 << (lg_LDE - 1);
    num_block2 = (num_threads2 + block_size2 - 1) / block_size2;

    block_size3 = 256;
    num_threads3 = 1 << (lg_LDE - 1);
    num_block3 = (num_threads3 + block_size3 - 1) / block_size3;

    for(int i=0;i<num_block2;i++){
        for(int j=0;j<block_size2;j++){
            uint32_t tid = j + i*block_size2;
            calculate_indices(tid, stage1, iterations1, stage2, iterations2, lde_step2_map);
        }
    }

    for(int i=0;i<num_block3;i++){
        for(int j=0;j<block_size3;j++){
            uint32_t tid = j + i*block_size3;
            calculate_indices(tid, stage1, iterations3, stage3, iterations3, lde_step3_map);
        }
    }

    SyncedMemory ntt_cpu_map(n*fr::Limbs*sizeof(uint64_t));
    void* map_ = ntt_cpu_map.mutable_cpu_data();
    caffe_memset(ntt_cpu_map.size(), 0, map_);

    SyncedMemory lde_cpu_map(lde_size*fr::Limbs*sizeof(uint64_t));
    void* lde_map_ = lde_cpu_map.mutable_cpu_data();
    
    SyncedMemory lde_input(n*fr::Limbs*sizeof(uint64_t));
    void* input_ = lde_input.mutable_cpu_data();

    std::vector<SyncedMemory>global_buffer;

    for(int j=0;j<8;j++){
        SyncedMemory buffer(chunk_size);
        global_buffer.push_back(buffer);
    }
    
    std::vector<SyncedMemory>quotient_buffer;

    for(int j=0;j<22;j++){
        SyncedMemory buffer(chunk_size);
        void* buffer_ = buffer.mutable_cpu_data();
        quotient_buffer.push_back(buffer);
    }

    std::vector<SyncedMemory>ntt_cpu_buffer;
    
    for(int j=0;j<chunk_num;j++){
        SyncedMemory buffer(chunk_size);
        void* buffer_ = buffer.mutable_cpu_data();
        ntt_cpu_buffer.push_back(buffer);
    }

    std::vector<SyncedMemory>lde_cpu_buffer;
    
    for(int j=0;j<16;j++){
        SyncedMemory buffer(chunk_size*chunk_num);
        void* buffer_ = buffer.mutable_cpu_data();
        lde_cpu_buffer.push_back(buffer);
    }

    std::vector<SyncedMemory> f_chunks;
    f_chunks.reserve(chunk_num);

    for(int i=0;i<chunk_num;i++){
        SyncedMemory f(chunk_size);
        void* f_ = f.mutable_cpu_data();
        f_chunks.push_back(f);
    }

    std::vector<SyncedMemory> t_chunks;
    t_chunks.reserve(chunk_num);

    for(int i=0;i<chunk_num;i++){
        SyncedMemory t(chunk_size);
        void* t_ = t.mutable_cpu_data();
        t_chunks.push_back(t);
    }

    SyncedMemory w_l(n*fr::Limbs*sizeof(uint64_t));
    SyncedMemory w_r(n*fr::Limbs*sizeof(uint64_t));
    SyncedMemory w_o(n*fr::Limbs*sizeof(uint64_t));
    SyncedMemory w_4(n*fr::Limbs*sizeof(uint64_t));
    SyncedMemory l1(n*fr::Limbs*sizeof(uint64_t));

    void* w_l_ = w_l.mutable_cpu_data();
    void* w_r_ = w_r.mutable_cpu_data();
    void* w_o_ = w_o.mutable_cpu_data();
    void* w_4_ = w_4.mutable_cpu_data(); 
    void* l1_ = l1.mutable_cpu_data(); 

    const char* w_l_f = "/home/zhiyuan/w_l-15.bin";
    const char* w_r_f = "/home/zhiyuan/w_r-15.bin";
    const char* w_o_f = "/home/zhiyuan/w_o-15.bin";
    const char* w_4_f = "/home/zhiyuan/w_4-15.bin";

    read_file(w_l_f, w_l_);
    read_file(w_r_f, w_r_);
    read_file(w_o_f, w_o_);
    read_file(w_4_f, w_4_);

    std::vector<AffinePointG1> h1_commits;
    std::vector<AffinePointG1> h2_commits;
    h1_commits.reserve(chunk_num);
    h2_commits.reserve(chunk_num);

    void* buffer0_ = global_buffer[0].mutable_gpu_data();
    void* buffer1_ = global_buffer[1].mutable_gpu_data();
    void* buffer2_ = global_buffer[2].mutable_gpu_data();
    void* buffer3_ = global_buffer[3].mutable_gpu_data();
    void* buffer4_ = global_buffer[4].mutable_gpu_data();
    void* buffer5_ = global_buffer[5].mutable_gpu_data();
    void* buffer6_ = global_buffer[6].mutable_gpu_data();
    void* buffer7_ = global_buffer[7].mutable_gpu_data();

    for(int i=0;i<chunk_num;i++){
        void* ck_ = ck[i].mutable_gpu_data();
    }

    void* pp_ = pp.mutable_gpu_data();
    // -384 MB
    Intt INTT(fr::TWO_ADICITY);

    msmstruct chunk_msm = msm_init(n/chunk_num);
    msmstruct full_msm = msm_init(n);
    SyncedMemory chunk_msm_workspace(chunk_msm.msm_worksize);
    SyncedMemory chunk_msm_out(chunk_msm.msm_outsize);
    SyncedMemory msm_workspace(full_msm.msm_worksize);
    SyncedMemory full_out(full_msm.msm_outsize);

    std::vector<SyncedMemory> wl_commits_step1;
    for(int i=0;i<chunk_num;i++){
        SyncedMemory w(chunk_msm.msm_outsize);
        void* w_ = w.mutable_cpu_data();
        wl_commits_step1.push_back(w);
    }

    std::vector<SyncedMemory> wr_commits_step1;
    for(int i=0;i<chunk_num;i++){
        SyncedMemory w(chunk_msm.msm_outsize);
        void* w_ = w.mutable_cpu_data();
        wr_commits_step1.push_back(w);
    }

    std::vector<SyncedMemory> wo_commits_step1;
    for(int i=0;i<chunk_num;i++){
        SyncedMemory w(chunk_msm.msm_outsize);
        void* w_ = w.mutable_cpu_data();
        wo_commits_step1.push_back(w);
    }

    std::vector<SyncedMemory> w4_commits_step1;
    for(int i=0;i<chunk_num;i++){
        SyncedMemory w(chunk_msm.msm_outsize);
        void* w_ = w.mutable_cpu_data();
        w4_commits_step1.push_back(w);
    }

    std::vector<SyncedMemory> quotient_commits_step1;
    for(int i=0;i<8*chunk_num;i++){
        SyncedMemory q(chunk_msm.msm_outsize);
        void* q_ = q.mutable_cpu_data();
        quotient_commits_step1.push_back(q);
    }

    void* chunk_workspace_ = chunk_msm_workspace.mutable_gpu_data();
    void* out_ = chunk_msm_out.mutable_gpu_data();
    

    std::vector<SyncedMemory> h_1_chunks;
    std::vector<SyncedMemory> h_2_chunks;
    std::vector<SyncedMemory> h_1_poly_chunks;
    std::vector<SyncedMemory> h_2_poly_chunks;

    std::vector<SyncedMemory> h_1_poly_cpu;
    std::vector<SyncedMemory> h_2_poly_cpu;

    for(int i = 0; i<chunk_num;i++){
        SyncedMemory h_1_part(chunk_size);
        SyncedMemory h_2_part(chunk_size);
        void* h_1_ = h_1_part.mutable_cpu_data();
        void* h_2_ = h_2_part.mutable_cpu_data();
        caffe_memset(chunk_size, 0, h_1_);
        caffe_memset(chunk_size, 0, h_2_);
        h_1_chunks.push_back(h_1_part);
        h_2_chunks.push_back(h_2_part);
    }

    for(int i = 0; i<chunk_num;i++){
        SyncedMemory h_1_part(chunk_size);
        SyncedMemory h_2_part(chunk_size);
        void* h_1_ = h_1_part.mutable_cpu_data();
        void* h_2_ = h_2_part.mutable_cpu_data();
        h_1_poly_chunks.push_back(h_1_part);
        h_2_poly_chunks.push_back(h_2_part);
    }

    for(int i = 0; i<chunk_num;i++){
        SyncedMemory h_1_part(chunk_size);
        SyncedMemory h_2_part(chunk_size);
        void* h_1_ = h_1_part.mutable_cpu_data();
        void* h_2_ = h_2_part.mutable_cpu_data();
        h_1_poly_cpu.push_back(h_1_part);
        h_2_poly_cpu.push_back(h_2_part);
    }

    std::vector<SyncedMemory> z_chunks;
    for(int i = 0; i<chunk_num;i++){
        SyncedMemory z(chunk_size);
        void* z_ = z.mutable_cpu_data();
        z_chunks.push_back(z);
    }

    std::vector<SyncedMemory> z2_chunks;
    for(int i = 0; i<chunk_num;i++){
        SyncedMemory z(chunk_size);
        void* z_ = z.mutable_cpu_data();
        z2_chunks.push_back(z);
    }

    std::vector<SyncedMemory> table_chunks;
    for(int i = 0; i<chunk_num;i++){
        SyncedMemory table(chunk_size);
        void* table_ = table.mutable_cpu_data();
        table_chunks.push_back(table);
        caffe_memset(chunk_size, 0, table_);
    }

    SyncedMemory pi_poly(chunk_size * chunk_num);
    void* pi_ = pi_poly.mutable_cpu_data();

    SyncedMemory pi_cpu(chunk_size * chunk_num);
    void* pi_cpu_ = pi_cpu.mutable_cpu_data();


    std::vector<SyncedMemory> f_commits_step1;
    for(int i=0;i<chunk_num;i++){
        SyncedMemory f(chunk_msm.msm_outsize);
        void* f_ = f.mutable_cpu_data();
        f_commits_step1.push_back(f);
    }

    std::vector<SyncedMemory> h1_commits_step1;
    for(int i=0;i<chunk_num;i++){
        SyncedMemory h(chunk_msm.msm_outsize);
        void* h_ = h.mutable_cpu_data();
        h1_commits_step1.push_back(h);
    }

    std::vector<SyncedMemory> h2_commits_step1;
    for(int i=0;i<chunk_num;i++){
        SyncedMemory h(chunk_msm.msm_outsize);
        void* h_ = h.mutable_cpu_data();
        h2_commits_step1.push_back(h);
    }

    std::vector<SyncedMemory> z_commits_step1;
    for(int i=0;i<chunk_num;i++){
        SyncedMemory z(chunk_msm.msm_outsize);
        void* z_ = z.mutable_cpu_data();
        z_commits_step1.push_back(z);
    }

    std::vector<SyncedMemory> z2_commits_step1;
    for(int i=0;i<chunk_num;i++){
        SyncedMemory z(chunk_msm.msm_outsize);
        void* z_ = z.mutable_cpu_data();
        z2_commits_step1.push_back(z);
    }

    std::vector<AffinePointG1> z2_commits;

    SyncedMemory w_l_8n((lde_size + 8) *fr::Limbs*sizeof(uint64_t));
    SyncedMemory w_r_8n((lde_size + 8) *fr::Limbs*sizeof(uint64_t));
    SyncedMemory w_o_8n(lde_size *fr::Limbs*sizeof(uint64_t));
    SyncedMemory w_4_8n((lde_size + 8) *fr::Limbs*sizeof(uint64_t));
    SyncedMemory pi_8n(lde_size *fr::Limbs*sizeof(uint64_t));
    SyncedMemory z_eval_8n((lde_size + 8) *fr::Limbs*sizeof(uint64_t));
    SyncedMemory z2_eval_8n((lde_size + 8) *fr::Limbs*sizeof(uint64_t));
    SyncedMemory table_eval_8n((lde_size + 8) *fr::Limbs*sizeof(uint64_t));
    SyncedMemory h1_eval_8n((lde_size + 8) *fr::Limbs*sizeof(uint64_t));
    SyncedMemory h2_eval_8n(lde_size *fr::Limbs*sizeof(uint64_t));
    SyncedMemory l1_eval_8n(lde_size *fr::Limbs*sizeof(uint64_t));
    SyncedMemory f_eval_8n(lde_size *fr::Limbs*sizeof(uint64_t));

    void* w_l_8n_ = w_l_8n.mutable_cpu_data();
    void* w_r_8n_ = w_r_8n.mutable_cpu_data();
    void* w_o_8n_ = w_o_8n.mutable_cpu_data();
    void* w_4_8n_ = w_4_8n.mutable_cpu_data();
    void* pi_8n_ = pi_8n.mutable_cpu_data();
    void* z_8n_ = z_eval_8n.mutable_cpu_data();
    void* z2_8n_ = z2_eval_8n.mutable_cpu_data();
    void* table_8n_ = table_eval_8n.mutable_cpu_data();
    void* h1_8n_ = h1_eval_8n.mutable_cpu_data();
    void* h2_8n_ = h2_eval_8n.mutable_cpu_data();
    void* l1_8n_ = l1_eval_8n.mutable_cpu_data();
    void* f_8n_ = f_eval_8n.mutable_cpu_data();

    // warm up
    cuda::msm_zkp_cuda_(ck[0], global_buffer[0], chunk_msm_workspace, chunk_msm_out, smcount);

    void* workspace_ = msm_workspace.mutable_gpu_data();
    // - 461.26 MB
    void* full_out_ = full_out.mutable_gpu_data();

    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

    // 1. Compute witness Polynomials
    std::vector<SyncedMemory> w_l_chunks = cs.w_l;
    std::vector<SyncedMemory> w_r_chunks = cs.w_r;
    std::vector<SyncedMemory> w_o_chunks = cs.w_o;
    std::vector<SyncedMemory> w_4_chunks = cs.w_4;
    
    w_l_ = w_l.mutable_gpu_data_async(stream1);
    
    INTT._forward_(w_l, stream1);

    // 2. Derive lookup polynomials

    // Generate table compression factor
    SyncedMemory zeta = transcript.challenge_scalar("zeta");
    transcript.append("zeta", zeta);
    void* zeta_gpu = zeta.mutable_gpu_data();
    // Compress lookup table into vector of single elements
    {
        void* table1_ = pk.lookup_coeffs.table1[0].mutable_cpu_data();
        void* table2_ = pk.lookup_coeffs.table2[0].mutable_cpu_data();
        void* table3_ = pk.lookup_coeffs.table3[0].mutable_cpu_data();
        void* table4_ = pk.lookup_coeffs.table4[0].mutable_cpu_data();

        caffe_gpu_memcpy_async(chunk_size, table1_, buffer0_, stream2);
        caffe_gpu_memcpy_async(chunk_size, table2_, buffer1_, stream2);
        caffe_gpu_memcpy_async(chunk_size, table3_, buffer2_, stream2);
        caffe_gpu_memcpy_async(chunk_size, table4_, buffer3_, stream2);
        void* t_1 = t_chunks[0].mutable_gpu_data_async(stream2);
        
        void* table5_ = pk.lookup_coeffs.table1[1].mutable_cpu_data();
        void* table6_ = pk.lookup_coeffs.table2[1].mutable_cpu_data();
        void* table7_ = pk.lookup_coeffs.table3[1].mutable_cpu_data();
        void* table8_ = pk.lookup_coeffs.table4[1].mutable_cpu_data();

        caffe_gpu_memcpy_async(chunk_size, table5_, buffer4_, stream2);
        caffe_gpu_memcpy_async(chunk_size, table6_, buffer5_, stream2);
        caffe_gpu_memcpy_async(chunk_size, table7_, buffer6_, stream2);
        caffe_gpu_memcpy_async(chunk_size, table8_, buffer7_, stream2);
        void* t_2 = t_chunks[1].mutable_gpu_data_async(stream2);

        compress(t_chunks[0], global_buffer[0], global_buffer[1], global_buffer[2], global_buffer[3], zeta, stream1);
        compress(t_chunks[1], global_buffer[4], global_buffer[5], global_buffer[6], global_buffer[7], zeta, stream1);
    }
    
    // Compute query table f
    compute_query_table(
        f_chunks,
        global_buffer,
        cs.cs_q_lookup,
        w_l_chunks,
        w_r_chunks,
        w_o_chunks,
        w_4_chunks,
        t_chunks,
        zeta,
        chunk_num,
        stream2,
        stream3);
    
    // Commit to query polynomial
    cudaEvent_t event;
    cudaEventCreate(&event);
    void* t_cpu_0_ = t_chunks[0].mutable_cpu_data_async(stream1);
    cuda::msm_zkp_cuda_(ck[0], f_chunks[0], chunk_msm_workspace, chunk_msm_out, smcount, stream2);

    void* f_0_ = chunk_msm_out.mutable_gpu_data_async(stream2);
    void* f_0_cpu_ = f_commits_step1[0].mutable_cpu_data_async(stream2);
    caffe_gpu_memcpy_async(chunk_msm_out.size(), f_0_, f_0_cpu_, stream2);

    cudaEventRecord(event, stream2);
    cuda::msm_zkp_cuda_(ck[1], f_chunks[1], chunk_msm_workspace, chunk_msm_out, smcount, stream2);
    cudaStreamWaitEvent(stream1, event);
    void* t_cpu_1_ = t_chunks[1].mutable_cpu_data_async(stream1);
    void* f_1_ = chunk_msm_out.mutable_gpu_data_async(stream2);
    void* f_1_cpu_ = f_commits_step1[1].mutable_cpu_data_async(stream2);
    cudaMemcpyAsync(f_1_cpu_, f_1_, chunk_msm_out.size(), cudaMemcpyDeviceToHost, stream2);
    
    cudaDeviceSynchronize();

    for(int i = 0;i<chunk_num;i++){
        SyncedMemory step2_res = cpu::msm_collect_cpu(f_commits_step1[i], chunk_size/(fr::Limbs*sizeof(uint64_t)));
        ProjectivePointG1 commit_pro = to_point(step2_res);
        AffinePointG1 f_commit_affine = to_affine(commit_pro);
        transcript.append_chunk("f", f_commit_affine, i);
    }

    // Compute s, as the sorted and concatenated version of f and t
    // we skipped the combine_split because both f and t are all zeros in this scenario.
    
    for(int i = 0; i<chunk_num;i++){
        cudaEvent_t event;
        cudaEventCreate(&event);
        void* h_1_gpu = h_1_chunks[i].mutable_gpu_data_async(stream1);
        void* h_2_gpu = h_2_chunks[i].mutable_gpu_data_async(stream1);
        void* h_1_ = h_1_poly_chunks[i].mutable_gpu_data_async(stream1);
        void* h_2_ = h_2_poly_chunks[i].mutable_gpu_data_async(stream1);
        cudaEventRecord(event, stream1);
        INTT.forward_(h_1_chunks[i], h_1_poly_chunks[i], stream1);
        INTT.forward_(h_2_chunks[i], h_2_poly_chunks[i], stream1);
        cudaStreamWaitEvent(stream2, event);
        void* f_ = f_chunks[i].mutable_cpu_data_async(stream2);
    }

    cudaDeviceSynchronize();
    // Commit to h polys

    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);

    cuda::msm_zkp_cuda_(ck[0], h_1_poly_chunks[0], chunk_msm_workspace, chunk_msm_out, smcount, stream2);
    void* h1_0_ = chunk_msm_out.mutable_gpu_data_async(stream2);
    cudaEventRecord(event1, stream2);
    void* h1_0_cpu_ = h1_commits_step1[0].mutable_cpu_data_async(stream1);
    void* h_1_0 = h_1_chunks[0].mutable_cpu_data_async(stream1);
    void* h_1_1 = h_1_chunks[1].mutable_cpu_data_async(stream1);
    cudaStreamWaitEvent(stream1, event1);
    cudaMemcpyAsync(h1_0_cpu_, h1_0_, chunk_msm_out.size(), cudaMemcpyDeviceToHost, stream1);

    cuda::msm_zkp_cuda_(ck[1], h_1_poly_chunks[1], chunk_msm_workspace, chunk_msm_out, smcount, stream2);
    void* h1_1_ = chunk_msm_out.mutable_gpu_data_async(stream2);
    cudaEventRecord(event2, stream2);
    void* h1_1_cpu_ = h1_commits_step1[1].mutable_cpu_data_async(stream1);
    void* h_2_0 = h_2_chunks[0].mutable_cpu_data_async(stream1);
    cudaStreamWaitEvent(stream1, event2);
    cudaMemcpyAsync(h1_1_cpu_, h1_1_, chunk_msm_out.size(), cudaMemcpyDeviceToHost, stream1);

    cudaStreamSynchronize(stream2);
    // Add h polynomials to transcript
    for(int i = 0;i<chunk_num;i++){
        SyncedMemory step2_res = cpu::msm_collect_cpu(h1_commits_step1[i], chunk_size/(fr::Limbs*sizeof(uint64_t)));
        ProjectivePointG1 commit_pro = to_point(step2_res);
        AffinePointG1 h1_commit_affine = to_affine(commit_pro);
        transcript.append_chunk("h1", h1_commit_affine, i);
    }

    cuda::msm_zkp_cuda_(ck[0], h_2_poly_chunks[0], chunk_msm_workspace, chunk_msm_out, smcount, stream2);
    void* h2_0_ = chunk_msm_out.mutable_gpu_data_async(stream2);
    cudaEventRecord(event1, stream2);
    void* h2_0_cpu_ = h2_commits_step1[0].mutable_cpu_data_async(stream1);
    for(int i = 0;i<chunk_num;i++){
        void* h_1_poly_cpu_ = h_1_poly_cpu[i].mutable_cpu_data();
        void* h_1_poly_gpu_ = h_1_poly_chunks[i].mutable_gpu_data();
        caffe_gpu_memcpy_async(chunk_size, h_1_poly_gpu_, h_1_poly_cpu_, stream1);
    }
    cudaStreamWaitEvent(stream1, event1);
    cudaMemcpyAsync(h2_0_cpu_, h2_0_, chunk_msm_out.size(), cudaMemcpyDeviceToHost, stream1);

    cuda::msm_zkp_cuda_(ck[1], h_2_poly_chunks[1], chunk_msm_workspace, chunk_msm_out, smcount, stream2);
    void* h2_1_ = chunk_msm_out.mutable_gpu_data_async(stream2);
    cudaEventRecord(event1, stream2);
    void* h2_1_cpu_ = h2_commits_step1[1].mutable_cpu_data_async(stream1);
    void* h_2_1 = h_2_chunks[1].mutable_cpu_data_async(stream1);
    cudaStreamWaitEvent(stream1, event1);
    cudaMemcpyAsync(h2_1_cpu_, h2_1_, chunk_msm_out.size(), cudaMemcpyDeviceToHost, stream1);
    
    cudaStreamSynchronize(stream2);
    for(int i = 0;i<chunk_num;i++){
        SyncedMemory step2_res = cpu::msm_collect_cpu(h2_commits_step1[i], chunk_size/(fr::Limbs*sizeof(uint64_t)));
        ProjectivePointG1 commit_pro = to_point(step2_res);
        AffinePointG1 h2_commit_affine = to_affine(commit_pro);
        h2_commits.push_back(h2_commit_affine);
        transcript.append_chunk("h2", h2_commit_affine, i);
    }
    // 3. Compute permutation polynomial
    // Compute permutation challenge `beta`.

    SyncedMemory beta = transcript.challenge_scalar("beta");
    transcript.append("beta", beta);
    void* beta_gpu = beta.mutable_gpu_data();

    // Compute permutation challenge `gamma`.
    SyncedMemory gamma = transcript.challenge_scalar("gamma");
    transcript.append("gamma", gamma);
    void* gamma_gpu = gamma.mutable_gpu_data();

    // Compute permutation challenge `delta`.
    SyncedMemory delta = transcript.challenge_scalar("delta");
    transcript.append("delta", delta);
    void* delta_gpu = delta.mutable_gpu_data();

    // Compute permutation challenge `epsilon`.
    SyncedMemory epsilon = transcript.challenge_scalar("epsilon");
    transcript.append("epsilon", epsilon);
    void* epsilon_gpu = epsilon.mutable_gpu_data();

    // Challenges must be different
    assert(!fr::is_equal(beta, gamma) && "challenges must be different");
    assert(!fr::is_equal(beta, delta) && "challenges must be different");
    assert(!fr::is_equal(beta, epsilon) && "challenges must be different");
    assert(!fr::is_equal(gamma, delta) && "challenges must be different");
    assert(!fr::is_equal(gamma, epsilon) && "challenges must be different");
    assert(!fr::is_equal(delta, epsilon) && "challenges must be different");
        
    // 8 buffers, ck, INTT Params, w_l_poly, h2_poly on GPU ~1.6G
    std::vector<SyncedMemory> z_gpu = compute_permutation_poly(domain, global_buffer, w_l_chunks, w_r_chunks, w_o_chunks, 
                                                   w_4_chunks, beta, gamma, pk.permutation_coeffs, chunk_num, ntt_step1_map, ntt_step2_map, ntt_step3_map,
                                                   ntt_cpu_map, ntt_cpu_buffer, h_2_poly_cpu, h_2_poly_chunks, w_l, INTT,
                                                   stream1, stream2);
    cudaDeviceSynchronize();
    // 8 buffers, ck, INTT Params, z_poly on GPU ~1.5G
    // Commit to permutation polynomial.
    cuda::msm_zkp_cuda_(ck[0], z_gpu[0], chunk_msm_workspace, chunk_msm_out, smcount, stream2);
    h_1_0 = h_1_chunks[0].mutable_gpu_data_async(stream1);
    h_1_1 = h_1_chunks[1].mutable_gpu_data_async(stream1);
    cudaEventRecord(event, stream2);
    cudaStreamWaitEvent(stream1, event);
    void* z_ = chunk_msm_out.mutable_gpu_data_async(stream1);
    void* z_cpu_ = z_commits_step1[0].mutable_cpu_data_async(stream1);
    caffe_gpu_memcpy_async(chunk_msm_out.size(), z_, z_cpu_, stream1);
    cuda::msm_zkp_cuda_(ck[1], z_gpu[1], chunk_msm_workspace, chunk_msm_out, smcount, stream2);
    h_2_0 = h_2_chunks[0].mutable_gpu_data_async(stream1);
    h_2_1 = h_2_chunks[1].mutable_gpu_data_async(stream1);
    cudaEventRecord(event, stream2);
    cudaStreamSynchronize(stream1);
    SyncedMemory z0_step2_res = cpu::msm_collect_cpu(z_commits_step1[0], chunk_size/(fr::Limbs*sizeof(uint64_t)));
    ProjectivePointG1 commit_pro_z0 = to_point(z0_step2_res);
    AffinePointG1 z0_commit_affine = to_affine(commit_pro_z0);
    transcript.append_chunk("z", z0_commit_affine, 0);

    cudaStreamWaitEvent(stream1, event);
    z_ = chunk_msm_out.mutable_gpu_data_async(stream1);
    z_cpu_ = z_commits_step1[1].mutable_cpu_data_async(stream1);
    caffe_gpu_memcpy_async(chunk_msm_out.size(), z_, z_cpu_, stream1);
    cudaStreamSynchronize(stream1);
    SyncedMemory z1_step2_res = cpu::msm_collect_cpu(z_commits_step1[1], chunk_size/(fr::Limbs*sizeof(uint64_t)));
    ProjectivePointG1 commit_pro_z1 = to_point(z1_step2_res);
    AffinePointG1 z1_commit_affine = to_affine(commit_pro_z1);
    transcript.append_chunk("z", z1_commit_affine, 1);
    
    // 8 buffers, ck, INTT Params, z_poly, h1, h2 on GPU ~1.8G
    // Compute mega permutation polynomial.
    // Compute lookup permutation poly
    std::vector<SyncedMemory> z2_gpu = compute_lookup_permutation_poly(n, global_buffer, f_chunks, t_chunks, 
                                            h_1_chunks, h_2_chunks, delta, epsilon, 
                                            ntt_step1_map, ntt_step2_map, ntt_step3_map, ntt_cpu_map, ntt_cpu_buffer, z_chunks, z_gpu,
                                            chunk_num, INTT, stream1, stream2);

    // 8 buffers, ck, INTT Params, z2, h1, h2, t, f on GPU  ~2G
    // Commit to lookup permutation polynomial.
    cuda::msm_zkp_cuda_(ck[0], z2_gpu[0], chunk_msm_workspace, chunk_msm_out, smcount, stream2);
    for(int i=0;i<chunk_num;i++){
        void* t_ = t_chunks[i].mutable_cpu_data_async(stream1);
        void* f_ = f_chunks[i].mutable_cpu_data_async(stream1);
        void* h1_ = h_1_chunks[i].mutable_cpu_data_async(stream1);
        void* h2_ = h_2_chunks[i].mutable_cpu_data_async(stream1);
    }
    // this time need to be eliminated
    for(int i=0; i<12; i++){
        void* buffer_ = quotient_buffer[i].mutable_gpu_data_async(stream1);
    }

    void* z2_ = chunk_msm_out.mutable_gpu_data_async(stream1);
    void* z2_cpu_ = z2_commits_step1[0].mutable_cpu_data_async(stream1);
    caffe_gpu_memcpy_async(chunk_msm_out.size(), z2_, z2_cpu_, stream1);
    cudaStreamSynchronize(stream1);
    cuda::msm_zkp_cuda_(ck[1], z2_gpu[1], chunk_msm_workspace, chunk_msm_out, smcount, stream2);
    pi_ = pi_poly.mutable_gpu_data_async(stream1);
    w_r_ = w_r.mutable_gpu_data_async(stream1);
    w_o_ = w_o.mutable_gpu_data_async(stream1);
    w_4_ = w_4.mutable_gpu_data_async(stream1);
    caffe_gpu_memcpy_async(chunk_size, z2_gpu[0].mutable_gpu_data(), z2_chunks[0].mutable_cpu_data(), stream1);
    SyncedMemory z2_0_step2_res = cpu::msm_collect_cpu(z2_commits_step1[0], chunk_size/(fr::Limbs*sizeof(uint64_t)));
    ProjectivePointG1 commit_pro_z2_0 = to_point(z2_0_step2_res);
    z2_commits.push_back(to_affine(commit_pro_z2_0));

    cudaStreamSynchronize(stream2);
    z2_ = chunk_msm_out.mutable_gpu_data_async(stream1);
    z2_cpu_ = z2_commits_step1[1].mutable_cpu_data_async(stream1);
    caffe_gpu_memcpy_async(chunk_msm_out.size(), z2_, z2_cpu_, stream1);
    cudaStreamSynchronize(stream1);
    SyncedMemory z2_1_step2_res = cpu::msm_collect_cpu(z2_commits_step1[1], chunk_size/(fr::Limbs*sizeof(uint64_t)));
    ProjectivePointG1 commit_pro_z2_1 = to_point(z2_1_step2_res);
    z2_commits.push_back(to_affine(commit_pro_z2_1));
    caffe_gpu_memcpy_async(chunk_size, z2_gpu[1].mutable_gpu_data(), z2_chunks[1].mutable_cpu_data(), stream1);
    INTT._forward_(w_r, stream2);
    global_buffer.clear();
    // - 8 buffer

    // ck, INTT params, w_r, w_o, w_4 on GPU
    // 3. Compute public inputs polynomial
    
    into_dense_poly_(cs.public_inputs, pi_poly, w_r, cs.intended_pi_pos, n, INTT, stream1, stream2);
    INTT._forward_(w_o, stream2);
    INTT._forward_(w_4, stream2);
    w_o_ = w_o.mutable_cpu_data_async(stream1);
    w_4_ = w_4.mutable_cpu_data_async(stream1);
    pi_ = pi_poly.mutable_gpu_data();
    pi_cpu_ = pi_cpu.mutable_cpu_data();
    caffe_gpu_memcpy_async(pi_cpu.size(), pi_, pi_cpu_, stream1);

    for(int i=12; i<22; i++){
        void* buffer_ = quotient_buffer[i].mutable_gpu_data_async(stream1);
    }

    pi_poly = SyncedMemory();
    INTT.Params = SyncedMemory();
    
    // 4. Compute quotient polynomial
    // Compute quotient challenge `alpha`, and gate-specific separation challenges.
    SyncedMemory alpha = transcript.challenge_scalar("alpha");
    transcript.append("alpha", alpha);

    SyncedMemory range_sep_challenge = transcript.challenge_scalar("range separation challenge");
    transcript.append("range seperation challenge", range_sep_challenge);

    SyncedMemory logic_sep_challenge = transcript.challenge_scalar("logic separation challenge");
    transcript.append("logic seperation challenge", logic_sep_challenge);

    SyncedMemory fixed_base_sep_challenge = transcript.challenge_scalar("fixed base separation challenge");
    transcript.append("fixed base separation challenge", fixed_base_sep_challenge);

    SyncedMemory var_base_sep_challenge = transcript.challenge_scalar("variable base separation challenge");
    transcript.append("variable base separation challenge", var_base_sep_challenge);

    SyncedMemory lookup_sep_challenge = transcript.challenge_scalar("lookup separation challenge");
    transcript.append("lookup separation challenge", lookup_sep_challenge);
    
    void* alpha_ = alpha.mutable_gpu_data_async(stream3);   
    void* range_ = range_sep_challenge.mutable_gpu_data_async(stream3);
    void* logic_ = logic_sep_challenge.mutable_gpu_data_async(stream3);
    void* fix_ = fixed_base_sep_challenge.mutable_gpu_data_async(stream3);
    void* var_ = var_base_sep_challenge.mutable_gpu_data_async(stream3);
    void* lookup_ = lookup_sep_challenge.mutable_gpu_data_async(stream3);
 
    // 8 quotient buffer, ck, on GPU ~ 1.4 Gb
    std::vector<SyncedMemory> quotient_chunks = compute_quotient_poly(
                                                n,
                                                pk,
                                                ck,
                                                chunk_msm_workspace, chunk_msm_out,
                                                wl_commits_step1,
                                                wr_commits_step1,
                                                wo_commits_step1,
                                                w4_commits_step1,
                                                quotient_buffer,
                                                w_l, 
                                                w_r, 
                                                w_o, 
                                                w_4,
                                                pi_cpu,
                                                w_l_8n,
                                                w_r_8n,
                                                w_o_8n,
                                                w_4_8n,
                                                pi_8n,
                                                z_eval_8n,
                                                z2_eval_8n,
                                                table_eval_8n,
                                                h1_eval_8n,
                                                h2_eval_8n,
                                                l1_eval_8n,
                                                f_eval_8n,
                                                z_chunks,
                                                z2_chunks,
                                                f_chunks,
                                                table_chunks,
                                                h_1_chunks,
                                                h_2_chunks,
                                                alpha,
                                                beta,
                                                gamma,
                                                delta,
                                                epsilon,
                                                zeta,
                                                range_sep_challenge,
                                                logic_sep_challenge,
                                                fixed_base_sep_challenge,
                                                var_base_sep_challenge,
                                                lookup_sep_challenge,
                                                lde_step1_map,
                                                lde_step2_map,
                                                lde_step3_map,
                                                lde_cpu_map,
                                                lde_input,
                                                lde_cpu_buffer,
                                                lg_LDE,
                                                chunk_num,
                                                transcript,
                                                stream1,
                                                stream2);
    for(int i = 0;i<quotient_chunks.size();i++){
        cuda::msm_zkp_cuda_(ck[i], quotient_chunks[i], chunk_msm_workspace, chunk_msm_out, smcount, stream2);
        void* q_ = chunk_msm_out.mutable_gpu_data();
        void* q_cpu_ = quotient_commits_step1[i].mutable_cpu_data();
        cudaStreamSynchronize(stream2);
        caffe_gpu_memcpy_async(chunk_msm_out.size(), q_, q_cpu_, stream1);

        SyncedMemory step2_res = cpu::msm_collect_cpu(quotient_commits_step1[i], chunk_size/(fr::Limbs*sizeof(uint64_t)));
        ProjectivePointG1 commit_pro = to_point(step2_res);
        AffinePointG1 q_commit_affine = to_affine(commit_pro);
        transcript.append_chunk(("t_" + std::to_string(i/2 + 1)).c_str(), q_commit_affine, i%2);
    }
}

bool readProverKey(const std::string& filename, ProverKeyC& proverKey) {
    // 打开二进制文件
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "无法打开文件!" << std::endl;
        return false;
    }

    // 读取数据到结构体中的每个字段
    size_t dataSize = size_t(1) << 24;
    size_t cosetSize = size_t(1) << 27;
    // 分配内存
    proverKey.q_m_evals = new uint64_t[cosetSize];

    proverKey.q_l_coeffs = new uint64_t[dataSize];
    proverKey.q_l_evals = new uint64_t[cosetSize];

    proverKey.q_r_coeffs = new uint64_t[dataSize];
    proverKey.q_r_evals = new uint64_t[cosetSize];

    proverKey.q_o_coeffs = new uint64_t[dataSize];
    proverKey.q_o_evals = new uint64_t[cosetSize];

    proverKey.q_4_coeffs = new uint64_t[dataSize];
    proverKey.q_4_evals = new uint64_t[cosetSize];

    proverKey.q_c_coeffs = new uint64_t[dataSize];
    proverKey.q_c_evals = new uint64_t[cosetSize];

    proverKey.q_hl_coeffs = new uint64_t[dataSize];
    proverKey.q_hl_evals = new uint64_t[cosetSize];

    proverKey.q_hr_coeffs = new uint64_t[dataSize];
    proverKey.q_hr_evals = new uint64_t[cosetSize];

    proverKey.q_h4_coeffs = new uint64_t[dataSize];
    proverKey.q_h4_evals = new uint64_t[cosetSize];

    proverKey.q_arith_coeffs = new uint64_t[dataSize];
    proverKey.q_arith_evals = new uint64_t[cosetSize];

    proverKey.range_selector_evals = new uint64_t[cosetSize];

    proverKey.logic_selector_evals = new uint64_t[cosetSize];

    proverKey.fixed_group_add_selector_evals = new uint64_t[cosetSize];

    proverKey.variable_group_add_selector_evals = new uint64_t[cosetSize];

    proverKey.q_lookup_evals = new uint64_t[cosetSize];

    proverKey.table1 = new uint64_t[dataSize];
    proverKey.table2 = new uint64_t[dataSize];
    proverKey.table3 = new uint64_t[dataSize];
    proverKey.table4 = new uint64_t[dataSize];

    proverKey.left_sigma_coeffs = new uint64_t[dataSize];
    proverKey.left_sigma_evals = new uint64_t[cosetSize];

    proverKey.right_sigma_coeffs = new uint64_t[dataSize];
    proverKey.right_sigma_evals = new uint64_t[cosetSize];

    proverKey.out_sigma_coeffs = new uint64_t[dataSize];
    proverKey.out_sigma_evals = new uint64_t[cosetSize];

    proverKey.fourth_sigma_coeffs = new uint64_t[dataSize];
    proverKey.fourth_sigma_evals = new uint64_t[cosetSize];

    proverKey.linear_evaluations = new uint64_t[cosetSize];
    proverKey.v_h_coset_8n = new uint64_t[cosetSize];

    // 读取文件数据
    file.read(reinterpret_cast<char*>(proverKey.q_m_evals), sizeof(uint64_t) * cosetSize);

    file.read(reinterpret_cast<char*>(proverKey.q_l_coeffs), sizeof(uint64_t) * dataSize);
    file.read(reinterpret_cast<char*>(proverKey.q_l_evals), sizeof(uint64_t) * cosetSize);

    file.read(reinterpret_cast<char*>(proverKey.q_r_coeffs), sizeof(uint64_t) * dataSize);
    file.read(reinterpret_cast<char*>(proverKey.q_r_evals), sizeof(uint64_t) * cosetSize);

    file.read(reinterpret_cast<char*>(proverKey.q_o_coeffs), sizeof(uint64_t) * dataSize);
    file.read(reinterpret_cast<char*>(proverKey.q_o_evals), sizeof(uint64_t) * cosetSize);

    file.read(reinterpret_cast<char*>(proverKey.q_4_coeffs), sizeof(uint64_t) * dataSize);
    file.read(reinterpret_cast<char*>(proverKey.q_4_evals), sizeof(uint64_t) * cosetSize);

    file.read(reinterpret_cast<char*>(proverKey.q_c_coeffs), sizeof(uint64_t) * dataSize);
    file.read(reinterpret_cast<char*>(proverKey.q_c_evals), sizeof(uint64_t) * cosetSize);

    file.read(reinterpret_cast<char*>(proverKey.q_hl_coeffs), sizeof(uint64_t) * dataSize);
    file.read(reinterpret_cast<char*>(proverKey.q_hl_evals), sizeof(uint64_t) * cosetSize);

    file.read(reinterpret_cast<char*>(proverKey.q_hr_coeffs), sizeof(uint64_t) * dataSize);
    file.read(reinterpret_cast<char*>(proverKey.q_hr_evals), sizeof(uint64_t) * cosetSize);

    file.read(reinterpret_cast<char*>(proverKey.q_h4_coeffs), sizeof(uint64_t) * dataSize);
    file.read(reinterpret_cast<char*>(proverKey.q_h4_evals), sizeof(uint64_t) * cosetSize);

    file.read(reinterpret_cast<char*>(proverKey.q_arith_coeffs), sizeof(uint64_t) * dataSize);
    file.read(reinterpret_cast<char*>(proverKey.q_arith_evals), sizeof(uint64_t) * cosetSize);

    file.read(reinterpret_cast<char*>(proverKey.range_selector_evals), sizeof(uint64_t) * cosetSize);

    file.read(reinterpret_cast<char*>(proverKey.logic_selector_evals), sizeof(uint64_t) * cosetSize);

    file.read(reinterpret_cast<char*>(proverKey.q_lookup_evals), sizeof(uint64_t) * cosetSize);

    file.read(reinterpret_cast<char*>(proverKey.table1), sizeof(uint64_t) * dataSize);
    file.read(reinterpret_cast<char*>(proverKey.table2), sizeof(uint64_t) * dataSize);
    file.read(reinterpret_cast<char*>(proverKey.table3), sizeof(uint64_t) * dataSize);
    file.read(reinterpret_cast<char*>(proverKey.table4), sizeof(uint64_t) * dataSize);

    file.read(reinterpret_cast<char*>(proverKey.fixed_group_add_selector_evals), sizeof(uint64_t) * cosetSize);

    file.read(reinterpret_cast<char*>(proverKey.variable_group_add_selector_evals), sizeof(uint64_t) * cosetSize);

    file.read(reinterpret_cast<char*>(proverKey.left_sigma_coeffs), sizeof(uint64_t) * dataSize);
    file.read(reinterpret_cast<char*>(proverKey.left_sigma_evals), sizeof(uint64_t) * cosetSize);

    file.read(reinterpret_cast<char*>(proverKey.right_sigma_coeffs), sizeof(uint64_t) * dataSize);
    file.read(reinterpret_cast<char*>(proverKey.right_sigma_evals), sizeof(uint64_t) * cosetSize);

    file.read(reinterpret_cast<char*>(proverKey.out_sigma_coeffs), sizeof(uint64_t) * dataSize);
    file.read(reinterpret_cast<char*>(proverKey.out_sigma_evals), sizeof(uint64_t) * cosetSize);

    file.read(reinterpret_cast<char*>(proverKey.fourth_sigma_coeffs), sizeof(uint64_t) * dataSize);
    file.read(reinterpret_cast<char*>(proverKey.fourth_sigma_evals), sizeof(uint64_t) * cosetSize);

    file.read(reinterpret_cast<char*>(proverKey.linear_evaluations), sizeof(uint64_t) * cosetSize);

    file.read(reinterpret_cast<char*>(proverKey.v_h_coset_8n), sizeof(uint64_t) * cosetSize);

    file.close();
    return true;
}


int main(){
    ProverKeyC pkc;

    const char* pp_f = "/home/zhiyuan/params-15.bin";
    uint64_t N = 1<<22;
    int chunk_num = 2;
    SyncedMemory pp(N*fq::Limbs*sizeof(uint64_t)*2);
    void* pp_ = pp.mutable_cpu_data();

    read_file(pp_f, pp_);

    SyncedMemory q_lookup(N*fr::Limbs*sizeof(uint64_t));
    SyncedMemory w_l(N*fr::Limbs*sizeof(uint64_t));
    SyncedMemory w_r(N*fr::Limbs*sizeof(uint64_t));
    SyncedMemory w_o(N*fr::Limbs*sizeof(uint64_t));
    SyncedMemory w_4(N*fr::Limbs*sizeof(uint64_t));

    void* q_lookup_ = q_lookup.mutable_cpu_data();
    void* w_l_ = w_l.mutable_cpu_data();
    void* w_r_ = w_r.mutable_cpu_data();
    void* w_o_ = w_o.mutable_cpu_data();
    void* w_4_ = w_4.mutable_cpu_data(); 

    const char* w_l_f = "/home/zhiyuan/w_l_poly-15.bin";
    const char* w_r_f = "/home/zhiyuan/w_r_poly-15.bin";
    const char* w_o_f = "/home/zhiyuan/w_o_poly-15.bin";
    const char* w_4_f = "/home/zhiyuan/w_4_poly-15.bin";

    read_file(w_l_f, w_l_);
    read_file(w_r_f, w_r_);
    read_file(w_o_f, w_o_);
    read_file(w_4_f, w_4_);
    caffe_memset(q_lookup.size(),0,q_lookup_);
    std::vector<SyncedMemory> ck = chunk(static_cast<uint64_t*>(pp_), pp.size()/chunk_num, chunk_num);
    std::vector<SyncedMemory> w_l_chunks = chunk(static_cast<uint64_t*>(w_l_), w_l.size()/chunk_num, chunk_num);
    std::vector<SyncedMemory> w_r_chunks = chunk(static_cast<uint64_t*>(w_r_), w_r.size()/chunk_num, chunk_num);
    std::vector<SyncedMemory> w_o_chunks = chunk(static_cast<uint64_t*>(w_o_), w_o.size()/chunk_num, chunk_num);
    std::vector<SyncedMemory> w_4_chunks = chunk(static_cast<uint64_t*>(w_4_), w_4.size()/chunk_num, chunk_num);
    std::vector<SyncedMemory> q_lookup_chunks = chunk(static_cast<uint64_t*>(q_lookup_), q_lookup.size()/chunk_num, chunk_num);

    SyncedMemory pi(fr::Limbs*sizeof(uint64_t));
    uint64_t pi_64[4] = {10967591153831532710, 17363069844447244526, 15073226941702976164, 6960990843264712954};
    void* pi_cpu = pi.mutable_cpu_data();
    memcpy(pi_cpu, pi_64, pi.size());
    uint64_t intended_pi_pos = 3161923;

    Circuit cs(N, intended_pi_pos, q_lookup_chunks, pi, w_l_chunks, w_r_chunks, w_o_chunks, w_4_chunks);

    if (readProverKey("/home/zhiyuan/pk-15.bin", pkc)) {
       prove(pkc, cs, ck, pp);
    }
}