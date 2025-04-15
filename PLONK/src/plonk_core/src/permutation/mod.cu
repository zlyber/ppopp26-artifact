#include "mod.cuh"
#include "../../../arithmetic.cuh"

SyncedMemory numerator_irreducible(SyncedMemory root, SyncedMemory w, SyncedMemory k, SyncedMemory beta, SyncedMemory gamma, cudaStream_t stream = (cudaStream_t)0) {
    SyncedMemory mid1 = mul_mod(beta, k, stream); 
    SyncedMemory mid2 = mul_mod(mid1, root, stream); 
    SyncedMemory mid3 = add_mod(w, mid2, stream); 
    SyncedMemory numerator = add_mod(mid3, gamma, stream);
    return numerator; 
}

void numerator_irreducible_(SyncedMemory root, SyncedMemory w, SyncedMemory k, SyncedMemory beta, SyncedMemory gamma, SyncedMemory buffer, cudaStream_t stream = (cudaStream_t)0) {
    mul_mod_(buffer, beta, stream);
    mul_mod_(buffer, k, stream);
    mul_mod_(buffer, root, stream);
    mul_mod_(buffer, w, stream);
    add_mod_(buffer, gamma, stream);
}

SyncedMemory denominator_irreducible(SyncedMemory w, SyncedMemory sigma, SyncedMemory beta, SyncedMemory gamma, cudaStream_t stream = (cudaStream_t)0) {
    SyncedMemory mid1 = mul_mod_scalar(sigma, beta, stream); 
    SyncedMemory mid2 = add_mod(w, mid1, stream);
    SyncedMemory denominator = add_mod(mid2, gamma, stream); 
    return denominator; 
}

void denominator_irreducible_(SyncedMemory w, SyncedMemory sigma, SyncedMemory beta, SyncedMemory gamma, SyncedMemory buffer, cudaStream_t stream = (cudaStream_t)0) {
    mul_mod_(buffer, sigma, stream);
    mul_mod_(buffer, beta, stream);
    mul_mod_(buffer, w, stream);
    add_mod_(buffer, gamma, stream);
}

SyncedMemory lookup_ratio(SyncedMemory one, SyncedMemory delta, SyncedMemory epsilon, SyncedMemory f, SyncedMemory t, SyncedMemory t_next,
                  SyncedMemory h_1, SyncedMemory h_1_next, SyncedMemory h_2, cudaStream_t stream = (cudaStream_t)0) {

    SyncedMemory one_plus_delta = add_mod(delta, one, stream); 
    SyncedMemory epsilon_one_plus_delta = mul_mod(epsilon, one_plus_delta, stream); 

    SyncedMemory mid1 = add_mod(epsilon, f, stream); 
    SyncedMemory mid2 = add_mod(epsilon_one_plus_delta, t, stream); 
    SyncedMemory mid3 = mul_mod(delta, t_next, stream); 
    SyncedMemory mid4 = add_mod(mid2, mid3, stream); 
    SyncedMemory mid5 = mul_mod(one_plus_delta, mid1, stream); 
    SyncedMemory result = mul_mod(mid4, mid5, stream); 
    
    SyncedMemory mid6 = mul_mod(h_2, delta, stream); 
    SyncedMemory mid7 = add_mod(epsilon_one_plus_delta, h_1, stream); 
    SyncedMemory mid8 = add_mod(mid6, mid7, stream); 
    SyncedMemory mid9 = add_mod(epsilon_one_plus_delta, h_2, stream); 
    SyncedMemory mid10 = mul_mod(h_1_next, delta, stream); 
    SyncedMemory mid11 = add_mod(mid9, mid10, stream); 
    SyncedMemory mid12 = mul_mod(mid8, mid11, stream); 
    SyncedMemory mid13 = div_mod(one, mid12, stream); 
    mul_mod_(result, mid13, stream); 
    return result; 
}

std::vector<SyncedMemory> compute_permutation_poly(Radix2EvaluationDomain domain, std::vector<SyncedMemory> global_buffer, 
std::vector<SyncedMemory> w_l, std::vector<SyncedMemory> w_r, std::vector<SyncedMemory> w_o, std::vector<SyncedMemory> w_4, 
SyncedMemory beta, SyncedMemory gamma, Permutation sigma_polys, int chunk_num, uint32_t* step1_map, uint32_t* step2_map, uint32_t* step3_map,
SyncedMemory ntt_cpu_map, std::vector<SyncedMemory>ntt_cpu_buffer, std::vector<SyncedMemory> h_2_poly_cpu, 
std::vector<SyncedMemory> h_2_poly_chunks, SyncedMemory w_l_poly, Intt INTT,
cudaStream_t stream1, cudaStream_t stream2) 
{
    uint64_t n = domain.size;
    uint64_t chunk_size = global_buffer[0].size();  
    SyncedMemory one = fr::one();
    void* one_gpu_data = one.mutable_gpu_data();
    // Constants defining cosets H, k1H, k2H, etc
    std::vector<SyncedMemory> ks;
    SyncedMemory obj1 = fr::one();
    SyncedMemory obj2 = K1();
    SyncedMemory obj3 = K2();
    SyncedMemory obj4 = K3();
    void* obj1_gpu_data = obj1.mutable_gpu_data();
    void* obj2_gpu_data = obj2.mutable_gpu_data();
    void* obj3_gpu_data = obj3.mutable_gpu_data();
    void* obj4_gpu_data = obj4.mutable_gpu_data();

    ks.push_back(obj1);
    ks.push_back(obj2);
    ks.push_back(obj3);
    ks.push_back(obj4);

    void* buffer0_ = global_buffer[0].mutable_gpu_data();
    void* buffer1_ = global_buffer[1].mutable_gpu_data();
    void* buffer2_ = global_buffer[2].mutable_gpu_data();
    void* buffer3_ = global_buffer[3].mutable_gpu_data();
    void* buffer4_ = global_buffer[4].mutable_gpu_data();
    void* buffer5_ = global_buffer[5].mutable_gpu_data();
    void* buffer6_ = global_buffer[6].mutable_gpu_data();
    void* buffer7_ = global_buffer[7].mutable_gpu_data();

    void* cpu_buffer0_ = ntt_cpu_buffer[0].mutable_cpu_data();
    void* cpu_buffer1_ = ntt_cpu_buffer[1].mutable_cpu_data();

    SyncedMemory extend_ks(chunk_size);
    SyncedMemory group_gen(domain.group_gen.size());
    void* domain_group_gen_cpu = domain.group_gen.mutable_cpu_data();
    void* extend_ks_ = extend_ks.mutable_gpu_data();
    void* map_ = ntt_cpu_map.mutable_cpu_data();
    void* group_gen_ = group_gen.mutable_gpu_data();
    caffe_gpu_memcpy(group_gen.size(), domain_group_gen_cpu, group_gen_);
    SyncedMemory roots(chunk_size);
    void* roots_ = roots.mutable_gpu_data();

    for(int i = 0;i<chunk_num;i++){
        void* h2_gpu_ = h_2_poly_chunks[i].mutable_gpu_data();
        void* h2_cpu_ = h_2_poly_cpu[i].mutable_cpu_data();
        caffe_gpu_memcpy_async(chunk_size, h2_gpu_, h2_cpu_, stream1);
    }
    Ntt NTT(fr::TWO_ADICITY, stream2);
    gen_sequence_(n/chunk_num, group_gen, roots, stream2);
    //-3.3ms warm up
    void* w_l_poly_ = w_l_poly.mutable_cpu_data_async(stream1);
    

    cudaEvent_t event[4];
    cudaEventCreate(&event[0]);
    cudaEventCreate(&event[1]);
    cudaEventCreate(&event[2]);
    cudaEventCreate(&event[3]);

    /*
      Transpose wires and sigma values to get "rows" in the form [wl_i,
      wr_i, wo_i, ... ] where each row contains the wire and sigma
      values for a single gate
     Compute all roots, same as calculating twiddles, but doubled in size
    */

    void* beta_ = beta.mutable_gpu_data();
    void* gamma_ = gamma.mutable_gpu_data();

    SyncedMemory extend_beta = repeat_to_poly(beta, n/chunk_num);
    SyncedMemory extend_gamma = repeat_to_poly(gamma, n/chunk_num);
    SyncedMemory extend_one = repeat_to_poly(one, n/chunk_num);
    
    void* ones_ = extend_one.mutable_gpu_data();

    caffe_gpu_memcpy(chunk_size, ones_, buffer3_);
    caffe_gpu_memcpy(chunk_size, ones_, buffer5_);

    extend_one = SyncedMemory();
    void* z_0_ = extend_beta.mutable_gpu_data();
    void* z_1_ = extend_gamma.mutable_gpu_data();
    std::vector<SyncedMemory> z_chunk;
    z_chunk.reserve(chunk_num);

    std::vector<SyncedMemory> wires;

    wires.push_back(w_l[0]);
    wires.push_back(w_l[1]);
    wires.push_back(w_r[0]);
    wires.push_back(w_r[1]);
    wires.push_back(w_o[0]);
    wires.push_back(w_o[1]);
    wires.push_back(w_4[0]);
    wires.push_back(w_4[1]);

    for (int index = 0; index < ks.size(); index++) {
        // compute sigma mappings

        void* chunk0_ = sigma_polys[index][0].mutable_cpu_data();
        void* chunk1_ = sigma_polys[index][1].mutable_cpu_data();

        void* wire_gpu0_ = wires[index*2].mutable_gpu_data_async(stream1);
        void* wire_gpu1_ = wires[index*2 + 1].mutable_gpu_data_async(stream1);
        caffe_gpu_memcpy_async(chunk_size, ones_, buffer2_, stream1);
        caffe_gpu_memcpy_async(chunk_size, ones_, buffer4_, stream1);
        caffe_gpu_memcpy_async(chunk_size, ones_, buffer6_, stream1);
        caffe_gpu_memcpy_async(chunk_size, ones_, buffer7_, stream1);
        bit_rev_step1_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(chunk0_), step1_map, 21, 0);
        
        caffe_gpu_memcpy_async(chunk_size, map_, buffer0_, stream1);
        repeat_to_poly_(ks[index], extend_ks, n/chunk_num, stream2);
        bit_rev_step1_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(chunk1_), step1_map, 21, n/2);
        
        NTT.forward1(global_buffer[0], 22, 0, 0, stream2);
        caffe_gpu_memcpy_async(chunk_size, map_ + chunk_size, buffer1_, stream1);
        cudaEventRecord(event[0], stream1);
        // numerator_temps0
        mul_mod_(global_buffer[2], beta, stream2);
        numerator_irreducible_(roots, wires[index*2], extend_ks, extend_beta, extend_gamma, global_buffer[2], stream2);
        // numerator_product0
        mul_mod_(global_buffer[3], global_buffer[2], stream2);
        cudaStreamWaitEvent(stream2, event[0]);
        NTT.forward1(global_buffer[1], 22, 0, 1, stream2);
        caffe_gpu_memcpy_async(chunk_size, buffer0_, cpu_buffer0_, stream1);
        cudaStreamSynchronize(stream1);
        cudaEventRecord(event[3], stream1);
        // numerator_temps1
        numerator_irreducible_(roots, wires[index*2+1], extend_ks, extend_beta, extend_gamma, global_buffer[4], stream2);
        // numerator_product1
        mul_mod_(global_buffer[5], global_buffer[3], stream2);
        caffe_gpu_memcpy_async(chunk_size, buffer1_, cpu_buffer1_, stream1);
        
        bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer0_), step2_map, 21, 0);
        caffe_gpu_memcpy_async(chunk_size, map_, buffer0_, stream1);
        bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer1_), step2_map, 21, n/2);
        NTT.forward1(global_buffer[0], 22, 7, 0, stream2);
        caffe_gpu_memcpy_async(chunk_size, map_ + chunk_size, buffer1_, stream1);
        cudaEventRecord(event[1], stream1);
        cudaStreamWaitEvent(stream2, event[1]);
        NTT.forward1(global_buffer[1], 22, 7, 1, stream2);
    
        caffe_gpu_memcpy_async(chunk_size, buffer0_, cpu_buffer0_, stream1);
        cudaDeviceSynchronize();
        caffe_gpu_memcpy_async(chunk_size, buffer1_, cpu_buffer1_, stream1);
        bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer0_), step3_map, 21, 0);
        caffe_gpu_memcpy_async(chunk_size, map_, buffer0_, stream1);
        bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer1_), step3_map, 21, n/2);
        NTT.forward1(global_buffer[0], 22, 7, 0, stream2);
        denominator_irreducible_(wires[index*2], global_buffer[0], beta, extend_gamma, global_buffer[6], stream2);
        div_mod_(global_buffer[3], global_buffer[6], stream2);
        caffe_gpu_memcpy_async(chunk_size, map_ + chunk_size, buffer1_, stream1);
        cudaEventRecord(event[2], stream1);
        cudaStreamWaitEvent(stream2, event[2]);
        NTT.forward1(global_buffer[1], 22, 7, 1, stream2);
        
        denominator_irreducible_(wires[index*2+1], global_buffer[1], beta, extend_gamma, global_buffer[7], stream2);
        div_mod_(global_buffer[5], global_buffer[7], stream2);
    }
    caffe_gpu_memset_async(chunk_size, 0, z_0_, stream1);
    cudaEventRecord(event[0], stream1);
    caffe_gpu_memset_async(chunk_size, 0, z_1_, stream1);
    cudaStreamWaitEvent(stream2, event[0]);
    accumulate_mul_poly_(global_buffer[3], extend_beta, stream2);
    cudaEventRecord(event[1], stream2);
    accumulate_mul_poly_(global_buffer[5], extend_gamma, stream2);
    cudaStreamWaitEvent(stream1, event[1]);
    caffe_gpu_memcpy_async(chunk_size, z_0_, cpu_buffer0_, stream1);
    cudaStreamSynchronize(stream1);
    caffe_gpu_memcpy_async(chunk_size, z_1_, cpu_buffer1_, stream1);
    

    bit_rev_step1_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer0_), step1_map, 21, 0);   
    caffe_gpu_memcpy_async(chunk_size, map_, z_0_, stream1);
    bit_rev_step1_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer1_), step1_map, 21, n/2);
    INTT.forward1(extend_beta, 22, 0, 0, stream2);
    caffe_gpu_memcpy_async(chunk_size, map_ + chunk_size, z_1_, stream1);
    cudaStreamSynchronize(stream1);
    cudaEventRecord(event[2], stream1);
    cudaStreamWaitEvent(stream2, event[2]);
    INTT.forward1(extend_gamma, 22, 0, 1, stream2);
    caffe_gpu_memcpy_async(chunk_size, z_0_, cpu_buffer0_, stream1);
    cudaStreamSynchronize(stream1);
    caffe_gpu_memcpy_async(chunk_size, z_1_, cpu_buffer1_, stream1);
    bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer0_), step2_map, 21, 0);
    caffe_gpu_memcpy_async(chunk_size, map_, z_0_, stream1);
    bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer1_), step2_map, 21, n/2);
    INTT.forward1(extend_beta, 22, 7, 0, stream2);
    caffe_gpu_memcpy_async(chunk_size, map_ + chunk_size, z_1_, stream1);
    cudaEventRecord(event[3], stream1);
    cudaStreamWaitEvent(stream2, event[3]);
    INTT.forward1(extend_gamma, 22, 7, 1, stream2);
    caffe_gpu_memcpy_async(chunk_size, z_0_, cpu_buffer0_, stream1);
    cudaStreamSynchronize(stream1);
    caffe_gpu_memcpy_async(chunk_size, z_1_, cpu_buffer1_, stream1);
    bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer0_), step3_map, 21, 0);
    caffe_gpu_memcpy_async(chunk_size, map_, z_0_, stream1);
    bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer1_), step3_map, 21, n/2);
    INTT.forward1(extend_beta, 22, 7, 0, stream2);
    caffe_gpu_memcpy_async(chunk_size, map_ + chunk_size, z_1_, stream1);
    cudaStreamSynchronize(stream1);
    INTT.forward1(extend_gamma, 22, 7, 1, stream2);

    z_chunk.push_back(extend_beta);
    z_chunk.push_back(extend_gamma);

    cudaEventDestroy(event[0]);
    cudaEventDestroy(event[1]);
    cudaEventDestroy(event[2]);
    cudaEventDestroy(event[3]);

    return z_chunk;
}

std::vector<SyncedMemory> compute_lookup_permutation_poly(uint64_t n, std::vector<SyncedMemory> global_buffer,
std::vector<SyncedMemory> f, std::vector<SyncedMemory> t, std::vector<SyncedMemory> h_1, std::vector<SyncedMemory> h_2,
SyncedMemory delta, SyncedMemory epsilon, uint32_t* step1_map, uint32_t* step2_map, uint32_t* step3_map,
SyncedMemory ntt_cpu_map, std::vector<SyncedMemory>ntt_cpu_buffer, std::vector<SyncedMemory>z_chunks, std::vector<SyncedMemory>z_gpu, 
int chunk_num, Intt INTT, cudaStream_t stream1, cudaStream_t stream2) {

    size_t chunk_size = global_buffer[0].size();

    std::vector<SyncedMemory> z2_chunks;
    for(int i = 0; i< chunk_num;i++){
        SyncedMemory z2(chunk_size);
        z2_chunks.push_back(z2);
    }
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);

    SyncedMemory one = fr::one();
    void* one_ = one.mutable_gpu_data();
    void* delta_ = delta.mutable_gpu_data();
    void* epsilon_ = epsilon.mutable_gpu_data();


    void* buffer0_ = global_buffer[0].mutable_gpu_data();
    void* buffer1_ = global_buffer[1].mutable_gpu_data();
    void* buffer2_ = global_buffer[2].mutable_gpu_data();
    void* buffer3_ = global_buffer[3].mutable_gpu_data();
    void* buffer4_ = global_buffer[4].mutable_gpu_data();
    void* buffer5_ = global_buffer[5].mutable_gpu_data();
    void* map_ = ntt_cpu_map.mutable_cpu_data();
    void* cpu_buffer0_ = ntt_cpu_buffer[0].mutable_cpu_data();
    void* cpu_buffer1_ = ntt_cpu_buffer[1].mutable_cpu_data();

    caffe_gpu_memset_async(chunk_size, 0, buffer4_, stream1);
    caffe_gpu_memset_async(chunk_size, 0, buffer5_, stream1);
    
    void* z2_0_ = z2_chunks[0].mutable_gpu_data_async(stream1);
    void* z2_1_ = z2_chunks[1].mutable_gpu_data_async(stream1);

    for(int i=0;i<chunk_num;i++){
        lookup_ratio_step1_(h_1[i], h_2[i], global_buffer[4], one, delta, epsilon, global_buffer[i*2], stream2);
        void* t_ = t[i].mutable_gpu_data_async(stream1);
        void* f_ = f[i].mutable_gpu_data_async(stream1);
        lookup_ratio_step2_(f[i], t[i], global_buffer[5], one, delta, epsilon, global_buffer[i*2], stream2);

        accumulate_mul_poly_(global_buffer[i*2], global_buffer[i*2 + 1], stream2);
    }
    
    for(int i =0;i<chunk_num;i++){
        void* z_gpu_ = z_gpu[i].mutable_gpu_data();
        void* z_cpu_ = z_chunks[i].mutable_cpu_data();
        caffe_gpu_memcpy_async(chunk_size, z_gpu_, z_cpu_, stream1);
    }

    caffe_gpu_memcpy_async(chunk_size, buffer1_, cpu_buffer0_, stream1);
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    caffe_gpu_memcpy_async(chunk_size, buffer3_, cpu_buffer1_, stream1);  
    bit_rev_step1_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer0_), step1_map, 21, 0);
    caffe_gpu_memcpy_async(chunk_size, map_, z2_0_, stream1);
    cudaEventRecord(event1, stream1);
    bit_rev_step1_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer1_), step1_map, 21, n/2);
    cudaDeviceSynchronize();
    caffe_gpu_memcpy_async(chunk_size, map_ + chunk_size, z2_1_, stream1);
    cudaEventRecord(event2, stream1);
    cudaStreamWaitEvent(stream2, event1);
    INTT.forward1(z2_chunks[0], 22, 0, 0, stream2);
    cudaStreamWaitEvent(stream2, event2);
    INTT.forward1(z2_chunks[1], 22, 0, 1, stream2);
    
    caffe_gpu_memcpy_async(chunk_size, z2_0_, cpu_buffer0_, stream1);
    cudaStreamSynchronize(stream1);
    caffe_gpu_memcpy_async(chunk_size, z2_1_, cpu_buffer1_, stream1);
    bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer0_), step2_map, 21, 0);
    caffe_gpu_memcpy_async(chunk_size, map_, z2_0_, stream1);
    bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer1_), step2_map, 21, n/2);
    INTT.forward1(z2_chunks[0], 22, 7, 0, stream2);
    caffe_gpu_memcpy_async(chunk_size, map_ + chunk_size, z2_1_, stream1);
    cudaStreamSynchronize(stream1);
    INTT.forward1(z2_chunks[1], 22, 7, 1, stream2);
    caffe_gpu_memcpy_async(chunk_size, z2_0_, cpu_buffer0_, stream1);
    cudaStreamSynchronize(stream1);
    caffe_gpu_memcpy_async(chunk_size, z2_1_, cpu_buffer1_, stream1);
    bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer0_), step3_map, 21, 0);
    caffe_gpu_memcpy_async(chunk_size, map_, z2_0_, stream1);
    bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer1_), step3_map, 21, n/2);
    INTT.forward1(z2_chunks[0], 22, 14, 0, stream2);
    caffe_gpu_memcpy_async(chunk_size, map_ + chunk_size, z2_1_, stream1);
    cudaStreamSynchronize(stream1);
    INTT.forward1(z2_chunks[1], 22, 14, 1, stream2);

    cudaEventDestroy(event1);
    cudaEventDestroy(event2);
    return z2_chunks;
}