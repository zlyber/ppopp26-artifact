#include "function.cuh"

bool gt_zkp(SyncedMemory a, SyncedMemory b){
    int64_t numel = a.size()/sizeof(uint64_t);
    uint64_t* a_ = reinterpret_cast<uint64_t*>(a.mutable_cpu_data());
    uint64_t* b_ = reinterpret_cast<uint64_t*>(b.mutable_cpu_data());
    bool gt = false;
    for(int64_t i = numel-1; i >= 0; i--){
        if(a_[i] > b_[i]){
            gt = true;
            break;
        }
        else if(a_[i] < b_[i]){
            gt = false;
            break;
        }
        else{
            continue;
        }
    }
    return gt;
}

void compress(SyncedMemory output, SyncedMemory t_0, SyncedMemory t_1, SyncedMemory t_2, SyncedMemory t_3, 
                       SyncedMemory challenge, cudaStream_t stream){
    cuda::compress_cuda(output, t_0, t_1, t_2, t_3, challenge, stream);
}

void compute_query_table(std::vector<SyncedMemory> f_chunks, std::vector<SyncedMemory> global_buffer, std::vector<SyncedMemory> q_lookup, 
            std::vector<SyncedMemory> w_l_scalar, std::vector<SyncedMemory> w_r_scalar, std::vector<SyncedMemory> w_o_scalar, std::vector<SyncedMemory> w_4_scalar, 
            std::vector<SyncedMemory> t_poly, SyncedMemory challenge, int chunk_num,
            cudaStream_t stream1, cudaStream_t stream2){
    uint64_t chunk_size = w_l_scalar[0].size();
    int64_t n = w_l_scalar[0].size() / (cuda::fr_LIMBS*sizeof(uint64_t));
    cudaEvent_t event;
    cudaEventCreate(&event);
    for(int i=0;i<chunk_num;i++){
        SyncedMemory concatenated_f_scalars(chunk_size*4);
    
        void* buffer0_ = global_buffer[0].mutable_gpu_data();
        void* buffer1_ = global_buffer[1].mutable_gpu_data();
        void* buffer2_ = global_buffer[2].mutable_gpu_data();
        void* buffer3_ = global_buffer[3].mutable_gpu_data();

        void* w_l_ = w_l_scalar[i].mutable_cpu_data();
        void* w_r_ = w_r_scalar[i].mutable_cpu_data();
        void* w_o_ = w_o_scalar[i].mutable_cpu_data();
        void* w_4_ = w_4_scalar[i].mutable_cpu_data();

        void* f_chunks_ = f_chunks[i].mutable_gpu_data_async(stream2);
        cudaMemsetAsync(f_chunks_, 0, chunk_size, stream2);
        cudaEventRecord(event, stream2);
        
        caffe_gpu_memcpy_async(chunk_size, w_l_, buffer0_, stream2);
        caffe_gpu_memcpy_async(chunk_size, w_r_, buffer1_, stream2);
        caffe_gpu_memcpy_async(chunk_size, w_o_, buffer2_, stream2);
        caffe_gpu_memcpy_async(chunk_size, w_4_, buffer3_, stream2);

        
        SyncedMemory padded_q_lookup = cuda::pad_poly_cuda(q_lookup[i], n, stream1);
        cuda::compute_query_table_cuda(concatenated_f_scalars, padded_q_lookup, global_buffer[0], global_buffer[1], global_buffer[2], global_buffer[3], t_poly[i], stream1);
        cudaStreamWaitEvent(stream1, event);
        cuda::compress_cuda_2(concatenated_f_scalars, f_chunks[i], challenge, n * cuda::fr_LIMBS, stream1);
    }
}



std::vector<SyncedMemory> split_tx_poly(uint64_t n, SyncedMemory t_poly){
    std::vector<SyncedMemory> t_x;
    void* t_gpu = t_poly.mutable_gpu_data();
    for(int i=0; i<8; i++){
        SyncedMemory t_(n*cuda::fr_LIMBS*sizeof(uint64_t));
        void* t_x_gpu = t_.mutable_gpu_data();
        caffe_gpu_memcpy(t_.size(), t_gpu + i*t_.size(), t_x_gpu);
        t_x.push_back(t_);
    }
    return t_x;
}

void COSET_INTT(Intt_coset INTT, 
    int lg_LDE,
    int chunk_num,
    std::vector<SyncedMemory> inout,
    std::vector<SyncedMemory> global_buffer,
    std::vector<SyncedMemory> cpu_buffer,
    uint32_t* step1_map,
    uint32_t* step2_map,
    uint32_t* step3_map,
    SyncedMemory cpu_map,
    cudaStream_t stream1,
    cudaStream_t stream2)
{
    void* map_ = cpu_map.mutable_cpu_data();

    void* buffer1_ = global_buffer[0].mutable_gpu_data();
    void* buffer2_ = global_buffer[1].mutable_gpu_data();
    void* buffer3_ = global_buffer[2].mutable_gpu_data();
    void* buffer4_ = global_buffer[3].mutable_gpu_data();
    void* buffer5_ = global_buffer[4].mutable_gpu_data();
    void* buffer6_ = global_buffer[5].mutable_gpu_data();
    void* buffer7_ = global_buffer[6].mutable_gpu_data();
    void* buffer8_ = global_buffer[7].mutable_gpu_data();
    void* buffer9_ = global_buffer[8].mutable_gpu_data();
    void* buffer10_ = global_buffer[9].mutable_gpu_data();
    void* buffer11_ = global_buffer[10].mutable_gpu_data();
    void* buffer12_ = global_buffer[11].mutable_gpu_data();
    void* buffer13_ = global_buffer[12].mutable_gpu_data();
    void* buffer14_ = global_buffer[13].mutable_gpu_data();
    void* buffer15_ = global_buffer[14].mutable_gpu_data();
    void* buffer16_ = global_buffer[15].mutable_gpu_data();

    void* cpu_buffer1_ = cpu_buffer[0].mutable_cpu_data();
    void* cpu_buffer2_ = cpu_buffer[1].mutable_cpu_data();
    void* cpu_buffer3_ = cpu_buffer[2].mutable_cpu_data();
    void* cpu_buffer4_ = cpu_buffer[3].mutable_cpu_data();
    void* cpu_buffer5_ = cpu_buffer[4].mutable_cpu_data();
    void* cpu_buffer6_ = cpu_buffer[5].mutable_cpu_data();
    void* cpu_buffer7_ = cpu_buffer[6].mutable_cpu_data();
    void* cpu_buffer8_ = cpu_buffer[7].mutable_cpu_data();
    void* cpu_buffer9_ = cpu_buffer[8].mutable_cpu_data();
    void* cpu_buffer10_ = cpu_buffer[9].mutable_cpu_data();
    void* cpu_buffer11_ = cpu_buffer[10].mutable_cpu_data();
    void* cpu_buffer12_ = cpu_buffer[11].mutable_cpu_data();
    void* cpu_buffer13_ = cpu_buffer[12].mutable_cpu_data();
    void* cpu_buffer14_ = cpu_buffer[13].mutable_cpu_data();
    void* cpu_buffer15_ = cpu_buffer[14].mutable_cpu_data();
    void* cpu_buffer16_ = cpu_buffer[15].mutable_cpu_data();

    int lg_step1 = 21;
    int lg_step2 = 21;
    int lg_step3 = 21;

    uint64_t chunk_size = global_buffer[0].size();

    cudaEvent_t step1, step2, step3;
    cudaEventCreate(&step1);
    cudaEventCreate(&step2);
    cudaEventCreate(&step3);

    
    for(int i = 0;i<chunk_num;i++){
        void* buffer_ = global_buffer[i].mutable_gpu_data();
        void* inout_ = inout[i].mutable_cpu_data_async(stream1);
        bit_rev_step1_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(inout_), step1_map, lg_step1, i * 1<<lg_step1);
        cudaEventRecord(step1, 0);
        cudaStreamWaitEvent(stream1, step1);
        caffe_gpu_memcpy_async(chunk_size, map_ + i*chunk_size , buffer_, stream1);
        cudaEventRecord(step1, stream1);
        cudaStreamWaitEvent(stream2, step1);
        INTT.init(global_buffer[i], lg_LDE, i, true, stream2);
    }
    


}


void LDE_forward(Ntt_coset NTT, 
                int lg_LDE,
                SyncedMemory input,
                SyncedMemory output,
                std::vector<SyncedMemory> global_buffer,
                std::vector<SyncedMemory> cpu_buffer,
                uint32_t* step1_map,
                uint32_t* step2_map,
                uint32_t* step3_map,
                SyncedMemory cpu_map,
                SyncedMemory step1_in,
                bool tail,
                cudaStream_t stream1,
                cudaStream_t stream2)
{   

    void* input_ = input.mutable_cpu_data();
    void* map_ = cpu_map.mutable_cpu_data();
    void* output_ = output.mutable_cpu_data();
    void* step1_in_ = step1_in.mutable_cpu_data();

    void* buffer1_ = global_buffer[0].mutable_gpu_data();
    void* buffer2_ = global_buffer[1].mutable_gpu_data();
    void* buffer3_ = global_buffer[2].mutable_gpu_data();
    void* buffer4_ = global_buffer[3].mutable_gpu_data();
    void* buffer5_ = global_buffer[4].mutable_gpu_data();
    void* buffer6_ = global_buffer[5].mutable_gpu_data();
    void* buffer7_ = global_buffer[6].mutable_gpu_data();
    void* buffer8_ = global_buffer[7].mutable_gpu_data();
    void* buffer9_ = global_buffer[8].mutable_gpu_data();
    void* buffer10_ = global_buffer[9].mutable_gpu_data();
    void* buffer11_ = global_buffer[10].mutable_gpu_data();
    void* buffer12_ = global_buffer[11].mutable_gpu_data();
    void* buffer13_ = global_buffer[12].mutable_gpu_data();
    void* buffer14_ = global_buffer[13].mutable_gpu_data();
    void* buffer15_ = global_buffer[14].mutable_gpu_data();
    void* buffer16_ = global_buffer[15].mutable_gpu_data();

    void* cpu_buffer1_ = cpu_buffer[0].mutable_cpu_data();
    void* cpu_buffer2_ = cpu_buffer[1].mutable_cpu_data();
    void* cpu_buffer3_ = cpu_buffer[2].mutable_cpu_data();
    void* cpu_buffer4_ = cpu_buffer[3].mutable_cpu_data();
    void* cpu_buffer5_ = cpu_buffer[4].mutable_cpu_data();
    void* cpu_buffer6_ = cpu_buffer[5].mutable_cpu_data();
    void* cpu_buffer7_ = cpu_buffer[6].mutable_cpu_data();
    void* cpu_buffer8_ = cpu_buffer[7].mutable_cpu_data();

    void* cpu_buffer9_ = cpu_buffer[8].mutable_cpu_data();
    void* cpu_buffer10_ = cpu_buffer[9].mutable_cpu_data();
    void* cpu_buffer11_ = cpu_buffer[10].mutable_cpu_data();
    void* cpu_buffer12_ = cpu_buffer[11].mutable_cpu_data();
    void* cpu_buffer13_ = cpu_buffer[12].mutable_cpu_data();
    void* cpu_buffer14_ = cpu_buffer[13].mutable_cpu_data();
    void* cpu_buffer15_ = cpu_buffer[14].mutable_cpu_data();
    void* cpu_buffer16_ = cpu_buffer[15].mutable_cpu_data();

    int lg_step0 = 21;
    int lg_step1 = 20;
    int lg_step2 = 21;
    int lg_step3 = 21;

    uint64_t chunk_size = global_buffer[0].size();

    cudaEvent_t step0, step1, step2, step3;
    cudaEventCreate(&step0);
    cudaEventCreate(&step1);
    cudaEventCreate(&step2);
    cudaEventCreate(&step3);

    // step0
    // bit_rev_step1_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(step1_in_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(input_), step1_map, lg_step0, 0);
    // cudaDeviceSynchronize();
    // caffe_gpu_memcpy_async(chunk_size, step1_in_, buffer1_, stream1);
    // cudaEventRecord(step0, stream1);
    // bit_rev_step1_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(step1_in_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(input_ + chunk_size), step1_map, lg_step0, 1<<lg_step0);
    // caffe_gpu_memcpy_async(chunk_size, step1_in_ + chunk_size, buffer2_, stream1);
    // cudaStreamWaitEvent(stream2, step0);
    // NTT.init(global_buffer[0], lg_LDE, 0, stream2);
    // cudaEventRecord(step0, stream2);
    // cudaStreamWaitEvent(stream1, step0);
    // caffe_gpu_memcpy_async(chunk_size, buffer1_, step1_in_, stream1);
    // NTT.init(global_buffer[1], lg_LDE, 1, stream2);

    caffe_gpu_memcpy_async(chunk_size, input_, buffer15_, stream1);
    cudaEventRecord(step0, stream1);
    caffe_gpu_memcpy_async(chunk_size, input_ + chunk_size, buffer16_, stream1);
    cudaStreamWaitEvent(stream2, step0);
    NTT.init_with_bitrev(global_buffer[14], lg_step0, stream2);
    cudaEventRecord(step0, stream1);
    NTT.init_with_bitrev(global_buffer[15], lg_step0, stream2);
    cudaStreamWaitEvent(stream2, step0);
    caffe_gpu_memcpy_async(chunk_size, buffer15_, step1_in_, stream1);
    cudaStreamSynchronize(stream1);
    caffe_gpu_memcpy_async(chunk_size, buffer16_, step1_in_ + chunk_size, stream1);
    pad_and_transpose(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(step1_in_), lg_step1, 0);
    cudaDeviceSynchronize();
    caffe_gpu_memcpy_async(chunk_size, map_ , buffer1_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + chunk_size, buffer2_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 2 * chunk_size, buffer3_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 3 * chunk_size, buffer4_, stream1);
    pad_and_transpose(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(step1_in_), lg_step1, 1<<lg_step1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 4 * chunk_size, buffer5_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 5 * chunk_size, buffer6_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 6 * chunk_size, buffer7_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 7 * chunk_size, buffer8_, stream1);
    cudaEventRecord(step1, stream1);
    pad_and_transpose(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(step1_in_), lg_step1, 2 * 1<<lg_step1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 8 * chunk_size, buffer9_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 9 * chunk_size, buffer10_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 10 * chunk_size, buffer11_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 11 * chunk_size, buffer12_, stream1);
    cudaStreamWaitEvent(stream2, step1);
    NTT.forward1(global_buffer[0], lg_LDE, 0, 0, stream2);
    NTT.forward1(global_buffer[1], lg_LDE, 0, 1, stream2);
    NTT.forward1(global_buffer[2], lg_LDE, 0, 2, stream2);
    NTT.forward1(global_buffer[3], lg_LDE, 0, 3, stream2);
    NTT.forward1(global_buffer[4], lg_LDE, 0, 4, stream2);
    NTT.forward1(global_buffer[5], lg_LDE, 0, 5, stream2);
    NTT.forward1(global_buffer[6], lg_LDE, 0, 6, stream2);
    NTT.forward1(global_buffer[7], lg_LDE, 0, 7, stream2);
    pad_and_transpose(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(step1_in_), lg_step1, 3 * 1<<lg_step1);
    cudaStreamSynchronize(stream1);
    NTT.forward1(global_buffer[8], lg_LDE, 0, 8, stream2);
    NTT.forward1(global_buffer[9], lg_LDE, 0, 9, stream2);
    NTT.forward1(global_buffer[10], lg_LDE, 0, 10, stream2);
    NTT.forward1(global_buffer[11], lg_LDE, 0, 11, stream2);
    caffe_gpu_memcpy_async(chunk_size, map_ + 12 * chunk_size, buffer13_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 13 * chunk_size, buffer14_, stream1);
    cudaStreamSynchronize(stream1);
    NTT.forward1(global_buffer[12], lg_LDE, 0, 12, stream2);
    NTT.forward1(global_buffer[13], lg_LDE, 0, 13, stream2);
    caffe_gpu_memcpy_async(chunk_size, map_ + 14 * chunk_size, buffer15_, stream1);
    cudaStreamSynchronize(stream1);
    NTT.forward1(global_buffer[14], lg_LDE, 0, 14, stream2);
    caffe_gpu_memcpy_async(chunk_size, map_ + 15 * chunk_size, buffer16_, stream1);
    NTT.forward1(global_buffer[15], lg_LDE, 0, 15, stream1);

    caffe_gpu_memcpy_async(chunk_size, buffer1_, cpu_buffer1_, stream2);
    cudaStreamSynchronize(stream2);
    caffe_gpu_memcpy_async(chunk_size, buffer2_, cpu_buffer2_, stream1);
    caffe_gpu_memcpy_async(chunk_size, buffer3_, cpu_buffer3_, stream1);
    caffe_gpu_memcpy_async(chunk_size, buffer4_, cpu_buffer4_, stream1);
    bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer1_), step2_map, lg_step2, 0);
    caffe_gpu_memcpy_async(chunk_size, buffer5_, cpu_buffer5_, stream1);
    caffe_gpu_memcpy_async(chunk_size, buffer6_, cpu_buffer6_, stream1);
    
    bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer2_), step2_map, lg_step2, 1<<lg_step2);
    caffe_gpu_memcpy_async(chunk_size, buffer7_, cpu_buffer7_, stream1);
    caffe_gpu_memcpy_async(chunk_size, buffer8_, cpu_buffer8_, stream1);
    
    bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer3_), step2_map, lg_step2, 2 * 1<<lg_step2);
    caffe_gpu_memcpy_async(chunk_size, buffer9_, cpu_buffer9_, stream1);
    caffe_gpu_memcpy_async(chunk_size, buffer10_, cpu_buffer10_, stream1);
    
    bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer4_), step2_map, lg_step2, 3 * 1<<lg_step2);
    caffe_gpu_memcpy_async(chunk_size, buffer11_, cpu_buffer11_, stream1);
    caffe_gpu_memcpy_async(chunk_size, buffer12_, cpu_buffer12_, stream1);
   
    bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer5_), step2_map, lg_step2, 4 * 1<<lg_step2);
    caffe_gpu_memcpy_async(chunk_size, buffer13_, cpu_buffer13_, stream1);
    caffe_gpu_memcpy_async(chunk_size, buffer14_, cpu_buffer14_, stream1);
    
    bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer6_), step2_map, lg_step2, 5 * 1<<lg_step2);
    caffe_gpu_memcpy_async(chunk_size, buffer15_, cpu_buffer15_, stream1);
    caffe_gpu_memcpy_async(chunk_size, buffer16_, cpu_buffer16_, stream1);
    
    bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer7_), step2_map, lg_step2, 6 * 1<<lg_step2);
    caffe_gpu_memcpy_async(chunk_size, map_, buffer1_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + chunk_size, buffer2_, stream1);
    

    bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer8_), step2_map, lg_step2, 7 * 1<<lg_step2);
    caffe_gpu_memcpy_async(chunk_size, map_ + 2 * chunk_size, buffer3_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 3 * chunk_size, buffer4_, stream1);
    
    bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer9_), step2_map, lg_step2, 8 * 1<<lg_step2);
    caffe_gpu_memcpy_async(chunk_size, map_ + 4 * chunk_size, buffer5_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 5 * chunk_size, buffer6_, stream1);
    
    bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer10_), step2_map, lg_step2, 9 * 1<<lg_step2);
    caffe_gpu_memcpy_async(chunk_size, map_ + 6 * chunk_size, buffer7_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 7 * chunk_size, buffer8_, stream1);
    cudaEventRecord(step2, stream1);
    
    bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer11_), step2_map, lg_step2, 10 * 1<<lg_step2);
    caffe_gpu_memcpy_async(chunk_size, map_ + 8 * chunk_size, buffer9_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 9 * chunk_size, buffer10_, stream1);
    cudaEventRecord(step2, stream1);
    bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer12_), step2_map, lg_step2, 11 * 1<<lg_step2);
    cudaStreamWaitEvent(stream2, step2);
    NTT.forward1(global_buffer[0], lg_LDE, 8, 0, stream2);
    NTT.forward1(global_buffer[1], lg_LDE, 8, 1, stream2);
    NTT.forward1(global_buffer[2], lg_LDE, 8, 2, stream2);
    NTT.forward1(global_buffer[3], lg_LDE, 8, 3, stream2);
    NTT.forward1(global_buffer[4], lg_LDE, 8, 4, stream2);
    NTT.forward1(global_buffer[5], lg_LDE, 8, 5, stream2);
    NTT.forward1(global_buffer[6], lg_LDE, 8, 6, stream2);
    NTT.forward1(global_buffer[7], lg_LDE, 8, 7, stream2);
    
    caffe_gpu_memcpy_async(chunk_size, map_ + 10 * chunk_size, buffer11_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 11 * chunk_size, buffer12_, stream1);
    
    bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer13_), step2_map, lg_step2, 12 * 1<<lg_step2);
    cudaStreamWaitEvent(stream2, step2);
    NTT.forward1(global_buffer[8], lg_LDE, 8, 8, stream2);
    NTT.forward1(global_buffer[9], lg_LDE, 8, 9, stream2);
    
    caffe_gpu_memcpy_async(chunk_size, map_ + 12 * chunk_size, buffer13_, stream1);
    cudaEventRecord(step2, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 13 * chunk_size, buffer14_, stream1);
    
    bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer14_), step2_map, lg_step2, 13 * 1<<lg_step2);
    cudaStreamWaitEvent(stream2, step2);
    NTT.forward1(global_buffer[10], lg_LDE, 8, 10, stream2);
    NTT.forward1(global_buffer[11], lg_LDE, 8, 11, stream2);
    NTT.forward1(global_buffer[12], lg_LDE, 8, 12, stream2);
    
    
    caffe_gpu_memcpy_async(chunk_size, map_ + 14 * chunk_size, buffer15_, stream1);
    cudaEventRecord(step2, stream1);
    bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer15_), step2_map, lg_step2, 14 * 1<<lg_step2);
    bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer16_), step2_map, lg_step2, 15 * 1<<lg_step2);
    cudaStreamWaitEvent(stream2, step2);
    NTT.forward1(global_buffer[13], lg_LDE, 8, 13, stream2);
    NTT.forward1(global_buffer[14], lg_LDE, 8, 14, stream2);
    caffe_gpu_memcpy_async(chunk_size, map_ + 15 * chunk_size, buffer16_, stream1);
    NTT.forward1(global_buffer[15], lg_LDE, 8, 15, stream1);
    

    caffe_gpu_memcpy_async(chunk_size, buffer1_, cpu_buffer1_, stream2);
    cudaStreamSynchronize(stream2);
    caffe_gpu_memcpy_async(chunk_size, buffer2_, cpu_buffer2_, stream1);
    caffe_gpu_memcpy_async(chunk_size, buffer3_, cpu_buffer3_, stream1);
    bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer1_), step3_map, lg_step3, 0);
    
    caffe_gpu_memcpy_async(chunk_size, buffer4_, cpu_buffer4_, stream1);
    caffe_gpu_memcpy_async(chunk_size, buffer5_, cpu_buffer5_, stream1);
    bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer2_), step3_map, lg_step3, 1<<lg_step3);
    
    caffe_gpu_memcpy_async(chunk_size, buffer6_, cpu_buffer6_, stream1);
    caffe_gpu_memcpy_async(chunk_size, buffer7_, cpu_buffer7_, stream1);
    bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer3_), step3_map, lg_step3, 2 * 1<<lg_step3);
    
    caffe_gpu_memcpy_async(chunk_size, buffer8_, cpu_buffer8_, stream1);
    caffe_gpu_memcpy_async(chunk_size, buffer9_, cpu_buffer9_, stream1);
    bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer4_), step3_map, lg_step3, 3 * 1<<lg_step3);
    
    caffe_gpu_memcpy_async(chunk_size, buffer10_, cpu_buffer10_, stream1);
    caffe_gpu_memcpy_async(chunk_size, buffer11_, cpu_buffer11_, stream1);
    bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer5_), step3_map, lg_step3, 4 * 1<<lg_step3);
    
    caffe_gpu_memcpy_async(chunk_size, buffer12_, cpu_buffer12_, stream1);
    caffe_gpu_memcpy_async(chunk_size, buffer13_, cpu_buffer13_, stream1);
    bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer6_), step3_map, lg_step3, 5 * 1<<lg_step3);
    
    caffe_gpu_memcpy_async(chunk_size, buffer14_, cpu_buffer14_, stream1);
    caffe_gpu_memcpy_async(chunk_size, buffer15_, cpu_buffer15_, stream1);
    bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer7_), step3_map, lg_step3, 6 * 1<<lg_step3);
    
    caffe_gpu_memcpy_async(chunk_size, buffer16_, cpu_buffer16_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_, buffer1_, stream1);

    bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer8_), step3_map, lg_step3, 7 * 1<<lg_step3);
    caffe_gpu_memcpy_async(chunk_size, map_ + chunk_size, buffer2_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 2 * chunk_size, buffer3_, stream1);

    bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer9_), step3_map, lg_step3, 8 * 1<<lg_step3);
    caffe_gpu_memcpy_async(chunk_size, map_ + 3 * chunk_size, buffer4_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 4 * chunk_size, buffer5_, stream1);

    bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer10_), step3_map, lg_step3, 9 * 1<<lg_step3);
    caffe_gpu_memcpy_async(chunk_size, map_ + 5 * chunk_size, buffer6_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 6 * chunk_size, buffer7_, stream1);

    bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer11_), step3_map, lg_step3, 10 * 1<<lg_step3);
    caffe_gpu_memcpy_async(chunk_size, map_ + 7 * chunk_size, buffer8_, stream1);
    cudaEventRecord(step3, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 8 * chunk_size, buffer9_, stream1);
    
    bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer12_), step3_map, lg_step3, 11 * 1<<lg_step3);
    cudaStreamWaitEvent(stream2, step3);
    NTT.forward1(global_buffer[0], lg_LDE, 8, 0, stream2);
    NTT.forward1(global_buffer[1], lg_LDE, 8, 1, stream2);
    NTT.forward1(global_buffer[2], lg_LDE, 8, 2, stream2);
    NTT.forward1(global_buffer[3], lg_LDE, 8, 3, stream2);
    NTT.forward1(global_buffer[4], lg_LDE, 8, 4, stream2);
    NTT.forward1(global_buffer[5], lg_LDE, 8, 5, stream2);
    NTT.forward1(global_buffer[6], lg_LDE, 8, 6, stream2);
    NTT.forward1(global_buffer[7], lg_LDE, 8, 7, stream2);
    caffe_gpu_memcpy_async(chunk_size, map_ + 9 * chunk_size, buffer10_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 10 * chunk_size, buffer11_, stream1);
    cudaEventRecord(step3, stream1);
    cudaStreamWaitEvent(stream2, step3);
    NTT.forward1(global_buffer[8], lg_LDE, 8, 8, stream2);
    NTT.forward1(global_buffer[9], lg_LDE, 8, 9, stream2);
    caffe_gpu_memcpy_async(chunk_size, map_ + 11 * chunk_size, buffer12_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 12 * chunk_size, buffer13_, stream1);
    cudaEventRecord(step3, stream1);
    cudaStreamWaitEvent(stream2, step3);
    NTT.forward1(global_buffer[10], lg_LDE, 8, 10, stream2);
    NTT.forward1(global_buffer[11], lg_LDE, 8, 11, stream2);
    NTT.forward1(global_buffer[12], lg_LDE, 8, 12, stream2);
    bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer13_), step3_map, lg_step3, 12 * 1<<lg_step3);
    bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer14_), step3_map, lg_step3, 13 * 1<<lg_step3);
    caffe_gpu_memcpy_async(chunk_size, map_ + 13 * chunk_size, buffer14_, stream1);
    caffe_gpu_memcpy_async(chunk_size, map_ + 14 * chunk_size, buffer15_, stream1);
    cudaEventRecord(step3, stream1);
    cudaStreamWaitEvent(stream2, step3);
    NTT.forward1(global_buffer[13], lg_LDE, 8, 13, stream2);
    NTT.forward1(global_buffer[14], lg_LDE, 8, 14, stream2);
    bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer15_), step3_map, lg_step3, 14 * 1<<lg_step3);
    bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer16_), step3_map, lg_step3, 15 * 1<<lg_step3);
    
    caffe_gpu_memcpy_async(chunk_size, map_ + 15 * chunk_size, buffer16_, stream1);
    NTT.forward1(global_buffer[15], lg_LDE, 8, 15, stream1);
    
    for(int i = 0;i<16;i++){
        void* d_ = global_buffer[i].mutable_gpu_data();
        caffe_gpu_memcpy_async(chunk_size, d_, output_ + i * chunk_size, stream2);
    }

    cudaEventDestroy(step0);
    cudaEventDestroy(step1);
    cudaEventDestroy(step2);
    cudaEventDestroy(step3);
}

void LDE_forward_chunk(Ntt_coset NTT, 
    int lg_LDE,
    int chunk_num,
    std::vector<SyncedMemory> input,
    SyncedMemory output,
    std::vector<SyncedMemory> global_buffer,
    std::vector<SyncedMemory> cpu_buffer,
    uint32_t* step1_map,
    uint32_t* step2_map,
    uint32_t* step3_map,
    SyncedMemory cpu_map,
    SyncedMemory step1_in,
    bool tail,
    cudaStream_t stream1,
    cudaStream_t stream2)
{   

void* input0_ = input[0].mutable_cpu_data();
void* input1_ = input[1].mutable_cpu_data();
void* map_ = cpu_map.mutable_cpu_data();
void* output_ = output.mutable_cpu_data();
void* step1_in_ = step1_in.mutable_cpu_data();

void* buffer1_ = global_buffer[0].mutable_gpu_data();
void* buffer2_ = global_buffer[1].mutable_gpu_data();
void* buffer3_ = global_buffer[2].mutable_gpu_data();
void* buffer4_ = global_buffer[3].mutable_gpu_data();
void* buffer5_ = global_buffer[4].mutable_gpu_data();
void* buffer6_ = global_buffer[5].mutable_gpu_data();
void* buffer7_ = global_buffer[6].mutable_gpu_data();
void* buffer8_ = global_buffer[7].mutable_gpu_data();
void* buffer9_ = global_buffer[8].mutable_gpu_data();
void* buffer10_ = global_buffer[9].mutable_gpu_data();
void* buffer11_ = global_buffer[10].mutable_gpu_data();
void* buffer12_ = global_buffer[11].mutable_gpu_data();
void* buffer13_ = global_buffer[12].mutable_gpu_data();
void* buffer14_ = global_buffer[13].mutable_gpu_data();
void* buffer15_ = global_buffer[14].mutable_gpu_data();
void* buffer16_ = global_buffer[15].mutable_gpu_data();

void* cpu_buffer1_ = cpu_buffer[0].mutable_cpu_data();
void* cpu_buffer2_ = cpu_buffer[1].mutable_cpu_data();
void* cpu_buffer3_ = cpu_buffer[2].mutable_cpu_data();
void* cpu_buffer4_ = cpu_buffer[3].mutable_cpu_data();
void* cpu_buffer5_ = cpu_buffer[4].mutable_cpu_data();
void* cpu_buffer6_ = cpu_buffer[5].mutable_cpu_data();
void* cpu_buffer7_ = cpu_buffer[6].mutable_cpu_data();
void* cpu_buffer8_ = cpu_buffer[7].mutable_cpu_data();

void* cpu_buffer9_ = cpu_buffer[8].mutable_cpu_data();
void* cpu_buffer10_ = cpu_buffer[9].mutable_cpu_data();
void* cpu_buffer11_ = cpu_buffer[10].mutable_cpu_data();
void* cpu_buffer12_ = cpu_buffer[11].mutable_cpu_data();
void* cpu_buffer13_ = cpu_buffer[12].mutable_cpu_data();
void* cpu_buffer14_ = cpu_buffer[13].mutable_cpu_data();
void* cpu_buffer15_ = cpu_buffer[14].mutable_cpu_data();
void* cpu_buffer16_ = cpu_buffer[15].mutable_cpu_data();

int lg_step0 = 21;
int lg_step1 = 20;
int lg_step2 = 21;
int lg_step3 = 21;

uint64_t chunk_size = global_buffer[0].size();

cudaEvent_t step0, step1, step2, step3;
cudaEventCreate(&step0);
cudaEventCreate(&step1);
cudaEventCreate(&step2);
cudaEventCreate(&step3);

// step0
// bit_rev_step1_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(step1_in_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(input_), step1_map, lg_step0, 0);
// cudaDeviceSynchronize();
// caffe_gpu_memcpy_async(chunk_size, step1_in_, buffer1_, stream1);
// cudaEventRecord(step0, stream1);
// bit_rev_step1_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(step1_in_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(input_ + chunk_size), step1_map, lg_step0, 1<<lg_step0);
// caffe_gpu_memcpy_async(chunk_size, step1_in_ + chunk_size, buffer2_, stream1);
// cudaStreamWaitEvent(stream2, step0);
// NTT.init(global_buffer[0], lg_LDE, 0, stream2);
// cudaEventRecord(step0, stream2);
// cudaStreamWaitEvent(stream1, step0);
// caffe_gpu_memcpy_async(chunk_size, buffer1_, step1_in_, stream1);
// NTT.init(global_buffer[1], lg_LDE, 1, stream2);

caffe_gpu_memcpy_async(chunk_size, input0_, buffer15_, stream1);
cudaEventRecord(step0, stream1);
caffe_gpu_memcpy_async(chunk_size, input1_, buffer16_, stream1);
cudaStreamWaitEvent(stream2, step0);
NTT.init_with_bitrev(global_buffer[14], lg_step0, stream2);
cudaEventRecord(step0, stream1);
NTT.init_with_bitrev(global_buffer[15], lg_step0, stream2);
cudaStreamWaitEvent(stream2, step0);
caffe_gpu_memcpy_async(chunk_size, buffer15_, step1_in_, stream1);
cudaStreamSynchronize(stream1);
caffe_gpu_memcpy_async(chunk_size, buffer16_, step1_in_ + chunk_size, stream1);
pad_and_transpose(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(step1_in_), lg_step1, 0);
cudaDeviceSynchronize();
caffe_gpu_memcpy_async(chunk_size, map_ , buffer1_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + chunk_size, buffer2_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 2 * chunk_size, buffer3_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 3 * chunk_size, buffer4_, stream1);
pad_and_transpose(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(step1_in_), lg_step1, 1<<lg_step1);
caffe_gpu_memcpy_async(chunk_size, map_ + 4 * chunk_size, buffer5_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 5 * chunk_size, buffer6_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 6 * chunk_size, buffer7_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 7 * chunk_size, buffer8_, stream1);
cudaEventRecord(step1, stream1);
pad_and_transpose(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(step1_in_), lg_step1, 2 * 1<<lg_step1);
caffe_gpu_memcpy_async(chunk_size, map_ + 8 * chunk_size, buffer9_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 9 * chunk_size, buffer10_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 10 * chunk_size, buffer11_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 11 * chunk_size, buffer12_, stream1);
cudaStreamWaitEvent(stream2, step1);
NTT.forward1(global_buffer[0], lg_LDE, 0, 0, stream2);
NTT.forward1(global_buffer[1], lg_LDE, 0, 1, stream2);
NTT.forward1(global_buffer[2], lg_LDE, 0, 2, stream2);
NTT.forward1(global_buffer[3], lg_LDE, 0, 3, stream2);
NTT.forward1(global_buffer[4], lg_LDE, 0, 4, stream2);
NTT.forward1(global_buffer[5], lg_LDE, 0, 5, stream2);
NTT.forward1(global_buffer[6], lg_LDE, 0, 6, stream2);
NTT.forward1(global_buffer[7], lg_LDE, 0, 7, stream2);
pad_and_transpose(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(step1_in_), lg_step1, 3 * 1<<lg_step1);
cudaStreamSynchronize(stream1);
NTT.forward1(global_buffer[8], lg_LDE, 0, 8, stream2);
NTT.forward1(global_buffer[9], lg_LDE, 0, 9, stream2);
NTT.forward1(global_buffer[10], lg_LDE, 0, 10, stream2);
NTT.forward1(global_buffer[11], lg_LDE, 0, 11, stream2);
caffe_gpu_memcpy_async(chunk_size, map_ + 12 * chunk_size, buffer13_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 13 * chunk_size, buffer14_, stream1);
cudaStreamSynchronize(stream1);
NTT.forward1(global_buffer[12], lg_LDE, 0, 12, stream2);
NTT.forward1(global_buffer[13], lg_LDE, 0, 13, stream2);
caffe_gpu_memcpy_async(chunk_size, map_ + 14 * chunk_size, buffer15_, stream1);
cudaStreamSynchronize(stream1);
NTT.forward1(global_buffer[14], lg_LDE, 0, 14, stream2);
caffe_gpu_memcpy_async(chunk_size, map_ + 15 * chunk_size, buffer16_, stream1);
NTT.forward1(global_buffer[15], lg_LDE, 0, 15, stream1);

caffe_gpu_memcpy_async(chunk_size, buffer1_, cpu_buffer1_, stream2);
cudaStreamSynchronize(stream2);
caffe_gpu_memcpy_async(chunk_size, buffer2_, cpu_buffer2_, stream1);
caffe_gpu_memcpy_async(chunk_size, buffer3_, cpu_buffer3_, stream1);
caffe_gpu_memcpy_async(chunk_size, buffer4_, cpu_buffer4_, stream1);
bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer1_), step2_map, lg_step2, 0);
caffe_gpu_memcpy_async(chunk_size, buffer5_, cpu_buffer5_, stream1);
caffe_gpu_memcpy_async(chunk_size, buffer6_, cpu_buffer6_, stream1);

bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer2_), step2_map, lg_step2, 1<<lg_step2);
caffe_gpu_memcpy_async(chunk_size, buffer7_, cpu_buffer7_, stream1);
caffe_gpu_memcpy_async(chunk_size, buffer8_, cpu_buffer8_, stream1);

bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer3_), step2_map, lg_step2, 2 * 1<<lg_step2);
caffe_gpu_memcpy_async(chunk_size, buffer9_, cpu_buffer9_, stream1);
caffe_gpu_memcpy_async(chunk_size, buffer10_, cpu_buffer10_, stream1);

bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer4_), step2_map, lg_step2, 3 * 1<<lg_step2);
caffe_gpu_memcpy_async(chunk_size, buffer11_, cpu_buffer11_, stream1);
caffe_gpu_memcpy_async(chunk_size, buffer12_, cpu_buffer12_, stream1);

bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer5_), step2_map, lg_step2, 4 * 1<<lg_step2);
caffe_gpu_memcpy_async(chunk_size, buffer13_, cpu_buffer13_, stream1);
caffe_gpu_memcpy_async(chunk_size, buffer14_, cpu_buffer14_, stream1);

bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer6_), step2_map, lg_step2, 5 * 1<<lg_step2);
caffe_gpu_memcpy_async(chunk_size, buffer15_, cpu_buffer15_, stream1);
caffe_gpu_memcpy_async(chunk_size, buffer16_, cpu_buffer16_, stream1);

bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer7_), step2_map, lg_step2, 6 * 1<<lg_step2);
caffe_gpu_memcpy_async(chunk_size, map_, buffer1_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + chunk_size, buffer2_, stream1);


bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer8_), step2_map, lg_step2, 7 * 1<<lg_step2);
caffe_gpu_memcpy_async(chunk_size, map_ + 2 * chunk_size, buffer3_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 3 * chunk_size, buffer4_, stream1);

bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer9_), step2_map, lg_step2, 8 * 1<<lg_step2);
caffe_gpu_memcpy_async(chunk_size, map_ + 4 * chunk_size, buffer5_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 5 * chunk_size, buffer6_, stream1);

bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer10_), step2_map, lg_step2, 9 * 1<<lg_step2);
caffe_gpu_memcpy_async(chunk_size, map_ + 6 * chunk_size, buffer7_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 7 * chunk_size, buffer8_, stream1);
cudaEventRecord(step2, stream1);

bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer11_), step2_map, lg_step2, 10 * 1<<lg_step2);
caffe_gpu_memcpy_async(chunk_size, map_ + 8 * chunk_size, buffer9_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 9 * chunk_size, buffer10_, stream1);
cudaEventRecord(step2, stream1);
bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer12_), step2_map, lg_step2, 11 * 1<<lg_step2);
cudaStreamWaitEvent(stream2, step2);
NTT.forward1(global_buffer[0], lg_LDE, 8, 0, stream2);
NTT.forward1(global_buffer[1], lg_LDE, 8, 1, stream2);
NTT.forward1(global_buffer[2], lg_LDE, 8, 2, stream2);
NTT.forward1(global_buffer[3], lg_LDE, 8, 3, stream2);
NTT.forward1(global_buffer[4], lg_LDE, 8, 4, stream2);
NTT.forward1(global_buffer[5], lg_LDE, 8, 5, stream2);
NTT.forward1(global_buffer[6], lg_LDE, 8, 6, stream2);
NTT.forward1(global_buffer[7], lg_LDE, 8, 7, stream2);

caffe_gpu_memcpy_async(chunk_size, map_ + 10 * chunk_size, buffer11_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 11 * chunk_size, buffer12_, stream1);

bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer13_), step2_map, lg_step2, 12 * 1<<lg_step2);
cudaStreamWaitEvent(stream2, step2);
NTT.forward1(global_buffer[8], lg_LDE, 8, 8, stream2);
NTT.forward1(global_buffer[9], lg_LDE, 8, 9, stream2);

caffe_gpu_memcpy_async(chunk_size, map_ + 12 * chunk_size, buffer13_, stream1);
cudaEventRecord(step2, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 13 * chunk_size, buffer14_, stream1);

bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer14_), step2_map, lg_step2, 13 * 1<<lg_step2);
cudaStreamWaitEvent(stream2, step2);
NTT.forward1(global_buffer[10], lg_LDE, 8, 10, stream2);
NTT.forward1(global_buffer[11], lg_LDE, 8, 11, stream2);
NTT.forward1(global_buffer[12], lg_LDE, 8, 12, stream2);


caffe_gpu_memcpy_async(chunk_size, map_ + 14 * chunk_size, buffer15_, stream1);
cudaEventRecord(step2, stream1);
bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer15_), step2_map, lg_step2, 14 * 1<<lg_step2);
bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer16_), step2_map, lg_step2, 15 * 1<<lg_step2);
cudaStreamWaitEvent(stream2, step2);
NTT.forward1(global_buffer[13], lg_LDE, 8, 13, stream2);
NTT.forward1(global_buffer[14], lg_LDE, 8, 14, stream2);
caffe_gpu_memcpy_async(chunk_size, map_ + 15 * chunk_size, buffer16_, stream1);
NTT.forward1(global_buffer[15], lg_LDE, 8, 15, stream1);


caffe_gpu_memcpy_async(chunk_size, buffer1_, cpu_buffer1_, stream2);
cudaStreamSynchronize(stream2);
caffe_gpu_memcpy_async(chunk_size, buffer2_, cpu_buffer2_, stream1);
caffe_gpu_memcpy_async(chunk_size, buffer3_, cpu_buffer3_, stream1);
bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer1_), step3_map, lg_step3, 0);

caffe_gpu_memcpy_async(chunk_size, buffer4_, cpu_buffer4_, stream1);
caffe_gpu_memcpy_async(chunk_size, buffer5_, cpu_buffer5_, stream1);
bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer2_), step3_map, lg_step3, 1<<lg_step3);

caffe_gpu_memcpy_async(chunk_size, buffer6_, cpu_buffer6_, stream1);
caffe_gpu_memcpy_async(chunk_size, buffer7_, cpu_buffer7_, stream1);
bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer3_), step3_map, lg_step3, 2 * 1<<lg_step3);

caffe_gpu_memcpy_async(chunk_size, buffer8_, cpu_buffer8_, stream1);
caffe_gpu_memcpy_async(chunk_size, buffer9_, cpu_buffer9_, stream1);
bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer4_), step3_map, lg_step3, 3 * 1<<lg_step3);

caffe_gpu_memcpy_async(chunk_size, buffer10_, cpu_buffer10_, stream1);
caffe_gpu_memcpy_async(chunk_size, buffer11_, cpu_buffer11_, stream1);
bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer5_), step3_map, lg_step3, 4 * 1<<lg_step3);

caffe_gpu_memcpy_async(chunk_size, buffer12_, cpu_buffer12_, stream1);
caffe_gpu_memcpy_async(chunk_size, buffer13_, cpu_buffer13_, stream1);
bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer6_), step3_map, lg_step3, 5 * 1<<lg_step3);

caffe_gpu_memcpy_async(chunk_size, buffer14_, cpu_buffer14_, stream1);
caffe_gpu_memcpy_async(chunk_size, buffer15_, cpu_buffer15_, stream1);
bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer7_), step3_map, lg_step3, 6 * 1<<lg_step3);

caffe_gpu_memcpy_async(chunk_size, buffer16_, cpu_buffer16_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_, buffer1_, stream1);

bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer8_), step3_map, lg_step3, 7 * 1<<lg_step3);
caffe_gpu_memcpy_async(chunk_size, map_ + chunk_size, buffer2_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 2 * chunk_size, buffer3_, stream1);

bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer9_), step3_map, lg_step3, 8 * 1<<lg_step3);
caffe_gpu_memcpy_async(chunk_size, map_ + 3 * chunk_size, buffer4_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 4 * chunk_size, buffer5_, stream1);

bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer10_), step3_map, lg_step3, 9 * 1<<lg_step3);
caffe_gpu_memcpy_async(chunk_size, map_ + 5 * chunk_size, buffer6_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 6 * chunk_size, buffer7_, stream1);

bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer11_), step3_map, lg_step3, 10 * 1<<lg_step3);
caffe_gpu_memcpy_async(chunk_size, map_ + 7 * chunk_size, buffer8_, stream1);
cudaEventRecord(step3, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 8 * chunk_size, buffer9_, stream1);

bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer12_), step3_map, lg_step3, 11 * 1<<lg_step3);
cudaStreamWaitEvent(stream2, step3);
NTT.forward1(global_buffer[0], lg_LDE, 8, 0, stream2);
NTT.forward1(global_buffer[1], lg_LDE, 8, 1, stream2);
NTT.forward1(global_buffer[2], lg_LDE, 8, 2, stream2);
NTT.forward1(global_buffer[3], lg_LDE, 8, 3, stream2);
NTT.forward1(global_buffer[4], lg_LDE, 8, 4, stream2);
NTT.forward1(global_buffer[5], lg_LDE, 8, 5, stream2);
NTT.forward1(global_buffer[6], lg_LDE, 8, 6, stream2);
NTT.forward1(global_buffer[7], lg_LDE, 8, 7, stream2);
caffe_gpu_memcpy_async(chunk_size, map_ + 9 * chunk_size, buffer10_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 10 * chunk_size, buffer11_, stream1);
cudaEventRecord(step3, stream1);
cudaStreamWaitEvent(stream2, step3);
NTT.forward1(global_buffer[8], lg_LDE, 8, 8, stream2);
NTT.forward1(global_buffer[9], lg_LDE, 8, 9, stream2);
caffe_gpu_memcpy_async(chunk_size, map_ + 11 * chunk_size, buffer12_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 12 * chunk_size, buffer13_, stream1);
cudaEventRecord(step3, stream1);
cudaStreamWaitEvent(stream2, step3);
NTT.forward1(global_buffer[10], lg_LDE, 8, 10, stream2);
NTT.forward1(global_buffer[11], lg_LDE, 8, 11, stream2);
NTT.forward1(global_buffer[12], lg_LDE, 8, 12, stream2);
bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer13_), step3_map, lg_step3, 12 * 1<<lg_step3);
bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer14_), step3_map, lg_step3, 13 * 1<<lg_step3);
caffe_gpu_memcpy_async(chunk_size, map_ + 13 * chunk_size, buffer14_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 14 * chunk_size, buffer15_, stream1);
cudaEventRecord(step3, stream1);
cudaStreamWaitEvent(stream2, step3);
NTT.forward1(global_buffer[13], lg_LDE, 8, 13, stream2);
NTT.forward1(global_buffer[14], lg_LDE, 8, 14, stream2);
bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer15_), step3_map, lg_step3, 14 * 1<<lg_step3);
bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer16_), step3_map, lg_step3, 15 * 1<<lg_step3);

caffe_gpu_memcpy_async(chunk_size, map_ + 15 * chunk_size, buffer16_, stream1);
NTT.forward1(global_buffer[15], lg_LDE, 8, 15, stream1);

for(int i = 0;i<16;i++){
    void* d_ = global_buffer[i].mutable_gpu_data();
    caffe_gpu_memcpy_async(chunk_size, d_, output_ + i * chunk_size, stream2);
}

cudaEventDestroy(step0);
cudaEventDestroy(step1);
cudaEventDestroy(step2);
cudaEventDestroy(step3);
}


void LDE_forward_with_commit(
    Ntt_coset NTT, 
    int lg_LDE,
    SyncedMemory input,
    std::vector<SyncedMemory> ck,
    SyncedMemory chunk_msm_workspace, SyncedMemory chunk_msm_out,
    std::vector<SyncedMemory> commits,
    std::vector<SyncedMemory> global_buffer,
    std::vector<SyncedMemory> cpu_buffer,
    uint32_t* step1_map,
    uint32_t* step2_map,
    uint32_t* step3_map,
    SyncedMemory cpu_map,
    SyncedMemory step1_in,
    bool tail,
    cudaStream_t stream1,
    cudaStream_t stream2)
{   

void* input_ = input.mutable_cpu_data();
void* map_ = cpu_map.mutable_cpu_data();
void* step1_in_ = step1_in.mutable_cpu_data();

void* buffer1_ = global_buffer[0].mutable_gpu_data();
void* buffer2_ = global_buffer[1].mutable_gpu_data();
void* buffer3_ = global_buffer[2].mutable_gpu_data();
void* buffer4_ = global_buffer[3].mutable_gpu_data();
void* buffer5_ = global_buffer[4].mutable_gpu_data();
void* buffer6_ = global_buffer[5].mutable_gpu_data();
void* buffer7_ = global_buffer[6].mutable_gpu_data();
void* buffer8_ = global_buffer[7].mutable_gpu_data();
void* buffer9_ = global_buffer[8].mutable_gpu_data();
void* buffer10_ = global_buffer[9].mutable_gpu_data();
void* buffer11_ = global_buffer[10].mutable_gpu_data();
void* buffer12_ = global_buffer[11].mutable_gpu_data();
void* buffer13_ = global_buffer[12].mutable_gpu_data();
void* buffer14_ = global_buffer[13].mutable_gpu_data();
void* buffer15_ = global_buffer[14].mutable_gpu_data();
void* buffer16_ = global_buffer[15].mutable_gpu_data();

void* cpu_buffer1_ = cpu_buffer[0].mutable_cpu_data();
void* cpu_buffer2_ = cpu_buffer[1].mutable_cpu_data();
void* cpu_buffer3_ = cpu_buffer[2].mutable_cpu_data();
void* cpu_buffer4_ = cpu_buffer[3].mutable_cpu_data();
void* cpu_buffer5_ = cpu_buffer[4].mutable_cpu_data();
void* cpu_buffer6_ = cpu_buffer[5].mutable_cpu_data();
void* cpu_buffer7_ = cpu_buffer[6].mutable_cpu_data();
void* cpu_buffer8_ = cpu_buffer[7].mutable_cpu_data();

void* cpu_buffer9_ = cpu_buffer[8].mutable_cpu_data();
void* cpu_buffer10_ = cpu_buffer[9].mutable_cpu_data();
void* cpu_buffer11_ = cpu_buffer[10].mutable_cpu_data();
void* cpu_buffer12_ = cpu_buffer[11].mutable_cpu_data();
void* cpu_buffer13_ = cpu_buffer[12].mutable_cpu_data();
void* cpu_buffer14_ = cpu_buffer[13].mutable_cpu_data();
void* cpu_buffer15_ = cpu_buffer[14].mutable_cpu_data();
void* cpu_buffer16_ = cpu_buffer[15].mutable_cpu_data();


int lg_step0 = 21;
int lg_step1 = 20;
int lg_step2 = 21;
int lg_step3 = 21;

uint64_t chunk_size = global_buffer[0].size();

cudaEvent_t step0, step1, step2, step3;
cudaEventCreate(&step0);
cudaEventCreate(&step1);
cudaEventCreate(&step2);
cudaEventCreate(&step3);

// step0
// bit_rev_step1_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(step1_in_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(input_), step1_map, lg_step0, 0);
// cudaDeviceSynchronize();
// caffe_gpu_memcpy_async(chunk_size, step1_in_, buffer1_, stream1);
// cudaEventRecord(step0, stream1);
// bit_rev_step1_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(step1_in_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(input_ + chunk_size), step1_map, lg_step0, 1<<lg_step0);
// caffe_gpu_memcpy_async(chunk_size, step1_in_ + chunk_size, buffer2_, stream1);
// cudaStreamWaitEvent(stream2, step0);
// NTT.init(global_buffer[0], lg_LDE, 0, stream2);
// cudaEventRecord(step0, stream2);
// cudaStreamWaitEvent(stream1, step0);
// caffe_gpu_memcpy_async(chunk_size, buffer1_, step1_in_, stream1);
// NTT.init(global_buffer[1], lg_LDE, 1, stream2);

caffe_gpu_memcpy_async(chunk_size, input_, buffer15_, stream1);
cudaEventRecord(step0, stream1);
caffe_gpu_memcpy_async(chunk_size, input_ + chunk_size, buffer16_, stream1);
cudaStreamWaitEvent(stream2, step0);
NTT.init_with_bitrev(global_buffer[14], lg_step0, stream2);
cudaEventRecord(step0, stream1);
NTT.init_with_bitrev(global_buffer[15], lg_step0, stream2);
cudaStreamWaitEvent(stream2, step0);
caffe_gpu_memcpy_async(chunk_size, buffer15_, step1_in_, stream1);
cuda::msm_zkp_cuda_(ck[0], global_buffer[14], chunk_msm_workspace, chunk_msm_out, 114, stream2);
void* commit0_ = commits[0].mutable_cpu_data();
caffe_gpu_memcpy_async(chunk_msm_out.size(), chunk_msm_out.mutable_gpu_data(), commit0_, stream2);
cudaStreamSynchronize(stream1);
caffe_gpu_memcpy_async(chunk_size, buffer16_, step1_in_ + chunk_size, stream1);
pad_and_transpose(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(step1_in_), lg_step1, 0);
cudaEventRecord(step0, stream1);
cudaStreamWaitEvent(stream1, step0);
caffe_gpu_memcpy_async(chunk_size, map_ , buffer1_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + chunk_size, buffer2_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 2 * chunk_size, buffer3_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 3 * chunk_size, buffer4_, stream1);
pad_and_transpose(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(step1_in_), lg_step1, 1<<lg_step1);
caffe_gpu_memcpy_async(chunk_size, map_ + 4 * chunk_size, buffer5_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 5 * chunk_size, buffer6_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 6 * chunk_size, buffer7_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 7 * chunk_size, buffer8_, stream1);
cudaEventRecord(step1, stream1);
pad_and_transpose(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(step1_in_), lg_step1, 2 * 1<<lg_step1);
caffe_gpu_memcpy_async(chunk_size, map_ + 8 * chunk_size, buffer9_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 9 * chunk_size, buffer10_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 10 * chunk_size, buffer11_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 11 * chunk_size, buffer12_, stream1);
cudaStreamWaitEvent(stream2, step1);
NTT.forward1(global_buffer[0], lg_LDE, 0, 0, stream2);
NTT.forward1(global_buffer[1], lg_LDE, 0, 1, stream2);
NTT.forward1(global_buffer[2], lg_LDE, 0, 2, stream2);
NTT.forward1(global_buffer[3], lg_LDE, 0, 3, stream2);
NTT.forward1(global_buffer[4], lg_LDE, 0, 4, stream2);
NTT.forward1(global_buffer[5], lg_LDE, 0, 5, stream2);
NTT.forward1(global_buffer[6], lg_LDE, 0, 6, stream2);
NTT.forward1(global_buffer[7], lg_LDE, 0, 7, stream2);
pad_and_transpose(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(step1_in_), lg_step1, 3 * 1<<lg_step1);
cudaStreamSynchronize(stream1);
NTT.forward1(global_buffer[8], lg_LDE, 0, 8, stream2);
NTT.forward1(global_buffer[9], lg_LDE, 0, 9, stream2);
NTT.forward1(global_buffer[10], lg_LDE, 0, 10, stream2);
NTT.forward1(global_buffer[11], lg_LDE, 0, 11, stream2);
caffe_gpu_memcpy_async(chunk_size, map_ + 12 * chunk_size, buffer13_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 13 * chunk_size, buffer14_, stream1);
cudaStreamSynchronize(stream1);
NTT.forward1(global_buffer[12], lg_LDE, 0, 12, stream2);
NTT.forward1(global_buffer[13], lg_LDE, 0, 13, stream2);
caffe_gpu_memcpy_async(chunk_size, map_ + 14 * chunk_size, buffer15_, stream1);
cudaStreamSynchronize(stream1);
NTT.forward1(global_buffer[14], lg_LDE, 0, 14, stream2);
cuda::msm_zkp_cuda_(ck[1], global_buffer[15], chunk_msm_workspace, chunk_msm_out, 114, stream2);
void* commit1_ = commits[1].mutable_cpu_data();
caffe_gpu_memcpy_async(chunk_msm_out.size(), chunk_msm_out.mutable_gpu_data(), commit1_, stream2);


caffe_gpu_memcpy_async(chunk_size, buffer1_, cpu_buffer1_, stream1);
cudaStreamSynchronize(stream1);
caffe_gpu_memcpy_async(chunk_size, buffer2_, cpu_buffer2_, stream1);
caffe_gpu_memcpy_async(chunk_size, buffer3_, cpu_buffer3_, stream1);
caffe_gpu_memcpy_async(chunk_size, buffer4_, cpu_buffer4_, stream1);
bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer1_), step2_map, lg_step2, 0);
caffe_gpu_memcpy_async(chunk_size, buffer5_, cpu_buffer5_, stream1);
caffe_gpu_memcpy_async(chunk_size, buffer6_, cpu_buffer6_, stream1);

bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer2_), step2_map, lg_step2, 1<<lg_step2);
caffe_gpu_memcpy_async(chunk_size, buffer7_, cpu_buffer7_, stream1);
caffe_gpu_memcpy_async(chunk_size, buffer8_, cpu_buffer8_, stream1);

bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer3_), step2_map, lg_step2, 2 * 1<<lg_step2);
caffe_gpu_memcpy_async(chunk_size, buffer9_, cpu_buffer9_, stream1);
caffe_gpu_memcpy_async(chunk_size, buffer10_, cpu_buffer10_, stream1);

bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer4_), step2_map, lg_step2, 3 * 1<<lg_step2);
caffe_gpu_memcpy_async(chunk_size, buffer11_, cpu_buffer11_, stream1);
caffe_gpu_memcpy_async(chunk_size, buffer12_, cpu_buffer12_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 15 * chunk_size, buffer16_, stream1);
cudaEventRecord(step1, stream1);
cudaStreamWaitEvent(stream2, step1);
NTT.forward1(global_buffer[15], lg_LDE, 0, 15, stream2);
bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer5_), step2_map, lg_step2, 4 * 1<<lg_step2);
caffe_gpu_memcpy_async(chunk_size, buffer13_, cpu_buffer13_, stream1);
caffe_gpu_memcpy_async(chunk_size, buffer14_, cpu_buffer14_, stream1);

bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer6_), step2_map, lg_step2, 5 * 1<<lg_step2);
caffe_gpu_memcpy_async(chunk_size, buffer15_, cpu_buffer15_, stream1);
caffe_gpu_memcpy_async(chunk_size, buffer16_, cpu_buffer16_, stream1);

bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer7_), step2_map, lg_step2, 6 * 1<<lg_step2);
caffe_gpu_memcpy_async(chunk_size, map_, buffer1_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + chunk_size, buffer2_, stream1);


bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer8_), step2_map, lg_step2, 7 * 1<<lg_step2);
caffe_gpu_memcpy_async(chunk_size, map_ + 2 * chunk_size, buffer3_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 3 * chunk_size, buffer4_, stream1);

bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer9_), step2_map, lg_step2, 8 * 1<<lg_step2);
caffe_gpu_memcpy_async(chunk_size, map_ + 4 * chunk_size, buffer5_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 5 * chunk_size, buffer6_, stream1);

bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer10_), step2_map, lg_step2, 9 * 1<<lg_step2);
caffe_gpu_memcpy_async(chunk_size, map_ + 6 * chunk_size, buffer7_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 7 * chunk_size, buffer8_, stream1);
cudaEventRecord(step2, stream1);

bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer11_), step2_map, lg_step2, 10 * 1<<lg_step2);
caffe_gpu_memcpy_async(chunk_size, map_ + 8 * chunk_size, buffer9_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 9 * chunk_size, buffer10_, stream1);
cudaEventRecord(step2, stream1);
bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer12_), step2_map, lg_step2, 11 * 1<<lg_step2);
cudaStreamWaitEvent(stream2, step2);
NTT.forward1(global_buffer[0], lg_LDE, 8, 0, stream2);
NTT.forward1(global_buffer[1], lg_LDE, 8, 1, stream2);
NTT.forward1(global_buffer[2], lg_LDE, 8, 2, stream2);
NTT.forward1(global_buffer[3], lg_LDE, 8, 3, stream2);
NTT.forward1(global_buffer[4], lg_LDE, 8, 4, stream2);
NTT.forward1(global_buffer[5], lg_LDE, 8, 5, stream2);
NTT.forward1(global_buffer[6], lg_LDE, 8, 6, stream2);
NTT.forward1(global_buffer[7], lg_LDE, 8, 7, stream2);

caffe_gpu_memcpy_async(chunk_size, map_ + 10 * chunk_size, buffer11_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 11 * chunk_size, buffer12_, stream1);

bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer13_), step2_map, lg_step2, 12 * 1<<lg_step2);
cudaStreamWaitEvent(stream2, step2);
NTT.forward1(global_buffer[8], lg_LDE, 8, 8, stream2);
NTT.forward1(global_buffer[9], lg_LDE, 8, 9, stream2);

caffe_gpu_memcpy_async(chunk_size, map_ + 12 * chunk_size, buffer13_, stream1);
cudaEventRecord(step2, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 13 * chunk_size, buffer14_, stream1);

bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer14_), step2_map, lg_step2, 13 * 1<<lg_step2);
cudaStreamWaitEvent(stream2, step2);
NTT.forward1(global_buffer[10], lg_LDE, 8, 10, stream2);
NTT.forward1(global_buffer[11], lg_LDE, 8, 11, stream2);
NTT.forward1(global_buffer[12], lg_LDE, 8, 12, stream2);


caffe_gpu_memcpy_async(chunk_size, map_ + 14 * chunk_size, buffer15_, stream1);
cudaEventRecord(step2, stream1);
bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer15_), step2_map, lg_step2, 14 * 1<<lg_step2);
bit_rev_step2_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer16_), step2_map, lg_step2, 15 * 1<<lg_step2);
cudaStreamWaitEvent(stream2, step2);
NTT.forward1(global_buffer[13], lg_LDE, 8, 13, stream2);
NTT.forward1(global_buffer[14], lg_LDE, 8, 14, stream2);
caffe_gpu_memcpy_async(chunk_size, map_ + 15 * chunk_size, buffer16_, stream1);
NTT.forward1(global_buffer[15], lg_LDE, 8, 15, stream1);


caffe_gpu_memcpy_async(chunk_size, buffer1_, cpu_buffer1_, stream2);
cudaStreamSynchronize(stream2);
caffe_gpu_memcpy_async(chunk_size, buffer2_, cpu_buffer2_, stream1);
caffe_gpu_memcpy_async(chunk_size, buffer3_, cpu_buffer3_, stream1);
bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer1_), step3_map, lg_step3, 0);

caffe_gpu_memcpy_async(chunk_size, buffer4_, cpu_buffer4_, stream1);
caffe_gpu_memcpy_async(chunk_size, buffer5_, cpu_buffer5_, stream1);
bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer2_), step3_map, lg_step3, 1<<lg_step3);

caffe_gpu_memcpy_async(chunk_size, buffer6_, cpu_buffer6_, stream1);
caffe_gpu_memcpy_async(chunk_size, buffer7_, cpu_buffer7_, stream1);
bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer3_), step3_map, lg_step3, 2 * 1<<lg_step3);

caffe_gpu_memcpy_async(chunk_size, buffer8_, cpu_buffer8_, stream1);
caffe_gpu_memcpy_async(chunk_size, buffer9_, cpu_buffer9_, stream1);
bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer4_), step3_map, lg_step3, 3 * 1<<lg_step3);

caffe_gpu_memcpy_async(chunk_size, buffer10_, cpu_buffer10_, stream1);
caffe_gpu_memcpy_async(chunk_size, buffer11_, cpu_buffer11_, stream1);
bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer5_), step3_map, lg_step3, 4 * 1<<lg_step3);

caffe_gpu_memcpy_async(chunk_size, buffer12_, cpu_buffer12_, stream1);
caffe_gpu_memcpy_async(chunk_size, buffer13_, cpu_buffer13_, stream1);
bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer6_), step3_map, lg_step3, 5 * 1<<lg_step3);

caffe_gpu_memcpy_async(chunk_size, buffer14_, cpu_buffer14_, stream1);
caffe_gpu_memcpy_async(chunk_size, buffer15_, cpu_buffer15_, stream1);
bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer7_), step3_map, lg_step3, 6 * 1<<lg_step3);

caffe_gpu_memcpy_async(chunk_size, buffer16_, cpu_buffer16_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_, buffer1_, stream1);

bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer8_), step3_map, lg_step3, 7 * 1<<lg_step3);
caffe_gpu_memcpy_async(chunk_size, map_ + chunk_size, buffer2_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 2 * chunk_size, buffer3_, stream1);

bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer9_), step3_map, lg_step3, 8 * 1<<lg_step3);
caffe_gpu_memcpy_async(chunk_size, map_ + 3 * chunk_size, buffer4_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 4 * chunk_size, buffer5_, stream1);

bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer10_), step3_map, lg_step3, 9 * 1<<lg_step3);
caffe_gpu_memcpy_async(chunk_size, map_ + 5 * chunk_size, buffer6_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 6 * chunk_size, buffer7_, stream1);

bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer11_), step3_map, lg_step3, 10 * 1<<lg_step3);
caffe_gpu_memcpy_async(chunk_size, map_ + 7 * chunk_size, buffer8_, stream1);
cudaEventRecord(step3, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 8 * chunk_size, buffer9_, stream1);

bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer12_), step3_map, lg_step3, 11 * 1<<lg_step3);
cudaStreamWaitEvent(stream2, step3);
NTT.forward1(global_buffer[0], lg_LDE, 8, 0, stream2);
NTT.forward1(global_buffer[1], lg_LDE, 8, 1, stream2);
NTT.forward1(global_buffer[2], lg_LDE, 8, 2, stream2);
NTT.forward1(global_buffer[3], lg_LDE, 8, 3, stream2);
NTT.forward1(global_buffer[4], lg_LDE, 8, 4, stream2);
NTT.forward1(global_buffer[5], lg_LDE, 8, 5, stream2);
NTT.forward1(global_buffer[6], lg_LDE, 8, 6, stream2);
NTT.forward1(global_buffer[7], lg_LDE, 8, 7, stream2);
caffe_gpu_memcpy_async(chunk_size, map_ + 9 * chunk_size, buffer10_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 10 * chunk_size, buffer11_, stream1);
cudaEventRecord(step3, stream1);
cudaStreamWaitEvent(stream2, step3);
NTT.forward1(global_buffer[8], lg_LDE, 8, 8, stream2);
NTT.forward1(global_buffer[9], lg_LDE, 8, 9, stream2);
caffe_gpu_memcpy_async(chunk_size, map_ + 11 * chunk_size, buffer12_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 12 * chunk_size, buffer13_, stream1);
cudaEventRecord(step3, stream1);
cudaStreamWaitEvent(stream2, step3);
NTT.forward1(global_buffer[10], lg_LDE, 8, 10, stream2);
NTT.forward1(global_buffer[11], lg_LDE, 8, 11, stream2);
NTT.forward1(global_buffer[12], lg_LDE, 8, 12, stream2);
bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer13_), step3_map, lg_step3, 12 * 1<<lg_step3);
bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer14_), step3_map, lg_step3, 13 * 1<<lg_step3);
caffe_gpu_memcpy_async(chunk_size, map_ + 13 * chunk_size, buffer14_, stream1);
caffe_gpu_memcpy_async(chunk_size, map_ + 14 * chunk_size, buffer15_, stream1);
cudaEventRecord(step3, stream1);
cudaStreamWaitEvent(stream2, step3);
NTT.forward1(global_buffer[13], lg_LDE, 8, 13, stream2);
NTT.forward1(global_buffer[14], lg_LDE, 8, 14, stream2);
bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer15_), step3_map, lg_step3, 14 * 1<<lg_step3);
bit_rev_step3_parallel(reinterpret_cast<cpu::BLS12_381_Fr_G1*>(map_), reinterpret_cast<cpu::BLS12_381_Fr_G1*>(cpu_buffer16_), step3_map, lg_step3, 15 * 1<<lg_step3);

caffe_gpu_memcpy_async(chunk_size, map_ + 15 * chunk_size, buffer16_, stream1);
NTT.forward1(global_buffer[15], lg_LDE, 8, 15, stream1);


cudaEventDestroy(step0);
cudaEventDestroy(step1);
cudaEventDestroy(step2);
cudaEventDestroy(step3);
}