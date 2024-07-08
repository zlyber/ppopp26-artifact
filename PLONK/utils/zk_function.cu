#include "function.cuh"

bool gt_zkp(SyncedMemory& a, SyncedMemory& b){
    int64_t numel = a.size()/sizeof(uint64_t);
    void* a_ = a.mutable_cpu_data();
    void* b_ = b.mutable_cpu_data();
    bool gt = false;
    for(int64_t i = 0; i<numel; i++){
        gt = static_cast<uint64_t*>(a_)[i] >= static_cast<uint64_t*>(b_)[i];
        if(gt == true){
            break;
        }
    }
    return gt;
}

SyncedMemory& compress(SyncedMemory& t_0, SyncedMemory& t_1, SyncedMemory& t_2, SyncedMemory& t_3, 
                       SyncedMemory& challenge){
    return cuda::compress_cuda(t_0, t_1, t_2, t_3, challenge);
}

SyncedMemory& compute_query_table(SyncedMemory& q_lookup, 
                        SyncedMemory& w_l_scalar ,SyncedMemory& w_r_scalar , SyncedMemory& w_o_scalar,
                        SyncedMemory& w_4_scalar, SyncedMemory& t_poly, SyncedMemory& challenge){
    int64_t n = w_l_scalar.size() / (cuda::fr_LIMBS*sizeof(uint64_t));
    SyncedMemory& padded_q_lookup = cuda::pad_poly_cuda(q_lookup, n);
    SyncedMemory& concatenated_f_scalars = cuda::compute_query_table_cuda(padded_q_lookup, w_l_scalar, w_r_scalar, w_o_scalar, w_4_scalar, t_poly);
    return cuda::compress_cuda_2(concatenated_f_scalars, challenge, n * cuda::fr_LIMBS);  
}