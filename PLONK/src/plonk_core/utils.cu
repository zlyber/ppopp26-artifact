#include "utils.cuh"

// Linear combination of a series of values
// For values [v_0, v_1,... v_k] returns:
// v_0 + challenge * v_1 + ... + challenge^k  * v_k
SyncedMemory lc(std::vector<SyncedMemory> values, SyncedMemory challenge) {
    SyncedMemory res(values[values.size()-1].size());
    void* res_gpu = res.mutable_gpu_data();
    caffe_gpu_memcpy(res.size(), values[values.size()-1].mutable_gpu_data(), res_gpu);
    for (int i = values.size() - 2; i; i--) {
        mul_mod_scalar_(res, challenge);
        add_mod_(res, values[i]); 
    }
    return res;
}