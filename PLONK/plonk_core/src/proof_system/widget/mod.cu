#include <cuda_runtime.h>
#include "PLONK/plonk_core/src/permutation/constants.cu"
#include "PLONK/utils/function.cuh"
#include "caffe/syncedmem.hpp"
#include "PLONK/src/structure.cuh"
#include "PLONK/src/bls12_381/fr.hpp"
#include "PLONK/src/bls12_381/fq.hpp"
#include "PLONK/src/domain.cuh"
#include "PLONK/src/domain.cu"

class WitnessValues {
public:
    WitnessValues( SyncedMemory& a_val,  SyncedMemory& b_val,  SyncedMemory& c_val,  SyncedMemory& d_val) 
        : a_val(a_val), b_val(b_val), c_val(c_val), d_val(d_val) {}


    SyncedMemory& a_val;
    SyncedMemory& b_val;
    SyncedMemory& c_val;
    SyncedMemory& d_val;
};

SyncedMemory& delta( SyncedMemory& f) {
    SyncedMemory& one = fr::one();  
    SyncedMemory& two = fr::make_tensor(2);  
    SyncedMemory& three = fr::make_tensor(3);  

    void* one_gpu_data=one.mutable_gpu_data();
    void* two_gpu_data=two.mutable_gpu_data();
    SyncedMemory& f_1 = sub_mod_scalar(f, one);  // f - 1
    SyncedMemory& f_2 = sub_mod_scalar(f, two);  // f - 2
    SyncedMemory& mid = mul_mod(f_1, f_2);  // (f - 1) * (f - 2)
    // delete f_1,f_2;
    void* three_gpu_data=three.mutable_gpu_data();
    SyncedMemory& f_3 = sub_mod_scalar(f, three);  // f - 3
    mid = mul_mod(mid, f_3);  // (f - 1) * (f - 2) * (f - 3)
    // delete f_3;

    SyncedMemory& res = mul_mod(f, mid);  // f * (f - 1) * (f - 2) * (f - 3)
    return res;
}