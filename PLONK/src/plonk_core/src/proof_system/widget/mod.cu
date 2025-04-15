#include "mod.cuh"

SyncedMemory delta(SyncedMemory f) {
    SyncedMemory one = fr::one();  
    SyncedMemory two = fr::make_tensor(2);  
    SyncedMemory three = fr::make_tensor(3);  

    void* one_gpu_data = one.mutable_gpu_data();
    void* two_gpu_data = two.mutable_gpu_data();
    SyncedMemory mid;
    {
        SyncedMemory mid_temp;
        {
            SyncedMemory f_1 = sub_mod_scalar(f, one);  // f - 1
            SyncedMemory f_2 = sub_mod_scalar(f, two);  // f - 2
            mid_temp = mul_mod(f_1, f_2);  // (f - 1) * (f - 2)
        }

        void* three_gpu_data = three.mutable_gpu_data();
        SyncedMemory f_3 = sub_mod_scalar(f, three);  // f - 3
        mid = mul_mod(mid_temp, f_3);  // (f - 1) * (f - 2) * (f - 3)
    }

    SyncedMemory res = mul_mod(f, mid);  // f * (f - 1) * (f - 2) * (f - 3)
    return res;
}