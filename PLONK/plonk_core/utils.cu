#include <vector>
#include "PLONK/utils/function.cuh"
#include "caffe/syncedmem.hpp"
#include "PLONK/src/structure.cuh"
// Linear combination of a series of values
// For values [v_0, v_1,... v_k] returns:
// v_0 + challenge * v_1 + ... + challenge^k  * v_k
SyncedMemory& lc( std::vector<SyncedMemory&>&values, SyncedMemory& challenge) {
    
    SyncedMemory& kth_val = values.back();
    for (auto it = values.rbegin() + 1; it != values.rend(); ++it) {
        kth_val = mul_mod_scalar(kth_val, challenge);
        kth_val = add_mod(kth_val, *it); 
    }
    return kth_val;
}