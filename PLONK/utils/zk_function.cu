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

