#pragma once
#include <iostream>
#include "../../../caffe/interface.hpp"


class fq{
    public:

    static constexpr int MODULUS_BITS = 381;
    static constexpr int BYTE_SIZE = (MODULUS_BITS + 7) / 8;
    static constexpr int Limbs = (MODULUS_BITS + 63) / 64;
    static constexpr int TWO_ADICITY = 32;
    static constexpr int REPR_SHAVE_BITS = 1;

    static SyncedMemory zero() {
        SyncedMemory zero(Limbs * sizeof(uint64_t), false);
        void* zero_ = zero.mutable_cpu_data();
        caffe_memset(zero.size(), 0, zero_);
        return zero;
    }
    
    static SyncedMemory one(){
        SyncedMemory one(Limbs * sizeof(uint64_t), false);
        void* one_ = one.mutable_cpu_data();
        uint64_t fq_one[Limbs] = {8505329371266088957UL, 
                                  17002214543764226050UL, 
                                  6865905132761471162UL, 
                                  8632934651105793861UL,
                                  6631298214892334189UL,
                                  1582556514881692819UL};
        memcpy(one_, fq_one, one.size());
        return one;
    }

    static bool is_equal(SyncedMemory a, SyncedMemory b){
        void* a_ = a.mutable_cpu_data();
        void* b_ = b.mutable_cpu_data();
        bool equal = true;
        for(int i=0; i<Limbs; i++){
            equal = equal && (static_cast<uint64_t*>(a_)[i] == static_cast<uint64_t*>(b_)[i]);
        }
        return equal;
    }
};