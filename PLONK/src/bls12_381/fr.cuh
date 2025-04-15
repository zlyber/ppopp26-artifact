#pragma once
#include "../../../caffe/interface.hpp"
#include <cassert>
#include <limits>
#include "../../utils/function.cuh"

class fr{
    public:
    
    static constexpr int MODULUS_BITS = 255;
    static constexpr int BYTE_SIZE = (MODULUS_BITS + 7) / 8;
    static constexpr int Limbs = (MODULUS_BITS + 63) / 64;
    static constexpr int TWO_ADICITY = 32;
    static constexpr int REPR_SHAVE_BITS = 1;

    static SyncedMemory zero() {
        SyncedMemory zero(Limbs * sizeof(uint64_t));
        void* zero_ = zero.mutable_cpu_data();
        caffe_memset(zero.size(), 0, zero_);
        return zero;
    }
    
    static SyncedMemory one() {
        SyncedMemory one(Limbs * sizeof(uint64_t));
        void* one_ = one.mutable_cpu_data();
        uint64_t fr_one[Limbs] = {8589934590UL, 6378425256633387010UL, 11064306276430008309UL, 1739710354780652911UL};
        memcpy(one_,fr_one,one.size());
        return one;
    }

    static SyncedMemory MODULUS() {
        SyncedMemory mod(Limbs * sizeof(uint64_t));
        void* mod_ = mod.mutable_cpu_data();
        uint64_t fr_mod[Limbs] = {18446744069414584321UL, 6034159408538082302UL, 3691218898639771653UL, 8353516859464449352UL};
        memcpy(mod_,fr_mod,mod.size());
        return mod;
    }

    static SyncedMemory TWO_ADIC_ROOT_OF_UNITY() {
        SyncedMemory two_adic(Limbs * sizeof(uint64_t));
        void* two_adic_ = two_adic.mutable_cpu_data();
        uint64_t fr_two_adic[Limbs] = {13381757501831005802UL, 6564924994866501612UL, 789602057691799140UL, 6625830629041353339UL};
        memcpy(two_adic_,fr_two_adic,two_adic.size());
        return two_adic;
    }

    static SyncedMemory GENERATOR() {
        SyncedMemory generator(Limbs * sizeof(uint64_t));
        void* generator_ = generator.mutable_cpu_data();
        uint64_t fr_generator[Limbs] = {64424509425UL, 1721329240476523535UL, 18418692815241631664UL, 3824455624000121028UL};
        memcpy(generator_,fr_generator,generator.size());
        return generator;
    }

    static SyncedMemory make_tensor(uint64_t x, int n = 1) {
        assert(x < std::numeric_limits<uint64_t>::max());
        SyncedMemory tensor(Limbs * sizeof(uint64_t));
        void* tensor_ = tensor.mutable_cpu_data();
        caffe_memset(tensor.size(), 0, tensor_);
        memcpy(tensor_, &x, sizeof(uint64_t));
        return to_mont(tensor);
    }

    static bool is_equal(SyncedMemory a, SyncedMemory b){
        void* a_ = a.mutable_cpu_data();
        void* b_ = b.mutable_cpu_data();
        bool equal = false;
        for(int i=0; i<Limbs; i++){
            equal = (reinterpret_cast<uint64_t*>(a_)[i] == reinterpret_cast<uint64_t*>(b_)[i]);
            if (!equal){
                break;
            }
        }
        return equal;
    }
};