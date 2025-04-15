#pragma once

#include "ff/bls12-377.hpp"
#include "ff/bls12-381.hpp"

namespace cpu{
    constexpr int fr_LIMBS = 4;
    constexpr int fq_LIMBS = 6;
    constexpr int fr_BITS = 255;
    constexpr int fq_BITS = 381;
    using fr = BLS12_381_Fr_G1;
    using fq = BLS12_381_Fq_G1;
    
    template <typename T>
    int num_uint64(T* a) { return (T::bit_length() + 63)/64;}
}//namespace::cpu