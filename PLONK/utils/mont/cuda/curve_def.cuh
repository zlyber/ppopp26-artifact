#pragma once

#include "ff/bls12-377.hpp"
#include "ff/bls12-381.hpp"

namespace cuda{
    using fr = BLS12_381_Fr_G1;
    using fq = BLS12_381_Fq_G1;
    constexpr int fr_LIMBS = 4;
    constexpr int fq_LIMBS = 6;
    constexpr int fr_BITS = 255;
    constexpr int fq_BITS = 381;

    constexpr uint32_t num_threads() { return 256; }
    constexpr int thread_work_size() { return 4; }
    constexpr int block_work_size() { return thread_work_size() * num_threads(); }

    template <typename T>
    int num_uint64(T* a) { return (T::bit_length() + 63)/64;}
}
