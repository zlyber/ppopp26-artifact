#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <cassert>
#include <type_traits>
#include "PLONK/src/transcript/flags.hpp"
#include "PLONK/src/structure.cuh"
#include "PLONK/src/bls12_381/fr.hpp"
using namespace caffe; 

// type traits
template <typename T>
struct is_synced_memory : std::false_type {};

template <typename T>
struct is_affine_point_g1 : std::false_type {};

template <typename T>
struct is_btree_map : std::false_type {};

template <>
struct is_synced_memory<SyncedMemory> : std::true_type {};

template <>
struct is_affine_point_g1<AffinePointG1> : std::true_type {};

template <>
struct is_btree_map<BTreeMap> : std::true_type {};

template<typename T>
void serialize(uint8_t* buffer, T& item, int flag = EmptyFlags::BIT_SIZE) {
    if constexpr(is_synced_memory<T>::value) {
        assert(flag <= 8 && "not enough space");
        SyncedMemory& item_base = to_base(item);
        void* item_ = item_base.mutable_cpu_data();
        memcpy(buffer, item_, item.size());
        buffer[-1] |= flag;
        return;
    }
    else if constexpr(std::is_same_v<T, uint64_t>) {
        memcpy(buffer, &item, 8);
        return;
    }
    else if constexpr(is_btree_map<T>::value) {
        uint64_t len = 1;
        serialize(buffer, len);
        serialize(buffer + 8, item.pos);
        serialize(buffer + 16, item.item);
        return;
    }
    else if constexpr(is_affine_point_g1<T>::value) {
        if (AffinePointG1::is_zero(item)) {
            flag = SWFlags::infinity().flag;
            serialize(buffer, fq::zero(), flag);
        } else {
            SyncedMemory& a = to_base(item.y);
            SyncedMemory& b = to_base(neg_mod(item.y));
            flag = SWFlags::from_y_sign(gt_zkp(a, b)).flag;
            serialize(buffer, item.x, flag);
        }
        return;
    }
    else {
    throw std::runtime_error("unsupported type");
    }
}

// Deserialize function
SyncedMemory& deserialize(uint8_t* x, size_t length) {
    assert(EmptyFlags::BIT_SIZE <= 8 && "empty flags too large");

    uint8_t aligned[64] = {0};
    memcpy(aligned, x, length);
    memset(aligned + length, 0, 64 - length);

    SyncedMemory scalar_in_uint64(sizeof(uint64_t)*fr::Limbs);
    void* scalar_ = scalar_in_uint64.mutable_cpu_data();
    memcpy(scalar_, aligned, sizeof(scalar_in_uint64));
    return to_mont(scalar_in_uint64);
}
