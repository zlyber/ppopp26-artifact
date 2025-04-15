#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <cassert>
#include <type_traits>
#include "transcript/flags.hpp"
#include "structure.cuh"
#include "bls12_381/fr.cuh"
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

template<typename T, typename flag>
void serialize(std::vector<uint8_t>& buffer, T item, flag flags = EmptyFlags(0), size_t offset = 0) {
    if constexpr(is_synced_memory<T>::value) {
        assert(flags.BIT_SIZE <= 8 && "not enough space");
        SyncedMemory item_base = to_base(item);
        void* item_ = item_base.mutable_cpu_data();
        memcpy(buffer.data() + offset, item_, item.size());
        buffer.back() |= flags.u8_bitmask();
        return;
    }
    else if constexpr(std::is_same_v<T, uint64_t>) {
        memcpy(buffer.data() + offset, &item, 8);
        return;
    }
    else if constexpr(is_btree_map<T>::value) {
        uint64_t len = 1;
        serialize(buffer, len, EmptyFlags(0));
        serialize(buffer, item.pos, EmptyFlags(0), 8);
        serialize(buffer, item.item, EmptyFlags(0), 16);
        return;
    }
    else if constexpr(is_affine_point_g1<T>::value) {
        if (AffinePointG1::is_zero(item)) {
            SWFlags flags = SWFlags::infinity();
            serialize(buffer, fq::zero(), flags);
        } else {
            SyncedMemory neg_a = neg_mod(item.y);
            SyncedMemory a = to_base(item.y);
            SyncedMemory b = to_base(neg_a);
            SWFlags flags = SWFlags::from_y_sign(gt_zkp(a, b));
            serialize(buffer, item.x, flags);
        }
        return;
    }
    else {
    throw std::runtime_error("unsupported type");
    }
}
