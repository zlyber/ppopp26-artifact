#include "PLONK/src/bls12_381/fr.hpp"
#include "PLONK/src/bls12_381/fq.hpp"
SyncedMemory& K1() {
    // 创建一个大小为7的张量
    SyncedMemory& K1 = fr::make_tensor(7);
    return K1;
}

SyncedMemory& K2() {
    // 创建一个大小为13的张量
    SyncedMemory& K2 = fr::make_tensor(13);
    return K2;
}

SyncedMemory& K3() {
    // 创建一个大小为17的张量
    SyncedMemory& K3 = fr::make_tensor(17);
    return K3;
}