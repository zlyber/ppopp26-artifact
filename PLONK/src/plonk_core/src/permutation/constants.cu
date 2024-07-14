#include "PLONK/src/bls12_381/fr.hpp"
#include "PLONK/src/bls12_381/fq.hpp"
SyncedMemory& K1() {
    
    SyncedMemory& K1 = fr::make_tensor(7);
    return K1;
}

SyncedMemory& K2() {
    
    SyncedMemory& K2 = fr::make_tensor(13);
    return K2;
}

SyncedMemory& K3() {
    
    SyncedMemory& K3 = fr::make_tensor(17);
    return K3;
}